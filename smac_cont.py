from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model import *

import csv
from subprocess import call

import ConfigSpace as CS
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband

cudaid = "cuda:0"
device = torch.device(cudaid)

def train_step(model,optimizer,features,labels,adj,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,features,labels,adj,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(split,data_name,config,checkpt_file,variant):
    # use gpu
    assert torch.cuda.is_available()

    splitstr = 'splits/'+data_name+'_split_0.6_0.2_'+str(split)+'.npz'

    adj, features, labels, _ , _ , idx_test, num_features, num_labels = full_load_data(data_name,splitstr)
    features = features.to(device)
    adj = adj.to(device)

    model = GCNII(nfeat=num_features,
        nlayers=64,
        nhidden=64,
        nclass=num_labels,
        dropout=config["dropout"],
        lamda = 0.5, 
        alpha=config["alpha"],
        variant=variant,
        act_fn="relu").to(device)

    model.load_state_dict(torch.load(checkpt_file))

    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()

def train_gcn(config, seed=42, budget=100):
    # use gpu
    assert torch.cuda.is_available()
    
    data_name = config["data"]
    splitstr = 'splits/'+data_name+'_split_0.6_0.2_'+str(config["split"])+'.npz'

    adj, features, labels, idx_train, idx_val, _ , num_features, num_labels = full_load_data(data_name,splitstr)
    features = features.to(device)
    adj = adj.to(device)

    # define model
    # not sure how to pass arguments using smac, so hardcoding values here :(
    model = GCNII(nfeat=num_features,
        nlayers=64,
        nhidden=64,
        nclass=num_labels,
        dropout=config["dropout"],
        lamda = 0.5, 
        alpha=config["alpha"],
        variant=config["variant"] == 'True',
        act_fn="relu").to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                weight_decay=config["weight_decay"])

    # Terrible way of checkpointing, but this is the only way I could do it
    # Cannot add a checkpoint to the search space since the value cannot be randomised (even after calling random)
    # And the train function seems to be only able to return the error and nothing else.

    ckpt_str = str(config["lr"]) + str(config["alpha"]) + \
                str(config["dropout"]) + str(config["weight_decay"]) + data_name
    checkpt_file = 'pretrained/'+ckpt_str+'.pt'

    best = 999999999
    epochs =100
    for epoch in range(epochs):
        loss_tra,acc_tra = train_step(model,optimizer,features,labels,adj,idx_train)
        loss_val,acc_val = validate_step(model,features,labels,adj,idx_val)

        if loss_val < best:
            torch.save(model.state_dict(), checkpt_file)
            best = loss_val

    return best


def train_one_split(split, args):
    cs = CS.ConfigurationSpace(
        seed=args.seed,
        space={
            "lr": CS.Float('lr', bounds=(1e-5, 1e-1), log=True),
            "alpha": CS.Float('alpha', bounds=(0.1, 0.9)),
            "dropout": CS.Float('dropout', (0, 0.5), q=0.05),
            "weight_decay": CS.Float('weight_decay', bounds=(1e-7, 1e-4), log=True),
            "epochs": 100,
            "data": args.data,
            "variant": str(args.variant),
            "split": split,
        }
    )

    scenario = Scenario(
        configspace=cs,
        walltime_limit=7200,
        n_trials=args.num_samples,
        min_budget=20,  # train for at least 20 epochs
        max_budget=100, # train for max 100 epochs
        n_workers=8,
    )

    # Run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    # Create the intensifier (scheduler)
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario,
        train_gcn,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    )

    incumbent = smac.optimize()

    return incumbent

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--data', default='cora', help='Dataset.')
    parser.add_argument('--algo', default='smac', help='Search algorithm.')                     # kept as an arg for convenience
    parser.add_argument('--scheduler', default='hyperband', help='Scheduler.')                  # kept as an arg for convenience
    parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
    parser.add_argument('--num_samples', type=int, default=150, help='Number of samples to be used by the HPO algorithm.')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    start_time = time.time()

    split = 8
    best_result = train_one_split(split, args)

    ckpt_str = str(best_result["lr"]) + str(best_result["alpha"]) + \
                str(best_result["dropout"]) + str(best_result["weight_decay"]) + args.data
    checkpt_file = 'pretrained/'+ckpt_str+'.pt'

    acc_list = []
    for split in range(10):
        # Evaluate on test set
        test_acc = test_step(split, 
                            args.data, 
                            best_result, 
                            checkpt_file,
                            args.variant)[1]
        acc_list.append(test_acc)
    
    time_taken = time.time() - start_time
    mean_test_acc = np.mean(acc_list)

    print("Time taken: {:.4f}s".format(time_taken))
    print("Mean Test acc.:{:.2f}".format(mean_test_acc))

    # write to csv file
    with open('logs.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        data_list = [args.data, args.algo, args.scheduler, time_taken, mean_test_acc, dict(best_result), checkpt_file]
        data_list.extend(acc_list)
        # dataset, algorithm, scheduler, time taken, mean test acc, best result config, best result path, test accuracies for each run
        writer.writerow([x for x in data_list]) 

if __name__ == "__main__":
    main()