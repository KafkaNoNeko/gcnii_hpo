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

import os
import tempfile
import csv
from subprocess import call

from ray.train.torch import get_device
from ray import train as ray_train
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.train import Checkpoint

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.nevergrad import NevergradSearch
import nevergrad as ng
import optuna

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


def test_step(split,data_name,config,variant,ckpt_dir,args):
    # use gpu
    assert torch.cuda.is_available()
    device = get_device()
    assert device == torch.device("cuda:0")

    splitstr = 'splits/'+data_name+'_split_0.6_0.2_'+str(split)+'.npz'

    adj, features, labels, _ , _ , idx_test, num_features, num_labels = full_load_data(data_name,splitstr)
    features = features.to(device)
    adj = adj.to(device)

    model = GCNII(nfeat=num_features,
        nlayers=args.num_layers,
        nhidden=args.hidden_channels,
        nclass=num_labels,
        dropout=config["dropout"],
        lamda = args.lamda, 
        alpha=config["alpha"],
        variant=variant,
        act_fn="relu").to(device)

    with ckpt_dir.as_directory() as checkpoint_dir:
        checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))    
        model.load_state_dict(checkpoint_dict["model_state_dict"])
    
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()


def train_gcn(config, data, variant, split, args):
    # use gpu
    assert torch.cuda.is_available()
    device = get_device()
    assert device == torch.device("cuda:0")

    # set seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    
    data_name = data
    splitstr = 'splits/'+data_name+'_split_0.6_0.2_'+str(split)+'.npz'

    adj, features, labels, idx_train, idx_val, _ , num_features, num_labels = full_load_data(data_name,splitstr)
    features = features.to(device)
    adj = adj.to(device)

    # define model
    model = GCNII(nfeat=num_features,
        nlayers=args.num_layers,
        nhidden=args.hidden_channels,
        nclass=num_labels,
        dropout=config["dropout"],
        lamda = args.lamda, 
        alpha=config["alpha"],
        variant=variant,
        act_fn="relu").to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                weight_decay=config["weight_decay"])

    # Load the checkpoint, if there is any.
    checkpoint = ray_train.get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state_dict"])

            # Load optimizer state (needed since we are using momentum),
            # then set the `lr` and `momentum` according to the config.
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            for param_group in optimizer.param_groups:
                if "lr" in config:
                    param_group["lr"] = config["lr"]
                if "momentum" in config:
                    param_group["momentum"] = config["momentum"]

    for epoch in range(start_epoch, args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,features,labels,adj,idx_train)
        loss_val,acc_val = validate_step(model,features,labels,adj,idx_val)

        # Create the checkpoint if needed and report metrics
        metrics = {"val_acc":acc_val, 
                    "val_loss":loss_val,
                    "epoch": epoch,
                    "train_acc": acc_tra,
                    "train_loss":loss_tra}

        if epoch % args.ckpt_interval == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                ray_train.report(metrics,  
                                checkpoint=Checkpoint.from_directory(tempdir),)
        else:
            ray_train.report(metrics)


def train_one_split(split, args):
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    perturbation_interval = args.ckpt_interval                              # for PBT. Recommended to be equal to the checkpoint interval.
    num_samples = args.num_samples

    # set the search space
    # RayTune docs: "it is beneficial that the seeds differ between different training runs."
    # https://docs.ray.io/en/latest/tune/faq.html#how-can-i-reproduce-experiments

    if args.algo == "grid":
        search_space = {
            "seed": tune.randint(0, 10000),
            "lr": tune.grid_search([1e-3, 1e-2]),                      
            "alpha": tune.grid_search([0.1, 0.5, 0.9]),                                              
            "dropout": tune.grid_search([0.3, 0.5]), 
            "weight_decay": tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6]),    # (L2 loss on parameters)   
        }
        num_samples = 1
    else:
        search_space = {
            "seed": tune.randint(0, 10000),
            "lr": tune.loguniform(1e-5, 1e-1),                                           
            "alpha": tune.uniform(0.1, 0.9),                                                  
            "dropout": tune.quniform(0, 0.5, 0.05),                                     
            "weight_decay": tune.loguniform(1e-7, 1e-4),                    # (L2 loss on parameters)   
        }

    # search algorithms
    if args.algo == "tpe":
        search_algo = HyperOptSearch()
    elif args.algo == "bohb":
        search_algo = tune.search.ConcurrencyLimiter(TuneBOHB(), max_concurrent=4)
        args.scheduler = "HyperBandForBOHB"                                 # switch to required scheduler for BOHB
    elif args.algo == "grid" or args.algo == "random":
        search_algo = None
    elif args.algo == "cmaes":
        cmaes_sampler = optuna.samplers.CmaEsSampler()
        search_algo = tune.search.ConcurrencyLimiter(OptunaSearch(sampler=cmaes_sampler), max_concurrent=4)
    elif args.algo == "pso":
        optimizer = ng.optimizers.registry["PSO"]
        search_algo = tune.search.ConcurrencyLimiter(NevergradSearch(optimizer=optimizer), max_concurrent=4)

    # schedulers
    if args.scheduler == "asha":
        scheduler = ASHAScheduler()
    elif args.scheduler == "pbt":
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=perturbation_interval,
            hyperparam_mutations=search_space,
        )
        
        # search algorithms cannot be used with PopulationBasedTraining schedulers
        search_algo = None
        args.algo = "None"
    elif args.scheduler == "HyperBandForBOHB":
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=100,
            reduction_factor=4,
            stop_last_trials=False,
        )
        
    # define trainable
    # not using cpu since the SMAC code only uses the gpu
    trainable_with_gpu = tune.with_resources(train_gcn, {"gpu": 1})

    tuner = tune.Tuner(
        tune.with_parameters(
            trainable_with_gpu, 
            data=args.data,
            variant= args.variant,
            split=split,
            args=args),
        tune_config=tune.TuneConfig(
            metric="val_loss", 
            mode="min",
            num_samples=num_samples,
            search_alg=search_algo,
            scheduler=scheduler,
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    best_result  = results.get_best_result(metric="val_loss", mode="min")
    
    return best_result

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--data', default='cora', help='Dataset.')
    parser.add_argument('--algo', default='tpe', help='Search algorithm.')
    parser.add_argument('--scheduler', default='asha', help='Scheduler.')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    parser.add_argument('--num_layers', type=int, default=64, help='Number of layers.')
    parser.add_argument('--hidden_channels', type=int, default=64, help='hidden dimensions.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--ckpt_interval', type=int, default=5, help='Checkpoint interval. Also defines the perturbation interval for PBT.')
    parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
    parser.add_argument('--num_samples', type=int, default=150, help='Number of samples to be used by the HPO algorithm.')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    start_time = time.time()

    # focusing on one specific split instead of repeating for 10 splits.
    split = 8
    best_results = []
    # for split in range(10):
    #     best_results.append(train_one_split(split, args))
    best_results.append(train_one_split(split, args))
        
    # identify the best model
    best_val_losses = [x.metrics['val_loss'] for x in best_results]
    best_result_idx = np.argmin(best_val_losses)
    best_result = best_results[best_result_idx]

    acc_list = []
    for split in range(10):
        # Evaluate on test set (10 splits)
        test_acc = test_step(split, 
                            args.data, 
                            best_result.metrics['config'], 
                            args.variant,
                            best_result.checkpoint,
                            args)[1]
        acc_list.append(test_acc)
    
    time_taken = time.time() - start_time
    mean_test_acc = np.mean(acc_list)

    print("Time taken: {:.4f}s".format(time_taken))
    print("Mean Test acc.:{:.2f}".format(mean_test_acc))

    # write to csv file
    with open('logs.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        data_list = [args.data, args.algo, args.scheduler, time_taken, mean_test_acc, best_result.config, best_result.path]
        data_list.extend(acc_list)
        # dataset, algorithm, scheduler, time taken, mean test acc, best result config, best result path, test accuracies for each run
        writer.writerow([x for x in data_list]) 

if __name__ == "__main__":
    main()