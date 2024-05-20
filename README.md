# Benchmarking Hyperparameter Optimization Methods for Node Classification

This repository contains a hyperparameter optimization implementation for the [GCNII](https://github.com/chennnM/GCNII) model proposed by Chen *et al* as described in my honours thesis.

We examine the performance of various hyperparameter optimization algorithms on the performance of the GCNII model and compare our results to those reported in the original paper which only performs a grid search on a small search space. Most of the implementation has been done using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). [SMAC3](https://automl.github.io/SMAC3/main/) is used to implement SMAC.

We compare the effects of the nature of the search space as well as the number of samples used. Details about the implementation decisions made are found in the thesis.

## Algorithms and Libraries

Grid search, random search, and several Bayesian- and heuristic- based methods were used.

| Algorithm     | Scheduler        | Library              |
|---------------|------------------|----------------------|
| Random Search | ASHA             | RayTune              |
| Grid Search   | ASHA             | RayTune              |
| TPE           | ASHA             | Hyperopt (RayTune)   |
| PSO           | ASHA             | Nevergrad (RayTune)  |
| CMA-ES        | ASHA             | Optuna (RayTune)     |
| BOHB          | HyperBandForBOHB | HpBandSter (RayTune) |
| PBT           | -                | RayTune              |
| SMAC          | Hyperband        | SMAC3                |

## Search Spaces

The details for the search spaces used are given below. Note that grid search uses a subset of the search spaces due to limited time and resources.

The notation $[a,s,b]$ denotes the set $\{a+is: i=0..(b-a)/s\}$.

| Hyperparameter       | Type        | Mixed                  | Continuous           |
|----------------------|-------------|------------------------|----------------------|
| Learning rate        | Real-valued | $[1e^{-5}, 1e^{-1}]$   | $[1e^{-5}, 1e^{-1}]$ |
| Dropout              | Real-valued | $[0, 0.05, 0.5]$       | $[0, 0.05, 0.5]$     |
| Hidden dimension     | Discrete    | $\{64,256\}$           | 64                   |
| No. of layers        | Discrete    | $\{4, 8, 16, 32, 64\}$ | 64                   |
| $\alpha_l$           | Real-valued | $[0.1,0.9]$            | $[0.1,0.9]$          |
| $\lambda$            | Real-valued | $\{0.5, 1.0, 1.5\}$    | 0.5                  |
| $L_2$ regularization | Real-valued | $[1e^{-7}, 1e^{-4}]$   | $[1e^{-7}, 1e^{-4}]$ |
| Activation function  | Categorical | \{ReLU, PReLU, ELU\}   | ReLU                 |
| Optimizer            | Categorical | \{Adam, SGD\}          | Adam                 |
| Momentum (SGD)       | Real-valued | $(0,1)$                | -                    |

## Main Notes on Implementation details

- We adapted the original full-supervised GCNII code for a fairer comparison.
  - We fix the number of epochs to 100 instead of using 1500 since the GCNII model proved to generalise well on the test set using only a small number of training epochs.
  - Similarly, instead of training the model using all 10 splits of a dataset, we use only one.
- For simplicity, no conditional hyperparameters were used.
- The number of samples per run of each HPO algorithm is fixed to 150.

## Results

### Mean classification accuracy and run times on a fixed number of samples ($n=150$).

**Table 1**: Mean classification accuracy (\%) when using a mixed search space. Each algorithm (other than PBT) has been used in conjunction with an early stopping algorithm.

| Method         | Cora            | Cite.           | Pubm.           | Cham.           | Corn.           | Texa.           | Wisc.           |
|:---------------|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|
| RS             | 91.21           | 82.67           | 90.91           | 43.12           | 86.94           | 85.32           | 86.27           |
| GS             | 90.86           | **85.81**       | 90.71           | **69.99**       | 90.99           | 87.12           | 88.17           |
| TPE            | 90.43           | 82.73           | 90.57           | 58.12           | 81.98           | 87.75           | 89.80           |
| PSO            | **91.52**       | 81.68           | 91.06           | 38.66           | 84.23           | 88.29           | 81.90           |
| CMA-ES         | 87.45           | 81.49           | **91.51**       | 52.95           | 88.92           | 85.77           | 84.58           |
| BOHB           | 90.17           | 84.99           | 91.29           | 67.05           | 87.39           | 83.78           | 87.97           |
| PBT            | 91.27           | 83.86           | 90.65           | 67.06           | 90.09           | 86.40           | 84.97           |
| SMAC           | 91.10           | 83.56           | 91.11           | 66.64           | **92.52**       | **89.01**       | **90.92**       |
| Orig. GCNII    | 88.49           | 77.08           | 89.57           | 60.61           | 74.86           | 69.46           | 74.12           |

**Table 2**: Mean classification accuracy (\%) averaged over 3 runs when using a search space only consisting of continuous HPs. Each algorithm (other than PBT) has been used in conjunction with an early stopping algorithm.

| Method         | Cora            | Cite.           | Pubm.           | Cham.           | Corn.           | Texa.           | Wisc.           |
|:---------------|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|
| RS             | 91.95           | 82.17           | 91.55           | **73.97**       | 87.66           | 84.32           | 86.08           |
| GS             | 88.80           | 83.77           | 88.36           | 61.68           | 85.14           | 85.50           | 82.03           |
| TPE            | 92.19           | **86.14**       | **91.57**       | 70.37           | 87.39           | 85.05           | 86.47           |
| PSO            | 91.19           | 84.09           | 91.14           | 57.15           | 87.03           | 81.98           | 85.82           |
| CMA-ES         | **92.33**       | 83.66           | 91.63           | 65.70           | 85.41           | 82.70           | 85.36           |
| BOHB           | 91.35           | 81.95           | 91.14           | 58.76           | 87.39           | 86.31           | 85.69           |
| PBT            | 91.31           | 83.07           | 90.92           | 66.86           | 86.40           | 86.85           | 87.65           |
| SMAC           | 91.24           | 84.14           | 91.41           | 70.35           | **92.16**       | **89.82**       | **91.05**       |
| Orig. GCNII    | 88.49           | 77.08           | 89.57           | 60.61           | 74.86           | 69.46           | 74.12           |

**Table 3**: Average run time in seconds (out of 3 runs) when using a mixed search space for each dataset and a fixed number of samples ($n=150$). Each algorithm (other than PBT which is itself implemented as a scheduler) has been used in conjunction with an early stopping algorithm.

| Method | Cora  | Cite. | Pubm.  | Cham. | Corn. | Texa. | Wisc. |
|:-------|------:|------:|-------:|------:|------:|------:|------:|
| RS     | 272   | 275   | 489    | 279   | 189   | 199   | 204   |
| GS     | 2,024 | 2,596 | 3,982  | 2,524 | 1,312 | 1,431 | 1,547 |
| TPE    | 307   | 364   | 645    | 386   | 234   | 372   | 391   |
| PSO    | 267   | 280   | 537    | 309   | 161   | 133   | 192   |
| CMA-ES | 378   | 411   | 731    | 347   | 210   | 344   | 260   |
| BOHB   | 826   | 958   | 1,580  | 1,298 | 887   | 1,089 | 896   |
| PBT    | 5,003 | 6,466 | 11,804 | 7,217 | 4,289 | 4,324 | 4,250 |
| SMAC   | 1,809 | 1,896 | 1,995  | 1,637 | 1,721 | 1,476 | 1,807 |

**Table 4**: Average run time in seconds (out of 3 runs) when using a continuous search space for each dataset and a fixed number of samples ($n=150$). Each algorithm (other than PBT which is itself implemented as a scheduler) has been used in conjunction with an early stopping algorithm.

| Method | Cora   | Cite.  | Pubm.  | Cham.  | Corn.  | Texa.  | Wisc.  |
|:-------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| RS     | 448    | 490    | 736    | 461    | 443    | 338    | 350    |
| GS     | 430    | 409    | 437    | 370    | 269    | 343    | 312    |
| TPE    | 429    | 671    | 813    | 456    | 410    | 666    | 530    |
| PSO    | 417    | 425    | 625    | 412    | 375    | 353    | 291    |
| CMA-ES | 843    | 646    | 1311   | 757    | 671    | 276    | 946    |
| BOHB   | 2,295  | 2,394  | 2,529  | 2,445  | 2,401  | 2,433  | 2,428  |
| PBT    | 10,303 | 11,381 | 15,390 | 12,373 | 10,107 | 10,165 | 10,183 |
| SMAC   | 3,822  | 3,462  | 5,879  | 4,451  | 3,065  | 3,146  | 2,399  |

### Effects of the number of samples

**Table 5**: Mean classification accuracy (\%) averaged over 3 runs when using a mixed search space on the Chameleon dataset for different number of samples.

| Method | $n=8$           | $n=15$          | $n=30$          | $n=60$          | $n=120$         | $n=150$         |
|:-------|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|
| RS     | 23.50           | 54.24           | 53.93           | 42.02           | 66.92           | 43.12           |
| TPE    | **65.10**       | 32.10           | 41.58           | **66.54**       | 61.51           | 58.12           |
| PSO    | 46.26           | 53.36           | 61.35           | 57.08           | 41.37           | 38.66           |
| CMA-ES | 51.94           | 41.55           | 38.45           | 59.36           | 53.09           | 52.95           |
| BOHB   | 44.77           | 47.11           | 46.67           | 37.19           | **69.79**       | 67.05           |
| PBT    | 28.57           | 63.07           | **66.86**       | 62.48           | 68.21           | **67.06**       |
| SMAC   | 64.74           | **67.53**       | 62.38           | 61.70           | 68.37           | 66.64           |

**Table 6**: Average run time in seconds averaged over 3 runs when using a mixed search space on the Chameleon dataset for different number of samples.

| Method | $n=8$ | $n=15$ | $n=30$  | $n=60$  | $n=120$ | $n=150$ |
|:-------|------:|-------:|--------:|--------:|--------:|--------:|
| RS     | 44    | 122    | 92      | 157     | 278     | 279     |
| TPE    | 60    | 88     | 151     | 182     | 296     | 386     |
| PSO    | 63    | 70     | 120     | 142     | 297     | 309     |
| CMA-ES | 90    | 93     | 145     | 155     | 297     | 347     |
| BOHB   | 81    | 101    | 118     | 150     | 656     | 1,289   |
| PBT    | 446   | 751    | 1,483   | 2,914   | 5,929   | 7,217   |
| SMAC   | 73    | 114    | 236     | 473     | 844     | 1,637   |

## Code Structure

```
.
├── data                            # Cora, Citeseer, Pubmed datasets
├── logs                            # Raw logs of HPO run results
├── new_data                        # Chameleon, Cornell, Texas, and Wisconsin datasets
├── splits                          # Dataset splits
├── .gitignore
├── commands.sh                     # Commands for reproducing the experiments
├── model.py                        # GCNII model file
├── process.py                      # Data loaders
├── ray_cont.py                     # Ray Tune implementation for a continuous search space
├── ray_mixed.py                    # Ray Tune implementation for a mixed search space
├── README.md                       # This file!
├── requirements.txt
├── smac_cont.py                    # SMAC3 implementation for a continuous search space
├── smac_mixed.py                   # SMAC3 implementation for a mixed search space
└── utils.py                        # Helper functions
```

## Usage

- Create a virtual environment and install the requirements found in `requirements.txt`.
- Run `sh commands.sh` to replicate one run of the experiments or run any single command from the file.
