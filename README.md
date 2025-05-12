# Time-Series Forecasting Pipeline

This project implements a reproducible, parallelized time-series forecasting pipeline using RNN-based models (RNN, LSTM, SLSTM, PhasedLSTM) alongside a naive‐forecast baseline.

## Key Components

- **`run_workflow_main.py`**  
  Central CLI script that spawns separate Python processes to train and evaluate each model under cross-validation for full reproducibility and parallelism.

- **`models_utils.py`**  
  Defines RNN-based model classes, their architectures, hyperparameter search spaces, and training/inference helper functions.

- **`workflowfunction_utils.py`**  
  Provides data loading, train/validation/test splitting, scaling, PCA, PyTorch DataLoader construction, metric logging (RMSE matrix), global Friedman χ² testing, and the `cross_validate_time_series()` orchestrator.

- **`naive_forecast.py`**  
  Supplies a simple forecasting baseline.

## Installation

```bash
conda env create -f requirements.yml
conda activate ep_forecasting_env



## Usage in terminal with command line arguments 

Run models: 
Run Model on device:
    python run_workflow_main.py --mode model/group --device mps/cpu

Example: LSTM on mps 
  python run_workflow_main.py --mode lstm  --device mps

Run global Friedman-test 

    python run_workflow_main.py --global_test


