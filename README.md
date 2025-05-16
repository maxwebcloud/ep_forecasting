# Time-Series Forecasting Pipeline

This project implements a reproducible time-series forecasting pipeline using RNN-based models
(RNN, LSTM, SLSTM, PhasedLSTM) alongside a naive-forecast baseline.

Data preprocessing workflow used from:
• Roussis, D., 2021. Electricity Price Forecasting with DNNs (+ EDA). Kaggle Notebook,
  available at: https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda (Accessed May 12, 2025)

PLSTM implementation adapted from:
• YellowRobot Share, 2022. Index of /1658316327-phased (model_3_PLSTM.py),
  available at: http://share.yellowrobot.xyz/1658316327-phased/ (Accessed May 12, 2025)
• YellowRobot.XYZ, 2021. Diff #21 – PyTorch Phased-LSTM implementation from scratch,
  available at: https://www.youtube.com/watch?v=So4Ro2CfZZY (Accessed May 12, 2025)



## Key Components

- **`run_workflow_main.py`**  
  Central CLI script that spawns separate Python processes to train and evaluate each model under cross-validation for full reproducibility

- **`models_utils.py`**  
  Defines RNN-based model classes, their architectures, hyperparameter search spaces

- **`workflowfunction_utils.py`**  
  Provides data loading, train/validation/test splitting, scaling, PCA, PyTorch DataLoader construction, metrics (RMSE matrix), in the `cross_validate_time_series()` wrapper

- **`naive_forecast.py`**  
  Supplies a simple forecasting baseline.

## Installation

bash
conda env create -f requirements.yml
conda activate ep_forecasting_env

## Usage 
In terminal with command line arguments 

Run models: 
Run Model on device:
    python run_workflow_main.py --mode model/group --device mps/cpu

Example: LSTM on mps 
  python run_workflow_main.py --mode lstm  --device mps

Run global Friedman-test 

    python run_workflow_main.py --global_test


## Reproducibility – Hardware Assumptions
	•	LSTM / SLSTM: Apple Silicon GPU (MPS backend)
	•	RNN / PLSTM: CPU

Running on other hardware or back-ends might yield performance-measurement differences