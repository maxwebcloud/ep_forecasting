"""
run_all_models_cv.py

This script runs selected model classes (defined in models_utils.py) on a given 
time series dataset and evaluates them using cross-validation.

The core process is handled by the function `cross_validate_time_series` 
(from cv_right.py), which performs preprocessing, sequence creation, 
hyperparameter tuning (via Optuna), training, evaluation, and result saving.

For each model and seed combination, the function:
    - Splits the data into training, validation, and test folds
    - Scales input features and target values (MinMax)
    - Applies PCA for dimensionality reduction
    - Builds sequences
    - Tunes hyperparameters and trains the model 
    - Evaluates performance and stores the best model per seed

Available CLI options:
----------------------
--mode      (str) : Select which model(s) to run. Choices:
    - "standard"   → Runs all four models (rnn, lstm, slstm, plstm)
    - "3models"    → Runs rnn, lstm, and slstm (excludes plstm)
    - "half_a_gpu"     → Runs lstm and slstm (50 / 50 split, LSTM‑heavy)
    - "half_b_cpu"     → Runs rnn and plstm (50 / 50 split, RNN‑heavy)
    - "rnn"        → Runs only the SimpleRNN model
    - "lstm"       → Runs only the LSTM model
    - "slstm"      → Runs only the StackedLSTM model
    - "plstm"      → Runs only the PhasedLSTM model

--device    (str) : Choose the device for training. Choices:
    - "cpu"        → Force model(s) to run on CPU
    - "mps"        → Use Apple MPS backend if available (for Apple Silicon GPUs)

Examples:
---------
Run LSTM on MPS:
    python run_all_models_cv_main.py --mode lstm --device mps

Run all models on CPU:
    python run_all_models_cv_main.py --mode standard --device cpu
"""

# ============================================================================
# Imports
# ============================================================================
import os
from re import S
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import numpy as np
import random
import time
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
from models_utils import *
from workflowfunctions_utils import get_device
from cv_right import cross_validate_time_series

# ============================================================================
# Configuration: Models groups and cv-params 
# ============================================================================
USE_CLI = True

MODELS = {
    "standard": [SimpleRNN, LSTMModel, StackedLSTMModel, PhasedLSTMModel],
    "3models":  [SimpleRNN, LSTMModel, StackedLSTMModel],
    "half_a_gpu":   [LSTMModel, StackedLSTMModel],
    "half_b_cpu":   [SimpleRNN, PhasedLSTMModel],
    "rnn":      [SimpleRNN],
    "lstm":     [LSTMModel],
    "slstm":    [StackedLSTMModel],
    "plstm":    [PhasedLSTMModel],
}

CV_PARAMS = {
    "n_folds": 5,
    "train_size": 0.77,
    "val_size": 0.11,
    "test_size": 0.12,
    "sequence_length": 24,
    "step_size": 1,
    "variance_ratio": 0.8,
    "single_step": True
}

# ============================================================================
# Command Line Arguments
# Enables dynamic configuration via CLI (model group, device)
# ============================================================================
if USE_CLI:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODELS.keys()), default="standard")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    mode = args.mode
    device_choice = args.device
    summary_dir = Path("model_metrics_overview")  
else:
    mode = "standard"
    device_choice = "cpu"
    summary_dir = Path("model_metrics_overview")

selected_models = MODELS[mode]

# ============================================================================
# Seed Initialization
# ============================================================================
SEED = 42
N_SEEDS = 5
random.seed(SEED)
np.random.seed(SEED)
seeds_list = random.sample(range(0, 100), N_SEEDS)

# ============================================================================
# Startup Information
# ============================================================================
def print_settings():
    """Combined output of basic and advanced settings"""
    import inspect
    import cv_right
    import re
    
    print("\n=== Starting run_all_models_cv.py ===")
    print(f"Device selected: {device_choice}")
    print(f"Model group: {mode}")
    
    print("\nSelected models to run:")
    for model in selected_models:
        print(f"  - {model.__name__}")
    
    print("\n" + "="*50)
    print("Settings for: CV, Tuning, Training")
    print("="*50)
    
    # CV-Settings (from CV_PARAMS)
    print("\nCV Settings")
    print(f"  Number of folds:      {CV_PARAMS['n_folds']}")
    print(f"  Train/Val/Test split: {CV_PARAMS['train_size']:.1f}/{CV_PARAMS['val_size']:.1f}/{CV_PARAMS['test_size']:.1f}")
    print(f"  Sequence length:      {CV_PARAMS['sequence_length']}")
    print(f"  Step size:            {CV_PARAMS['step_size']}")
    print(f"  Variance ratio (PCA): {CV_PARAMS['variance_ratio']:.1f}")
    print(f"  Single step forecast: {CV_PARAMS['single_step']}")
    
    # Dynamic settings from cv_right.py
    try:
        # Extract values from cv_right.py functions
        batch_size_source = inspect.getsource(cv_right.get_dataloaders) 
        hyperparameter_tuning_source = inspect.getsource(cv_right.hyperparameter_tuning)
        cv_time_series_source = inspect.getsource(cv_right.cross_validate_time_series)
        
        # Batch size settings
        batch_size_match = re.search(r'batch_size\s*=\s*(\d+)', batch_size_source)
        batch_size = int(batch_size_match.group(1)) if batch_size_match else "N/A"
        
        # Pruner settings
        pruner_match = re.search(r'pruner\s*=\s*optuna\.pruners\.HyperbandPruner\s*\(\s*min_resource\s*=\s*(\d+)\s*,\s*max_resource\s*=\s*(\d+)\s*,\s*reduction_factor\s*=\s*(\d+)', hyperparameter_tuning_source)
        min_resource = int(pruner_match.group(1)) if pruner_match else "N/A"
        max_resource = int(pruner_match.group(2)) if pruner_match else "N/A"
        reduction_factor = int(pruner_match.group(3)) if pruner_match else "N/A"
        
        # n_trials
        n_trials_match = re.search(r'n_trials\s*=\s*(\d+)', hyperparameter_tuning_source)
        n_trials = int(n_trials_match.group(1)) if n_trials_match else "N/A"
        
        # Tuning model_train settings
        hp_epochs_match = re.search(r'model_train\(.*?num_epochs\s*=\s*(\d+)\s*,\s*patience\s*=\s*(\d+)', hyperparameter_tuning_source, re.DOTALL)
        hp_epochs = int(hp_epochs_match.group(1)) if hp_epochs_match else "N/A"
        hp_patience = int(hp_epochs_match.group(2)) if hp_epochs_match else "N/A"
        
        # Final training settings
        final_epochs_match = re.search(r'model_train\(.*?num_epochs\s*=\s*(\d+)\s*,\s*patience\s*=\s*(\d+).*?final\s*=\s*True', cv_time_series_source, re.DOTALL)
        final_epochs = int(final_epochs_match.group(1)) if final_epochs_match else "N/A"
        final_patience = int(final_epochs_match.group(2)) if final_epochs_match else "N/A"


        # Hyperparameter Tuning Settings
        print("\nHyperparameter Tuning")
        print(f"  Number of trials:         {n_trials}")
        print(f"  Pruner min resources:     {min_resource}")
        print(f"  Pruner max resources:     {max_resource}")
        print(f"  Pruner reduction factor:  {reduction_factor}")
        print(f"  Training epochs:          {hp_epochs}")
        print(f"  Early stopping patience:  {hp_patience}")
        
        # Final Model Training
        print("\nFinal Model Training")
        print(f"  Training epochs:          {final_epochs}")
        print(f"  Early stopping patience:  {final_patience}")


        # Selected Batch size Settings
        print("\nSelected Batch size Settings")
        print(f"  Batch Size:               {batch_size}")
    except Exception as e:
        print(f"\n[Warning] Could not extract advanced settings: {str(e)}")
    
    print("\n" + "="*50)

# Output all settings
print_settings()


# ============================================================================
# Data Loading
# ============================================================================
print("\nLoading data and extract feature matrix (X) and target vector (Y)... ")
with open("data/df_final_eng.pkl", "rb") as f:
    df_final = pickle.load(f)

X = df_final[df_final.columns.drop('price actual')].values
y = df_final['price actual'].values.reshape(-1, 1)

# ============================================================================
# Device Setup 
# ============================================================================
use_gpu = device_choice == "mps"
device = get_device(use_gpu=use_gpu)

# ============================================================================
# Model Training & Cross-Validation
# ============================================================================
model_runtimes = {}          # model_name  ->  minutes
start_time = time.time()
results = []

for model_class in selected_models:
    model_name = getattr(model_class, "name", model_class.__name__)
    model_start = time.perf_counter() 
    
    # MPS memory is cleared before each model if using MPS
    if device.type == "mps":
        # Force garbage collection to free up memory
        import gc
        gc.collect()
        torch.mps.empty_cache()
        assert torch.mps.current_allocated_memory() == 0, "⚠️ MPS memory not cleared!"

    # Update the function call to match the signature in cv_right.py
    model_results = cross_validate_time_series(
        models=[model_class],
        seeds=seeds_list,
        X=X,                            # Changed from data=X
        y=y,                            # Changed from target=y
        device=device,                  # Same device for all operations
        train_size=CV_PARAMS["train_size"],
        val_size=CV_PARAMS["val_size"],
        test_size=CV_PARAMS["test_size"],
        sequence_length=CV_PARAMS["sequence_length"],
        step_size=CV_PARAMS["step_size"],
        n_folds=CV_PARAMS["n_folds"],
        variance_ratio=CV_PARAMS["variance_ratio"],
        single_step=CV_PARAMS["single_step"]
    )

    model_end = time.perf_counter()       
    model_runtimes[model_name] = (model_end - model_start) / 60  


    # Handle the results which might be a DataFrame
    if isinstance(model_results, pd.DataFrame):
        results.extend(model_results.to_dict('records'))
    else:
        results.extend(model_results)

# ============================================================================
# Evaluation Summary
# ============================================================================
end_time = time.time()
total_minutes = (end_time - start_time) / 60          # total runtime in minutes
total_hours   = total_minutes / 60                    # and in hours

summary_lines = ["\n=== RMSE Summary (Cross-Validation) ==="]

# ---------------------------------
# Collect results by model & seed
# ---------------------------------
processed_results = {}
for result in results:
    name  = result["model"]
    seed  = result["seed"]
    rmse_scaled  = result["rmse_scaled"]
    rmse_orig  = result["rmse_orig"]
    processed_results.setdefault(name, []).append((seed, rmse_scaled, rmse_orig))

# ---------------------------------
# Print section per model
# ---------------------------------
for model_name, results_list in processed_results.items():
    rmses_scaled = [rmse_scaled for _, rmse_scaled, _ in results_list]
    rmses_orig = [rmse_orig for _, _, rmse_orig in results_list]
    
    summary_lines.append(f"\nModel: {model_name}")
    summary_lines.append(f" Device: {device_choice}")
    
    for seed, rmse_scaled, rmse_orig in results_list:
        summary_lines.append(f" Seed {seed}: OOS-RMSE (scaled) = {rmse_scaled:.4f}, OOS-RMSE (original) = {rmse_orig:.4f}")
    
    summary_lines.append(f" Runtime: {model_runtimes[model_name]:.2f} min")
    
    summary_lines.append(
        f"\033[91mMean Out-of-Sample Performance (RMSE scaled) "
        f"across {len(rmses_scaled)} seeds: {np.mean(rmses_scaled):.4f}\033[0m"
    )
    summary_lines.append(
        f"\033[94mMean Out-of-Sample Performance (RMSE original) "
        f"across {len(rmses_orig)} seeds: {np.mean(rmses_orig):.4f}\033[0m"
    )
# ---------------------------------
# Overall runtime
# ---------------------------------
summary_lines.append(
    f"\nTotal runtime (all models & seeds): "
    f"{total_minutes:.2f} min  |  {total_hours:.2f} h"
)

# ---------------------------------
# Join to single string
# ---------------------------------
summary_text = "\n".join(summary_lines)

# ============================================================================
# Save Results in .txt file which will be summarized later 
# ============================================================================
print(summary_text)
summary_dir.mkdir(exist_ok=True)
filename = f"summary_cv_{mode}_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
(summary_dir / filename).write_text(summary_text, encoding="utf-8")
print(f"\nRMSE-Summary saved to: {(summary_dir / filename).resolve()}")