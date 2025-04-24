"""
run_all_models_cv.py

This script runs selected model classes (defined in models_utils.py) on a given 
time series dataset and evaluates them using cross-validation.

The core process is handled by the function `cross_validate_time_series` 
(from cv_core.py), which performs preprocessing, sequence creation, 
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
    - "half_a_gpu"     → Runs lstm and slstm (50 / 50 split, LSTM‑heavy)
    - "half_b_cpu"     → Runs rnn and plstm (50 / 50 split, RNN‑heavy)
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
    python run_all_models_cv.py --mode lstm --device mps

Run all models on CPU:
    python run_all_models_cv.py --mode standard --device cpu
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
from pathlib import Path
from datetime import datetime
from models_utils import *
from workflowfunctions_utils_gpu import get_device
from cv_core import *

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
    "train_size": 0.6,
    "val_size": 0.2,
    "test_size": 0.2,
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
N_SEEDS = 1
random.seed(SEED)
np.random.seed(SEED)
seeds_list = random.sample(range(0, 100), N_SEEDS)


# ============================================================================
# Startup Information
# ============================================================================
print("\n=== Starting run_all_models_cv.py ===")
print(f"Device selected: {device_choice}")
print(f"Model group: {mode}")
print("Selected models to run:")
for model in selected_models:
    print(f"  - {model.__name__}")

print("\nCross-validation settings:")
for key, value in CV_PARAMS.items():
    print(f"  {key}: {value}")

# ============================================================================
# Data Loading
# ============================================================================
print("\nLoading data and extract feature matrix (X) and target vector (Y)... ")
with open("data/df_final_eng.pkl", "rb") as f:
    df_final = pickle.load(f)

X = df_final[df_final.columns.drop('price actual')].values
y = df_final['price actual'].values.reshape(-1, 1)
# ============================================================================
# Model Training & Cross-Validation
# ============================================================================
start_time = time.time()
results = []

for model_class in selected_models:
    model_name = getattr(model_class, "name", model_class.__name__)
    use_gpu = args.device == "mps"
    device = get_device(use_gpu=use_gpu)

    if device.type == "mps":
        assert torch.mps.current_allocated_memory() == 0, "⚠️ MPS memory not cleared!"

    model_results = cross_validate_time_series(
        models=[model_class],
        seeds=seeds_list,
        data=X,
        target=y,
        train_size=CV_PARAMS["train_size"],
        val_size=CV_PARAMS["val_size"],
        test_size=CV_PARAMS["test_size"],
        sequence_length=CV_PARAMS["sequence_length"],
        step_size=CV_PARAMS["step_size"],
        n_folds=CV_PARAMS["n_folds"],
        sliding_window=True,
        variance_ratio=CV_PARAMS["variance_ratio"],
        single_step=CV_PARAMS["single_step"],
        use_gpu=use_gpu
    )

    results.extend(model_results)

# ============================================================================
# Evaluation Summary
# ============================================================================
end_time = time.time()
total_runtime = (end_time - start_time) / 60

summary_lines = ["\n=== RMSE Summary (Cross-Validation) ==="]

processed_results = {}
for result in results:
    name = result["model"]
    seed = result["seed"]
    rmse = result["rmse"]
    processed_results.setdefault(name, []).append((seed, rmse))

avg_runtime = total_runtime / len(selected_models) if selected_models else 0

for model_name, results_list in processed_results.items():
    rmses = [rmse for _, rmse in results_list]
    summary_lines.append(f"\nModel: {model_name}")
    summary_lines.append(f"  Device: {device_choice}")
    for seed, rmse in results_list:
        summary_lines.append(f"  Seed {seed}: RMSE = {rmse:.4f}")
    summary_lines.append(f"  Average RMSE: {np.mean(rmses):.4f}")
    summary_lines.append(f"  Runtime: {avg_runtime:.2f} minutes")

summary_text = "\n".join(summary_lines)

# ============================================================================
# Save Results in .txt file which will be summarized later 
# ============================================================================
print(summary_text)
summary_dir.mkdir(exist_ok=True)
filename = f"summary_cv_{mode}_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
(summary_dir / filename).write_text(summary_text, encoding="utf-8")
print(f"\nRMSE-Summary saved to: {(summary_dir / filename).resolve()}")