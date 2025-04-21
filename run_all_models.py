"""
run_all_models.py

This script automatically runs selected model classes defined in models_utils.py.
It generates random seeds, trains each model using optimized hyperparameters,
evaluates performance, and prints a detailed RMSE summary.

You can control the execution via command-line arguments (CLI):

Available CLI options:
----------------------
--mode      (str) : Select which model(s) to run. Choices:
    - "standard"   → Runs all four models (rnn, lstm, slstm, plstm)
    - "3models"    → Runs rnn, lstm, and slstm (excludes plstm)
    - "half_a"     → Runs lstm and slstm (50 / 50 split, LSTM‑heavy)
    - "half_b"     → Runs rnn and plstm (50 / 50 split, RNN‑heavy)
    - "rnn"        → Runs only the SimpleRNN model
    - "lstm"       → Runs only the LSTM model
    - "slstm"      → Runs only the StackedLSTM model
    - "plstm"      → Runs only the PhasedLSTM model

--device    (str) : Choose the device for training. Choices:
    - "cpu"        → Force model(s) to run on CPU
    - "mps"        → Use Apple MPS backend if available (for Apple Silicon GPUs)

Examples:
---------
Run all models on CPU:
    python run_all_models.py --mode standard --device cpu

Run only the LSTM model on MPS:
    python run_all_models.py --mode lstm --device mps
"""

import numpy as np
import random
import time
from models_utils import *          # brings SimpleRNN, LSTMModel, StackedLSTMModel, PhasedLSTMModel
from models_main import generate_evaluate_models

# Toggle: use CLI or not --> needed for simultaneous runs on CPU and MPS
USE_CLI = True      # ← set to False to ignore command‑line arguments

# 1) Mapping of mode strings to class lists
MODELS = {
    "standard": [SimpleRNN, LSTMModel, StackedLSTMModel, PhasedLSTMModel],
    "3models":  [SimpleRNN, LSTMModel, StackedLSTMModel],
    "half_a":   [LSTMModel, StackedLSTMModel],   # LSTM‑only variants
    "half_b":   [SimpleRNN, PhasedLSTMModel],    # RNN and PLSTM variants
    "rnn":      [SimpleRNN],
    "lstm":     [LSTMModel],
    "slstm":    [StackedLSTMModel],
    "plstm":    [PhasedLSTMModel],
}

if USE_CLI:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(MODELS.keys()),
        default="standard",
        help=(
            "standard = all models; "
            "3models = rnn, lstm, slstm; "
            "half_a = lstm + slstm; "
            "half_b = rnn + plstm; "
            "rnn, lstm, slstm, plstm = single models"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Run models on CPU or Apple‑Silicon MPS",
    )
    args = parser.parse_args()
    mode = args.mode
    use_gpu_flag = args.device == "mps"
else:
    mode = "standard"
    use_gpu_flag = True  # MPS as fallback

# 2) Select model classes
selected_classes = MODELS[mode]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#N_SEEDS = 5
#seeds_list = random.sample(range(0, 100), N_SEEDS)
seeds_list = [81]  # Fixed seed list for reproducibility

# build (class, use_gpu) tuples
model_configs = [(cls, use_gpu_flag) for cls in selected_classes]

print("Model configurations:")
for model_class, use_gpu in model_configs:
    print(f"  {model_class.__name__} (use_gpu={use_gpu})")

# Training & evaluation
start_time = time.time()
results, runtimes, devices = generate_evaluate_models(model_configs, seeds_list)

# Output summary
print("\n=== RMSE Summary ===")
for model_name, rmse_list in results.items():
    mean_rmse = np.mean(rmse_list)
    print(f"\nModel: {model_name}")
    print(f"  Device: {devices[model_name]}")
    for seed, rmse in zip(seeds_list, rmse_list):
        print(f"  Seed {seed}: RMSE = {rmse:.4f}")
    print(f"  Average RMSE: {mean_rmse:.4f}")
    print(f"  Runtime: {runtimes[model_name]:.2f} minutes")

total_runtime = sum(runtimes.values())
overall_runtime = (time.time() - start_time) / 60
print(f"\nTotal runtime across all models: {total_runtime:.2f} minutes")
