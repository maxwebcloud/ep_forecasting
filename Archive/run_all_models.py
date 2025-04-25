


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
import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import numpy as np
import random
import time
from pathlib import Path              # ➊ neu
from datetime import datetime         # ➋ neu

from models_utils import *
from models_main import generate_evaluate_models

# ---------------------------------------------------------------------------
# Konfiguration (unverändert)
# ---------------------------------------------------------------------------
USE_CLI = True  # ← set to False to ignore command‑line arguments

MODELS = {
    "standard": [SimpleRNN, LSTMModel, StackedLSTMModel, PhasedLSTMModel],
    "3models":  [SimpleRNN, LSTMModel, StackedLSTMModel],
    "half_a":   [LSTMModel, StackedLSTMModel],
    "half_b":   [SimpleRNN, PhasedLSTMModel],
    "rnn":      [SimpleRNN],
    "lstm":     [LSTMModel],
    "slstm":    [StackedLSTMModel],
    "plstm":    [PhasedLSTMModel],
}

if USE_CLI:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODELS.keys()), default="standard")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--summary-path", default="model_metrics_overview",  # ➌ neu
                        help="Ordner für RMSE‑Summary‑Dateien")
    args = parser.parse_args()
    mode = args.mode
    use_gpu_flag = args.device == "mps"
    summary_dir = Path(args.summary_path)           # ➍ neu
else:
    mode = "standard"
    use_gpu_flag = True
    summary_dir = Path("model_metrics_overview")   # ➍ neu

selected_classes = MODELS[mode]

SEED = 42
N_SEEDS=1
random.seed(SEED)
np.random.seed(SEED)
seeds_list = random.sample(range(0, 100), N_SEEDS)
#seeds_list = [81]


model_configs = [(cls, use_gpu_flag) for cls in selected_classes]

print("Model configurations:")
for model_class, use_gpu in model_configs:
    print(f"  {model_class.__name__} (use_gpu={use_gpu})")

# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
start_time = time.time()
results, runtimes, devices = generate_evaluate_models(model_configs, seeds_list)

# ---------------------------------------------------------------------------
# RMSE‑Summary → Terminal + Datei --------------------------- ➎ komplett neu
# ---------------------------------------------------------------------------
summary_lines: list[str] = []
summary_lines.append("\n=== RMSE Summary ===")
for model_name, rmse_list in results.items():
    mean_rmse = np.mean(rmse_list)
    summary_lines.append(f"\nModel: {model_name}")
    summary_lines.append(f"  Device: {devices[model_name]}")
    for seed, rmse in zip(seeds_list, rmse_list):
        summary_lines.append(f"  Seed {seed}: RMSE = {rmse:.4f}")
    summary_lines.append(f"  Average RMSE: {mean_rmse:.4f}")
    summary_lines.append(f"  Runtime: {runtimes[model_name]:.2f} minutes")

#summary_lines.append(
    #f"\nTotal runtime across all models: {sum(runtimes.values()):.2f} minutes")

summary_text = "\n".join(summary_lines)
print(summary_text)

summary_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"summary_{mode}_{'mps' if use_gpu_flag else 'cpu'}_{stamp}.txt"
(summary_dir / filename).write_text(summary_text, encoding="utf‑8")
print(f"\nRMSE‑Summary gespeichert unter: {(summary_dir/filename).resolve()}")
