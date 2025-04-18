"""
run_all_models.py

Automatically runs all model classes defined in models_utils.py.
Generates seeds, evaluates models, and reports RMSE values.

Toggle CLI on/off with USE_CLI.
"""

import numpy as np
import random
import time
from models_utils import *          # brings SimpleRNN, LSTMModel, StackedLSTMModel, PhasedLSTMModel
from models_main import generate_evaluate_models



#Toggle: use CLI or not --> needed for simukltaneous runs on CPU und GPU 
USE_CLI = True      # ← set to False to ignore command‑line arguments

if USE_CLI:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--mode",
    choices=["standard", "3models", "phased", "rnn", "lstm", "slstm", "plstm"],
    default="standard",
    help="standard = all four models; "
         "3models = rnn, lstm, slstm; "
         "phased = plstm only; "
         "rnn, lstm, slstm, plstm = run individual model only",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Run models on CPU or GPU (boolean flag per model_configs entry)",
    )
    args = parser.parse_args()
    mode = args.mode
    use_gpu_flag = args.device == "gpu"
else:
    mode = "standard"       # fallback when CLI is disabled
    use_gpu_flag = True    # run on CPU by default

# Seed of this .py file 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#N_SEEDS = 5
#seeds_list = random.sample(range(0, 100), N_SEEDS)
seeds_list = [81]  # uncomment for a fixed single seed

#Model selection
#Model selection
if mode == "standard":          
    selected_classes = [SimpleRNN, LSTMModel, StackedLSTMModel, PhasedLSTMModel]
elif mode == "3models":         
    selected_classes = [SimpleRNN, LSTMModel, StackedLSTMModel]
elif mode == "rnn":
    selected_classes = [SimpleRNN]
elif mode == "lstm":
    selected_classes = [LSTMModel]
elif mode == "slstm":
    selected_classes = [StackedLSTMModel]
elif mode == "plstm":
    selected_classes = [PhasedLSTMModel]
else:
    raise ValueError(f"Unknown mode: {mode}")

# build (class, use_gpu) tuples in one line with list comprehension 
model_configs = [(cls, use_gpu_flag) for cls in selected_classes]

print("Model configurations:")
for model_class, use_gpu in model_configs:
    print(f"  {model_class.__name__} (use_gpu={use_gpu})")


# Training & evaluation
start_time = time.time()
results, runtimes = generate_evaluate_models(model_configs, seeds_list)


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