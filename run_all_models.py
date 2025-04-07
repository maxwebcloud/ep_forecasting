# Main script to run all models with iteration over multiple seeds

# Package imports
import importlib
import numpy as np
import random

# Central runner for all models: iterates over seeds and collects evaluation metrics
SEED= 42
random.seed(SEED)

n = 5
seeds_list = random.sample(range(0, 100), n)
# print(seeds_list)

# List of model modules to be imported
model_modules = [
    "rnn_model",
    # "lstm_model",
    #",
    # "phased_lstm_pytorch"
]

results = {}  # Empty dictionary for each models M-RMSE

# Iteration over all models (and all seeds within each model)
for module_name in model_modules:
    module = importlib.import_module(module_name)
    rmse_list = []
    print(f"\nModel: {module_name}")  # Print the name of the model
    for seed in seeds_list:
        rmse = module.main(seed=seed)
        print(f"  Seed {seed} -> RMSE: {rmse:.4f}")
        rmse_list.append(rmse)
    if rmse_list:
        mean_rmse = np.mean(rmse_list)
        results[module_name] = mean_rmse  # model name as key and corresponding RMSE as value
        print(f"=> Average RMSE ({module_name}): {mean_rmse:.4f}")

# Summary
print("\n\n=== Summary ===")
for model, avg_rmse in results.items():
    print(f"{model}: {avg_rmse:.4f}")






    