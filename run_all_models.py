# Main script to run all models with list of seeds as argument


"""
Runs each model with a list of seeds
Each model gets the full list, runs once per seed, and returns RMSE values.  
The average RMSE is then calculated for each model
"""

import importlib
import numpy as np
import random

# Central seed for reproducible seed list generation
SEED = 42
random.seed(SEED)

n = 5
seeds_list = random.sample(range(0, 100), n)

# List of model modules
model_modules = [
    ("rnn_model_main", "main_rnn"),
    # ("lstm_model_main", "main_lstm"),
    # ("slstm_model_main", "main_slstm"),
    # ("plstm_model_main", "main_plstm"),
]

results = {}  # Dictionary: model name  (mean_rmse, list_of_rmse)

# Go through all models
for module_name, function_name in model_modules:
    print(f"\nModel: {module_name} → calling {function_name}(seeds_list)")

    # Import the module and get the function
    module = importlib.import_module(module_name)
    rmse_func = getattr(module, function_name)

    # Get the list of RMSE values from the model
    rmse_list = rmse_func(seeds=seeds_list)

    # Print RMSE for each seed
    for seed, rmse in zip(seeds_list, rmse_list):
        print(f"  Seed {seed} → RMSE: {rmse:.4f}")
    
    # Calculate the average RMSE + save it
    if rmse_list:
        mean_rmse = np.mean(rmse_list)
        clean_name = module_name.replace("_main", "")
        results[clean_name] = (mean_rmse, rmse_list)
        print(f"=> Average RMSE ({clean_name}): {mean_rmse:.4f}")

# Summary
print("\n=== Summary ===")
for model, (avg_rmse, rmse_vals) in results.items():
    print(f"{model}: {avg_rmse:.4f}")




