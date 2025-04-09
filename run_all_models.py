"""
run_all_models.py

Automatically runs all model classes defined in models_utils.py.
Generates seeds, evaluates models, and reports RMSE values.
"""

import numpy as np
import random
import inspect
import models_utils
from models_main import generate_evaluate_models

# Central seed for reproducibility
SEED = 42
random.seed(SEED)

# Generate a list of random seeds
n = 5
seeds_list = random.sample(range(0, 100), n)

# Dynamically extract all relevant model classes from models_utils.py
models = [
    obj for name, obj in inspect.getmembers(models_utils, inspect.isclass)
    if obj.__module__ == models_utils.__name__
]
print("Models list:", models)

# Call the central evaluation function with dynamic model classes
results = generate_evaluate_models(models, seeds_list)

# Display RMSE values and averages per model
print("\n=== RMSE Summary ===")
for model_class, rmse_list in results.items():
    model_name = model_class.__name__
    mean_rmse = np.mean(rmse_list)
    print(f"\nModel: {model_name}")
    for seed, rmse in zip(seeds_list, rmse_list):
        print(f"  Seed {seed}: RMSE = {rmse:.4f}")
    print(f"  Average RMSE: {mean_rmse:.4f}")