"""
run_all_models.py

Automatically runs all model classes defined in models_utils.py.
Generates seeds, evaluates models, and reports RMSE values.
"""

import numpy as np
import random
import inspect
import time
#from models_utils import * #nur wenn manuelle seed lsite 
import models_utils
from models_main import generate_evaluate_models

start_time = time.time() 
# Central seed for reproducibility
SEED = 42
random.seed(SEED)

# Generate a list of random seeds
n = 5
seeds_list = random.sample(range(0, 100), n)
#seeds_list = [42]

# Dynamically extract all relevant model classes from models_utils.py
models = [
    obj for name, obj in inspect.getmembers(models_utils, inspect.isclass)
    if obj.__module__ == models_utils.__name__
]

#
# models = [LSTMModel]


print("Models list:", models)

# Call the central evaluation function with dynamic model classes
results = generate_evaluate_models(models, seeds_list)
print(results)

end_time = time.time()
runtime = end_time - start_time


# Display RMSE values and averages per model
print("\n=== RMSE Summary ===")
for model_name, rmse_list in results.items():
    mean_rmse = np.mean(rmse_list)
    print(f"\nModel: {model_name}")
    for seed, rmse in zip(seeds_list, rmse_list):
        print(f"  Seed {seed}: RMSE = {rmse:.4f}")
    print(f"  Average RMSE: {mean_rmse:.4f}")

runtime_minutes = runtime / 60
print(f"\nTotal Runtime: {runtime_minutes:.2f} minutes")