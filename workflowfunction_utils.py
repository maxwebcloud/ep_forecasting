
"""
workflow_utils.py 
Last changes: 16.05.2025

Provides utilities to:
  • select device (CPU/GPU)
  • ensure reproducibility (seed setting)
  • split time series into train/validation/test sets
  • scale data and apply PCA
  • tranform data into PyTorch DataLoaders
  • perform hyperparameter tuning and final model training
  • make predictions and compute model performance (e.g., out-of-sample RMSE)
  • generate naive forecast baseline
  • plot residuals 
  • wraps the entire workflow via `cross_validate_time_series()`
"""


# ============================================================================
# Imports
# ============================================================================

# Standard Libraries
import json

# Numerical Libraries
import numpy as np
import scipy.stats as stats

# Scikit-Learn: Dimensionality Reduction, Scaling & TimeSeries-Splits
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameter Optimization
import optuna
from optuna.pruners import HyperbandPruner

# Rich CLI & PPrinting
from rich import print
from rich.console import Console

# Custom Forecast & Utility Modules
from naive_forecast import naive_forecast
from models_utils import *



# ============================================================================
# 1. Environment & Reproducibility Setup
# ============================================================================

def get_device(use_gpu=True):
    """
    Returns the appropriate torch.device based on use_gpu flag.
    Apple MPS or fallback to CPU
    """
    if use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

def set_seed(SEED, device=None):
    import torch
    import numpy as np
    import random
    import os

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    if device is not None and device.type == "mps":
        try:
            torch.mps.manual_seed(SEED)
        except AttributeError:
            pass  

    torch.use_deterministic_algorithms(True)

# ============================================================================
# 2. Time Series Split Indices
# ============================================================================

def get_time_series_split_indices(n_samples, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1."

    train_end = int(n_samples * train_frac)
    val_end = train_end + int(n_samples * val_frac)

    return {
        "train_idx": (0, train_end),
        "val_idx": (train_end, val_end),
        "test_idx": (val_end, n_samples)
    }


def split_data_time_series_sliding_auto_folds(data, target, n_folds=5, slide_fraction=0.2, 
                                               train_frac=0.8, val_frac=0.2):
    """
    Splits time series data into train/validation sets using a sliding window for a desired number of folds.
    The window size is determined automatically.

    Args:
        data (np.array): Input data 
        target (np.array): Target
        n_folds (int): Specified number of folds
        slide_fraction (float): Fraction by which the window is shifted per fold (e.g. 0.2)
        train_frac, val_frac (float): Fractions used to split each fold (must sum to 1)
    Returns:
        list of dicts: train/val splits for every fold
    """
    data_len = len(data)
    fold_data = []

    # Determine window size automatically
    window_size = int(data_len / (1 + (n_folds - 1) * slide_fraction))
    slide_step = int(window_size * slide_fraction)

    for fold in range(n_folds):
        start = fold * slide_step
        end = start + window_size
        if end > data_len:
            break

        train_end = start + int(train_frac * window_size)

        train_data = data[start:train_end]
        val_data = data[train_end:end]

        train_target = target[start:train_end]
        val_target = target[train_end:end]

        fold_data.append({
            "train": (train_data, train_target),
            "val": (val_data, val_target)
        })

    return fold_data

# ============================================================================
# 3. Preprocessing
# ============================================================================

def fit_minmax_scalers(X_train, y_train, feature_range=(0, 1)):
    """
    Fit MinMaxScaler for features (X) und target (y) in train data.

    Args:
        X_train (np.array oder pd.DataFrame): feature matrix
        y_train (np.array oder pd.Series): target vector
        feature_range (tuple): range for scaling

    Returns:
        tuple: (X_scaler, y_scaler)
    """
    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)

    scaler_X.fit(X_train)
    scaler_y.fit(y_train.reshape(-1, 1))  # Reshape for compatibility

    return scaler_X, scaler_y


def preprocessing(folds, variance_ratio=0.8, return_pca_scaler= False):
    """
    Scales and transforms the data within each fold:
        Scales X and y separately using MinMaxScaler
        Applies PCA to X (fitted on X_train)
        Appends y as an additional feature to the PCA components

    Args:
        folds (list of dicts): Fold data containing train/val(/test) sets
        variance_ratio (float): Retained variance ratio for PCA
        return_pca_scaler (boolean): Determin wheather fitted scalers and pca should be returned

    Returns:
    list of dicts: Preprocessed folds
    If `return_pca_scaler` is True, also returns:
            list of PCA objects,
            list of fitted X scalers (MinMaxScaler),
            list of fitted y scalers (MinMaxScaler).
    """
    processed_folds = []
    pcas, scalers_X, scalers_y = [], [], []
        

    for fold in folds:
        # Extract data
        X_train, y_train = fold["train"]
        X_val, y_val = fold["val"]
        X_test, y_test = fold.get("test", (None, None))  # optional

        # Scaling
        scaler_X, scaler_y = fit_minmax_scalers(X_train, y_train)
        X_train = scaler_X.transform(X_train)
        y_train = scaler_y.transform(y_train) 
        X_val = scaler_X.transform(X_val)
        y_val = scaler_y.transform(y_val)
        if X_test is not None:
            X_test = scaler_X.transform(X_test)
            y_test = scaler_y.transform(y_test)

        # PCA
        pca = PCA(n_components= variance_ratio)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)
        if X_test is not None:
            X_test = pca.transform(X_test)
   

        # Appends y as an additional feature to the PCA components
        X_train = np.concatenate((X_train, y_train), axis=1)
        X_val = np.concatenate((X_val, y_val), axis=1)
        if X_test is not None:
            X_test = np.concatenate((X_test, y_test), axis=1)

        processed_fold = {
            "train": (X_train, y_train),
            "val": (X_val, y_val)
        }
        if X_test is not None:
            processed_fold["test"] = (X_test, y_test)

        processed_folds.append(processed_fold)

        if return_pca_scaler:
            pcas.append(pca)
            scalers_X.append(scaler_X)
            scalers_y.append(scaler_y)

    if return_pca_scaler:
        return processed_folds, pcas, scalers_X, scalers_y
    else:
        return processed_folds

# ============================================================================
# 4. Sequence creation & TensorDataset conversion
# ============================================================================

def create_sequences_for_folds(folds, history_size, target_size, step, single_step):
    """
    Creates input/target sequences for each fold's train/val and optionally test split.

    Args:
        folds (list of dicts): List of folds containing "train", "val", and optionally "test" data.
        history_size (int): Length of the input sequence.
        target_size (int): Length of the target sequence.
            Set to 0 if you want the target to be the immediate next observation after the input sequence.
        step (int): Step size within the input sequence.
        single_step (bool): Whether only a single target value is predicted (True) or a sequence (False).

    Returns:
        list of dicts: Each fold contains sequenced data for each available split.
    """
    sequenced_folds = []

    for fold in folds:
        fold_seq = {}

        for split_name in ["train", "val", "test"]:
            split_data = fold.get(split_name)
            if split_data is not None:
                X, y = split_data
                X_seq, y_seq = create_sequences(X, y, history_size, target_size, step, single_step)
                fold_seq[split_name] = (X_seq, y_seq)

        sequenced_folds.append(fold_seq)

    return sequenced_folds


def create_sequences(X, y, history_size, target_size, step, single_step):
    """
    Creates sequences from given features (X) and targets (y).

    Args:
        X (np.array): Input data (e.g., X_train, X_val, X_test)
        y (np.array): Target values (e.g., y_train, y_val, y_test)
        history_size (int): Length of the input sequence
        target_size (int): Length of the target sequence
        step (int): Step size within the input sequence
        single_step (bool): If True, use a single target value; otherwise, use a sequence

    Returns:
        Tuple (np.array, np.array): Sequences (samples, timesteps, features), target values
    """
    data = []
    labels = []

    for i in range(history_size, len(X) - target_size):
        indices = range(i - history_size, i, step)
        data.append(X[indices])

        if single_step:
            if i + target_size >= len(y):
                break
            labels.append(y[i + target_size])
        else:
            labels.append(y[i : i + target_size])

    return np.array(data), np.array(labels)


def get_tensordatasets_from_folds(folds):
    """
    Converts all splits in each fold into TensorDatasets.

    Args:
        folds (list of dicts): Each fold contains "train", "val", and optionally "test" splits.

    Returns:
        list of dicts: Each fold contains TensorDatasets for "train", "val", and optionally "test".
    """
    tensor_folds = []

    for fold in folds:
        fold_tensor = {}

        for split_name in ["train", "val", "test"]:
            split_data = fold.get(split_name)
            if split_data is not None:
                X, y = split_data
                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32)
                if y_tensor.ndim == 1:
                    y_tensor = y_tensor.view(-1, 1)
                fold_tensor[split_name] = TensorDataset(X_tensor, y_tensor)

        tensor_folds.append(fold_tensor)

    return tensor_folds


def get_dataloaders(seed, train_dataset, val_dataset, test_dataset=None, shuffle_train=True, batch_size=128):
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False,
        num_workers=0,
        generator=g,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        generator=g,
        pin_memory=True
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            generator=g,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader

# ============================================================================
# 5. Hyperparamter Tuning 
# ============================================================================

# Hyperparameter Tuning

def hyperparameter_tuning(Model, folds, seed, device, 
                          max_epochs=15, n_trials=10):
    """
    Optuna tuning with Successive-Halving pruner.
    After every epoch we report the mean val-loss over all folds to Optuna.
    """
    import time, numpy as np, torch, optuna, json
    from torch import nn

    # Helper functions
    def _train_single_epoch(model, loader, optimizer, criterion):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb.to(device))
            loss.backward()
            optimizer.step()

    def _validate(model, loader, criterion):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for Xb, yb in loader:
                total += criterion(model(Xb.to(device)), yb.to(device)).item()
        return total / len(loader)

    # Callback for timing
    start = time.time()
    trial_times, last_start = {}, None
    def callback(study, trial):
        nonlocal last_start
        if last_start is not None:
            trial_times[trial.number] = time.time() - last_start
        last_start = time.time()

    # Objective function
    def objective(trial):
        hp = Model.suggest_hyperparameters(trial)
        fold_states = []
        for fold in folds:
            set_seed(seed, device)
            tr_ds, va_ds = fold["train"], fold["val"]
            tr_loader, va_loader, _ = get_dataloaders(seed, tr_ds, va_ds)
            m = Model(input_size=tr_ds.tensors[0].shape[-1], hp=hp).to(device)
            fold_states.append({
                "model": m,
                "opt": torch.optim.Adam(m.parameters(), lr=hp["learning_rate"]),
                "crit": nn.MSELoss(),
                "tr": tr_loader,
                "va": va_loader
            })

        best_val = float("inf")

        for epoch in range(max_epochs):
            losses = []
            for st in fold_states:
                _train_single_epoch(st["model"], st["tr"], st["opt"], st["crit"])
                losses.append(_validate(st["model"], st["va"], st["crit"]))
            mean_val = float(np.mean(losses))
            best_val = min(best_val, mean_val)

            # report & prune
            trial.report(mean_val, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return best_val

    # Study creation & optimization
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[callback])

    # Print results
    import json
    from pprint import pprint
    total_tuning_time = time.time() - start

    print("\nSummary of trial times:")
    pprint(trial_times) 

    if trial_times:
        avg_time = sum(trial_times.values()) / len(trial_times)
        print(f"\nAverage trial time: {avg_time:.2f} seconds")

    print(f"\nTotal hyperparameter tuning time: {total_tuning_time:.2f} seconds "
        f"({total_tuning_time/60:.2f} minutes)")

    print("\nBest trial parameters:")
    print(json.dumps(study.best_trial.params, indent=4))

    return study, study.best_trial.params



# Save best hyperparameters
def save_best_hp(model, study, SEED):
    best_hp = study.best_trial.params
    with open(f"best_hp_all_models/best_hp_{model.name}_{SEED}.json", "w") as f:
        json.dump(best_hp, f)

# ============================================================================
# 7. Final Model Trainng 
# ============================================================================

# Load best hyperparameters 
def load_best_hp(model, SEED):
    with open(f"best_hp_all_models/best_hp_{model.name}_{SEED}.json", "r") as f:
        best_hp = json.load(f)
    return best_hp

# Final model training
def final_train(model, criterion, optimizer, val_loader, train_loader, device, num_epochs, patience, seed):
    import time
    
    set_seed(seed, device)

    # Start time measurement for final training
    start_time = time.time()
    print(f"Starting final training with {num_epochs} epochs and patience {patience}...")

    best_val_loss = float('inf')
    early_stopping_counter = 0

    train_loss_history = []
    val_loss_history = []
    best_model = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model = model
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
        
    training_time = time.time() - start_time
    print(f'Final training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)')
    return best_model, train_loss_history, val_loss_history

    
# Plot training and validation loss over the training process
def train_history_plot(train_loss_history, val_loss_history, model, SEED):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epochen")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Trainings- and Validation Loss {model.name.upper()} ({SEED})")
    
    filename = f"{model.name}_train_history_{SEED}.png"
    filepath = os.path.join("plots", filename)

    # Save & close
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    


# ============================================================================
# 8. Predictions and Plots 
# ============================================================================

# Make predictions
def get_predictions_in_batches(final_model, dataloader, device):
    final_model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            preds.append(final_model(X_batch).cpu().numpy())  # Immer .cpu() für späteres numpy
    return np.vstack(preds)

# Generate residual plots
def plot_residuals_with_index(y_true, y_pred, model, df_final,seq_length, test_index, seed, bins="auto"):
    """
    Generate and save three separate residual plots:
      1. Residuals over time
      2. Histogram of residuals with normal density
      3. Q-Q plot comparing residuals to a normal distribution

    """
    # Compute residuals and flatten to 1D
    residuals = (y_true - y_pred).ravel() #ravel for flattening 

    # Determine time index range for the test period
    start_idx = test_index + seq_length
    index_range = df_final.index[start_idx : start_idx + len(residuals)]

    # Ensure output directory exists
    os.makedirs("plots", exist_ok=True)

    # 1) Plot residuals over time
    plt.figure(figsize=(10, 4))
    plt.plot(index_range, residuals, label="Residuals")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"Residuals over Time — {model.name.upper()} (Seed {seed})")
    plt.xlabel("Time")
    plt.ylabel("Prediction Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{model.name}_residuals_time_{seed}.png", bbox_inches="tight")
    plt.close()

    # 2) Plot histogram of residuals with normal PDF overlay
    mu, sigma = residuals.mean(), residuals.std(ddof=1)
    x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=bins, density=True, edgecolor="black")
    plt.plot(x_vals, stats.norm.pdf(x_vals, mu, sigma), linestyle="--", label="Normal PDF")
    plt.title(f"Histogram of Residuals — {model.name.upper()} (Seed {seed})")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{model.name}_residuals_hist_{seed}.png", bbox_inches="tight")
    plt.close()

    # 3) Plot Q-Q plot of residuals
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of Residuals — {model.name.upper()} (Seed {seed})")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Ordered Values")
    plt.tight_layout()
    plt.savefig(f"plots/{model.name}_residuals_qq_{seed}.png", bbox_inches="tight")
    plt.close()


# Load trained model  
def load_model(model, input_size, seed, device):
    hp = load_best_hp(model, seed)
    model_final = model(input_size=input_size, hp=hp).to(device)
    model_final.load_state_dict(torch.load(f"saved_models/{model.name}_model_final_{seed}.pth", map_location=device, weights_only=True))
    model_final.eval()
    return model_final




# ============================================================================
# 10. Wrapper: End-to-End Cross-Validation Workflow
# ============================================================================

def cross_validate_time_series(models, seeds, df, device , train_size=0.6, val_size=0.2, test_size=0.2, 
                               sequence_length=24, target_size = 0, step_size=1, n_folds = 5, variance_ratio=0.8, single_step= True):
    
    console = Console() 

    # Splitting the data in target und features +  trainings-, val- und testset
    X = df[df.columns.drop('price actual')].values
    y = np.array(df['price actual']).reshape(-1,1)
    n = len(X)  
    splits = get_time_series_split_indices(n, train_frac = train_size, val_frac = val_size, test_frac = test_size)
    X_train_full = X[splits["train_idx"][0] : splits["train_idx"][1]]
    y_train_full = y[splits["train_idx"][0] : splits["train_idx"][1]]

    initial_split = [{
        "train": (X[splits["train_idx"][0] : splits["train_idx"][1]], y[splits["train_idx"][0] : splits["train_idx"][1]]),
        "val": (X[splits["val_idx"][0] : splits["val_idx"][1]], y[splits["val_idx"][0] : splits["val_idx"][1]]),
        "test": (X[splits["test_idx"][0] : splits["test_idx"][1]], y[splits["test_idx"][0] : splits["test_idx"][1]])
    }]

    # Generate folds only from the trainingsset (77%) 
    folds = split_data_time_series_sliding_auto_folds(
            X_train_full,  y_train_full,
            n_folds       = n_folds,
            slide_fraction= 0.2,     # 1.0, if no overlappimg wanted
            train_frac    = 0.8,
            val_frac      = 0.2)


    # Scaling + dimension reduction on folds
    folds_preprocessed = preprocessing(folds, variance_ratio)
    # Cut folds in sequences + transform in tensordatasets
    folds_sequenced = create_sequences_for_folds(folds_preprocessed, sequence_length, target_size, step_size, single_step)
    folds_tensordatasets = get_tensordatasets_from_folds(folds_sequenced)


    # Scaling + dimension reduction on initial split
    initial_split_preprocessed, _, _, scalers_y = preprocessing(initial_split, variance_ratio, return_pca_scaler = True)
    # Cut initial split in sequences + transform in tensordatasets
    initial_split_sequenced = create_sequences_for_folds(initial_split_preprocessed, sequence_length, target_size, step_size, single_step)
    initial_split_tensordatasets = get_tensordatasets_from_folds(initial_split_sequenced)

    final_results = []

    # Naive forecast
    mse_scaled, rmse_scaled, mse_orig, rmse_orig = naive_forecast(initial_split_preprocessed, scalers_y[0])
    rmse_naive_list = [rmse_scaled] * len(seeds) 
    final_results.append({
            "model": "naive",
            "seed": None,
            "rmse_scaled": rmse_scaled,
            "mse_scaled": mse_scaled,
            "rmse_orig": rmse_orig,
            "mse_orig": mse_orig,
            "rmse_train": None,
            "mse_train": None,
            "rmse_val": None,
            "mse_val": None, 
            "p_value": None
            })
  

    for model in models:
        for seed in seeds:
            set_seed(seed, device)
            console.rule(f"\n[medium_purple]{model.name.upper()} mit Seed {seed}[/medium_purple]")

            # Hyperparametertuning
            console.print("\n[bold turquoise2]Hyperparametertuning: [/bold turquoise2]")
            study, best_hp = hyperparameter_tuning(model, folds_tensordatasets, seed, device)
            save_best_hp(model, study, seed)


            # Final model training
            console.print("\n[bold turquoise2]Final Training[/bold turquoise2]")
            X_train_seq, _ = initial_split_sequenced[0]["train"]
            amount_features = X_train_seq.shape[2]
            final_model = model(input_size= amount_features, hp=best_hp).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(final_model.parameters(), lr=best_hp['learning_rate'])
            train_dataset = initial_split_tensordatasets[0]["train"]
            val_dataset = initial_split_tensordatasets[0]["val"]
            test_dataset = initial_split_tensordatasets[0]["test"]
            train_loader, val_loader, test_loader = get_dataloaders(seed, train_dataset, val_dataset, test_dataset)

            final_model, train_loss_history, val_loss_history = final_train(final_model, criterion, optimizer, val_loader, train_loader, device, 
                                                                            num_epochs = 50, patience = 10, seed = seed)
            torch.save(final_model.state_dict(), f"saved_models/{final_model.name}_model_final_{seed}.pth")
            train_history_plot(train_loss_history, val_loss_history, final_model, seed)

            # Model evaluation
            train_loader, val_loader, test_loader = get_dataloaders(seed, train_dataset, val_dataset, test_dataset, shuffle_train = False)

            _, y_test_seq = initial_split_sequenced[0]["test"]
            y_test_seq = y_test_seq.reshape(-1,1)
            test_predictions = get_predictions_in_batches(final_model, dataloader = test_loader, device =device)
            y_test_seq_rescaled = scalers_y[0].inverse_transform(y_test_seq)
            test_predictions_rescaled = scalers_y[0].inverse_transform(test_predictions)

            train_predictions = get_predictions_in_batches(final_model, dataloader = train_loader, device =device)
            _, y_train_seq = initial_split_sequenced[0]["train"]
            y_train_seq_rescaled = scalers_y[0].inverse_transform(y_train_seq)
            train_predictions_rescaled = scalers_y[0].inverse_transform(train_predictions)

            val_predictions = get_predictions_in_batches(final_model, dataloader = val_loader, device =device)
            _, y_val_seq = initial_split_sequenced[0]["val"]
            y_val_seq_rescaled = scalers_y[0].inverse_transform(y_val_seq)
            val_predictions_rescaled = scalers_y[0].inverse_transform(val_predictions)

            mse_scaled = np.mean((test_predictions - y_test_seq)**2)
            rmse_scaled = np.sqrt(mse_scaled)
            mse_orig = np.mean((test_predictions_rescaled - y_test_seq_rescaled)**2)
            rmse_orig = np.sqrt(mse_orig)
            mse_train = np.mean((train_predictions_rescaled - y_train_seq_rescaled)**2)
            rmse_train = np.sqrt(mse_train)
            mse_val = np.mean((val_predictions_rescaled - y_val_seq_rescaled)**2)
            rmse_val = np.sqrt(mse_val)

            console.print(f"[bold turquoise2]Out of Sample Performance: {rmse_scaled}[/bold turquoise2]")
            final_results.append({
            "model": model.name,
            "seed": seed, 
            "rmse_scaled": rmse_scaled,
            "mse_scaled": mse_scaled,
            "rmse_orig": rmse_orig,
            "mse_orig": mse_orig,
            "rmse_train": rmse_train,
            "mse_train": mse_train,
            "rmse_val": rmse_val,
            "mse_val": mse_val,
            "p_value": None
            })

            # Generating residual plots
            plot_residuals_with_index(y_test_seq_rescaled, test_predictions_rescaled, model, df, sequence_length, splits["test_idx"][0], seed)

        # Testing the significance of the model against the naive forecast
        rmses = [entry["rmse_scaled"] for entry in final_results if entry["model"] == model.name]
        stat, p_value = stats.wilcoxon(rmses, rmse_naive_list)
        console.print(f"\n[bold turquoise2]p-Value for significance in comparison to naive model: {round(p_value,4)} [/bold turquoise2]")

        p_value = round(p_value, 4)
        for entry in final_results:         
            if entry["model"] == model.name:
                 entry["p_value"] = p_value  
    
    final_eval_df = pd.DataFrame(final_results)
    return final_eval_df


