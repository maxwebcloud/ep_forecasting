# Import Libraries
# Standard Libraries
import random
import os
import pickle
import json

# Numerical Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt

# Machine Learning & Statistics
from sklearn.metrics import mean_squared_error

# Jupyter Notebook Imports
import import_ipynb

# PyTorch & Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Custom Modules
from phased_lstm_implementation import PLSTM, Model
import joblib
scaler_y = joblib.load("data/scaler_y.pkl")

# Hyperparameter Tuning
import optuna
from optuna.pruners import HyperbandPruner

import time 


"""

#GPU toggle switch 
USE_GPU = False           

if USE_GPU and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

"""


def get_device(use_gpu=True):
    """
    Returns the appropriate torch.device based on use_gpu flag.
    Supports Apple M1/M2 (MPS) or falls back to CPU.
    """
    if use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(SEED, device=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    if device is not None and device.type == "mps":
        torch.mps.manual_seed(SEED)

    torch.use_deterministic_algorithms(True)


# Data Import
def import_data():
    # Load X_train
    with open("data/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    # Load y_train
    with open("data/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)

    # Load X_val
    with open("data/X_val.pkl", "rb") as f:
        X_val = pickle.load(f)

    # Load y_val
    with open("data/y_val.pkl", "rb") as f:
        y_val = pickle.load(f)

    # Load X_test
    with open("data/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    # Load y_test
    with open("data/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    # Load df_final_viz
    with open("data/df_final_viz.pkl", "rb") as f:
        df_final_viz = pickle.load(f)

    return X_train, y_train, X_val, y_val, X_test, y_test, df_final_viz


"""

# Data Import (long versions)
with open("data/X_train_long.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("data/y_train_long.pkl", "rb") as f:
    y_train = pickle.load(f)
with open("data/X_val_long.pkl", "rb") as f:
    X_val = pickle.load(f)
with open("data/y_val_long.pkl", "rb") as f:
    y_val = pickle.load(f)
with open("data/X_test_long.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("data/y_test_long.pkl", "rb") as f:
    y_test = pickle.load(f)
with open("data/df_final_viz.pkl", "rb") as f:
    df_final_viz = pickle.load(f)


"""

""" 

# Data Import (red1 versions)
import pickle

with open("data/X_train_red1.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("data/y_train_red1.pkl", "rb") as f:
    y_train = pickle.load(f)
with open("data/X_val_red1.pkl", "rb") as f:
    X_val = pickle.load(f)
with open("data/y_val_red1.pkl", "rb") as f:
    y_val = pickle.load(f)
with open("data/X_test_red1.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("data/y_test_red1.pkl", "rb") as f:
    y_test = pickle.load(f)
with open("data/df_final_viz.pkl", "rb") as f:
    df_final_viz = pickle.load(f)


"""

"""
# Data Import (red2 versions)
import pickle

with open("data/X_train_red2.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("data/y_train_red2.pkl", "rb") as f:
    y_train = pickle.load(f)
with open("data/X_val_red2.pkl", "rb") as f:
    X_val = pickle.load(f)
with open("data/y_val_red2.pkl", "rb") as f:
    y_val = pickle.load(f)
with open("data/X_test_red2.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("data/y_test_red2.pkl", "rb") as f:
    y_test = pickle.load(f)
with open("data/df_final_viz.pkl", "rb") as f:
    df_final_viz = pickle.load(f)


"""


# Setting the Number of CPU Threads in PyTorch
def set_num_cpu_threads():
    num_threads = max(1, os.cpu_count() // 2)
    torch.set_num_threads(num_threads)
    print(f"PyTorch uses: {torch.get_num_threads()} Threads")




def get_tensordatasets(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    return train_dataset, val_dataset, test_dataset






def get_dataloaders(seed, train_dataset, val_dataset, test_dataset, shuffle_train=True):
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=shuffle_train, drop_last=False, num_workers=0, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0, generator=g, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0, generator=g, pin_memory=True)
    return train_loader, test_loader, val_loader



# Objective-function with Hyperband-Pruning 
def hyperparameter_tuning(X_train, Model, train_dataset, val_dataset, test_dataset, hp_function, SEED, device):
    
    def objective(trial):
        set_seed(SEED, device)  # device übergeben
        train_loader, _, val_loader = get_dataloaders(SEED, train_dataset, val_dataset, test_dataset)

        hp = hp_function(trial)
        
        rnn_model = Model(input_size=X_train.shape[2], hp=hp).to(device)
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=hp['learning_rate'])

        num_epochs = 15
        patience = 7
        best_val_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            rnn_model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = rnn_model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            rnn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = rnn_model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Optuna-Pruning Check
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    break

        return best_val_loss

    # Study with HyperbandPruner
    sampler = optuna.samplers.TPESampler(seed=SEED) 
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=15, reduction_factor=3)
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=10, n_jobs=1)

    # Show Best Result
    print("Best trial parameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    
    best_hp = study.best_trial.params
    return study, best_hp
    # Study with HyperbandPruner
    sampler = optuna.samplers.TPESampler(seed=SEED) 
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=15, reduction_factor=3)
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler= sampler)
    study.optimize(objective, n_trials=10, n_jobs=1)

    # Show Best Result
    print("Best trial parameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    
    best_hp = study.best_trial.params
    return study, best_hp


# save best hyperparameters
def save_best_hp(modelName, study, SEED):
    best_hp = study.best_trial.params
    with open(f"best_hp_all_models/best_hp_{modelName}_{SEED}.json", "w") as f:
        json.dump(best_hp, f)


# load best hyperparameters 
def load_best_hp(modelName, SEED):
    with open(f"best_hp_all_models/best_hp_{modelName}_{SEED}.json", "r") as f:
        best_hp = json.load(f)


# Bestes Modell mit den gefundenen Hyperparametern trainieren
def final_model_training(X_train, best_hp, Model, train_dataset, val_dataset, test_dataset, SEED, device):
    set_seed(SEED, device)
    train_loader, _, val_loader = get_dataloaders(SEED, train_dataset, val_dataset, test_dataset)

    final_model = Model(input_size=X_train.shape[2], hp=best_hp).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_hp['learning_rate'])

    num_epochs = 50
    train_loss_history = []
    val_loss_history = []
    patience = 10
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        final_model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = final_model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)

        # Validation Loss berechnen
        final_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)               
                y_pred = final_model(X_batch).to(device)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f"GPU memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # Reset Counter
            torch.save(final_model.state_dict(), f"saved_models/{Model.name}_model_final_{SEED}.pth") #save best weights

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping nach {epoch+1} Epochen.")
                break
    
    return train_loss_history, val_loss_history 


def train_history_plot(train_loss_history, val_loss_history, modelName, SEED):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epochen")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Trainings- and Validation Loss {modelName.upper()} ({SEED})")
    
    filename = f"{modelName}_train_history_{SEED}.png"
    filepath = os.path.join("plots", filename)

    # Speichern & schließen
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


# load trained model  
def load_model(Model, hp, X_train, SEED, device):
    model_final = Model(input_size=X_train.shape[2], hp=hp).to(device)
    model_final.load_state_dict(torch.load(f"saved_models/{Model.name}_model_final_{SEED}.pth", map_location=device))
    model_final.eval()
    return model_final



# Make predictions
def get_predictions_in_batches(final_model, dataloader, device):
    final_model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            preds.append(final_model(X_batch).cpu().numpy())  # Immer .cpu() für späteres numpy
    return np.vstack(preds)


def get_predictions(final_model, train_dataset, val_dataset, test_dataset, SEED, device):
    set_seed(SEED, device)
    train_loader, test_loader, val_loader = get_dataloaders(SEED, train_dataset, val_dataset, test_dataset, shuffle_train=False)

    train_predictions = get_predictions_in_batches(final_model, train_loader, device)
    validation_predictions = get_predictions_in_batches(final_model, val_loader, device)
    test_predictions = get_predictions_in_batches(final_model, test_loader, device)

    # Inverse transform scaled predictions and scaled target
    train_predictions_actual = scaler_y.inverse_transform(train_predictions)
    validation_predictions_actual = scaler_y.inverse_transform(validation_predictions)
    test_predictions_actual = scaler_y.inverse_transform(test_predictions)

    return train_predictions, validation_predictions, test_predictions, train_predictions_actual, validation_predictions_actual, test_predictions_actual

def get_unscaled_targets(y_train, y_val, y_test):
    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    return y_train_actual, y_val_actual, y_test_actual


def calculate_loss(y_train_actual, y_val_actual, y_test_actual, train_predictions_actual, val_predictions_actual, test_predictions_actual, y_test, test_predictions):
    # Calculate loss on original scale
    mse_train = mean_squared_error(y_train_actual, train_predictions_actual)
    rmse_train = np.sqrt(mse_train)

    mse_val = mean_squared_error(y_val_actual, val_predictions_actual)
    rmse_val = np.sqrt(mse_val)

    mse_test = mean_squared_error(y_test_actual, test_predictions_actual)
    rmse_test = np.sqrt(mse_test)

    print(f"Original Train MSE: {mse_train:.4f}, Original Train RMSE: {rmse_train:.4f}")
    print(f"Original Validation MSE: {mse_val:.4f}, Original Validation RMSE: {rmse_val:.4f}")
    print(f"Original Test MSE: {mse_test:.4f}, Original Test RMSE: {rmse_test:.4f}")

    # Calculate loss on scaled test data
    mse_test_scaled = mean_squared_error(y_test, test_predictions)
    rmse_test_scaled = np.sqrt(mse_test_scaled)
    print(f"Scaled Test MSE: {mse_test_scaled:.4f}, Scaled Test RMSE: {rmse_test_scaled:.4f}")

    return mse_train, rmse_train, mse_val, rmse_val, mse_test, rmse_test, mse_test_scaled, rmse_test_scaled


def plot_forecast(seq_length, df_final_viz, train_predictions_actual, val_predictions_actual, test_predictions_actual, modelName, SEED):

    # Plot of the forecast
    plt.figure(figsize=(10, 6))

    # Plot of actual values
    plt.plot(df_final_viz.index[seq_length:], df_final_viz['price actual'][seq_length:], label='Actual', color='blue')

    # Training predictions
    train_start = seq_length
    train_end = train_start + len(train_predictions_actual)
    plt.plot(df_final_viz.index[train_start:train_end], train_predictions_actual, label='Train Predictions', color='green', alpha=0.8)

    # Validation predictions 
    val_start = train_end + seq_length
    val_end = val_start + len(val_predictions_actual)
    plt.plot(df_final_viz.index[val_start:val_end], val_predictions_actual, label='Validation Predictions', color='red', alpha=0.8)

    # Test predictions 
    test_start = val_end + seq_length
    test_end = test_start + len(test_predictions_actual)
    plt.plot(df_final_viz.index[test_start:test_end], test_predictions_actual, label='Test Predictions', color='orange', alpha=0.8)

    plt.title(f'Electricity Price Time Series Forecasting {modelName.upper()} ({SEED})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    filename = f"{modelName}_forecast_{SEED}.png"
    filepath = os.path.join("plots", filename)

    # Speichern & schließen
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def plot_residuals_with_index(y_true, y_pred, df_final_viz, seq_length, modelName, SEED):

    residuals = y_true - y_pred
    start_idx = 31056 + seq_length # ab Beobachtung 31056 beginnen die Testdaten siehe feature_engineering
    index_range = df_final_viz.iloc[start_idx : start_idx + len(residuals)].index

    plt.figure(figsize=(12, 4))
    plt.plot(index_range, residuals, label='Residuals')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    plt.title(f"Prediction Residuals {modelName.upper()} ({SEED})")
    plt.xlabel('Time')
    plt.ylabel('Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"{modelName}_residuals_{SEED}.png"
    filepath = os.path.join("plots", filename)

    # Speichern & schließen
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    


def print_num_parameters(final_model, modelName):
    for name, param in final_model.named_parameters():
        print(f"{name}: {param.shape}")
        
    total_params = sum(p.numel() for p in final_model.parameters())
    print(f"Total number of model paramters {modelName.upper()}: {total_params:,}")
