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

#universal functions needed here
from universal_utils import set_seed, get_dataloaders



class SimpleRNN(nn.Module):
    def __init__(self, input_size, hp):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hp['rnn_units'], batch_first=True)
        self.dropout1 = nn.Dropout(hp['dropout_rate_rnn'])
        self.fc1 = nn.Linear(hp['rnn_units'], hp['dense_units'])
        self.dropout2 = nn.Dropout(hp['dropout_rate_dense'])
        self.fc2 = nn.Linear(hp['dense_units'], 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Nur der letzte Zeitschritt
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out


# Objective-function with Hyperband-Pruning for RNN
def hyperparameter_tuning_rnn(X_train, Model, train_dataset, val_dataset, test_dataset,SEED):
    
    def objective(trial):

        set_seed(SEED)
        train_loader, _, val_loader = get_dataloaders(SEED, train_dataset, val_dataset, test_dataset)
        
        hp = {
            'rnn_units': trial.suggest_int('rnn_units', 32, 256, step=16),
            'dropout_rate_rnn': trial.suggest_float('dropout_rate_rnn', 0.1, 0.5, step=0.1),
            'dense_units': trial.suggest_int('dense_units', 8, 64, step=8),
            'dropout_rate_dense': trial.suggest_float('dropout_rate_dense', 0.0, 0.4, step=0.1),
            'learning_rate': trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])
        }
        
        rnn_model = Model(input_size=X_train.shape[2], hp=hp)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(rnn_model.parameters(), lr=hp['learning_rate'])

        num_epochs = 3
        patience = 7
        best_val_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            rnn_model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch, y_batch
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
                    X_batch, y_batch = X_batch, y_batch
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
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler= sampler)
    study.optimize(objective, n_trials=3, n_jobs=1)

    # Show Best Result
    print("Best trial parameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    
    best_hp = study.best_trial.params
    return study, best_hp






