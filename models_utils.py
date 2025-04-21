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



# RNN Model ----------------------------------------------------------------------------------------------------------------------------------------------------

class SimpleRNN(nn.Module):
    name = "rnn"
    def __init__(self, input_size, hp):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hp['rnn_units'], batch_first=True)
        self.layer_norm = nn.LayerNorm(hp['rnn_units'])
        self.dropout1 = nn.Dropout(hp['dropout_rate_rnn'])
        self.fc1 = nn.Linear(hp['rnn_units'], hp['dense_units'])
        self.dropout2 = nn.Dropout(hp['dropout_rate_dense'])
        self.fc2 = nn.Linear(hp['dense_units'], 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Nur der letzte Zeitschritt
        out = self.layer_norm(out)
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out

def suggest_rnn_hyperparameters(trial):
    return{
            'rnn_units': trial.suggest_int('rnn_units', 32, 256, step=16),
            'dropout_rate_rnn': trial.suggest_float('dropout_rate_rnn', 0.1, 0.5, step=0.1),
            'dense_units': trial.suggest_int('dense_units', 8, 64, step=8),
            'dropout_rate_dense': trial.suggest_float('dropout_rate_dense', 0.0, 0.4, step=0.1),
            'learning_rate': trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])
    }


# LSTM Model ---------------------------------------------------------------------------------------------------------------------------------------------------

class LSTMModel(nn.Module):
    name = "lstm"
    def __init__(self, input_size, hp):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hp['lstm_units'], batch_first=True)
        self.layer_norm = nn.LayerNorm(hp['lstm_units'])
        self.dropout1 = nn.Dropout(hp['dropout_rate_lstm'])
        self.fc1 = nn.Linear(hp['lstm_units'], hp['dense_units'])
        self.dropout3 = nn.Dropout(hp['dropout_rate_dense'])
        self.fc2 = nn.Linear(hp['dense_units'], 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.layer_norm(out)
        out = self.dropout1(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        return out
    
def suggest_lstm_hyperparameters(trial):
    return{
        'lstm_units': trial.suggest_int('lstm_units', 32, 256, step=16),
        'dropout_rate_lstm': trial.suggest_float('dropout_rate_lstm', 0.1, 0.3, step=0.1),
        'dense_units': trial.suggest_int('dense_units', 16, 64, step=16),
        'dropout_rate_dense': trial.suggest_float('dropout_rate_dense', 0.0, 0.3, step=0.1),
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-3, 1e-4])
    }

# SLSTM Model ---------------------------------------------------------------------------------------------------------------------------------------------

class StackedLSTMModel(nn.Module):
    name = "slstm"
    def __init__(self, input_size, hp):
        super(StackedLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hp['lstm_units_1'], batch_first=True)
        self.norm1 = nn.LayerNorm(hp['lstm_units_1'])
        self.dropout1 = nn.Dropout(hp['dropout_rate_lstm_1'])
        self.lstm2 = nn.LSTM(hp['lstm_units_1'], hp['lstm_units_2'], batch_first=True)
        self.norm2 = nn.LayerNorm(hp['lstm_units_2'])
        self.dropout2 = nn.Dropout(hp['dropout_rate_lstm_2'])
        self.fc1 = nn.Linear(hp['lstm_units_2'], hp['dense_units'])
        self.dropout3 = nn.Dropout(hp['dropout_rate_dense'])
        self.fc2 = nn.Linear(hp['dense_units'], 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.norm1(out)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.norm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        return out

def suggest_slstm_hyperparameters(trial):
    return{
        'lstm_units_1': trial.suggest_int('lstm_units_1', 32, 256, step=16),
        'lstm_units_2': trial.suggest_int('lstm_units_2', 32, 256, step=16),
        'dropout_rate_lstm_1': trial.suggest_float('dropout_rate_lstm_1', 0.1, 0.3, step=0.1),
        'dropout_rate_lstm_2': trial.suggest_float('dropout_rate_lstm_2', 0.1, 0.3, step=0.1),
        'dense_units': trial.suggest_int('dense_units', 8, 64, step=8),
        'dropout_rate_dense': trial.suggest_float('dropout_rate_dense', 0.0, 0.4, step=0.1),
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])
    }


#PLSTM Model -----------------------------------------------------------------------------------------------------------------------------------------------

class PhasedLSTMModel(nn.Module):
    name = "plstm"
    def __init__(self, input_size, hp):
        super(PhasedLSTMModel, self).__init__()
        self.plstm = PLSTM(input_sz=input_size, hidden_sz=hp['lstm_units'])
        self.norm = nn.LayerNorm(hp['lstm_units'])
        self.dropout1 = nn.Dropout(hp['dropout_rate_lstm'])
        self.fc1 = nn.Linear(hp['lstm_units'], hp['dense_units'])
        self.dropout2 = nn.Dropout(hp['dropout_rate_dense'])
        self.fc2 = nn.Linear(hp['dense_units'], 1)

    def forward(self, x):
        out= self.plstm(x)
        out = self.norm(out)
        out = self.dropout1(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out

def suggest_plstm_hyperparameters(trial):
    return{
        'lstm_units': trial.suggest_int('lstm_units', 32, 256, step=16),
        'dropout_rate_lstm': trial.suggest_float('dropout_rate_lstm', 0.1, 0.3, step=0.1),
        'dense_units': trial.suggest_int('dense_units', 8, 64, step=8),
        'dropout_rate_dense': trial.suggest_float('dropout_rate_dense', 0.0, 0.4, step=0.1),
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-3, 1e-4])
    }