#Packages 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import import_ipynb
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error


# zusätzliche Funktionen die im CV Ablauf benötigt werden -----------------------------------------------------

# zum Splitten, wenn nichts verändert wird diese später aus workflowfunctions importieren
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i : i + target_size])

    return np.array(data), np.array(labels)

"""
Reihenfolge in cv:

für jedes Modell
    für jeden Seed
        1. splitten
        für jeden fold
            1. scaler fitten
            2. pca fitten
            3. Transformieren
            4. Sequnzen schneiden
            5. Trainieren
            6. Evaluieren
"""

def split_data_time_series_with_data(data, target, n_folds=5, train_size=0.6, val_size=0.2, test_size=0.2, sequence_length=24):
    """
    Rolling-Window-Splits für Zeitreihen-Daten mit vollständigem Train/Val/Test
    auf Basis von Daten und Zielvariablen (target).
    
    Args:
        data (np.array): Eingabedaten, z. B. PCA-Features
        target (np.array): Zielvariablen, z. B. Preisvorhersage
        n_folds (int): Anzahl der Folds
        train_size (float): Anteil des Trainingssatzes
        val_size (float): Anteil des Validierungssatzes
        test_size (float): Anteil des Testsatzes
        sequence_length (int): Länge der Zeitschritte in einer Sequenz
    
    Returns:
        list of dict: List mit train, val, test Sets für jeden Fold
    """
    fold_data = []
    data_len = len(data)
    fold_window = int(data_len * (train_size + val_size + test_size))
    step_size = int((data_len - fold_window) / max(n_folds - 1, 1))

    for fold in range(n_folds):
        start = fold * step_size
        end = start + fold_window
        if end > data_len:
            break
        
        train_end = start + int(train_size * fold_window)
        val_end = train_end + int(val_size * fold_window)
        
        # Trainings-, Validierungs- und Testdaten
        train_data = data[start:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:end]
        
        # Zielwerte für jedes Set
        train_target = target[start:train_end]
        val_target = target[train_end:val_end]
        test_target = target[val_end:end]
        
        # Fitting und Anwenden von Scaler und PCA
        train_data_scaled, val_data_scaled, test_data_scaled = fit_transform_scaler(train_data, val_data, test_data)
        train_data_pca, val_data_pca, test_data_pca = fit_transform_pca(train_data_scaled, val_data_scaled, test_data_scaled)

        # Jetzt Sequenzen schneiden
        train_seq, train_target_seq = create_sequences(train_data_pca, train_target, sequence_length)
        val_seq, val_target_seq = create_sequences(val_data_pca, val_target, sequence_length)
        test_seq, test_target_seq = create_sequences(test_data_pca, test_target, sequence_length)
        
        # Speichern der Folds
        fold_data.append({
            "train": (train_seq, train_target_seq),
            "val": (val_seq, val_target_seq),
            "test": (test_seq, test_target_seq)
        })
    
    return fold_data