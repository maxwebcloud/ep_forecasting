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
"""
Reihenfolge in cv:

für jedes Modell
    für jeden Seed
        1. splitten
        für jeden fold
            1. scaler fitten
            2. pca fitten
            3. Transformieren
            4. Sequenzen schneiden
            5. Trainieren
                5.1 get_tensor_datasets()
                5.2 load_best_hp()
                5.3 final_model_training --> funktion anpasses, sodass model zurückgegeben und nicht gespeichert
            6. Evaluieren
                6.1 get_predictions()
                6.2 calculate_loss()
"""


def split_data_time_series_sliding_auto_folds(data, target, n_folds=5, slide_fraction=0.2, 
                                               train_frac=0.6, val_frac=0.2, test_frac=0.2):
    """
    Splittet Zeitreihen-Daten mit Sliding-Window in train/val/test für eine gewünschte Anzahl an Folds.
    Die Fenstergröße wird automatisch bestimmt, sodass n_folds exakt möglich sind.

    Args:
        data (np.array): Eingabedaten 
        target (np.array): Zielwerte
        n_folds (int): Gewünschte Anzahl an Folds
        slide_fraction (float): Anteil, um den das Fenster pro Fold verschoben wird (z.B. 0.2)
        train_frac, val_frac, test_frac (float): Aufteilung des Folds (muss in Summe ≈ 1 sein)

    Returns:
        list of dicts: train/val/test Splits für jeden Fold
    """
    data_len = len(data)
    fold_data = []

    # Gesamtanteil der einzelnen Sets validieren
    total_frac = train_frac + val_frac + test_frac
    assert np.isclose(total_frac, 1.0), "train + val + test fractions müssen 1 ergeben"

    # Automatisch optimale Fenstergröße bestimmen
    slide_steps = n_folds - 1
    window_size = int(data_len / (1 + slide_steps * slide_fraction))

    slide_step = int(window_size * slide_fraction)

    for fold in range(n_folds):
        start = fold * slide_step
        end = start + window_size
        if end > data_len:
            break  # falls der letzte Fold nicht mehr reinpasst

        # Indices für die Sets
        train_end = start + int(train_frac * window_size)
        val_end = train_end + int(val_frac * window_size)

        train_data = data[start:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:end]

        train_target = target[start:train_end]
        val_target = target[train_end:val_end]
        test_target = target[val_end:end]

        fold_data.append({
            "train": (train_data, train_target),
            "val": (val_data, val_target),
            "test": (test_data, test_target)
        })

    return fold_data


def fit_minmax_scalers(X_train, y_train, feature_range=(0, 1)):
    """
    Fit MinMaxScaler für Features (X) und Zielvariable (y) auf Trainingsdaten.

    Args:
        X_train (np.array oder pd.DataFrame): Feature-Matrix
        y_train (np.array oder pd.Series): Zielvektor
        feature_range (tuple): Wertebereich für Skalierung

    Returns:
        tuple: (X_scaler, y_scaler)
    """
    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)

    scaler_X.fit(X_train)
    scaler_y.fit(y_train.reshape(-1, 1))  # reshape für Kompatibilität

    return scaler_X, scaler_y

def apply_scaler_to_splits(scaler, X_train, X_val, X_test):
    """
    Wendet einen gefitteten Scaler auf Train-, Val- und Testdaten an.

    Args:
        scaler: Gefitteter Sklearn-Scaler (z.B. MinMaxScaler).
        X_train, X_val, X_test (np.array): Datensplits.

    Returns:
        Tuple: Skalierte Versionen von (X_train, X_val, X_test)
    """
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


def fit_pca(X_norm_train, variance_ratio=0.80):
    """
    Führt PCA auf normierten Trainingsdaten durch und gibt das gefittete PCA-Objekt zurück.

    Args:
        X_norm_train (np.array): Normierte Trainingsdaten (z. B. durch MinMaxScaler oder StandardScaler).
        variance_ratio (float): Zielanteil der erklärten Varianz (zwischen 0 und 1).

    Returns:
        PCA: Gefittetes PCA-Objekt.
    """
    pca = PCA(n_components=variance_ratio)
    pca.fit(X_norm_train)
    return pca


def apply_pca_to_splits(pca, X_train, X_val, X_test):
    """
    Wendet eine gefittete PCA auf Train-, Val- und Testdaten an.

    Args:
        pca: Gefittetes PCA-Objekt.
        X_train, X_val, X_test (np.array): Normierte Datensplits.

    Returns:
        Tuple: PCA-transformierte Versionen von (X_train, X_val, X_test)
    """
    return pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)


def create_sequences(X, y, history_size, target_size, step=1, single_step=False):
    """
    Erstellt Sequenzen aus gegebenen Features (X) und Zielwerten (y).

    Args:
        X (np.array): Eingabedaten (z.B. X_train, X_val, X_test)
        y (np.array): Zielwerte (z.B. y_train, y_val, y_test)
        history_size (int): Länge der Eingabesequenz
        target_size (int): Länge der Zielsequenz
        step (int): Schrittweite innerhalb der Eingabesequenz
        single_step (bool): Wenn True, nimm nur einen Zielwert, sonst Sequenz

    Returns:
        Tuple (np.array, np.array): Sequenzen (samples, timesteps, features), Zielwerte
    """
    data = []
    labels = []

    for i in range(history_size, len(X) - target_size):
        indices = range(i - history_size, i, step)
        data.append(X[indices])

        if single_step:
            labels.append(y[i + target_size])
        else:
            labels.append(y[i : i + target_size])

    return np.array(data), np.array(labels)
