#Packages 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import import_ipynb
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import json

from workflowfunctions_utils_gpu import set_seed, get_dataloaders, get_tensordatasets, get_predictions, calculate_loss
from models_utils import *

#testweise Ausführung 

with open("data/df_final_eng.pkl", "rb") as f:
        df_final = pickle.load(f)

X = df_final[df_final.columns.drop('price actual')].values
y = df_final['price actual'].values
y = y.reshape(-1, 1)

result = cross_validate_time_series([LSTMModel], [42], X, y, train_size=0.6, val_size=0.2, test_size=0.2, 
                               sequence_length=24, step_size=1, n_folds = 5, n_components=0.8, single_step= True)

print(result)


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

def cross_validate_time_series(models, seeds, data, target, train_size=0.6, val_size=0.2, test_size=0.2, 
                               sequence_length=24, step_size=1, n_folds = 5, n_components=0.8, single_step= True):
    results = []

    for model in models:
        for seed in seeds:
            set_seed(seed)

            # Sliding Window Folds erzeugen
            folds = split_data_time_series_sliding_auto_folds(
                data, target, n_folds=n_folds, slide_fraction= test_size, 
                train_frac= train_size, val_frac= val_size, test_frac= test_size)

            #Schleife über jeden Fold
            for fold_idx, fold in enumerate(folds):
                # Daten extrahieren
                X_train, y_train = fold["train"]
                X_val, y_val = fold["val"]
                X_test, y_test = fold["test"]

                # Scaler fitten und anwenden
                scaler_X, scaler_y = fit_minmax_scalers(X_train, y_train)
                X_train = apply_scaler(scaler_X, X_train)
                X_val = apply_scaler(scaler_X, X_val)
                X_test = apply_scaler(scaler_X, X_test)
                y_train = apply_scaler(scaler_y, y_train)
                y_val = apply_scaler(scaler_y, y_val)
                y_test = apply_scaler(scaler_y, y_test)

                # PCA fitten und Daten transformieren
                pca = fit_pca(X_train, n_components=n_components)
                X_train = apply_pca(pca, X_train)
                X_val = apply_pca(pca, X_val)
                X_test = apply_pca(pca, X_test)

                #Feature-Set vervollständigen PCs + Price_Feature
                X_train = np.concatenate((X_train, y_train), axis=1)
                X_val = np.concatenate((X_val, y_val), axis=1)
                X_test = np.concatenate((X_test, y_test), axis=1)

                # Sequenzen schneiden
                X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length, step_size, single_step)
                X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length, step_size, single_step)
                X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length, step_size, single_step)

                # besten Hyperparameter für den Seed und das Modell laden
                hp = load_best_hp(model.name, seed)

                # Tensodatasets 
                train_dataset, val_dataset, test_dataset = get_tensordatasets(X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq)

                # Modell trainieren
                trained_model = final_model_training_flex(X_train, hp, model, train_dataset, val_dataset, test_dataset, seed, device = "cpu", return_final = True)

                # Modell evaluieren
                predictions = trained_model.predict(X_test_seq)
                y_test_inv = scaler_y.inverse_transform(y_test_seq)
                predictions_inv = scaler_y.inverse_transform(predictions)

                rmse = np.sqrt(np.mean((predictions_inv - y_test_inv)**2))

                results.append({
                    "model": model.name,
                    "seed": seed,
                    "fold": fold_idx,
                    "rmse": rmse
                })

    return results



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

def apply_scaler(scaler, data):
    """
    Wendet einen gefitteten Scaler auf Train-, Val- und Testdaten an.

    Args:
        scaler: Gefitteter Sklearn-Scaler (z.B. MinMaxScaler).
        data: Datensplit

    Returns:
        Tuple: Skalierte Versionen von (data)
    """
    return scaler.transform(data)


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


def apply_pca(pca, data):
    """
    Wendet eine gefittete PCA auf Train-, Val- und Testdaten an.

    Args:
        pca: Gefittetes PCA-Objekt.
        data: Normierte Datensplits.

    Returns:
        Tuple: PCA-transformierte Versionen von (data)
    """
    return pca.transform(data)


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


## vorhandene funktionen, die angepasst werden müssen

# load best hyperparameters 
def load_best_hp(modelName, SEED):
    with open(f"best_hp_all_models/best_hp_{modelName}_{SEED}.json", "r") as f:
        best_hp = json.load(f)
    return best_hp #Anpassung

#return final model
def final_model_training_flex(X_train, best_hp, Model, train_dataset, val_dataset, test_dataset, SEED, device, return_final = False):
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
            best_model = final_model
            torch.save(final_model.state_dict(), f"saved_models/{Model.name}_model_final_{SEED}.pth") #save best weights

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping nach {epoch+1} Epochen.")
                break
    
    if return_final:
        return best_model
    else:
        return train_loss_history, val_loss_history 
    

