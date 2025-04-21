#Packages 
import numpy as np
import pickle 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import optuna
from optuna.pruners import HyperbandPruner
from rich import print
from rich.console import Console
from workflowfunctions_utils_gpu import set_seed, get_dataloaders, get_predictions_in_batches, get_device
from models_utils import *


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
                               sequence_length=24, step_size=1, n_folds = 5, sliding_window = True, variance_ratio=0.8, single_step= True):
    results = []

    console = Console()

    suggest_functions = {
        "rnn": suggest_rnn_hyperparameters,
        "lstm": suggest_lstm_hyperparameters,
        "slstm": suggest_slstm_hyperparameters,
        "plstm": suggest_plstm_hyperparameters
        }
    
    device = get_device(use_gpu = False)

    for model in models:
        console.rule(f"[bold magenta]Starte Cross Validation für {model.name.upper()} auf {device}[/bold magenta]")
        for seed in seeds:
            fold_rmses = []

            console.rule(f"\n[medium_purple]{model.name.upper()} mit Seed {seed}[/medium_purple]")
            set_seed(seed)


            if sliding_window:
                # Sliding Window Folds erzeugen
                folds = split_data_time_series_sliding_auto_folds(
                    data, target, n_folds=n_folds, slide_fraction= test_size, 
                    train_frac= train_size, val_frac= val_size, test_frac= test_size)
            else:
                folds = split_data_time_series_expanding_window(data, target, n_splits = 5)
            
            best_rmse = float('inf')
            #Schleife über jeden Fold
            for fold_idx, fold in enumerate(folds):
                console.print(f"\n[bold turquoise2]Fold [{fold_idx+1}/{n_folds}]: [/bold turquoise2]")
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
                pca = fit_pca(X_train, variance_ratio= variance_ratio)
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
                #hp = load_best_hp(model.name, seed)


                # Tensodatasets
                train_dataset, val_dataset, test_dataset = get_tensordatasets(X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq)

                # Hyperparametertuning
                console.print("\n[bold turquoise2]Hyperparametertuning: [/bold turquoise2]")
                study, best_hp = hyperparameter_tuning(X_train_seq, model, train_dataset, val_dataset, test_dataset, suggest_functions[model.name], seed, torch.device("cpu"))

                # Modell trainieren
                console.print("\n[bold turquoise2]Final Training[/bold turquoise2]")
                trained_model = final_model_training_flex(X_train_seq, best_hp, model, train_dataset, val_dataset, test_dataset, seed, device = torch.device("cpu"), return_final = True)

                # Modell evaluieren
                train_loader, test_loader, val_loader = get_dataloaders(seed, train_dataset, val_dataset, test_dataset, shuffle_train = False, batch_size = 64)
                predictions = get_predictions_in_batches(trained_model, dataloader = test_loader, device = torch.device("cpu"))
                y_test_inv = scaler_y.inverse_transform(y_test_seq.reshape(-1,1))
                predictions_inv = scaler_y.inverse_transform(predictions)

                rmse = np.sqrt(np.mean((predictions - y_test_seq.reshape(-1,1))**2))
                fold_rmses.append(rmse)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = trained_model
                    with_seed = seed
                    with_hp = best_hp
            
            # Speichern der Gewichte des besten Modells pro Seed und dessen Hyperparameter
            torch.save(best_model.state_dict(), f"saved_models/{model.name}_model_final_{with_seed}_cv.pth")
            with open(f"best_hp_all_models/best_hp_{model.name}_{with_seed}_cv.json", "w") as f:
                json.dump(with_hp, f)

            avg_rmse = np.mean(fold_rmses)
            results.append({
                "model": model.name,
                "seed": seed,
                "fold": fold_rmses,
                "rmse": avg_rmse
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
    window_size = int(data_len / (1 + (n_folds - 1) * slide_fraction))
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

def split_data_time_series_expanding_window(X, y, n_splits,val_frac = 0.2):
    fold_data = []
    tscv = TimeSeriesSplit(n_splits= n_splits)
    for train_idx, test_idx in tscv.split(X):
        # vom Trainingsset Validation rausnehmen
        val_split = int(len(train_idx) * val_frac)
        val_idx = train_idx[-val_split:]
        real_train_idx = train_idx[:-val_split]
        
        X_train, y_train = X[real_train_idx], y[real_train_idx]
        X_val, y_val     = X[val_idx], y[val_idx]
        X_test, y_test   = X[test_idx], y[test_idx]

        fold_data.append({
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)
        })
    return fold_data

def hyperparameter_tuning(X_train, Model, train_dataset, val_dataset, test_dataset, hp_function, SEED, device):
    
    def objective(trial):
        set_seed(SEED)  # device übergeben
        train_loader, _, val_loader = get_dataloaders(SEED, train_dataset, val_dataset, test_dataset, batch_size = 64)

        hp = hp_function(trial)
        
        model = Model(input_size=X_train.shape[2], hp=hp).to(device)
        model = torch.compile(model)
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

        num_epochs = 15
        patience = 7
        best_val_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(num_epochs):
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

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
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
    pruner = optuna.pruners.HyperbandPruner(min_resource=4, max_resource=17, reduction_factor=3)
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=10, n_jobs=1)

    # Show Best Result
    print("Best trial parameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    
    best_hp = study.best_trial.params
    return study, best_hp

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
    set_seed(SEED)
    train_loader, _, val_loader = get_dataloaders(SEED, train_dataset, val_dataset, test_dataset, batch_size= 32)

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
        #print(f"GPU memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # Reset Counter
            best_model = final_model

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping nach {epoch+1} Epochen.")
                break
    
    if return_final:
        return best_model
    else:
        return train_loss_history, val_loss_history 
    
def get_tensordatasets(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)#.unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1)#.unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)#.unsqueeze(1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    return train_dataset, val_dataset, test_dataset
    
"""
#testweise Ausführung 

with open("data/df_final_eng.pkl", "rb") as f:
        df_final = pickle.load(f)

X = df_final[df_final.columns.drop('price actual')].values
y = np.array(df_final['price actual']).reshape(-1,1)

result = cross_validate_time_series([LSTMModel], [42], X, y, train_size=0.6, val_size=0.2, test_size=0.2, 
                               sequence_length=24, step_size=1, n_folds = 5, variance_ratio=0.8, single_step= True)

print(result)
"""