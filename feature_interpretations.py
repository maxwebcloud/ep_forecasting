import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import torch

#vorrübergehend
from sklearn.decomposition import PCA
from cv_right import fit_minmax_scalers, get_time_series_split_indices


with open("data/df_final_eng.pkl", "rb") as f:
    df_final = pickle.load(f)

featureNames = df_final[df_final.columns.drop('price actual')].columns

#Heatmap für PCA Loadings-----------------------------------------------------------------------------
def plot_top_pca_loadings(pca, feature_names, top_percent=0.3):
    # Lade PCA-Loadings 
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )

    # Mittelwert der absoluten Loadings über alle PCs
    mean_abs_loading = loadings.abs().mean(axis=1)

    # Top X % auswählen
    top_n = int(len(mean_abs_loading) * top_percent)
    top_features = mean_abs_loading.sort_values(ascending=False).head(top_n).index

    # Nur die Top-Features plotten
    top_loadings = loadings.loc[top_features]

    # Heatmap
    plt.figure(figsize=(12, 0.5 * len(top_features)))
    sns.heatmap(top_loadings, cmap='coolwarm', center=0, annot=False)
    plt.title(f'Top {int(top_percent*100)}% PCA-Loadings (nach mittlerem Einfluss)')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

#Features Interpretation ------------------------------------------------------------------------

def model_predict(inputFlat, finalModel, seqLen = 24, inputDim = 17):
    inputSeq = inputFlat.reshape((-1, seqLen, inputDim))
    tensor = torch.tensor(inputSeq, dtype = torch.float32)
    with torch.no_grad():
        finalModel.eval()
        output = finalModel(tensor).cpu().numpy()
    return output

def get_shap_feature_importance(finalModel, inputSample, pca, seqLen = 24, inputDim = 17):
    # SHAP will 2D: (samples, features) → also Sequenzen flatten
    inputFlat = inputSample.reshape((inputSample.shape[0], inputSample.shape[1]* inputSample.shape[2])) # shape: (inputSample len, 24*16)
    # KernelExplainer verwenden
    model_fn = lambda x: model_predict(x, finalModel, seqLen, inputDim)
    explainer = shap.KernelExplainer(model_fn, inputFlat)
    shapValues = explainer.shap_values(inputFlat, n_samples = 100)
    # Rückprojektion auf Originalfeatures (ungefähr)
    shapPcaSeq = np.array(shapValues).reshape(inputSample.shape)
    shapPcaOnly = shapPcaSeq[:, :, :16]  # SHAP-Werte für PCA-Komponenten
    shapTargetFeature = shapPcaSeq[:, :, 16]  # SHAP-Werte für den Preis direkt
    shapPcaOrig = np.matmul(shapPcaOnly, pca.components_)
    shapCombined = np.concatenate([shapPcaOrig, shapTargetFeature[..., np.newaxis]], axis=-1)

    return shapCombined

def plot_top_shap_features(shapValues, featureNames, topPercent=0.3):
    # Mittlerer absolute SHAP-Werte pro Feature
    meanAbsShap = np.mean(np.abs(shapValues), axis=(0, 1))
    
    # Top-N bestimmen
    nTop = int(len(featureNames) * topPercent)
    topIndices = np.argsort(meanAbsShap)[-nTop:][::-1]
    
    # Sortierte Namen und Werte
    topFeatures = [featureNames[i] for i in topIndices]
    topShapVals = meanAbsShap[topIndices]

    # Shap Werte in % ausdrücken
    shapTotal = np.sum(meanAbsShap)  # Summe aller Features
    topShapPerc = 100 * topShapVals / shapTotal

    # Plot
    plt.figure(figsize=(10, max(5, nTop // 2)))
    plt.barh(topFeatures[::-1], topShapPerc[::-1])
    plt.xlabel("Anteil an totaler SHAP-Wichtigkeit (%)")
    plt.title(f"Top {int(topPercent*100)}% wichtigste Features")
    plt.tight_layout()
    plt.show()


def preprocessing(folds, variance_ratio=0.8, return_pca_scaler= False):
    """
    Skaliert und transformiert die Daten in jedem Fold:
    - Skaliert X und y separat per MinMaxScaler
    - Führt PCA auf X durch (trainiert auf X_train)
    - Hängt y als zusätzliches Feature an die PCA-Komponenten

    Args:
        folds (list of dicts): Fold-Daten mit train/val(/test)
        variance_ratio (float): Beibehaltener Varianzanteil für PCA

    Returns:
        list of dicts: Preprozessierte Folds
    """
    processed_folds = []
    pcas, scalers_X, scalers_y = [], [], []
        

    for fold in folds:
        # Daten extrahieren
        X_train, y_train = fold["train"]
        X_val, y_val = fold["val"]
        X_test, y_test = fold.get("test", (None, None))  # optional

        # Skalierung
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
   

        # y als Feature anhängen
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
    

    """
    1.) Funktion zur PCA Loadings/Wichtigkeit mit einem Aufruf
    dann daten importieren
    in X und y teilen

    funktion aufrufen und übergeben: dataframe 
    X, y und featurenames extrahieren
    daten splitten und preprocessing 
    dann plot_top_pca:loadings


    2.) funktion zur Feature Interpretation mit einem Aufruf

    """