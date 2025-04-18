import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import torch

pca = joblib.load("data/pca.pkl")

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
