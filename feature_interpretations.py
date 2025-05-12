# The goal of the following functions is getting an insight into the features and their importance in the forecasting process even though the features are transformed by pca.

# ============================================================================
# Imports
# ============================================================================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import torch

from workflowfunction_utils import get_time_series_split_indices, create_sequences_for_folds, preprocessing, load_model

# ============================================================================
# Visualize and describe feature transformation with PCA 
# ============================================================================

# Plot Heatmap for PCA Loadings
def plot_top_pca_loadings(pca, feature_names, top_percent=0.3):
    # get PCA-Loadings 
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )

    # Mean of loadings over all PCs for each feature 
    mean_abs_loading = loadings.abs().mean(axis=1)

    # get X % top features
    top_n = int(len(mean_abs_loading) * top_percent)
    top_features = mean_abs_loading.sort_values(ascending=False).head(top_n).index

    # plot only the top features
    top_loadings = loadings.loc[top_features]

    # Heatmap
    plt.figure(figsize=(12, 0.5 * len(top_features)))
    sns.heatmap(top_loadings, cmap='coolwarm', center=0, annot=False)
    plt.title(f'Top {int(top_percent*100)}% PCA-Loadings (nach mittlerem Einfluss)')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    """
    filename = f"top_{top_percent *100}%_loadings.png"
    filepath = os.path.join("plots", filename)

    # Save & close
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    """

# Plot cumulated explained variance over amount of pcs
def plot_cumulative_explained_variance(pca, threshold_lines=[0.8, 0.9]):

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.arange(1, len(cum_var) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(num_components, cum_var, marker='o', label='Kumulative Varianz')
    for thresh in threshold_lines:
        plt.axhline(y=thresh, color='r', linestyle='--', label=f'{int(thresh*100)}% Schwelle')
    plt.xlabel('Anzahl Komponenten')
    plt.ylabel('Kumulative erklärte Varianz')
    plt.title('Kumulative erklärte Varianz der PCA')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    filename = f"cumulative_explained_variance_pca.png"
    filepath = os.path.join("plots", filename)

    # Save & close
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    """

# Wrapper to plot necessary pca information
def plot_pca_information(df_final, train_frac, val_frac, test_frac,variance_ratio, top_pca_percentage):

    # Divide data in PCA-features und target
    X = df_final[df_final.columns.drop('price actual')].values
    y = np.array(df_final['price actual']).reshape(-1,1)
    feature_names = df_final[df_final.columns.drop('price actual')].columns
    
    # Splitting Features 
    splits = get_time_series_split_indices(len(X), train_frac, val_frac, test_frac)
    initial_split = [{
    "train": (X[splits["train_idx"][0] : splits["train_idx"][1]], y[splits["train_idx"][0] : splits["train_idx"][1]]),
    "val": (X[splits["val_idx"][0] : splits["val_idx"][1]], y[splits["val_idx"][0] : splits["val_idx"][1]]),
    "test": (X[splits["test_idx"][0] : splits["test_idx"][1]], y[splits["test_idx"][0] : splits["test_idx"][1]])
    }]

    # Plot of the cumulative explained variance across the principal components
    initial_split_preprocessed, pcas, scalers_X, scalers_y = preprocessing(initial_split, variance_ratio = None, return_pca_scaler= True)
    plot_cumulative_explained_variance(pcas[0])

    # Top x% of features that contribute the most (on average) to the principal components
    initial_split_preprocessed, pcas, scalers_X, scalers_y = preprocessing(initial_split, variance_ratio, return_pca_scaler= True)
    plot_top_pca_loadings(pcas[0], feature_names, top_pca_percentage)


# ============================================================================
# Feature interpretation with Shap-Values
# ============================================================================

# Prediction function required by SHAP KernelExplainer for computing feature attributions
def model_predict(inputFlat, finalModel, seqLen, inputDim):
    inputSeq = inputFlat.reshape((-1, seqLen, inputDim))
    tensor = torch.tensor(inputSeq, dtype = torch.float32)
    with torch.no_grad():
        finalModel.eval()
        output = finalModel(tensor).cpu().numpy()
    return output

# Determine Shap_Values for all PCs and the price feature
def get_shap_feature_importance(finalModel, inputSample, pca, seqLen, inputDim):

    # SHAP needs 2D: (samples, features) → flatten sequences
    inputFlat = inputSample.reshape((inputSample.shape[0], inputSample.shape[1]* inputSample.shape[2])) # shape: (inputSample len, 24*16)

    # use KernelExplainer 
    model_fn = lambda x: model_predict(x, finalModel, seqLen, inputDim)
    explainer = shap.KernelExplainer(model_fn, inputFlat)
    shapValues = explainer.shap_values(inputFlat, n_samples = 100)

    # Approximate backprojection to original features
    num_pca_components = pca.components_.shape[0]
    shapPcaSeq = np.array(shapValues).reshape(inputSample.shape)
    shapPcaOnly = shapPcaSeq[:, :, :num_pca_components]  # SHAP-Values for PCA-Componenets
    shapTargetFeature = shapPcaSeq[:, :, num_pca_components]  # SHAP-Values for feature 'Price_actual'
    shapPcaOrig = np.matmul(shapPcaOnly, pca.components_)
    shapCombined = np.concatenate([shapPcaOrig, shapTargetFeature[..., np.newaxis]], axis=-1)

    return shapCombined

# Plot of most important features ranked by shap value
def plot_top_shap_features(shapValues, featureNames, model, seed, topPercent=0.3):

    # Mean shap value per feature
    meanAbsShap = np.mean(np.abs(shapValues), axis=(0, 1))
    
    # Determine Top N features
    nTop = int(len(featureNames) * topPercent)
    topIndices = np.argsort(meanAbsShap)[-nTop:][::-1]
    
    # sorted Names and Values
    topFeatures = [featureNames[i] for i in topIndices]
    topShapVals = meanAbsShap[topIndices]

    # Express Shap Werte in % 
    shapTotal = np.sum(meanAbsShap)  # sum over all Features
    topShapPerc = 100 * topShapVals / shapTotal

    # Plot
    plt.figure(figsize=(10, max(5, nTop // 2)))
    plt.barh(topFeatures[::-1], topShapPerc[::-1])
    plt.xlabel("Anteil an totaler SHAP-Wichtigkeit (%)")
    plt.title(f"Top {int(topPercent*100)}% wichtigste Features")
    plt.tight_layout()
    plt.show()
    """
    filename = f"shap_feature_importance_{model.name}_{seed}.png"
    filepath = os.path.join("plots", filename)

    # Save & close
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    """

# Wrapper to apply interpretation with shap values
def plot_feature_importance_with_shap(model, df_final, train_frac, val_frac, test_frac,variance_ratio, sequence_length, step_size, seed, top_percent, device):

    # Split data in PCA-Features und Target
    X = df_final[df_final.columns.drop('price actual')].values
    y = np.array(df_final['price actual']).reshape(-1,1)
    feature_names = df_final.columns.drop('price actual').tolist()
    feature_names.append('price actual')
    
    # Split features 
    splits = get_time_series_split_indices(len(X), train_frac, val_frac, test_frac)
    initial_split = [{
    "train": (X[splits["train_idx"][0] : splits["train_idx"][1]], y[splits["train_idx"][0] : splits["train_idx"][1]]),
    "val": (X[splits["val_idx"][0] : splits["val_idx"][1]], y[splits["val_idx"][0] : splits["val_idx"][1]]),
    "test": (X[splits["test_idx"][0] : splits["test_idx"][1]], y[splits["test_idx"][0] : splits["test_idx"][1]])
    }]

    # Preprocessing and create sequences
    initial_split_preprocessed, pcas, _, _  = preprocessing(initial_split, variance_ratio, return_pca_scaler= True)
    initial_split_sequenced = create_sequences_for_folds(initial_split_preprocessed, sequence_length, step_size, step=1, single_step=True)
    X_train_seq, _ = initial_split_sequenced[0]["train"]
    amount_features = X_train_seq.shape[2]

    # load model
    model = load_model(model, amount_features, seed, device)

    # Produce and Plot Shap-Values 
    X_sample = initial_split_sequenced[0]['val'][0][:100]
    shapValues = get_shap_feature_importance(model, X_sample, pcas[0], seqLen = sequence_length, inputDim = amount_features)
    plot_top_shap_features(shapValues, feature_names, model, seed, top_percent)

