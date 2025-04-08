from models_utils import *
from workflowfunctions_utils import *
from rich import print
from rich.console import Console



def generate_evaluate_all_models(seeds):

    # necessary list, library, etc
    models = [SimpleRNN, LSTMModel, StackedLSTMModel, PhasedLSTMModel]
    modelNames = ["rnn", "lstm", "slstm", "plstm"]
    suggest_functions = {
        "rnn": suggest_rnn_hyperparameters,
        "lstm": suggest_lstm_hyperparameters,
        "slstm": suggest_slstm_hyperparameters,
        "plstm": suggest_plstm_hyperparameters
    }

    results = {name: [] for name in modelNames}
    X_train, y_train, X_val, y_val, X_test, y_test, df_final_viz = import_data()
    set_num_cpu_threads()
    train_dataset, val_dataset, test_dataset= get_tensordatasets(X_train, y_train, X_val, y_val, X_test, y_test)
    seq_length = X_train.shape[1]

    console = Console()

    # tune, train and evaluate every model for all seeds
    for idx, model in enumerate(models):
        console.rule(f"[bold magenta]Starte Modellworkflow für {modelNames[idx].upper()}[/bold magenta]")
        for seed in seeds:
            set_seed(seed)
            console.rule(f"\n[medium_purple]{modelNames[idx].upper()} mit Seed {seed}[/medium_purple]")
        

            # hyperparametertuning
            console.print("\n[bold turquoise2]Hyperparametertuning: [/bold turquoise2]")
            study, best_hp = hyperparameter_tuning(X_train, model, train_dataset, val_dataset, test_dataset,suggest_functions[modelNames[idx]], seed)
            save_best_hp(modelNames[idx], study, seed)

            # final model training
            console.print("\n[bold turquoise2]Final Training[/bold turquoise2]")
            final_model, train_loss_history, val_loss_history = final_model_training(X_train, best_hp, model, modelNames[idx], train_dataset, val_dataset, test_dataset, seed)
            train_history_plot(train_loss_history, val_loss_history, modelNames[idx], seed)

            # final model evaluation
            console.print("\n[bold turquoise2]Model Evaluation[/bold turquoise2]")
            train_predictions, val_predictions, test_predictions, train_predictions_actual, val_predictions_actual, test_predictions_actual =get_predictions(final_model, train_dataset, val_dataset, test_dataset, seed)
            y_train_actual, y_val_actual, y_test_actual = get_unscaled_targets(y_train, y_val, y_test)
            _, _, _, _, _, _, _, rmse_test_scaled = calculate_loss(y_train_actual, y_val_actual, y_test_actual, train_predictions_actual, val_predictions_actual, test_predictions_actual, y_test, test_predictions)
            plot_forecast(seq_length, df_final_viz, train_predictions_actual, val_predictions_actual, test_predictions_actual, modelNames[idx], seed)
            plot_residuals_with_index(y_test_actual, test_predictions_actual, df_final_viz, seq_length, modelNames[idx], seed)

            # add model performance to the dictionary
            results[modelNames[idx]].append(rmse_test_scaled)

    return results


