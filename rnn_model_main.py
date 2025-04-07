from rnn_utils import *
from universal_utils import *

def main_rnn(seed):
    set_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test, df_final_viz = import_data()

    set_num_cpu_threads()

    train_dataset, val_dataset, test_dataset= get_tensordatasets(X_train, y_train, X_val, y_val, X_test, y_test)
    study, best_hp = hyperparameter_tuning(X_train, SimpleRNN, train_dataset, val_dataset, test_dataset,suggest_rnn_hyperparameters, seed)
    final_model, _, _ = final_model_training(X_train, best_hp, SimpleRNN, "rnn", train_dataset, val_dataset, test_dataset, seed)
    train_predictions, val_predictions, test_predictions, train_predictions_actual, val_predictions_actual, test_predictions_actual =get_predictions(final_model, train_dataset, val_dataset, test_dataset, seed)
    y_train_actual, y_val_actual, y_test_actual = get_unscaled_targets(y_train, y_val, y_test)
    mse_train, rmse_train, mse_val, rmse_val, mse_test, rmse_test, mse_test_scaled, rmse_test_scaled = calculate_loss(y_train_actual, y_val_actual, y_test_actual, train_predictions_actual, val_predictions_actual, test_predictions_actual, y_test, test_predictions)

    return rmse_test_scaled

rmse = main_rnn(42)
print(rmse)