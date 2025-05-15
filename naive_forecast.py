from workflowfunction_utils import *
from sklearn.metrics import mean_squared_error



# Generate a naive forecast with data preprocessed in the same way as for the other models
def naive_forecast(initial_split_preprocessed, scaler_y):

    _, y_test= initial_split_preprocessed[0]["test"]
    seasonal_lag = 24*7 # Lag of one week 
    predictions = y_test[:-seasonal_lag]  # for seasonal_lag=1 → y_{t-1}
    actuals = y_test[seasonal_lag:]

    mse_scaled = mean_squared_error(actuals, predictions)
    rmse_scaled = np.sqrt(mse_scaled)

    actuals = scaler_y.inverse_transform(actuals)
    predictions = scaler_y.inverse_transform(predictions)
    mse_orig = mean_squared_error(actuals, predictions)
    rmse_orig = np.sqrt(mse_orig)

    return mse_scaled, rmse_scaled, mse_orig, rmse_orig


