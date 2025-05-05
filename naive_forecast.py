from cv_right import *

def naive_forecast(initial_split_preprocessed, scaler_y):

    _, y_test= initial_split_preprocessed[0]["test"]
    seasonal_lag = 7*24 #lag of one week 
    y_test["predictions"] = y_test["actuals"].shift(seasonal_lag)
    y_test=y_test.iloc[168:]

    mse_scaled = mean_squared_error(y_test["actuals"], y_test["predictions"])
    rmse_scaled = np.sqrt(mse_scaled)

    y_test['actuals'] = scaler_y.inverse_transform(y_test['actuals'].values.reshape(-1, 1))
    y_test['predictions'] = scaler_y.inverse_transform(y_test['predictions'].values.reshape(-1, 1))
    mse_orig = mean_squared_error(y_test["actuals"], y_test["predictions"])
    rmse_orig = np.sqrt(mse_orig)

    return mse_scaled, rmse_scaled, mse_orig, rmse_orig