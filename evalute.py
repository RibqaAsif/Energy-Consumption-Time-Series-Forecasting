from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_models(actual, arima_pred, prophet_pred, xgb_pred):
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("ARIMA - MAE:", mean_absolute_error(actual, arima_pred))
    print("Prophet - MAE:", mean_absolute_error(actual, prophet_pred))
    print("XGBoost - MAE:", mean_absolute_error(actual, xgb_pred))

    print("ARIMA - RMSE:", mean_squared_error(actual, arima_pred, squared=False))
    print("Prophet - RMSE:", mean_squared_error(actual, prophet_pred, squared=False))
    print("XGBoost - RMSE:", mean_squared_error(actual, xgb_pred, squared=False))

    print("ARIMA - MAPE:", mape(actual, arima_pred))
    print("Prophet - MAPE:", mape(actual, prophet_pred))
    print("XGBoost - MAPE:", mape(actual, xgb_pred))