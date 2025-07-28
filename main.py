from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_arima import train_arima
from src.model_prophet import train_prophet
from src.model_xgboost import train_xgboost
from src.evaluate import evaluate_models
from src.visualize import plot_forecasts

if __name__ == '__main__':
    df = preprocess_data('data/raw/household_power_consumption.txt')
    df = engineer_features(df)
    
    arima_pred = train_arima(df)
    prophet_forecast = train_prophet(df)
    prophet_pred = prophet_forecast['yhat'][-24:].values
    
    xgb_pred, X_test, y_test = train_xgboost(df)
    
    evaluate_models(y_test.values, arima_pred, prophet_pred, xgb_pred)
    plot_forecasts(X_test.index, y_test.values, arima_pred, prophet_pred, xgb_pred)