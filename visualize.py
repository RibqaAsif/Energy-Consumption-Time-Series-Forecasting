import matplotlib.pyplot as plt

def plot_forecasts(test_index, actual, arima_pred, prophet_pred, xgb_pred):
    plt.figure(figsize=(14,6))
    plt.plot(test_index, actual, label='Actual')
    plt.plot(test_index, arima_pred, label='ARIMA')
    plt.plot(test_index, prophet_pred, label='Prophet')
    plt.plot(test_index, xgb_pred, label='XGBoost')
    plt.legend()
    plt.title('Actual vs Forecasted Energy Consumption')
    plt.xlabel('Time')
    plt.ylabel('Global Active Power (kilowatt)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/forecast_comparison.png')
    plt.show()