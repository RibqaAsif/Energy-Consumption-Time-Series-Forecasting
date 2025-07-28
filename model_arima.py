from statsmodels.tsa.arima.model import ARIMA

def train_arima(df):
    train = df['Global_active_power'][:-24]
    model = ARIMA(train, order=(5,1,0))  # (p,d,q)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=24)
    return forecast