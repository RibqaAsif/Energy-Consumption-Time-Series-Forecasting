from prophet import Prophet
import pandas as pd

def train_prophet(df):
    prophet_df = df.reset_index()[['Datetime', 'Global_active_power']]
    prophet_df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    return forecast