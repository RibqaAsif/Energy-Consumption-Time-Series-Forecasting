import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path, sep=';', parse_dates={'Datetime': ['Date', 'Time']},
                     infer_datetime_format=True, na_values='?', low_memory=False)
    df.dropna(inplace=True)
    df.set_index('Datetime', inplace=True)
    df = df.sort_index()
    df = df.resample('H').mean()  # Hourly resampling
    return df