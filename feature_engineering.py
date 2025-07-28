def engineer_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df.index.month
    df['day'] = df.index.day

    for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
        df[f'lag_{lag}'] = df['Global_active_power'].shift(lag)

    df['rolling_mean_24'] = df['Global_active_power'].rolling(24).mean()
    df['rolling_std_24'] = df['Global_active_power'].rolling(24).std()
    df.dropna(inplace=True)
    return df