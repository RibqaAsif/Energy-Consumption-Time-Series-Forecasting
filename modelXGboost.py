import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_xgboost(df):
    X = df.drop(columns=['Global_active_power'])
    y = df['Global_active_power']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)
    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
    return forecast, X_test, y_test