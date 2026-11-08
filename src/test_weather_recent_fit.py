import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

# Merge train with weather
df = pd.merge(train, w, on=['location_id', 'dt'], how='inner')
recent = df[df['dt'] >= '2026-10-25'].copy()
recent['hour'] = recent['dt'].dt.hour
recent['day'] = recent['dt'].dt.date

features = ['wind_speed', 'temp', 'pressure', 'stagnation']

print(f"Total merged recent rows (Oct 25 - Nov 2): {len(recent)}")
for loc, grp in recent.groupby('location_id'):
    print(f"\nStation {loc} ({grp['station_name'].iloc[0]}): N={len(grp)}")
    X = grp[features].values
    y = grp['pm25'].values
    model = Ridge(alpha=10.0)
    model.fit(X, y)
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"  Direct Weather Fit RMSE: {rmse:.2f}")
    for feat, coef in zip(features, model.coef_):
        print(f"    {feat:12s}: {coef:+.3f}")
    print(f"    Intercept: {model.intercept_:.2f}")

