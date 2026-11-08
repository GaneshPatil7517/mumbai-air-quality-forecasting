"""
Explore Autoregressive Horizon Forecasting from Nov 2 23:00 ground truth
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])

w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

print("=" * 70)
print("AUTOREGRESSIVE FORECASTING EVALUATION")
print("=" * 70)

# Build hourly continuous series per station
stations = sorted(train['location_id'].unique())
for loc in stations:
    st_train = train[train['location_id'] == loc].sort_values('dt').set_index('dt')
    # Reindex to hourly
    full_idx = pd.date_range(st_train.index.min(), '2026-11-02 23:00:00+05:30', freq='1h')
    st_train = st_train.reindex(full_idx)
    st_train['pm25'] = st_train['pm25'].interpolate(method='time')
    
    # Check last 48 hours of train
    last_48 = st_train['pm25'].iloc[-48:]
    print(f"Station {loc}: Last 48h Mean = {last_48.mean():.2f}, Std = {last_48.std():.2f}, Last value = {st_train['pm25'].iloc[-1]:.2f}")

