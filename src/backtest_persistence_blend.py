"""
Backtest blending v42 base with Lag-24/48 on Nov 2 ground truth
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])

nov1 = train[train['dt'].dt.date == pd.to_datetime('2026-11-01').date()].copy()
nov2 = train[train['dt'].dt.date == pd.to_datetime('2026-11-02').date()].copy()

nov1 = nov1[['location_id', 'dt', 'pm25']].rename(columns={'dt': 'dt_nov1', 'pm25': 'pm25_nov1'})
nov2['dt_lag24'] = nov2['dt'] - pd.Timedelta(days=1)

m = pd.merge(nov2, nov1, left_on=['location_id', 'dt_lag24'], right_on=['location_id', 'dt_nov1'], how='inner')

y_true = m['pm25'].values
y_lag = m['pm25_nov1'].values

# Shift lag to match true mean of Nov 2
shift = np.mean(y_true) - np.mean(y_lag)
y_lag_cal = y_lag + shift

rmse_lag = np.sqrt(mean_squared_error(y_true, y_lag_cal))
print(f"Lag-24 persistence RMSE on Nov 2: {rmse_lag:.4f} ug/m3")

# What if we blend with flat station mean?
st_means = m.groupby('location_id')['pm25'].transform('mean').values
for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    b = w * st_means + (1 - w) * y_lag_cal
    b = b - np.mean(b) + np.mean(y_true)
    r = np.sqrt(mean_squared_error(y_true, b))
    print(f"  Weight flat={w:.2f}, lag={1-w:.2f} -> RMSE = {r:.4f}")
