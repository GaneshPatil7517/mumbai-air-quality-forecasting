import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def as_ist(values):
    return pd.to_datetime(values, format='ISO8601').dt.tz_convert('Asia/Kolkata')

train = pd.read_csv('data/raw/Train.csv')
train['hour_end'] = as_ist(train['timestamp_ist'])
train['hour_start'] = train['hour_end'] - pd.Timedelta(hours=1)
train = train.sort_values(['location_id', 'hour_end']).reset_index(drop=True)

# Test on the last 48 hours of Train.csv (October 20-21)
val_end = train['hour_end'].max()  # 2026-10-22 00:00:00
val_start = val_end - pd.Timedelta(days=2)  # 2026-10-20 00:00:00

hist = train[train['hour_end'] <= val_start].copy()
valid = train[(train['hour_end'] > val_start) & (train['hour_end'] <= val_end)].copy()

print(f"Hist max: {hist['hour_end'].max()} | Valid range: {valid['hour_end'].min()} to {valid['hour_end'].max()}")
print(f"Valid count: {len(valid)}")

# Method 1: Recent station diurnal anomaly profile
# Over the last 14 days of history
recent_hist = hist[hist['hour_end'] > val_start - pd.Timedelta(days=14)].copy()
recent_hist['hour'] = recent_hist['hour_start'].dt.hour

st_recent_mean = recent_hist.groupby('location_id')['pm25'].mean()
st_hour_mean = recent_hist.groupby(['location_id', 'hour'])['pm25'].mean()
st_diurnal_anomaly = (st_hour_mean - st_recent_mean).to_dict()

# Recent levels
last_24h = hist[hist['hour_end'] > val_start - pd.Timedelta(hours=24)]
last_24h_mean = last_24h.groupby('location_id')['pm25'].mean().to_dict()
last_48h = hist[hist['hour_end'] > val_start - pd.Timedelta(hours=48)]
last_48h_mean = last_48h.groupby('location_id')['pm25'].mean().to_dict()

valid['hour'] = valid['hour_start'].dt.hour
valid['pred_24h_diurnal'] = valid.apply(
    lambda r: last_24h_mean.get(r['location_id'], r['pm25']) + st_diurnal_anomaly.get((r['location_id'], r['hour']), 0),
    axis=1
)
valid['pred_48h_diurnal'] = valid.apply(
    lambda r: last_48h_mean.get(r['location_id'], r['pm25']) + st_diurnal_anomaly.get((r['location_id'], r['hour']), 0),
    axis=1
)

rmse_24h = np.sqrt(mean_squared_error(valid['pm25'], valid['pred_24h_diurnal']))
rmse_48h = np.sqrt(mean_squared_error(valid['pm25'], valid['pred_48h_diurnal']))
print(f"Recent 24h + Diurnal Profile RMSE: {rmse_24h:.4f}")
print(f"Recent 48h + Diurnal Profile RMSE: {rmse_48h:.4f}")

# Latest hour reading
last_1h = hist.groupby('location_id').last()['pm25'].to_dict()
valid['step'] = ((valid['hour_end'] - val_start) / pd.Timedelta(hours=1)).astype(int)

for alpha in [0.80, 0.85, 0.90, 0.92, 0.95]:
    w = alpha ** valid['step']
    p = w * valid['location_id'].map(last_1h) + (1 - w) * valid['pred_24h_diurnal']
    print(f"Alpha {alpha:.2f} decay RMSE: {np.sqrt(mean_squared_error(valid['pm25'], p)):.4f}")
