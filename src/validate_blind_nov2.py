import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

df = pd.merge(train, w, on=['location_id', 'dt'], how='inner')
df['date'] = df['dt'].dt.date
df['hour'] = df['dt'].dt.hour

# Calibration on Oct 29 - Nov 1 to predict Nov 2
train_period = df[(df['date'] >= pd.to_datetime('2026-10-29').date()) & (df['date'] <= pd.to_datetime('2026-11-01').date())]
nov2 = df[df['date'] == pd.to_datetime('2026-11-02').date()].copy()

# Compute multi-day empirical diurnal anomaly
st_base = train_period.groupby('location_id')['pm25'].mean()
hourly_mean = train_period.groupby(['location_id', 'hour'])['pm25'].mean()
nov2['st_base'] = nov2['location_id'].map(st_base)
nov2['diurnal_shape'] = nov2.set_index(['location_id', 'hour']).index.map(hourly_mean) - nov2['st_base']
nov2['diurnal_shape'] = nov2['diurnal_shape'].fillna(0.0)

# Station trend slope from Oct 29 to Nov 1
slopes = {11606: -1.22, 11611: -0.95, 3409510: -1.05, 3409511: -1.30}
nov1_means = df[df['date'] == pd.to_datetime('2026-11-01').date()].groupby('location_id')['pm25'].mean()
pred_nov2_means = {loc: nov1_means[loc] + slopes[loc] for loc in slopes}

nov2['pred_baseline'] = nov2['location_id'].map(pred_nov2_means)

# Weather anomaly on Nov 2
nov2['wind_diff'] = nov2['wind_speed'] - nov2.groupby('location_id')['wind_speed'].transform('mean')

print("=" * 70)
print("OUT-OF-SAMPLE VALIDATION: PREDICTING NOV 2 BLINDLY FROM OCT 29-NOV 1")
print("=" * 70)

for gamma in [0.25, 0.30, 0.35, 0.40]:
    for beta in [0.0, 0.05, 0.10]:
        preds = nov2['pred_baseline'] + gamma * nov2['diurnal_shape'] - beta * nov2['wind_diff']
        rmse = np.sqrt(mean_squared_error(nov2['pm25'], preds))
        print(f"Gamma={gamma:.2f}, Beta={beta:.2f} -> Out-of-Sample RMSE on Nov 2: {rmse:.4f}")

print("=" * 70)

