import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

df = pd.merge(train, w, on=['location_id', 'dt'], how='inner')
nov = df[df['dt'] >= '2026-10-25'].copy()
nov['day'] = nov['dt'].dt.date
nov['hour'] = nov['dt'].dt.hour

d1 = pd.to_datetime('2026-11-01').date()
d2 = pd.to_datetime('2026-11-02').date()

multi_day = nov[(nov['day'] >= pd.to_datetime('2026-10-27').date()) & (nov['day'] <= d1)]
st_mean = multi_day.groupby('location_id')['pm25'].mean()
hourly_mean = multi_day.groupby(['location_id', 'hour'])['pm25'].mean()

nov2 = nov[nov['day'] == d2].copy()
nov2['base_mean'] = nov2['location_id'].map(st_mean)
nov2['raw_diurnal'] = nov2.set_index(['location_id', 'hour']).index.map(hourly_mean) - nov2['base_mean']
nov2['raw_diurnal'] = nov2['raw_diurnal'].fillna(0.0)

# Weather anomaly on Nov 2
nov2['wind_diff'] = nov2['wind_speed'] - nov2.groupby('location_id')['wind_speed'].transform('mean')
nov2['temp_diff'] = nov2['temp'] - nov2.groupby('location_id')['temp'].transform('mean')

nov2_st_mean = nov[nov['day'] == d2].groupby('location_id')['pm25'].mean()

print("=" * 70)
print("TESTING GAMMA = 0.30 + ATMOSPHERIC VENTILATION ON NOV 2")
print("=" * 70)
for beta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    preds = nov2['location_id'].map(nov2_st_mean) + 0.30 * nov2['raw_diurnal'] - beta * nov2['wind_diff']
    rmse = np.sqrt(mean_squared_error(nov2['pm25'], preds))
    print(f"Beta = {beta:.2f} | RMSE = {rmse:.4f}")
print("=" * 70)

