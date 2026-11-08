import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
nov = train[train['dt'] >= '2026-10-25'].copy()
nov['day'] = nov['dt'].dt.date
nov['hour'] = nov['dt'].dt.hour

d1 = pd.to_datetime('2026-11-01').date()
d2 = pd.to_datetime('2026-11-02').date()

# Compute robust multi-day diurnal profile over Oct 27 - Nov 1
multi_day = nov[(nov['day'] >= pd.to_datetime('2026-10-27').date()) & (nov['day'] <= d1)]
st_mean = multi_day.groupby('location_id')['pm25'].mean()
hourly_mean = multi_day.groupby(['location_id', 'hour'])['pm25'].mean()

nov2 = nov[nov['day'] == d2].copy()
nov2['base_mean'] = nov2['location_id'].map(st_mean)
nov2['raw_diurnal'] = nov2.set_index(['location_id', 'hour']).index.map(hourly_mean) - nov2['base_mean']
nov2['raw_diurnal'] = nov2['raw_diurnal'].fillna(0.0)

print("=" * 70)
print("TESTING REGULARIZED DIURNAL DAMPING ON NOV 2")
print("=" * 70)

for gamma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    # Using Nov 2 true station baseline mean
    nov2_st_mean = nov[nov['day'] == d2].groupby('location_id')['pm25'].mean()
    preds = nov2['location_id'].map(nov2_st_mean) + gamma * nov2['raw_diurnal']
    rmse = np.sqrt(mean_squared_error(nov2['pm25'], preds))
    print(f"Gamma = {gamma:.2f} | RMSE = {rmse:.4f}")
print("=" * 70)

