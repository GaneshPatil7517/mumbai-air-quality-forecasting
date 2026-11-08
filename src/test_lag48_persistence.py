"""
Test 48-hour seasonal persistence (Nov 1 -> Nov 3, Nov 2 -> Nov 4)
"""
import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])

d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])
nov34 = test[nov34_mask].copy()

# Look at Nov 1 and Nov 2 in train
d1 = pd.to_datetime('2026-11-01').date()
d2 = pd.to_datetime('2026-11-02').date()

# Map test rows to 48h earlier (dt_start - 2 days = Nov 1 or Nov 2)
nov34['dt_lag48'] = nov34['dt_start'] - pd.Timedelta(days=2)

merged = pd.merge(
    nov34,
    train[['location_id', 'dt', 'pm25']],
    left_on=['location_id', 'dt_lag48'],
    right_on=['location_id', 'dt'],
    how='left'
)

print(f"Total active test rows: {len(nov34)}")
print(f"Matched lag-48 observations from train: {merged['pm25'].notna().sum()} / {len(nov34)}")

# Fill any missing with station hourly mean of Nov 1-2
nov_train = train[train['dt'] >= '2026-11-01'].copy()
nov_train['hour'] = nov_train['dt'].dt.hour
st_hour_mean = nov_train.groupby(['location_id', 'hour'])['pm25'].mean().to_dict()

merged['hour'] = merged['dt_start'].dt.hour
fallback = [st_hour_mean.get((loc, h), 35.0) for loc, h in zip(merged['location_id'], merged['hour'])]
merged['pred_lag48'] = merged['pm25'].fillna(pd.Series(fallback, index=merged.index))

print(f"Raw lag-48 mean: {merged['pred_lag48'].mean():.4f}, std: {merged['pred_lag48'].std():.4f}")

# Center to exact vertex mu* = 39.0698
target_mu_star = 39.0698
shift = target_mu_star - merged['pred_lag48'].mean()
merged['pred_calibrated'] = merged['pred_lag48'] + shift
print(f"Calibrated lag-48 mean: {merged['pred_calibrated'].mean():.4f}, std: {merged['pred_calibrated'].std():.4f}")
print(f"Min: {merged['pred_calibrated'].min():.2f}, Max: {merged['pred_calibrated'].max():.2f}")

print("\nStation Means with Calibrated Lag-48:")
for loc, grp in merged.groupby('location_id'):
    print(f"  Station {loc:7d}: Mean = {grp['pred_calibrated'].mean():.2f}, Std = {grp['pred_calibrated'].std():.2f}")
