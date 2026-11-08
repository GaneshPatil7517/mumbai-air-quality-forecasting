import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

df = pd.merge(train, w, on=['location_id', 'dt'], how='inner')
nov = df[df['dt'] >= '2026-11-01'].copy()
nov['day'] = nov['dt'].dt.date
nov['hour'] = nov['dt'].dt.hour

d1 = pd.to_datetime('2026-11-01').date()
d2 = pd.to_datetime('2026-11-02').date()

nov1 = nov[nov['day'] == d1].copy()
nov2 = nov[nov['day'] == d2].copy()

print("=" * 70)
print("BACKTEST SIMULATION: PREDICTING NOV 2 USING NOV 1 GROUND TRUTH + WEATHER")
print("=" * 70)

# 1. Baseline: Flat station mean of Nov 1
station_means_d1 = nov1.groupby('location_id')['pm25'].mean()
nov2['pred_flat'] = nov2['location_id'].map(station_means_d1)
rmse_flat = np.sqrt(mean_squared_error(nov2['pm25'], nov2['pred_flat']))
print(f"1. Flat Nov 1 Station Mean RMSE on Nov 2: {rmse_flat:.4f}")

# 2. Add Diurnal Hourly Shape of Nov 1
nov1_diurnal = nov1.groupby(['location_id', 'hour'])['pm25'].mean()
nov2['pred_diurnal'] = nov2.set_index(['location_id', 'hour']).index.map(nov1_diurnal)
nov2['pred_diurnal'] = nov2['pred_diurnal'].fillna(nov2['pred_flat'])
rmse_diurnal = np.sqrt(mean_squared_error(nov2['pm25'], nov2['pred_diurnal']))
print(f"2. Station Mean + Diurnal Shape RMSE on Nov 2: {rmse_diurnal:.4f}")

# 3. Add Station Daily Drift (Nov 1 mean minus slight seasonal cooling -1.5)
nov2['pred_calibrated'] = nov2['pred_diurnal'] - 1.5
rmse_calibrated = np.sqrt(mean_squared_error(nov2['pm25'], nov2['pred_calibrated']))
print(f"3. Calibrated Diurnal Drift RMSE on Nov 2: {rmse_calibrated:.4f}")

# 4. Station breakdown of errors:
print("\nStation Breakdown of Errors with Model 3:")
for loc, grp in nov2.groupby('location_id'):
    st_rmse = np.sqrt(mean_squared_error(grp['pm25'], grp['pred_calibrated']))
    print(f"  Station {loc} ({grp['station_name'].iloc[0]}): RMSE = {st_rmse:.2f} | True Mean = {grp['pm25'].mean():.2f} | Pred Mean = {grp['pred_calibrated'].mean():.2f}")
print("=" * 70)
