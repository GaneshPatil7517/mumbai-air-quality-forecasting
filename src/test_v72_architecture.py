"""
Test and build Phase 3 v72 Champion Architecture
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

v65 = pd.read_csv('submission_v65_multivariate_atmospheric_apex.csv')

d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

# 1. Multi-day diurnal profile over Oct 29 - Nov 2
df_train = pd.merge(train, w, on=['location_id', 'dt'], how='inner')
df_train['date'] = df_train['dt'].dt.date
df_train['hour'] = df_train['dt'].dt.hour

calib_period = df_train[(df_train['date'] >= pd.to_datetime('2026-10-29').date()) & (df_train['date'] <= pd.to_datetime('2026-11-02').date())].copy()
st_base = calib_period.groupby('location_id')['pm25'].mean()
hourly_mean = calib_period.groupby(['location_id', 'hour'])['pm25'].mean()

# 2. Extract active test rows
nov34 = test[nov34_mask].copy()
nov34['day'] = nov34['dt_start'].dt.day
nov34['date'] = nov34['dt_start'].dt.date
nov34['hour'] = nov34['dt_start'].dt.hour

# Merge weather
nov34 = pd.merge(
    nov34,
    w[['location_id', 'dt', 'wind_speed', 'temp', 'pressure']],
    left_on=['location_id', 'dt_end'],
    right_on=['location_id', 'dt'],
    how='left'
)

# 3. Base mean per station and day with secular post-monsoon drift
nov2_date = pd.to_datetime('2026-11-02').date()
nov2_means = df_train[df_train['date'] == nov2_date].groupby('location_id')['pm25'].mean().to_dict()

# Drift per day: ~1.0 ug/m3
daily_drift = {
    11606: 1.10,   # Borivali East
    11611: 0.95,   # Malad West
    3409510: 1.00, # Byculla
    3409511: 1.25  # Shivaji Nagar
}

base_preds = []
for idx, r in nov34.iterrows():
    loc = r['location_id']
    d = r['date']
    if d == d3:
        b = nov2_means[loc] - daily_drift[loc]
    else:
        b = nov2_means[loc] - 2.0 * daily_drift[loc]
    base_preds.append(b)

nov34['pred_base'] = base_preds

# 4. Add Regularized Diurnal Profile (Gamma = 0.50)
diurnal_anom = []
for idx, r in nov34.iterrows():
    loc = r['location_id']
    h = r['hour']
    if (loc, h) in hourly_mean:
        anom = hourly_mean.loc[(loc, h)] - st_base[loc]
    else:
        anom = 0.0
    diurnal_anom.append(anom)

nov34['diurnal_anom'] = diurnal_anom
gamma = 0.50
nov34['pred_diurnal'] = nov34['pred_base'] + gamma * nov34['diurnal_anom']

# 5. Autoregressive Boundary Continuity Bridge for hours 00..03 on Nov 3
# Latest values on Nov 2 at 23:00 IST
nov2_23 = df_train[df_train['dt'] == '2026-11-02 23:00:00+05:30'].set_index('location_id')['pm25'].to_dict()

ar_adjusted = nov34['pred_diurnal'].values.copy()
bridge_hours = [0, 1, 2, 3]
bridge_weights = {0: 0.70, 1: 0.50, 2: 0.30, 3: 0.15}

for i, (_, r) in enumerate(nov34.iterrows()):
    if r['date'] == d3 and r['hour'] in bridge_hours:
        loc = r['location_id']
        h = r['hour']
        w_bridge = bridge_weights[h]
        # Target night value from continuous AR decay
        target_ar = nov2_23.get(loc, r['pred_diurnal'])
        ar_adjusted[i] = w_bridge * target_ar + (1.0 - w_bridge) * r['pred_diurnal']

nov34['pred_ar'] = ar_adjusted

# 6. Multivariate Weather Modulation (Wind, Temp, Pressure)
for c in ['wind_speed', 'temp', 'pressure']:
    nov34[f'{c}_diff'] = nov34[c] - nov34.groupby(['location_id', 'day'])[c].transform('mean')

features = ['wind_speed', 'temp', 'pressure']
for c in features:
    calib_period[f'{c}_diff'] = calib_period[c] - calib_period.groupby(['location_id', 'date'])[c].transform('mean')
calib_period['pm25_diff'] = calib_period['pm25'] - calib_period.groupby(['location_id', 'date'])['pm25'].transform('mean')

weather_mod = np.zeros(len(nov34))
for loc in sorted(nov34['location_id'].unique()):
    mask = (nov34['location_id'] == loc).values
    st = calib_period[calib_period['location_id'] == loc].dropna(subset=[f'{c}_diff' for c in features] + ['pm25_diff'])
    X = st[[f'{c}_diff' for c in features]].values
    y = st['pm25_diff'].values
    model = Ridge(alpha=40.0, fit_intercept=False)
    model.fit(X, y)
    X_test = nov34.loc[mask, [f'{c}_diff' for c in features]].values
    weather_mod[mask] = model.predict(X_test)

eta = 0.05
nov34['final_pred'] = nov34['pred_ar'] + eta * weather_mod
nov34['final_pred'] = np.clip(nov34['final_pred'], 0.0, None)

print("=" * 70)
print("PHASE 3 v72 PREDICTION AUDIT (NOV 3-4, 192 ROWS)")
print("=" * 70)
print(f"Overall Mean: {nov34['final_pred'].mean():.2f} ug/m3")
print(f"Overall Std:  {nov34['final_pred'].std():.2f} ug/m3")
print(f"Overall Min:  {nov34['final_pred'].min():.2f} ug/m3")
print(f"Overall Max:  {nov34['final_pred'].max():.2f} ug/m3")

d3_mean = nov34[nov34['date'] == d3]['final_pred'].mean()
d4_mean = nov34[nov34['date'] == d4]['final_pred'].mean()
print(f"Nov 3 Mean:   {d3_mean:.2f} ug/m3")
print(f"Nov 4 Mean:   {d4_mean:.2f} ug/m3")
print(f"Daily Drop:   {d3_mean - d4_mean:.2f} ug/m3")

print("\nStation Breakdown on Nov 3-4:")
for loc, grp in nov34.groupby('location_id'):
    name = grp['station_name'].iloc[0] if 'station_name' in grp.columns else f"Station {loc}"
    print(f"  Station {loc:7d} ({name:25s}) | Mean: {grp['final_pred'].mean():5.2f} | Std: {grp['final_pred'].std():4.2f} | Min: {grp['final_pred'].min():5.2f} | Max: {grp['final_pred'].max():5.2f}")
print("=" * 70)

