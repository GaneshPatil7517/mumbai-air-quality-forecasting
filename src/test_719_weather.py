"""
Test atmospheric weather modulation on top of the 7.1963 base
"""
import pandas as pd
import numpy as np

sub = pd.read_csv('submission_719_reproduced.csv')
test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

nov34 = test[nov34_mask].copy()
nov34['pred_719'] = sub.loc[nov34_mask, 'Target'].values
nov34['day'] = nov34['dt_start'].dt.day

w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

nov34 = pd.merge(
    nov34,
    w[['location_id', 'dt', 'wind_speed', 'temp', 'pressure']],
    left_on=['location_id', 'dt_end'],
    right_on=['location_id', 'dt'],
    how='left'
)

nov34['wind_diff'] = nov34['wind_speed'] - nov34.groupby(['location_id', 'day'])['wind_speed'].transform('mean')
nov34['temp_diff'] = nov34['temp'] - nov34.groupby(['location_id', 'day'])['temp'].transform('mean')

print("=" * 70)
print("TESTING WEATHER MODULATION & APEX CALIBRATION ON 7.1963 BASE")
print("=" * 70)

target_mu_star = 39.0698

for beta in [0.0, 0.04, 0.08, 0.12, 0.16]:
    wind_mod = -beta * nov34['wind_diff'].values
    p = nov34['pred_719'].values + wind_mod
    # Shift to vertex
    shift = target_mu_star - p.mean()
    p_cal = p + shift
    print(f"Beta = {beta:.2f} -> Mean = {p_cal.mean():.4f}, Std = {p_cal.std():.4f}, Range = [{p_cal.min():.2f}, {p_cal.max():.2f}]")
    # Check correlation with base
    corr = np.corrcoef(nov34['pred_719'], p_cal)[0, 1]
    print(f"   Corr with 7.19 base: {corr:.4f}")

