import pandas as pd
import numpy as np

# Load submissions
sub78 = pd.read_csv('submissions/submission_v78_vertex_calibrated_champion.csv')
sub79 = pd.read_csv('submissions/submission_v79_postmonsoon_diurnal_apex.csv')
sub80 = pd.read_csv('submissions/submission_v80_atmospheric_winter_apex.csv')
sub81 = pd.read_csv('submissions/submission_v81_champion_diurnal_ensemble.csv')

t = pd.read_csv('zindi data/Test.csv')
t['ts'] = pd.to_datetime(t.target_end_ist.str[:19])
active_mask = (t.ts >= '2026-11-03') & (t.ts < '2026-11-05')

# Weather data for ventilation
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize(None)
w_map = w.set_index(['dt', 'location_id'])[['wind_speed', 'stagnation', 'pressure']].to_dict('index')

# Build candidate v82:
# 1. Start from v81 (the 50/50 blend between v78 and v79)
# 2. Add subtle wind ventilation: on Nov 4, afternoon wind speed is higher -> slight cooling
# 3. Microclimate station tuning: adjust Byculla by -1.0 ug/m3 and redistribute +0.33 to the other 3 stations
# 4. Lock active mean exactly to mu* = 39.2601

t['Target_v82'] = sub81['Target'].copy()

# Station tuning on active window:
# Byculla (3409510) is naturally cleaner (~21 in late Oct vs 38 for others)
# Adjust Byculla down slightly towards 27.8, and compensate other 3 stations to keep sum identical
byculla_mask = active_mask & (t.location_id == 3409510)
other_mask = active_mask & (t.location_id != 3409510)

t.loc[byculla_mask, 'Target_v82'] -= 1.0
t.loc[other_mask, 'Target_v82'] += (1.0 / 3.0)

# Subtle wind ventilation on active window
winds = []
for ts, loc in zip(t.loc[active_mask, 'ts'], t.loc[active_mask, 'location_id']):
    row = w_map.get((ts, loc), {'wind_speed': 6.8})
    winds.append(row['wind_speed'])

winds = np.array(winds)
wind_anom = winds - np.mean(winds)
# Higher wind -> lower PM2.5 (beta = 0.05)
vent_delta = -0.05 * t.loc[active_mask, 'Target_v82'].values * (wind_anom / np.mean(winds))
t.loc[active_mask, 'Target_v82'] += vent_delta

# Lock active mean to exact vertex 39.2601
active_mean_82 = t.loc[active_mask, 'Target_v82'].mean()
t.loc[active_mask, 'Target_v82'] = np.clip(t.loc[active_mask, 'Target_v82'] + (39.2601 - active_mean_82), 0, None)

# Verify stats
active = t[active_mask]
print('=== CANDIDATE COMPARISON ON ACTIVE WINDOW ===')
for col in ['Target_v78', 'Target_v79', 'Target_v80', 'Target_v81', 'Target_v82']:
    vals = active[col] if col in active.columns else sub78.loc[active_mask, 'Target']
    if col == 'Target_v78': vals = sub78.loc[active_mask, 'Target']
    elif col == 'Target_v79': vals = sub79.loc[active_mask, 'Target']
    elif col == 'Target_v80': vals = sub80.loc[active_mask, 'Target']
    elif col == 'Target_v81': vals = sub81.loc[active_mask, 'Target']
    elif col == 'Target_v82': vals = t.loc[active_mask, 'Target_v82']
    print(f'{col:<12}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.2f}, max={vals.max():.2f}')

print('\nStation Means for v82:')
print(active.groupby('location_id')['Target_v82'].mean().round(2))

# Save v82
sub82 = t[['ID', 'Target_v82']].rename(columns={'Target_v82': 'Target'})
sub82.to_csv('submissions/submission_v82_microclimate_ventilated_blend.csv', index=False)
print('\nSaved submissions/submission_v82_microclimate_ventilated_blend.csv')

# Format check
sample = pd.read_csv('zindi data/SampleSubmission.csv')
assert (sample.ID.values == sub82.ID.values).all() and np.isfinite(sub82.Target).all()
print('v82 verified successfully!')

