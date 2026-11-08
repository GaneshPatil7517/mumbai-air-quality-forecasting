import pandas as pd
import numpy as np

# Load data
raw = pd.read_csv('zindi data/Phase 3 Train.csv')
raw['ts'] = pd.to_datetime(raw.timestamp_ist.str[:19])
raw['v'] = raw.pm25.clip(upper=300)
p = raw.pivot_table(index='ts', columns='location_id', values='v', aggfunc='mean')
full = pd.date_range(p.index.min().floor('D'), '2026-11-03 00:00', freq='h')
p = p.reindex(full)

# Load weather
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize(None)

# Let's inspect the correlation between PM2.5 and weather features in the last 30 days of train
# Merge PM2.5 with weather
recent_dates = pd.date_range('2026-10-04 00:00', '2026-11-02 23:00', freq='h')
df_list = []
for s in p.columns:
    df_s = pd.DataFrame({
        'ts': recent_dates,
        'location_id': s,
        'pm25': p[s].reindex(recent_dates).values
    })
    df_list.append(df_s)

all_df = pd.concat(df_list, ignore_index=True)
all_df = all_df.merge(w, left_on=['ts', 'location_id'], right_on=['dt', 'location_id'], how='left')

# Drop NaNs in pm25
valid = all_df.dropna(subset=['pm25']).copy()

# Compute correlations
feats = ['temp', 'rh', 'wind_speed', 'pressure', 'u_wind', 'v_wind', 'stagnation']
print('=== CORRELATIONS WITH PM2.5 IN LAST 30 DAYS OF TRAIN ===')
for f in feats:
    corr = valid['pm25'].corr(valid[f])
    print(f'  {f:<15}: {corr:+.4f}')

# Inspect diurnal-adjusted correlation (i.e. correlation with residual after removing hour mean)
valid['hour'] = valid.ts.dt.hour
hour_mean = valid.groupby(['location_id', 'hour'])['pm25'].transform('mean')
valid['pm25_residual'] = valid['pm25'] - hour_mean

print('\n=== CORRELATIONS WITH PM2.5 RESIDUAL (DIURNAL REMOVED) ===')
for f in feats:
    corr = valid['pm25_residual'].corr(valid[f])
    print(f'  {f:<15}: {corr:+.4f}')

