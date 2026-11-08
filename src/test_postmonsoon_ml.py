import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

# Load data
raw = pd.read_csv('zindi data/Phase 3 Train.csv')
raw['ts'] = pd.to_datetime(raw.timestamp_ist.str[:19])
raw['v'] = raw.pm25.clip(upper=300)
p = raw.pivot_table(index='ts', columns='location_id', values='v', aggfunc='mean')
full = pd.date_range(p.index.min().floor('D'), '2026-11-03 00:00', freq='h')
p = p.reindex(full)

# Load weather
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['ts'] = pd.to_datetime(w['time']).dt.tz_localize(None)

# Let's inspect the holdout: Oct 31 00:00 to Nov 2 23:00 (last 72h of train)
train_end = pd.Timestamp('2026-10-31 00:00')
valid_end = pd.Timestamp('2026-11-03 00:00')

print('Testing models on holdout:', train_end, 'to', valid_end)

# Let's build tabular dataset with rich post-monsoon features:
# 1. Hour of day (cyclical sin/cos + categorical)
# 2. Station ID
# 3. Rolling station means (m1, m3, m7, m14)
# 4. Station hourly diurnal profile (calculated from last 21-30 days)
# 5. Weather: wind_speed, temp, rh, stagnation, u_wind, v_wind
# 6. Interaction: wind * diurnal, temp * diurnal

rows = []
for s in p.columns:
    series = p[s]
    w_s = w[w.location_id == s].set_index('ts') if 'location_id' in w.columns else w.set_index('ts')
    
    # Let's build hourly rows for October 1 onwards
    dates = pd.date_range('2026-10-01 00:00', '2026-11-02 23:00', freq='h')
    for t in dates:
        # Features use only data strictly before or at cutoff if simulating cutoff,
        # or for standard forecasting, features use available lag
        pass

print('Ready to test tabular post-monsoon modeling.')

