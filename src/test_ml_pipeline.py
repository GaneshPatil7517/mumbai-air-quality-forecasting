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

# Let's inspect the correlation and predictability of direct horizon models
# For test set: cutoff is 2026-11-03 00:00, forecasting 48 hours ahead (Nov 3 - Nov 4)
# Let's simulate this exact setup on multiple cutoffs in October 2026!

cutoffs = [
    pd.Timestamp('2026-10-18 00:00'),
    pd.Timestamp('2026-10-21 00:00'),
    pd.Timestamp('2026-10-24 00:00'),
    pd.Timestamp('2026-10-27 00:00'),
    pd.Timestamp('2026-10-30 00:00'),
]

print('Testing direct multi-horizon and recursive models on October cutoffs...')

