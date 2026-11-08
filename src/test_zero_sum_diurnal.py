"""
Compute strictly zero-sum diurnal profile per station from Nov 1-2
and test its impact on v74
"""
import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
nov = train[train['dt'] >= '2026-11-01'].copy()
nov['hour'] = nov['dt'].dt.hour

test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

v74 = pd.read_csv('submission_v74_phase3_vertex_apex.csv')
nov34 = test[nov34_mask].copy()
nov34['v74'] = v74.loc[nov34_mask, 'Target'].values
nov34['hour'] = nov34['dt_start'].dt.hour
nov34['day'] = nov34['dt_start'].dt.day

# Compute zero-sum diurnal profile per station
diurnal_map = {}
for loc in sorted(nov['location_id'].unique()):
    st = nov[nov['location_id'] == loc]
    h_means = st.groupby('hour')['pm25'].mean()
    # Interpolate missing hours
    h_means = h_means.reindex(range(24)).interpolate(method='linear').bfill().ffill()
    # Strictly zero-sum: subtract mean of the 24 hours
    zero_sum_h = h_means - h_means.mean()
    print(f"Station {loc}: Zero-sum diurnal range [{zero_sum_h.min():+.2f}, {zero_sum_h.max():+.2f}], sum = {zero_sum_h.sum():.6f}")
    for h in range(24):
        diurnal_map[(loc, h)] = zero_sum_h.loc[h]

# Map to nov34
nov34['diurnal_mod'] = [diurnal_map.get((loc, h), 0.0) for loc, h in zip(nov34['location_id'], nov34['hour'])]

print("\nVerifying Zero-Sum Invariant on Active Window:")
for loc, grp in nov34.groupby('location_id'):
    for d, d_grp in grp.groupby('day'):
        print(f"  Station {loc}, Day {d}: sum of diurnal mod = {d_grp['diurnal_mod'].sum():.6f}")

# Test damping factor lambda on top of v74
print("\nTesting Diurnal Damping Lambda on v74:")
for lam in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
    test_pred = nov34['v74'] + lam * nov34['diurnal_mod']
    print(f"  Lambda = {lam:.2f} -> Mean = {test_pred.mean():.4f}, Std = {test_pred.std():.4f}, Min = {test_pred.min():.2f}, Max = {test_pred.max():.2f}")

