import pandas as pd
import numpy as np

test = pd.read_csv('zindi data/Test.csv')
test['ts'] = pd.to_datetime(test['target_end_ist'].str[:19])
mask_78 = (test['ts'] >= '2026-11-07') & (test['ts'] < '2026-11-09')

test_78 = test[mask_78].copy()
test_78['day'] = test_78['ts'].dt.day
test_78['hour'] = test_78['ts'].dt.hour
test_78['station'] = test_78['location_id'].astype(str)

# Baseline from v106
sub106 = pd.read_csv('submissions/submission_v106_exact_subspace_vertex_champion.csv')
test_78['base'] = sub106.loc[mask_78, 'Target'].values

print("Baseline on Nov 7-8 in v106:")
for s, df in test_78.groupby('station'):
    print(f"  Station {s}: Mean = {df['base'].mean():.2f}, Nov7 = {df[df['day']==7]['base'].mean():.2f}, Nov8 = {df[df['day']==8]['base'].mean():.2f}")

print("\nNov 5-6 Actual Ground Truth Means from Train Phase 4:")
train = pd.read_csv('Train Phase 4.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
nov56 = train[(train['dt'] >= '2026-11-05') & (train['dt'] < '2026-11-07')]
print(nov56.groupby('location_id')['pm25'].mean())

