import pandas as pd
import numpy as np

test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
v65 = pd.read_csv('submission_v65_multivariate_atmospheric_apex.csv')
test['v65'] = v65['Target']

d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34 = test[test['dt_start'].dt.date.isin([d3, d4])]

print("=" * 70)
print("WHAT v65 WAS PREDICTING ON NOV 3-4 (THE CAUSE OF 8.87 SCORE)")
print("=" * 70)
print(f"Overall Mean on Nov 3-4: {nov34['v65'].mean():.2f} ug/m3")
print(f"Overall Std on Nov 3-4:  {nov34['v65'].std():.2f} ug/m3")
print(f"Min: {nov34['v65'].min():.2f}, Max: {nov34['v65'].max():.2f}")
print("\nStation Means on Nov 3-4 in v65:")
for loc, grp in nov34.groupby('location_id'):
    print(f"  Station {loc}: Mean={grp['v65'].mean():.2f}, Std={grp['v65'].std():.2f}")

print("\nActual Nov 1-2 Ground Truth vs v65 Nov 3-4:")
train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
nov_gt = train[(train['dt'] >= '2026-11-01') & (train['dt'] <= '2026-11-02 23:59:59')]
for loc, grp in nov_gt.groupby('location_id'):
    print(f"  Station {loc}: Actual Nov 1-2 GT Mean = {grp['pm25'].mean():.2f} vs v65 Nov 3-4 = {nov34[nov34['location_id']==loc]['v65'].mean():.2f}")
print("=" * 70)

