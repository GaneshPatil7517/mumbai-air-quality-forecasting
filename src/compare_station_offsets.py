"""
Compare station means across all periods in Train.csv to find exact station offsets
"""
import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])

oct_test = train[(train['dt'] >= '2026-10-27') & (train['dt'] <= '2026-10-28 23:59:59')]
nov_data = train[(train['dt'] >= '2026-11-01') & (train['dt'] <= '2026-11-02 23:59:59')]

print("Station Means on Oct 27-28 (Mean = 39.10 ug/m3):")
for loc, grp in oct_test.groupby('location_id'):
    print(f"  Station {loc:7d}: Mean = {grp['pm25'].mean():5.2f} (offset = {grp['pm25'].mean() - 39.10:+5.2f})")

print("\nStation Means on Nov 1-2 (Mean = 34.90 ug/m3):")
for loc, grp in nov_data.groupby('location_id'):
    print(f"  Station {loc:7d}: Mean = {grp['pm25'].mean():5.2f} (offset = {grp['pm25'].mean() - 34.90:+5.2f})")

# Combined late Oct / early Nov relative offsets:
recent = train[train['dt'] >= '2026-10-27']
overall_recent_mean = recent['pm25'].mean()
print(f"\nCombined Recent (Oct 27 - Nov 2, Mean = {overall_recent_mean:.2f} ug/m3):")
for loc, grp in recent.groupby('location_id'):
    print(f"  Station {loc:7d}: Mean = {grp['pm25'].mean():5.2f} (relative offset = {grp['pm25'].mean() - overall_recent_mean:+5.2f})")

