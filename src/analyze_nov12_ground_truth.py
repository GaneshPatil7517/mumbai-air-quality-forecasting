import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])

nov_data = train[(train['dt'] >= '2026-11-01') & (train['dt'] <= '2026-11-02 23:59:59')].copy()
print("=" * 70)
print("NOVEMBER 1-2 GROUND TRUTH ATMOSPHERIC ANALYSIS")
print("=" * 70)
print(f"Total observations in Nov 1-2: {len(nov_data)}")
print("\nPer-Station Breakdown for Nov 1-2:")
for loc, grp in nov_data.groupby('location_id'):
    name = grp['station_name'].iloc[0]
    print(f"  Station {loc:7d} ({name:25s}): Mean = {grp['pm25'].mean():5.2f} | Std = {grp['pm25'].std():5.2f} | Min = {grp['pm25'].min():5.2f} | Max = {grp['pm25'].max():5.2f} | N = {len(grp)}")

print("\nDaily Means:")
nov_data['date'] = nov_data['dt'].dt.date
for d, grp in nov_data.groupby('date'):
    print(f"  Date {d}: Mean PM2.5 = {grp['pm25'].mean():5.2f} ug/m3 (N = {len(grp)})")

print("\nOct 27-28 vs Oct 29-31 vs Nov 1-2 Trajectory:")
for period, start, end in [
    ("Phase 2 Test (Oct 27-28)", "2026-10-27", "2026-10-28 23:59:59"),
    ("Late Oct (Oct 29-31)", "2026-10-29", "2026-10-31 23:59:59"),
    ("Early Nov (Nov 1-2)", "2026-11-01", "2026-11-02 23:59:59")
]:
    sub = train[(train['dt'] >= start) & (train['dt'] <= end)]
    print(f"  {period:25s}: Mean = {sub['pm25'].mean():.2f} ug/m3 | N = {len(sub)}")
    for loc, grp in sub.groupby('location_id'):
        print(f"     Station {loc}: {grp['pm25'].mean():.2f} ug/m3")
print("=" * 70)

