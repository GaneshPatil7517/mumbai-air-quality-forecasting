import pandas as pd

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
nov = train[train['dt'] >= '2026-11-01'].copy()
nov['date'] = nov['dt'].dt.date

d1 = pd.to_datetime('2026-11-01').date()
d2 = pd.to_datetime('2026-11-02').date()

print("Per-station daily change from Nov 1 to Nov 2:")
for loc, grp in nov.groupby('location_id'):
    m1 = grp[grp['date'] == d1]['pm25'].mean()
    m2 = grp[grp['date'] == d2]['pm25'].mean()
    print(f"Station {loc}: Nov 1 = {m1:.2f}, Nov 2 = {m2:.2f}, Change = {m2 - m1:+.2f}")

