import pandas as pd

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
print('Train max dt per station:')
for loc, grp in train.groupby('location_id'):
    latest_row = grp.sort_values('dt').iloc[-1]
    print(f"  Station {loc}: {latest_row['dt']} with pm25={latest_row['pm25']:.2f}")

test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34 = test[test['dt_start'].dt.date.isin([d3, d4])]
print('\nNov 3-4 dt_start per station in Test.csv:')
for loc, grp in nov34.groupby('location_id'):
    print(f"  Station {loc}: {grp['dt_start'].min()} to {grp['dt_start'].max()} ({len(grp)} hours)")

