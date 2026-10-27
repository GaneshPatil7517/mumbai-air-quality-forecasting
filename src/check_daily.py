import pandas as pd

train = pd.read_csv('data/raw/Train.csv')
train['hour_end'] = pd.to_datetime(train['timestamp_ist'], format='ISO8601')
train['date'] = train['hour_end'].dt.date

recent = train[train['hour_end'] >= '2026-10-15'].copy()
daily = recent.groupby(['date', 'location_id'])['pm25'].mean().unstack()
print("Daily mean PM2.5 per station leading up to Oct 22:")
print(daily)

print("\nOverall daily mean across all stations:")
print(recent.groupby('date')['pm25'].mean())
