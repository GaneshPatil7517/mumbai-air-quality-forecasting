import pandas as pd
import numpy as np

train = pd.read_csv('Train.csv')
train['ts'] = pd.to_datetime(train['timestamp_ist'])
recent = train[train['ts'] >= '2026-10-28'].copy()
recent['date'] = recent['ts'].dt.date
recent['hour'] = recent['ts'].dt.hour
recent['station'] = recent['location_id'].astype(str)

print("Daily Means by Station (Oct 28 - Nov 2):")
pivot_mean = recent.pivot_table(index='date', columns='station', values='pm25', aggfunc='mean')
print(pivot_mean)

print("\nDaily Overall Mean across 4 Stations:")
print(pivot_mean.mean(axis=1))

print("\nOverall Station Means (Oct 28 - Nov 2):")
print(recent.groupby('station')['pm25'].mean())

print("\nOverall Station Standard Deviations (Oct 28 - Nov 2):")
print(recent.groupby('station')['pm25'].std())

