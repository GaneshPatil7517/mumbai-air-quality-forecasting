import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
df = train[train['dt'] >= '2026-10-25'].copy()
df['date'] = df['dt'].dt.date

print("Daily Station Means (Oct 27 - Nov 2):")
pivot = df.pivot_table(index='date', columns='location_id', values='pm25', aggfunc='mean')
pivot['Overall'] = df.groupby('date')['pm25'].mean()
print(pivot.to_string())

print("\nLinear trend (slope per day) over the last 5 days (Oct 29 - Nov 2):")
for col in pivot.columns:
    s = pivot[col].dropna()
    x = np.arange(len(s))
    slope, intercept = np.polyfit(x, s.values, 1)
    print(f"  {col}: slope = {slope:+.3f} ug/m3 per day")

