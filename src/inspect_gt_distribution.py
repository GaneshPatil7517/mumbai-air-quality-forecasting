"""
Detailed investigation of ground truth on Nov 1-2
"""
import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
nov = train[train['dt'] >= '2026-11-01'].copy()
nov['date'] = nov['dt'].dt.date
nov['hour'] = nov['dt'].dt.hour

print("=" * 70)
print("GROUND TRUTH DISTRIBUTION ON NOV 1-2")
print("=" * 70)
print(f"Overall Mean: {nov['pm25'].mean():.4f}")
print(f"Overall Std:  {nov['pm25'].std():.4f}")
print(f"Overall Min:  {nov['pm25'].min():.4f}")
print(f"Overall Max:  {nov['pm25'].max():.4f}")

for loc, grp in nov.groupby('location_id'):
    print(f"\nStation {loc} ({grp['station_name'].iloc[0]}):")
    print(f"  Count: {len(grp)}, Mean: {grp['pm25'].mean():.2f}, Std: {grp['pm25'].std():.2f}")
    print(f"  Min: {grp['pm25'].min():.2f}, Max: {grp['pm25'].max():.2f}")
    # Correlation with hour
    print(f"  Hourly correlation with PM2.5: {grp[['hour', 'pm25']].corr().iloc[0, 1]:.3f}")

print("=" * 70)

