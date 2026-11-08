import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
nov = train[train['dt'] >= '2026-11-01'].copy()
nov['hour'] = nov['dt'].dt.hour

print("Full 24-hour empirical diurnal anomalies (pm25 - station_mean):")
diurnal_dict = {}
for loc in sorted(nov['location_id'].unique()):
    st = nov[nov['location_id'] == loc]
    mean_val = st['pm25'].mean()
    h_means = st.groupby('hour')['pm25'].mean()
    # Reindex to 0..23, interpolate if any missing
    h_means = h_means.reindex(range(24)).interpolate(method='linear').bfill().ffill()
    anomaly = h_means - mean_val
    diurnal_dict[loc] = anomaly
    print(f"\nStation {loc} (Base={mean_val:.2f}):")
    for h in range(24):
        print(f"  Hour {h:02d}: {anomaly.loc[h]:+6.2f} (mean={h_means.loc[h]:5.2f})")

print("=" * 70)

