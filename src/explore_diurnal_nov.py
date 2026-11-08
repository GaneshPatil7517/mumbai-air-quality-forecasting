import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

print("=" * 70)
print("PHASE 3 EMPIRICAL DIURNAL & TRANSITION DYNAMICS")
print("=" * 70)

# Filter recent late-Oct to Nov 2 data
recent = train[train['dt'] >= '2026-10-27'].copy()
recent['hour'] = recent['dt'].dt.hour
recent['day'] = recent['dt'].dt.date

# 1. Station diurnal deviations (hourly mean - station daily mean)
for loc in sorted(recent['location_id'].unique()):
    st_data = recent[recent['location_id'] == loc]
    name = st_data['station_name'].iloc[0]
    print(f"\nStation {loc} ({name}):")
    # Hourly mean in Nov 1-2
    nov_st = st_data[st_data['dt'] >= '2026-11-01']
    hourly_mean = nov_st.groupby('hour')['pm25'].mean()
    base_mean = nov_st['pm25'].mean()
    hourly_anomaly = hourly_mean - base_mean
    print(f"  Nov 1-2 Base Mean: {base_mean:.2f}")
    print(f"  Peak Morning Rush (8-10h):   {hourly_anomaly.loc[8:10].mean():+.2f}")
    print(f"  Afternoon Sea Breeze (14-16h): {hourly_anomaly.loc[14:16].mean():+.2f}")
    print(f"  Evening Peak (20-22h):       {hourly_anomaly.loc[20:22].mean():+.2f}")
    print(f"  Night Baseline (1-4h):       {hourly_anomaly.loc[1:4].mean():+.2f}")

print("=" * 70)

