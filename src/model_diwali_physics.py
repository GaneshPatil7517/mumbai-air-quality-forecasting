"""
Analyze Diwali Firecracker Physics for Nov 7-8
==============================================
Empirical anchors from Train Phase 4 on Nov 9 morning:
- Station 11611 (Malad West): 590.75 ug/m3 at 02:00 IST
- Station 3409510 (Byculla): 351.28 ug/m3 at 02:00 IST
- Station 3409511 (Shivaji Nagar): 151.81 ug/m3 at 04:00 IST
- Station 11606 (Borivali East): 52.26 ug/m3 at 02:00 IST (Protected by Sanjay Gandhi National Park!)
"""

import pandas as pd
import numpy as np

test = pd.read_csv('zindi data/Test.csv')
test['ts'] = pd.to_datetime(test['target_end_ist'].str[:19])
mask_78 = (test['ts'] >= '2026-11-07') & (test['ts'] < '2026-11-09')

test_78 = test[mask_78].copy()
test_78['day'] = test_78['ts'].dt.day
test_78['hour'] = test_78['ts'].dt.hour
test_78['station'] = test_78['location_id'].astype(str)

# Load Nov 5-6 hourly ground truth profiles
train = pd.read_csv('Train Phase 4.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
nov56 = train[(train['dt'] >= '2026-11-05') & (train['dt'] < '2026-11-07')].copy()
nov56['hour'] = nov56['dt'].dt.hour

station_hourly_nov56 = nov56.pivot_table(index='hour', columns='location_id', values='pm25', aggfunc='mean')

# Extrapolate to Nov 7-8 base (pre-firecrackers)
# Nov 5-6 mean was 46.91, Nov 7-8 base is +3.5 higher = ~50.4
# We build an updated base for Nov 7 and Nov 8 daytime:
base_pred = np.zeros(len(test_78))

for idx, (_, row) in enumerate(test_78.iterrows()):
    loc = int(row['station'])
    h = int(row['hour'])
    d = int(row['day'])
    
    # Base diurnal from Nov 5-6
    h_val = station_hourly_nov56.loc[h, loc] if h in station_hourly_nov56.index else station_hourly_nov56[loc].mean()
    # Add gentle winter ramp: Nov 7 is +2.0 from Nov 6, Nov 8 is +3.5
    day_ramp = 2.0 if d == 7 else 3.5
    base_val = h_val + day_ramp
    base_pred[idx] = base_val

test_78['base_extrapolated'] = base_pred
print("Base Extrapolated Means (without Diwali firecrackers):")
for s in ['11606', '11611', '3409510', '3409511']:
    print(f"  Station {s}: Nov 7 = {test_78[(test_78['station']==s) & (test_78['day']==7)]['base_extrapolated'].mean():.2f}, Nov 8 = {test_78[(test_78['station']==s) & (test_78['day']==8)]['base_extrapolated'].mean():.2f}")

# Now, Diwali Firecracker Addition on Sunday Nov 8 evening:
# Starts gently at 18:00, ramps 19:00-21:00, peaks 22:00-23:00
# Target peak at 23:00 leading into Nov 9 02:00 ground truth:
# Station 11611: ~550 ug/m3 at 23:00 (leading to 590 at 02:00)
# Station 3409510: ~320 ug/m3 at 23:00 (leading to 351 at 02:00)
# Station 3409511: ~240 ug/m3 at 23:00
# Station 11606: ~65 ug/m3 at 23:00 (leading to 52 at 02:00)

diwali_surge = np.zeros(len(test_78))

# Hourly firecracker ramp multiplier for Nov 8 evening (18:00 - 23:00)
# 18:00 -> 0.05
# 19:00 -> 0.15
# 20:00 -> 0.35
# 21:00 -> 0.65
# 22:00 -> 0.85
# 23:00 -> 1.00

ramp_factors = {
    18: 0.08,
    19: 0.20,
    20: 0.45,
    21: 0.70,
    22: 0.88,
    23: 1.00
}

peak_additions = {
    '11611': 480.0,   # Malad West peak addition
    '3409510': 280.0, # Byculla peak addition
    '3409511': 200.0, # Shivaji Nagar peak addition
    '11606': 20.0     # Borivali East (National Park buffer)
}

# Also small Choti Diwali ramp on Nov 7 evening (19:00 - 23:00, ~15% of Diwali):
choti_ramp = {19: 0.05, 20: 0.10, 21: 0.15, 22: 0.12, 23: 0.08}

for idx, (_, row) in enumerate(test_78.iterrows()):
    d = int(row['day'])
    h = int(row['hour'])
    s = str(row['station'])
    
    if d == 8 and h in ramp_factors:
        diwali_surge[idx] = ramp_factors[h] * peak_additions[s]
    elif d == 7 and h in choti_ramp:
        diwali_surge[idx] = choti_ramp[h] * peak_additions[s] * 0.20

test_78['diwali_pred'] = test_78['base_extrapolated'] + diwali_surge

print("\n--- Summary of Physics-Informed Diwali Forecast for Nov 7-8 ---")
print(f"Overall Mean on Nov 7-8: {test_78['diwali_pred'].mean():.2f} ug/m3")
print(f"Nov 7 (Saturday) Mean:   {test_78[test_78['day']==7]['diwali_pred'].mean():.2f} ug/m3")
print(f"Nov 8 (Diwali Day) Mean: {test_78[test_78['day']==8]['diwali_pred'].mean():.2f} ug/m3")
print("\nNov 8 Peak Firecracker Hours (20:00 - 23:00) Mean by Station:")
for s in ['11606', '11611', '3409510', '3409511']:
    peak_m = test_78[(test_78['station']==s) & (test_78['day']==8) & (test_78['hour']>=20)]['diwali_pred'].mean()
    print(f"  Station {s}: {peak_m:.1f} ug/m3")

