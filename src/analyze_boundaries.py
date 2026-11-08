import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
train['ts'] = pd.to_datetime(train['timestamp_ist'], utc=True)
train['date'] = train['ts'].dt.date
train['hour'] = train['ts'].dt.hour
train['year'] = train['ts'].dt.year
train['month'] = train['ts'].dt.month

# CRITICAL: Only use 2026 October data for boundary interpolation
oct2026 = train[(train['year']==2026) & (train['month']==10)].copy()
print(f"Oct 2026 data shape: {oct2026.shape}")

# Check which days exist
for loc in sorted(oct2026['location_id'].unique()):
    dates = sorted(oct2026[oct2026['location_id']==loc]['date'].unique())
    days = [d.day for d in dates]
    print(f"  Station {loc}: days = {days}")
print()

# IMPORTANT DISCOVERY: Oct 28 has data for some stations!
oct28_data = oct2026[oct2026['date'].apply(lambda d: d.day==28)]
if len(oct28_data) > 0:
    print(f"*** Oct 28 data exists! Shape: {oct28_data.shape}")
    for loc in sorted(oct28_data['location_id'].unique()):
        loc_28 = oct28_data[oct28_data['location_id']==loc]
        hours = sorted(loc_28['hour'].unique())
        mean_val = loc_28['pm25'].mean()
        print(f"  Station {loc}: {len(loc_28)} obs, hours={hours}, mean PM2.5={mean_val:.2f}")
    print()

# Get all boundary data (Oct 20-23 and Oct 28-31 in 2026)
boundary = oct2026[oct2026['date'].apply(lambda d: d.day in list(range(20,32)))].copy()
boundary['day'] = boundary['date'].apply(lambda d: d.day)

# Per-(station, hour) mean for each boundary day
print("=== Per-(station, hour) values for key days ===")
pivot_data = []
for loc in sorted(boundary['location_id'].unique()):
    loc_data = boundary[boundary['location_id']==loc]
    for h in range(24):
        hour_data = loc_data[loc_data['hour']==h]
        row = {'location_id': loc, 'hour': h}
        for day in [20,21,22,23,28,29,30,31]:
            day_vals = hour_data[hour_data['day']==day]['pm25']
            if len(day_vals) > 0:
                row[f'oct{day}'] = day_vals.values[0]
            else:
                row[f'oct{day}'] = np.nan
        pivot_data.append(row)

pivot_df = pd.DataFrame(pivot_data)
print(pivot_df.to_string(max_rows=20))
print()

# Method 1: Pure linear interp Oct23 -> Oct29 per (station, hour)
print("\n=== Method 1: Pure Linear Interp Oct23->Oct29 ===")
m1_v27 = []
m1_v28 = []
for _, row in pivot_df.iterrows():
    v23 = row.get('oct23', np.nan)
    v29 = row.get('oct29', np.nan)
    if pd.notna(v23) and pd.notna(v29):
        v27 = v23 + (v29 - v23) * 4/6
        v28 = v23 + (v29 - v23) * 5/6
    else:
        v27 = np.nan
        v28 = np.nan
    m1_v27.append(v27)
    m1_v28.append(v28)

all_m1 = [v for v in m1_v27 + m1_v28 if pd.notna(v)]
print(f"  Oct 27 mean: {np.nanmean(m1_v27):.4f}")
print(f"  Oct 28 mean: {np.nanmean(m1_v28):.4f}")
print(f"  Overall mean: {np.mean(all_m1):.4f}")
print(f"  Overall std: {np.std(all_m1):.4f}")
print(f"  N valid: {len(all_m1)}")

# Method 2: Weighted linear regression using ALL boundary days
print("\n=== Method 2: Weighted LR (all boundary days) ===")
m2_v27 = []
m2_v28 = []
for _, row in pivot_df.iterrows():
    days = []
    vals = []
    for day in [20,21,22,23,28,29,30,31]:
        v = row.get(f'oct{day}', np.nan)
        if pd.notna(v):
            days.append(day)
            vals.append(v)
    if len(days) >= 2:
        days_arr = np.array(days, dtype=float)
        vals_arr = np.array(vals, dtype=float)
        # Weight = 1 / (|day - 27.5| + 0.5)
        weights = 1.0 / (np.abs(days_arr - 27.5) + 0.5)
        
        w_sum = weights.sum()
        wx = (weights * days_arr).sum()
        wy = (weights * vals_arr).sum()
        wxx = (weights * days_arr**2).sum()
        wxy = (weights * days_arr * vals_arr).sum()
        
        denom = w_sum * wxx - wx**2
        if abs(denom) > 1e-10:
            slope = (w_sum * wxy - wx * wy) / denom
            intercept = (wy - slope * wx) / w_sum
            v27 = intercept + slope * 27
            v28 = intercept + slope * 28
        else:
            v27 = np.mean(vals_arr)
            v28 = np.mean(vals_arr)
    else:
        v27 = np.nan
        v28 = np.nan
    m2_v27.append(v27)
    m2_v28.append(v28)

all_m2 = [v for v in m2_v27 + m2_v28 if pd.notna(v)]
print(f"  Oct 27 mean: {np.nanmean(m2_v27):.4f}")
print(f"  Oct 28 mean: {np.nanmean(m2_v28):.4f}")
print(f"  Overall mean: {np.mean(all_m2):.4f}")
print(f"  Overall std: {np.std(all_m2):.4f}")

# Method 3: If Oct 28 has actual data, use Oct23->Oct28 interp for Oct27
# and Oct28 actual data
print("\n=== Method 3: Using Oct28 actual data if available ===")
for loc in sorted(pivot_df['location_id'].unique()):
    loc_pivot = pivot_df[pivot_df['location_id']==loc]
    n28 = loc_pivot['oct28'].notna().sum()
    print(f"  Station {loc}: {n28}/24 hours have Oct 28 data")

# Method 4: PCHIP interpolation using all boundary days  
from scipy.interpolate import PchipInterpolator
print("\n=== Method 4: PCHIP Interpolation ===")
m4_v27 = []
m4_v28 = []
for _, row in pivot_df.iterrows():
    days = []
    vals = []
    for day in [20,21,22,23,28,29,30,31]:
        v = row.get(f'oct{day}', np.nan)
        if pd.notna(v):
            days.append(day)
            vals.append(v)
    if len(days) >= 3:
        pchip = PchipInterpolator(days, vals)
        v27 = float(pchip(27))
        v28 = float(pchip(28))
    elif len(days) >= 2:
        # Linear fallback
        slope = (vals[-1] - vals[0]) / (days[-1] - days[0])
        v27 = vals[0] + slope * (27 - days[0])
        v28 = vals[0] + slope * (28 - days[0])
    else:
        v27 = np.nan
        v28 = np.nan
    m4_v27.append(v27)
    m4_v28.append(v28)

all_m4 = [v for v in m4_v27 + m4_v28 if pd.notna(v)]
print(f"  Oct 27 mean: {np.nanmean(m4_v27):.4f}")
print(f"  Oct 28 mean: {np.nanmean(m4_v28):.4f}")
print(f"  Overall mean: {np.mean(all_m4):.4f}")
print(f"  Overall std: {np.std(all_m4):.4f}")

# Check v31 submission
print("\n=== V31 (current best) stats ===")
v31 = pd.read_csv('submission_v31_boundary_bridge_champion.csv')
test = pd.read_csv('data/raw/Test.csv')
merged = test.merge(v31, on='ID')
w2 = merged[merged['scoring_block']=='public']
print(f"W2 shape: {w2.shape}")
print(f"W2 mean: {w2['Target'].mean():.4f}")
print(f"W2 std: {w2['Target'].std():.4f}")

# Per-station W2 stats
for loc in sorted(w2['location_id'].unique()):
    loc_w2 = w2[w2['location_id']==loc]
    print(f"  Station {loc}: mean={loc_w2['Target'].mean():.4f}, std={loc_w2['Target'].std():.4f}, n={len(loc_w2)}")
