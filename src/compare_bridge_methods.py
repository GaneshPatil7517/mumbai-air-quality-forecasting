import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('data/raw/Train.csv')
test = pd.read_csv('data/raw/Test.csv')

train['dt'] = pd.to_datetime(train['timestamp_ist'])
train['hour'] = train['dt'].dt.hour
train['date'] = train['dt'].dt.date

test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])

w2_mask = (test['dt_start'] >= '2026-10-27 00:00:00+05:30') & (test['dt_start'] <= '2026-10-28 23:00:00+05:30')
w2 = test[w2_mask].copy()

# ==============================================================================
# Method A: Continuous Hourly Bridge (like v31, but with FIXED end date indexing)
# ==============================================================================
full_dt = pd.date_range('2026-10-20 00:00:00+05:30', '2026-10-31 23:00:00+05:30', freq='h')

# 1. Unbiased Diurnal Harmonics (calculated per complete day so no missing-hour bias)
oct26 = train[(train['dt'].dt.year == 2026) & (train['dt'].dt.month == 10)].copy()
oct26['daily_mean'] = oct26.groupby(['location_id', 'date'])['pm25'].transform('mean')
oct26['daily_anom'] = oct26['pm25'] - oct26['daily_mean']

unbiased_diurnal = {}
for loc in test['location_id'].unique():
    st = oct26[(oct26['location_id'] == loc) & (oct26['date'] >= pd.to_datetime('2026-10-15').date())]
    # groupby hour mean of daily_anom
    anom_s = st.groupby('hour')['daily_anom'].mean()
    # Ensure it sums to exactly 0
    anom_s = anom_s - anom_s.mean()
    unbiased_diurnal[loc] = anom_s

# Linear time interpolation on continuous time series
method_a_preds = {}
for loc in test['location_id'].unique():
    s = train[(train['location_id'] == loc) & (train['dt'] >= pd.to_datetime('2026-10-20').tz_localize('Asia/Kolkata'))].copy()
    s = s.set_index('dt').reindex(full_dt)
    s_lin = s['pm25'].interpolate(method='time')
    
    for _, row in w2[w2['location_id'] == loc].iterrows():
        t = row['dt_end']
        anom = unbiased_diurnal[loc].get(t.hour, 0.0)
        # alpha = 0.50 diurnal harmonic
        method_a_preds[row['ID']] = s_lin.loc[t] + 0.50 * anom

w2['pred_a'] = w2['ID'].map(method_a_preds)

# ==============================================================================
# Method B: Pure Hour-Matched Diurnal Boundary Bridge
# Interpolate directly between the same hour on Oct 23 and Oct 29!
# For hour h on Oct 27 (day 4 of 6): Oct23[h] + 4/6 * (Oct29[h] - Oct23[h])
# For hour h on Oct 28 (day 5 of 6): Oct23[h] + 5/6 * (Oct29[h] - Oct23[h])
# ==============================================================================
method_b_preds = {}
for loc in test['location_id'].unique():
    st = train[(train['location_id'] == loc) & (train['dt'].dt.year == 2026) & (train['dt'].dt.month == 10)].copy()
    oct23_hours = st[st['date'] == pd.to_datetime('2026-10-23').date()].set_index('hour')['pm25'].to_dict()
    oct29_hours = st[st['date'] == pd.to_datetime('2026-10-29').date()].set_index('hour')['pm25'].to_dict()
    
    # If any hour is missing on Oct 29 (like hours 0, 1), impute from Oct 30 or neighboring hours
    oct30_hours = st[st['date'] == pd.to_datetime('2026-10-30').date()].set_index('hour')['pm25'].to_dict()
    
    for _, row in w2[w2['location_id'] == loc].iterrows():
        h = row['dt_end'].hour
        day = row['dt_start'].day
        frac = 4.0 / 6.0 if day == 27 else 5.0 / 6.0
        
        # Get start value (Oct 23)
        v_start = oct23_hours.get(h, np.nan)
        if np.isnan(v_start):
            # Impute from neighboring hours or Oct 22
            v_start = st[st['date'] == pd.to_datetime('2026-10-22').date()].set_index('hour')['pm25'].get(h, 55.0)
            
        # Get end value (Oct 29)
        v_end = oct29_hours.get(h, np.nan)
        if np.isnan(v_end):
            # For hours 0 and 1 missing on Oct 29, use Oct 30 adjusted by Oct 29/30 ratio
            v_end = oct30_hours.get(h, np.nan)
            if np.isnan(v_end):
                v_end = 35.0
                
        pred_val = v_start + frac * (v_end - v_start)
        method_b_preds[row['ID']] = pred_val

w2['pred_b'] = w2['ID'].map(method_b_preds)

# ==============================================================================
# Method C: Ventilation-Accelerated Physical Bridge
# Using actual Open-Meteo wind speed to model nonlinear atmospheric clearance!
# Wind: Oct 24=5.46, Oct 25=6.10, Oct 26=7.79, Oct 27=9.20, Oct 28=10.59, Oct 29=7.18
# ==============================================================================
# Cumulative dispersion integral from Oct 23 to Oct 29:
# Day 24: 5.46 (cum: 5.46)
# Day 25: 6.10 (cum: 11.56)
# Day 26: 7.79 (cum: 19.35)
# Day 27: 9.20 (cum: 28.55) -> 28.55 / 46.32 = 61.6% clearance
# Day 28: 10.59 (cum: 39.14) -> 39.14 / 46.32 = 84.5% clearance
# Day 29: 7.18 (cum: 46.32) -> 100% clearance
# Notice Oct 27 is 61.6% cleared, Oct 28 is 84.5% cleared!
# (vs linear which is 66.7% and 83.3%)
method_c_preds = {}
for loc in test['location_id'].unique():
    for _, row in w2[w2['location_id'] == loc].iterrows():
        h = row['dt_end'].hour
        day = row['dt_start'].day
        # Use physical clearance fraction based on wind accumulation:
        # Oct 27 hour h: fractional accumulation between day 26 24:00 (19.35/46.32=41.8%) and day 27 24:00 (28.55/46.32=61.6%)
        # Oct 28 hour h: fractional accumulation between day 27 24:00 (61.6%) and day 28 24:00 (84.5%)
        if day == 27:
            frac = (19.35 + (9.20 * (h + 1) / 24.0)) / 46.32
        else:
            frac = (28.55 + (10.59 * (h + 1) / 24.0)) / 46.32
            
        # Time-continuous linear interpolation value
        s = train[(train['location_id'] == loc) & (train['dt'] >= pd.to_datetime('2026-10-20').tz_localize('Asia/Kolkata'))].copy()
        s = s.set_index('dt').reindex(full_dt)
        # Start and end anchor values for station
        v_start = s.loc['2026-10-23 21:00:00+05:30', 'pm25'] if loc != 11611 else s.loc['2026-10-23 20:00:00+05:30', 'pm25']
        v_end = s.loc['2026-10-29 02:00:00+05:30', 'pm25']
        
        base_trend = v_start + frac * (v_end - v_start)
        anom = unbiased_diurnal[loc].get(h, 0.0)
        method_c_preds[row['ID']] = base_trend + 0.60 * anom

w2['pred_c'] = w2['ID'].map(method_c_preds)

print("=" * 70)
print(" COMPARISON OF INTERPOLATION APPROACHES ON W2 (192 rows)")
print("=" * 70)

v31 = pd.read_csv('submission_v31_boundary_bridge_champion.csv')
w2['v31'] = w2['ID'].map(v31.set_index('ID')['Target'])

for name, col in [('v31 (Actual Best)', 'v31'), 
                  ('Method A (Continuous + Unbiased Diurnal + Full 192)', 'pred_a'),
                  ('Method B (Hour-Matched Bridge)', 'pred_b'),
                  ('Method C (Ventilation-Accelerated Bridge)', 'pred_c')]:
    m = w2[col].mean()
    s = w2[col].std()
    d27_m = w2[w2['dt_start'].dt.day == 27][col].mean()
    d28_m = w2[w2['dt_start'].dt.day == 28][col].mean()
    drop = d27_m - d28_m
    print(f"{name:50s} | Mean: {m:.3f} | Std: {s:.3f} | Oct27: {d27_m:.2f} | Oct28: {d28_m:.2f} | Drop: {drop:.2f}")

print("\nStation Means Comparison:")
for loc in test['location_id'].unique():
    sub_df = w2[w2['location_id'] == loc]
    print(f"Station {loc}:")
    for name, col in [('v31', 'v31'), ('Method A', 'pred_a'), ('Method B', 'pred_b'), ('Method C', 'pred_c')]:
        print(f"   {name:10s} -> Mean: {sub_df[col].mean():.2f}, Std: {sub_df[col].std():.2f}")

