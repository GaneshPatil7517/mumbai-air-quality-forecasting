import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
test = pd.read_csv('data/raw/Test.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
train['date'] = train['dt'].dt.date
train['hour'] = train['dt'].dt.hour
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])

w2_idx = (test['dt_start'] >= '2026-10-27 00:00:00+05:30') & (test['dt_start'] <= '2026-10-28 23:00:00+05:30')
w2 = test[w2_idx].copy()

oct26 = train[(train['dt'].dt.year == 2026) & (train['dt'].dt.month == 10)].copy()
oct26['date'] = oct26['dt'].dt.date
oct26['hour'] = oct26['dt'].dt.hour
oct26['daily_mean'] = oct26.groupby(['location_id', 'date'])['pm25'].transform('mean')
oct26['daily_anom'] = oct26['pm25'] - oct26['daily_mean']

unbiased_diurnal = {}
for loc in test['location_id'].unique():
    st = oct26[(oct26['location_id'] == loc) & (oct26['date'] >= pd.to_datetime('2026-10-15').date())]
    anom_s = st.groupby('hour')['daily_anom'].mean()
    anom_s = anom_s - anom_s.mean()
    unbiased_diurnal[loc] = anom_s

full_dt = pd.date_range('2026-10-20 00:00:00+05:30', '2026-10-31 23:00:00+05:30', freq='h')

# Bridge 1 with alpha = 0.50
b1_preds = {}
for loc in test['location_id'].unique():
    s = train[(train['location_id'] == loc) & (train['dt'] >= pd.to_datetime('2026-10-20').tz_localize('Asia/Kolkata'))].copy()
    s = s.set_index('dt').reindex(full_dt)
    s_lin = s['pm25'].interpolate(method='time')
    for _, row in w2[w2['location_id'] == loc].iterrows():
        t = row['dt_end']
        b1_preds[row['ID']] = s_lin.loc[t] + 0.50 * unbiased_diurnal[loc].get(t.hour, 0.0)

# Bridge 2: Hour-matched
b2_preds = {}
for loc in test['location_id'].unique():
    st = train[(train['location_id'] == loc) & (train['dt'].dt.year == 2026) & (train['dt'].dt.month == 10)].copy()
    oct23_h = st[st['date'] == pd.to_datetime('2026-10-23').date()].set_index('hour')['pm25'].to_dict()
    oct29_h = st[st['date'] == pd.to_datetime('2026-10-29').date()].set_index('hour')['pm25'].to_dict()
    oct22_h = st[st['date'] == pd.to_datetime('2026-10-22').date()].set_index('hour')['pm25'].to_dict()
    oct30_h = st[st['date'] == pd.to_datetime('2026-10-30').date()].set_index('hour')['pm25'].to_dict()
    for _, row in w2[w2['location_id'] == loc].iterrows():
        h = row['dt_end'].hour
        day = row['dt_start'].day
        frac = 4.0 / 6.0 if day == 27 else 5.0 / 6.0
        v_start = oct23_h.get(h, np.nan)
        if np.isnan(v_start): v_start = oct22_h.get(h, 55.0)
        v_end = oct29_h.get(h, np.nan)
        if np.isnan(v_end): v_end = oct30_h.get(h, 35.0)
        b2_preds[row['ID']] = v_start + frac * (v_end - v_start)

b1_arr = np.array([b1_preds[i] for i in w2['ID']])
b2_arr = np.array([b2_preds[i] for i in w2['ID']])

print("Weight B1 (Continuous) vs B2 (Twin):")
day27_mask = (w2['dt_start'].dt.day == 27).values

for w_b1 in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
    blend = w_b1 * b1_arr + (1 - w_b1) * b2_arr
    shift = 38.7600 - blend.mean()
    cal = blend + shift
    d27_m = cal[day27_mask].mean()
    d28_m = cal[~day27_mask].mean()
    drop = d27_m - d28_m
    print(f"Weight B1={w_b1:.2f} -> Std: {cal.std():.4f}, Oct27: {d27_m:.2f}, Oct28: {d28_m:.2f}, Drop: {drop:.2f}")

