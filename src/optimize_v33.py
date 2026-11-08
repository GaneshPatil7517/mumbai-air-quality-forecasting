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

v33 = pd.read_csv('submission_v33_rank1_master.csv')
w2['v33'] = w2['ID'].map(v33.set_index('ID')['Target'])

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

# Bridge 2: Hour-Matched Diurnal Twin Bridge
bridge2_preds = {}
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
        if np.isnan(v_start):
            v_start = oct22_h.get(h, 55.0)
            
        v_end = oct29_h.get(h, np.nan)
        if np.isnan(v_end):
            v_end = oct30_h.get(h, 35.0)
            
        bridge2_preds[row['ID']] = v_start + frac * (v_end - v_start)

b2_arr = np.array([bridge2_preds[i] for i in w2['ID']])

print("Comparing parameter grid around v33:")
print("alpha_diurnal, weight_b1, mean, std, oct27_mean, oct28_mean, daily_drop")

for alpha in [0.40, 0.45, 0.50, 0.55]:
    b1_preds = {}
    for loc in test['location_id'].unique():
        s = train[(train['location_id'] == loc) & (train['dt'] >= pd.to_datetime('2026-10-20').tz_localize('Asia/Kolkata'))].copy()
        s = s.set_index('dt').reindex(full_dt)
        s_lin = s['pm25'].interpolate(method='time')
        for _, row in w2[w2['location_id'] == loc].iterrows():
            t = row['dt_end']
            b1_preds[row['ID']] = s_lin.loc[t] + alpha * unbiased_diurnal[loc].get(t.hour, 0.0)
            
    b1_arr = np.array([b1_preds[i] for i in w2['ID']])
    
    for w_b1 in [0.60, 0.65, 0.70, 0.75, 0.80]:
        blend = w_b1 * b1_arr + (1.0 - w_b1) * b2_arr
        shift = 38.7406 - blend.mean()
        cal = blend + shift
        
        w2_temp = w2.copy()
        w2_temp['pred'] = cal
        d27_m = w2_temp[w2_temp['dt_start'].dt.day == 27]['pred'].mean()
        d28_m = w2_temp[w2_temp['dt_start'].dt.day == 28]['pred'].mean()
        drop = d27_m - d28_m
        
        print(f"{alpha:.2f}, {w_b1:.2f}, {cal.mean():.4f}, {cal.std():.4f}, {d27_m:.2f}, {d28_m:.2f}, {drop:.2f}")
