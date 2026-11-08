import pandas as pd
import numpy as np

train = pd.read_csv('data/raw/Train.csv')
test = pd.read_csv('data/raw/Test.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])

w2_idx = (test['dt_start'] >= '2026-10-27 00:00:00+05:30') & (test['dt_start'] <= '2026-10-28 23:00:00+05:30')
w2 = test[w2_idx].copy()

train['date'] = train['dt'].dt.date
train['hour'] = train['dt'].dt.hour

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
stl_preds = {}

for loc in test['location_id'].unique():
    s = train[(train['location_id'] == loc) & (train['dt'] >= pd.to_datetime('2026-10-20').tz_localize('Asia/Kolkata'))].copy()
    s['hour'] = s['dt'].dt.hour
    s['anom'] = s['hour'].map(unbiased_diurnal[loc])
    s['deseasonalized'] = s['pm25'] - s['anom']
    
    s_ts = s.set_index('dt').reindex(full_dt)
    s_base = s_ts['deseasonalized'].interpolate(method='time')
    
    for _, row in w2[w2['location_id'] == loc].iterrows():
        t = row['dt_end']
        anom = unbiased_diurnal[loc].get(t.hour, 0.0)
        stl_preds[row['ID']] = s_base.loc[t] + anom

w2['stl'] = w2['ID'].map(stl_preds)

v33 = pd.read_csv('submission_v33_rank1_master.csv')
w2['v33'] = w2['ID'].map(v33.set_index('ID')['Target'])

print("STL Reconstruction vs v33 on W2 (192 rows):")
stl_mean = w2['stl'].mean()
stl_std = w2['stl'].std()
v33_mean = w2['v33'].mean()
v33_std = w2['v33'].std()
print(f"STL Mean: {stl_mean:.4f}, Std: {stl_std:.4f}")
print(f"v33 Mean: {v33_mean:.4f}, Std: {v33_std:.4f}")

# Station comparison
print("\nStation Breakdown:")
for loc in sorted(w2['location_id'].unique()):
    st_df = w2[w2['location_id'] == loc]
    sm = st_df['stl'].mean()
    ss = st_df['stl'].std()
    vm = st_df['v33'].mean()
    vs = st_df['v33'].std()
    print(f"Station {loc}: STL Mean={sm:.2f}, Std={ss:.2f} | v33 Mean={vm:.2f}, Std={vs:.2f}")

# Correlation
corr = np.corrcoef(w2['stl'], w2['v33'])[0, 1]
print(f"\nCorrelation(STL, v33): {corr:.4f}")
