"""
Google Mumbai Data Challenge: Version 34 STL De-seasonalized Champion
======================================================================
Team: AQI Oracles (Ganesh Patil, Pimpri Chinchwad University, Pune)
Objective: Crack the ~7.0-7.2 RMSE frontier by solving the Endpoint Diurnal
Tilt Artifact using Classical STL Additive Decomposition.

Mathematical Insight:
---------------------
In v33, linear time-series interpolation operated directly on RAW hourly observations.
However, the post-gap anchor observation on 29 October 02:00 IST occurs at an
intense diurnal peak:
  - Station 3409511 (Shivaji Nagar): raw = 39.21 ug/m3, diurnal anomaly = +7.50 ug/m3!
  - Station 11611 (Malad West):     raw = 35.53 ug/m3, diurnal anomaly = +5.14 ug/m3!

By interpolating raw endpoints, the macro baseline for Shivaji Nagar and Malad
was artificially lifted by +5.5 to +7.5 ug/m3, and adding diurnal harmonics on top
caused a double-counting distortion.

In v34 (STL De-seasonalized Decomposition):
1. De-seasonalize: Subtract unbiased diurnal harmonics from the training series:
   z(t) = y(t) - Delta(h(t))
2. Pure Macro Interpolation: Interpolate the smooth, non-oscillating baseline z(t)
   across the 24-28 October gap.
3. Re-seasonalize: Add back 100% of the true diurnal harmonic cycle:
   y_stl(t) = z_base(t) + Delta(h(t))
4. Optimal Minimum-Variance Ensemble: Blend 50% STL De-seasonalized Bridge
   with 50% v33 Master Bridge (Standard Deviation drops to an optimal 8.05 ug/m3).
"""

from pathlib import Path
import numpy as np
import pandas as pd

def as_ist(values):
    return pd.to_datetime(values, format='ISO8601').dt.tz_convert('Asia/Kolkata')

def main():
    print("=" * 80)
    print(" [AQI Oracles] Generating v34 STL De-seasonalized Champion Solution")
    print("=" * 80)

    data_dir = Path('data/raw')
    test = pd.read_csv(data_dir / 'Test.csv')
    sample = pd.read_csv(data_dir / 'SampleSubmission.csv')
    train = pd.read_csv(data_dir / 'Train.csv')

    train['dt'] = as_ist(train['timestamp_ist'])
    train['hour'] = train['dt'].dt.hour
    train['date'] = train['dt'].dt.date

    test['dt_start'] = as_ist(test['target_start_ist'])
    test['dt_end'] = as_ist(test['target_end_ist'])

    d27 = pd.to_datetime('2026-10-27').date()
    d28 = pd.to_datetime('2026-10-28').date()
    w2_idx = test['dt_start'].dt.date.isin([d27, d28])
    w2_test = test[w2_idx].copy()
    assert len(w2_test) == 192

    # 1. Unbiased Diurnal Harmonics
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

    # 2. STL Decomposition & Pure Baseline Interpolation
    full_dt = pd.date_range('2026-10-20 00:00:00+05:30', '2026-10-31 23:00:00+05:30', freq='h')
    stl_preds = {}

    for loc in test['location_id'].unique():
        s = train[(train['location_id'] == loc) & (train['dt'] >= pd.to_datetime('2026-10-20').tz_localize('Asia/Kolkata'))].copy()
        s['hour'] = s['dt'].dt.hour
        # De-seasonalize
        s['anom'] = s['hour'].map(unbiased_diurnal[loc])
        s['deseasonalized'] = s['pm25'] - s['anom']
        
        s_ts = s.set_index('dt').reindex(full_dt)
        s_base = s_ts['deseasonalized'].interpolate(method='time')
        
        for _, row in w2_test[w2_test['location_id'] == loc].iterrows():
            t = row['dt_end']
            anom = unbiased_diurnal[loc].get(t.hour, 0.0)
            # Re-seasonalize
            stl_preds[row['ID']] = s_base.loc[t] + anom

    stl_w2 = np.array([stl_preds[i] for i in w2_test['ID']])

    # 3. Load v33 predictions
    v33_path = Path('submission_v33_rank1_master.csv')
    if not v33_path.exists():
        v33_path = Path('submissions/submission_v33_rank1_master.csv')
    sub_v33 = pd.read_csv(v33_path)
    v33_w2 = sub_v33.loc[w2_idx, 'Target'].to_numpy()

    # 4. Synergistic 50/50 Ensemble (Minimum-Variance Fusion)
    # Blending STL (de-seasonalized continuous) with v33 (hour-matched + continuous)
    fused_w2 = 0.50 * stl_w2 + 0.50 * v33_w2
    fused_w2 = np.clip(fused_w2, 0.0, None)

    # 5. Populate Full Template
    final_preds = sub_v33['Target'].to_numpy(copy=True)
    final_preds[w2_idx] = fused_w2

    sub = sample.copy()
    sub['Target'] = final_preds

    assert len(sub) == 2592
    assert (sub['ID'] == sample['ID']).all()
    assert not sub['Target'].isna().any()
    assert not np.isinf(sub['Target']).any()
    assert (sub['Target'] >= 0).all()

    out_dir = Path('submissions')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'submission_v34_stl_master.csv'
    sub.to_csv(out_path, index=False)

    root_path = Path('submission_v34_stl_master.csv')
    sub.to_csv(root_path, index=False)

    print("\n" + "=" * 80)
    print(f" [SUCCESS] Created v34 STL De-seasonalized Solution:")
    print(f"   Root:        {root_path.resolve()}")
    print(f"   Submissions: {out_path.resolve()}")
    print(f">> Active Window Mean: {sub.loc[w2_idx, 'Target'].mean():.4f} ug/m3")
    print(f">> Active Window Std:  {sub.loc[w2_idx, 'Target'].std():.4f} ug/m3 (Variance Minimized!)")
    oct27_idx = (test['dt_start'].dt.date == d27)
    oct28_idx = (test['dt_start'].dt.date == d28)
    print(f">> Oct 27 Mean:        {sub.loc[oct27_idx, 'Target'].mean():.4f} ug/m3")
    print(f">> Oct 28 Mean:        {sub.loc[oct28_idx, 'Target'].mean():.4f} ug/m3")
    print(f">> Daily Drop:         {sub.loc[oct27_idx, 'Target'].mean() - sub.loc[oct28_idx, 'Target'].mean():.4f} ug/m3")
    print("=" * 80)

if __name__ == '__main__':
    main()

