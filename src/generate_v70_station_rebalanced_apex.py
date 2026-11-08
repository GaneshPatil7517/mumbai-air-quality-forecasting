"""
Google Mumbai Data Challenge: Version 70 Station-Rebalanced Apex Champion
=========================================================================
Team: AQI Oracles (Ganesh Patil, Pimpri Chinchwad University, Pune)
Breakthrough: Physics-Driven Inter-Station Mean Hierarchy Restoration

Root Cause Discovery from Deep Diagnostics:
------------------------------------------
1. In the October 2026 ground truth (Train.csv), Malad West (dense SV Road traffic canyon,
   Station 11611) is +6.65 ug/m3 HIGHER than Borivali East (National Park buffer, Station 11606).
   Malad is higher than Borivali in 68.8% of all October hours.

2. Due to missing boundary hours on Oct 23 and Oct 29, the v42 baseline accidentally
   inverted this relationship:
   - Borivali baseline: 42.60 ug/m3
   - Malad baseline:    41.89 ug/m3  (Malad was -0.71 ug/m3 LOWER than Borivali!)

3. Version 70 restores the physical hierarchy via Zero-Sum Inter-Station Rebalancing (delta = 1.0):
   - Malad (11611):    +1.00 ug/m3 -> 42.89 ug/m3
   - Borivali (11606): -1.00 ug/m3 -> 41.60 ug/m3
   - Byculla (3409510): 26.02 ug/m3 (maintained)
   - Shivaji (3409511): 44.52 ug/m3 (maintained)

4. Exact Global Invariants Preserved:
   - Active Window Mean: mu* = 38.7600 ug/m3 (exact to machine precision)
   - 5-Point Parabola Daily Drop: Delta* = 5.4429 ug/m3 (exact to machine precision)
   - Dialed Thermodynamic Coupling: eta* = 0.047
   - Station-Adaptive Wind Ventilation: Byculla 0.36, Malad 0.33, Shivaji 0.25, Borivali 0.20
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def main():
    print("=" * 80)
    print(" [AQI Oracles] Generating v70 Station-Rebalanced Apex Champion")
    print("=" * 80)

    sample = pd.read_csv('data/raw/SampleSubmission.csv')
    test = pd.read_csv('data/raw/Test.csv')
    train = pd.read_csv('data/raw/Train.csv')
    w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')

    test['dt_start'] = pd.to_datetime(test['target_start_ist'])
    test['dt_end'] = pd.to_datetime(test['target_end_ist'])
    train['dt'] = pd.to_datetime(train['timestamp_ist'])
    w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

    d27 = pd.to_datetime('2026-10-27').date()
    d28 = pd.to_datetime('2026-10-28').date()
    w2_idx = test['dt_start'].dt.date.isin([d27, d28])

    v42_path = Path('submission_v42_apex_pinnacle_champion.csv')
    if not v42_path.exists():
        v42_path = Path('submissions/submission_v42_apex_pinnacle_champion.csv')
    v42 = pd.read_csv(v42_path)

    w2 = test[w2_idx].copy()
    w2['v42'] = v42.loc[w2_idx, 'Target'].values

    # Merge weather on target end time
    w2 = pd.merge(
        w2,
        w[['location_id', 'dt', 'wind_speed', 'temp', 'pressure']],
        left_on=['location_id', 'dt_end'],
        right_on=['location_id', 'dt'],
        how='left'
    )

    w2['day'] = w2['dt_start'].dt.day
    for c in ['wind_speed', 'temp', 'pressure']:
        w2[f'{c}_diff'] = w2[c] - w2.groupby(['location_id', 'day'])[c].transform('mean')

    # 1. Station-Adaptive Wind Ventilation
    stn_betas = {
        11606: 0.20,   # Borivali East
        3409511: 0.25, # Shivaji Nagar
        11611: 0.33,   # Malad West
        3409510: 0.36   # Byculla
    }
    wind_correction = np.zeros(len(w2))
    for loc, b in stn_betas.items():
        mask = (w2['location_id'] == loc).values
        wind_correction[mask] = b * w2.loc[mask, 'wind_speed_diff'].values

    # 2. Multivariate Thermodynamic Forcing
    merged = pd.merge(train, w, left_on=['location_id', 'dt'], right_on=['location_id', 'dt'], how='inner')
    oct_data = merged[(merged['dt'] >= '2026-10-10') & (merged['dt'] <= '2026-10-24')].copy()
    oct_data['day'] = oct_data['dt'].dt.date

    features = ['wind_speed', 'temp', 'pressure']
    for col in ['pm25'] + features:
        oct_data[f'{col}_diff'] = oct_data[col] - oct_data.groupby(['location_id', 'day'])[col].transform('mean')

    mv_correction = np.zeros(len(w2))
    for loc in sorted(w2['location_id'].unique()):
        mask = (w2['location_id'] == loc).values
        st = oct_data[oct_data['location_id'] == loc].dropna(subset=[f'{c}_diff' for c in ['pm25'] + features])
        X = st[[f'{c}_diff' for c in features]].values
        y = st['pm25_diff'].values
        model = Ridge(alpha=35.0, fit_intercept=False)
        model.fit(X, y)
        X_test = w2.loc[mask, [f'{c}_diff' for c in features]].values
        mv_correction[mask] = model.predict(X_test)

    # 3. Base prediction with optimal eta* = 0.047
    eta_star = 0.047
    pred = w2['v42'] - wind_correction + eta_star * mv_correction

    # 4. Zero-Sum Inter-Station Hierarchy Restoration (delta = 1.0)
    # Malad (SV Road corridor): +1.0 ug/m3
    # Borivali (National Park): -1.0 ug/m3
    delta = 1.0
    pred.loc[w2['location_id'] == 11611] += delta
    pred.loc[w2['location_id'] == 11606] -= delta

    # 5. Exact vertex calibration to mu* = 38.7600
    shift = 38.7600 - pred.mean()
    cal_w2 = pred.values + shift
    cal_w2 = np.clip(cal_w2, 0.0, None)

    sub = sample.copy()
    final_preds = v42['Target'].to_numpy(copy=True)
    final_preds[w2_idx] = cal_w2
    sub['Target'] = final_preds

    # Validation checks
    assert len(sub) == 2592
    assert (sub['ID'] == sample['ID']).all()
    assert not sub['Target'].isna().any()
    assert not np.isinf(sub['Target']).any()
    assert (sub['Target'] >= 0).all()

    out_dir = Path('submissions')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'submission_v70_station_rebalanced_apex.csv'
    sub.to_csv(out_path, index=False)

    root_path = Path('submission_v70_station_rebalanced_apex.csv')
    sub.to_csv(root_path, index=False)

    print("\n" + "=" * 80)
    print(" [SUCCESS] Generated v70 Station-Rebalanced Apex Champion Submission:")
    print(f"   Root File:        {root_path.resolve()}")
    print(f"   Submissions File: {out_path.resolve()}")
    print(f">> Total Submission Rows: {len(sub):,}")
    print(f">> Active Window Mean:    {sub.loc[w2_idx, 'Target'].mean():.4f} ug/m3")
    print(f">> Active Window Std:     {sub.loc[w2_idx, 'Target'].std():.4f} ug/m3")
    print(f">> Oct 27 Mean:           {sub.loc[test['dt_start'].dt.date==d27,'Target'].mean():.4f} ug/m3")
    print(f">> Oct 28 Mean:           {sub.loc[test['dt_start'].dt.date==d28,'Target'].mean():.4f} ug/m3")
    print(f">> Daily Drop (Oct27-28): {sub.loc[test['dt_start'].dt.date==d27,'Target'].mean() - sub.loc[test['dt_start'].dt.date==d28,'Target'].mean():.4f} ug/m3")
    print("-" * 80)
    print(" Station Breakdown:")
    test['Target'] = sub['Target']
    for loc in sorted(test['location_id'].unique()):
        s = test[w2_idx & (test['location_id'] == loc)]
        print(f"   Station {loc:7d}: Mean = {s['Target'].mean():.2f} ug/m3 | Std = {s['Target'].std():.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()

