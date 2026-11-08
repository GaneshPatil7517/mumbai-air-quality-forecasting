"""
Google Mumbai Data Challenge: Version 72 Phase 3 Champion
=========================================================
Team: AQI Oracles (Ganesh Patil, Pimpri Chinchwad University, Pune)
Scoring Window: Phase 3 (3-4 November 2026, 92 valid hourly observations)

Scientific Architecture:
------------------------
1. Ground-Truth Anchored:
   - Synchronized with Zindi's newly released Train.csv (47,849 rows, through 2 Nov 23:00 IST).
   - Real starting station baselines from 2 Nov: Borivali 38.29, Malad 38.87, Byculla 22.28, Shivaji 37.16.
2. Secular Post-Monsoon Drift:
   - Empirical decay rate from late October / early November (-1.0 to -1.2 ug/m3 per day).
   - Nov 3 baseline mean = 33.76 ug/m3, Nov 4 baseline mean = 31.96 ug/m3.
3. Multi-Day Damped Diurnal Harmonics:
   - 4-day empirical diurnal anomaly profile (Oct 29 - Nov 2) regularized with gamma = 0.50.
4. Autoregressive Boundary Continuity Bridge:
   - Exponentially smoothed transition from 2 Nov 23:00 ground truth across hours 00-03 of 3 Nov.
5. Multivariate Atmospheric Weather Modulation:
   - Ridge-regularized wind ventilation, convective temperature, and pressure subsidence from Open-Meteo.
6. Baseline Invariants:
   - All 2,400 non-active rows preserved strictly from proven champion v65.
   - Total rows: 2,592, exact IDs, no NaNs, no negatives.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def main():
    print("=" * 80)
    print(" [AQI Oracles] Generating v72 Phase 3 Champion Model")
    print("=" * 80)

    sample = pd.read_csv('data/raw/SampleSubmission.csv')
    test = pd.read_csv('data/raw/Test.csv')
    train = pd.read_csv('data/raw/Train.csv')
    w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')

    test['dt_start'] = pd.to_datetime(test['target_start_ist'])
    test['dt_end'] = pd.to_datetime(test['target_end_ist'])
    train['dt'] = pd.to_datetime(train['timestamp_ist'])
    w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

    d3 = pd.to_datetime('2026-11-03').date()
    d4 = pd.to_datetime('2026-11-04').date()
    nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

    v65_path = Path('submission_v65_multivariate_atmospheric_apex.csv')
    if not v65_path.exists():
        v65_path = Path('submissions/submission_v65_multivariate_atmospheric_apex.csv')
    v65 = pd.read_csv(v65_path)

    # 1. Multi-day calibration period (Oct 29 - Nov 2)
    df_train = pd.merge(train, w, on=['location_id', 'dt'], how='inner')
    df_train['date'] = df_train['dt'].dt.date
    df_train['hour'] = df_train['dt'].dt.hour

    calib_start = pd.to_datetime('2026-10-29').date()
    calib_end = pd.to_datetime('2026-11-02').date()
    calib = df_train[(df_train['date'] >= calib_start) & (df_train['date'] <= calib_end)].copy()

    st_base = calib.groupby('location_id')['pm25'].mean()
    hourly_mean = calib.groupby(['location_id', 'hour'])['pm25'].mean()

    # 2. Extract active Nov 3-4 test rows
    nov34 = test[nov34_mask].copy()
    nov34['day'] = nov34['dt_start'].dt.day
    nov34['date'] = nov34['dt_start'].dt.date
    nov34['hour'] = nov34['dt_start'].dt.hour

    # Merge weather
    nov34 = pd.merge(
        nov34,
        w[['location_id', 'dt', 'wind_speed', 'temp', 'pressure']],
        left_on=['location_id', 'dt_end'],
        right_on=['location_id', 'dt'],
        how='left'
    )

    # 3. Base mean per station and day with secular post-monsoon drift
    nov2_means = df_train[df_train['date'] == calib_end].groupby('location_id')['pm25'].mean().to_dict()
    daily_drift = {
        11606: 1.10,   # Borivali East
        11611: 0.95,   # Malad West
        3409510: 1.00, # Byculla
        3409511: 1.25  # Shivaji Nagar
    }

    base_preds = []
    for idx, r in nov34.iterrows():
        loc = r['location_id']
        d = r['date']
        if d == d3:
            b = nov2_means[loc] - daily_drift[loc]
        else:
            b = nov2_means[loc] - 2.0 * daily_drift[loc]
        base_preds.append(b)

    nov34['pred_base'] = base_preds

    # 4. Add Regularized Diurnal Profile (Gamma = 0.50)
    diurnal_anom = []
    for idx, r in nov34.iterrows():
        loc = r['location_id']
        h = r['hour']
        if (loc, h) in hourly_mean:
            anom = hourly_mean.loc[(loc, h)] - st_base[loc]
        else:
            anom = 0.0
        diurnal_anom.append(anom)

    nov34['diurnal_anom'] = diurnal_anom
    gamma = 0.50
    nov34['pred_diurnal'] = nov34['pred_base'] + gamma * nov34['diurnal_anom']

    # 5. Autoregressive Boundary Continuity Bridge for hours 00..03 on Nov 3
    nov2_23 = df_train[df_train['dt'] == '2026-11-02 23:00:00+05:30'].set_index('location_id')['pm25'].to_dict()

    ar_adjusted = nov34['pred_diurnal'].values.copy()
    bridge_weights = {0: 0.70, 1: 0.50, 2: 0.30, 3: 0.15}

    for i, (_, r) in enumerate(nov34.iterrows()):
        if r['date'] == d3 and r['hour'] in bridge_weights:
            loc = r['location_id']
            h = r['hour']
            w_bridge = bridge_weights[h]
            target_ar = nov2_23.get(loc, r['pred_diurnal'])
            ar_adjusted[i] = w_bridge * target_ar + (1.0 - w_bridge) * r['pred_diurnal']

    nov34['pred_ar'] = ar_adjusted

    # 6. Multivariate Weather Modulation
    for c in ['wind_speed', 'temp', 'pressure']:
        nov34[f'{c}_diff'] = nov34[c] - nov34.groupby(['location_id', 'day'])[c].transform('mean')

    features = ['wind_speed', 'temp', 'pressure']
    for c in features:
        calib[f'{c}_diff'] = calib[c] - calib.groupby(['location_id', 'date'])[c].transform('mean')
    calib['pm25_diff'] = calib['pm25'] - calib.groupby(['location_id', 'date'])['pm25'].transform('mean')

    weather_mod = np.zeros(len(nov34))
    for loc in sorted(nov34['location_id'].unique()):
        mask = (nov34['location_id'] == loc).values
        st = calib[calib['location_id'] == loc].dropna(subset=[f'{c}_diff' for c in features] + ['pm25_diff'])
        X = st[[f'{c}_diff' for c in features]].values
        y = st['pm25_diff'].values
        model = Ridge(alpha=40.0, fit_intercept=False)
        model.fit(X, y)
        X_test = nov34.loc[mask, [f'{c}_diff' for c in features]].values
        weather_mod[mask] = model.predict(X_test)

    eta = 0.05
    nov34['final_pred'] = nov34['pred_ar'] + eta * weather_mod
    nov34['final_pred'] = np.clip(nov34['final_pred'], 0.0, None)

    # 7. Assemble Complete 2,592-Row Submission
    sub = sample.copy()
    final_preds = v65['Target'].to_numpy(copy=True)

    # Map predictions back by exact ID alignment
    pred_dict = dict(zip(nov34['ID'], nov34['final_pred']))
    for idx, row_id in enumerate(sub['ID']):
        if row_id in pred_dict:
            final_preds[idx] = pred_dict[row_id]

    sub['Target'] = final_preds

    # Validation Checks
    assert len(sub) == 2592, f"Expected 2592 rows, got {len(sub)}"
    assert (sub['ID'] == sample['ID']).all(), "ID alignment mismatch"
    assert not sub['Target'].isna().any(), "Target contains NaNs"
    assert not np.isinf(sub['Target']).any(), "Target contains Infs"
    assert (sub['Target'] >= 0).all(), "Target contains negative values"

    # Save to submissions/ and root
    out_dir = Path('submissions')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'submission_v72_phase3_champion.csv'
    sub.to_csv(out_path, index=False)

    root_path = Path('submission_v72_phase3_champion.csv')
    sub.to_csv(root_path, index=False)

    print("\n" + "=" * 80)
    print(" [SUCCESS] Generated v72 Phase 3 Champion Submission:")
    print(f"   Root File:        {root_path.resolve()}")
    print(f"   Submissions File: {out_path.resolve()}")
    print(f">> Total Submission Rows: {len(sub):,}")
    print(f">> Overall Dataset Mean:  {sub['Target'].mean():.4f} ug/m3")
    print("-" * 80)
    print(" ACTIVE EVALUATION WINDOW (3-4 Nov, 192 rows):")
    active_preds = sub.loc[nov34_mask, 'Target']
    print(f"   Active Window Mean: {active_preds.mean():.4f} ug/m3")
    print(f"   Active Window Std:  {active_preds.std():.4f} ug/m3")
    print(f"   Active Window Min:  {active_preds.min():.4f} ug/m3")
    print(f"   Active Window Max:  {active_preds.max():.4f} ug/m3")
    d3_mask = (test['dt_start'].dt.date == d3)
    d4_mask = (test['dt_start'].dt.date == d4)
    d3_m = sub.loc[d3_mask, 'Target'].mean()
    d4_m = sub.loc[d4_mask, 'Target'].mean()
    print(f"   Nov 3 Mean:         {d3_m:.4f} ug/m3")
    print(f"   Nov 4 Mean:         {d4_m:.4f} ug/m3")
    print(f"   Daily Drop:         {d3_m - d4_m:.4f} ug/m3")
    print("-" * 80)
    print(" Station Breakdown on Active Evaluation Window (Nov 3-4):")
    test['Target'] = sub['Target']
    active_eval = test[nov34_mask]
    for loc in sorted(test['location_id'].unique()):
        s = active_eval[active_eval['location_id'] == loc]
        name = s['station_name'].iloc[0] if 'station_name' in s.columns else f"Station {loc}"
        print(f"   Station {loc:7d} ({name:27s}) | Mean: {s['Target'].mean():.2f} ug/m3 | Std: {s['Target'].std():.2f} | Range: [{s['Target'].min():.2f}, {s['Target'].max():.2f}]")
    print("=" * 80)


if __name__ == '__main__':
    main()

