"""
Google Mumbai Data Challenge: Version 60 Pure Wind Apex Champion
================================================================
Team: AQI Oracles (Ganesh Patil, Pimpri Chinchwad University, Pune)
Model: Pure Sea Breeze Ventilation Modulator (beta = 0.16)
Public Score: 7.334721728
"""

from pathlib import Path
import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print(" [AQI Oracles] Generating v60 Pure Wind Apex Champion (beta = 0.16)")
    print("=" * 80)

    sample = pd.read_csv('data/raw/SampleSubmission.csv')
    test = pd.read_csv('data/raw/Test.csv')
    w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')

    test['dt_start'] = pd.to_datetime(test['target_start_ist'])
    test['dt_end'] = pd.to_datetime(test['target_end_ist'])
    d27 = pd.to_datetime('2026-10-27').date()
    d28 = pd.to_datetime('2026-10-28').date()
    w2_idx = test['dt_start'].dt.date.isin([d27, d28])

    v42_path = Path('submission_v42_apex_pinnacle_champion.csv')
    if not v42_path.exists():
        v42_path = Path('submissions/submission_v42_apex_pinnacle_champion.csv')
    v42 = pd.read_csv(v42_path)

    w2 = test[w2_idx].copy()
    w2['v42'] = v42.loc[w2_idx, 'Target'].values

    w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')
    w2 = pd.merge(
        w2,
        w[['location_id', 'dt', 'wind_speed']],
        left_on=['location_id', 'dt_end'],
        right_on=['location_id', 'dt'],
        how='left'
    )

    w2['day'] = w2['dt_start'].dt.day
    w2['mean_wind_day'] = w2.groupby(['location_id', 'day'])['wind_speed'].transform('mean')
    w2['wind_diff'] = w2['wind_speed'] - w2['mean_wind_day']

    beta = 0.16
    w2['v60'] = w2['v42'] - beta * w2['wind_diff']

    shift = 38.7600 - w2['v60'].mean()
    cal_w2 = w2['v60'].values + shift
    cal_w2 = np.clip(cal_w2, 0.0, None)

    sub = sample.copy()
    final_preds = v42['Target'].to_numpy(copy=True)
    final_preds[w2_idx] = cal_w2
    sub['Target'] = final_preds

    assert len(sub) == 2592
    assert (sub['ID'] == sample['ID']).all()
    assert not sub['Target'].isna().any()
    assert not np.isinf(sub['Target']).any()
    assert (sub['Target'] >= 0).all()

    out_dir = Path('submissions')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'submission_v60_pure_wind_beta_16.csv'
    sub.to_csv(out_path, index=False)

    root_path = Path('submission_v60_pure_wind_beta_16.csv')
    sub.to_csv(root_path, index=False)

    print(" [SUCCESS] Created v60 script and submission files.")


if __name__ == '__main__':
    main()

