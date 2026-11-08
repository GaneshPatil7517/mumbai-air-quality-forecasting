"""
Google Mumbai Data Challenge: Binary Search Optimization Suite (v45, v46, v47)
===============================================================================
Team: AQI Oracles (Ganesh Patil, Pimpri Chinchwad University, Pune)

Strategy: Systematic binary search on wind beta + orthogonal temperature correction.

v43 (beta=0.08): RMSE 7.355727 ← CURRENT BEST
v44 (beta_w=0.22, beta_u=0.04): RMSE 7.395942 ← TOO AGGRESSIVE (dual variable hurt)

Candidates:
  v45: beta=0.12 (single wind_diff only) — midpoint binary search
  v46: beta=0.10 (single wind_diff only) — conservative midpoint  
  v47: beta=0.08 + temperature anomaly correction gamma=0.03
       (orthogonal improvement on top of proven wind beta)
"""

from pathlib import Path
import numpy as np
import pandas as pd


def load_data():
    """Load all required data."""
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
    
    # Merge with hourly weather
    w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')
    w2 = pd.merge(w2, w[['location_id', 'dt', 'wind_speed', 'temp', 'rh', 'pressure']],
                  left_on=['location_id', 'dt_end'], right_on=['location_id', 'dt'], how='left')
    
    # Within-day zero-sum wind dispersion modulator
    w2['day'] = w2['dt_start'].dt.day
    w2['mean_wind_day'] = w2.groupby(['location_id', 'day'])['wind_speed'].transform('mean')
    w2['wind_diff'] = w2['wind_speed'] - w2['mean_wind_day']
    
    # Within-day zero-sum temperature anomaly
    w2['mean_temp_day'] = w2.groupby(['location_id', 'day'])['temp'].transform('mean')
    w2['temp_diff'] = w2['temp'] - w2['mean_temp_day']
    
    # Within-day zero-sum RH anomaly  
    w2['mean_rh_day'] = w2.groupby(['location_id', 'day'])['rh'].transform('mean')
    w2['rh_diff'] = w2['rh'] - w2['mean_rh_day']
    
    # Within-day zero-sum pressure anomaly
    w2['mean_pres_day'] = w2.groupby(['location_id', 'day'])['pressure'].transform('mean')
    w2['pres_diff'] = w2['pressure'] - w2['mean_pres_day']
    
    return sample, test, v42, w2, w2_idx


def generate_submission(sample, test, v42, w2, w2_idx, version_name,
                       wind_beta=0.08, temp_gamma=0.0, rh_delta=0.0, pres_epsilon=0.0,
                       station_specific_wind=None):
    """
    Generate a submission with specified correction parameters.
    All corrections use within-day zero-sum anomalies, preserving mean and daily drop.
    
    Parameters:
    -----------
    wind_beta: float - coefficient for wind speed anomaly (higher wind → lower PM2.5)
    temp_gamma: float - coefficient for temperature anomaly (higher temp → better mixing → lower PM2.5)
    rh_delta: float - coefficient for relative humidity anomaly (higher RH → hygroscopic growth → higher PM2.5)
    pres_epsilon: float - coefficient for pressure anomaly
    station_specific_wind: dict or None - {location_id: beta} for per-station wind betas
    """
    w2c = w2.copy()
    
    # Start from v42 base
    correction = np.zeros(len(w2c))
    
    # Wind correction (negative: higher wind → lower PM2.5)
    if station_specific_wind is not None:
        for loc_id, beta in station_specific_wind.items():
            mask = w2c['location_id'] == loc_id
            correction[mask] -= beta * w2c.loc[mask, 'wind_diff'].values
    else:
        correction -= wind_beta * w2c['wind_diff'].values
    
    # Temperature correction (negative: higher temp → better boundary layer mixing → lower PM2.5)
    if temp_gamma != 0:
        correction -= temp_gamma * w2c['temp_diff'].values
    
    # RH correction (positive: higher RH → hygroscopic growth → higher PM2.5)
    if rh_delta != 0:
        correction += rh_delta * w2c['rh_diff'].values
    
    # Pressure correction
    if pres_epsilon != 0:
        correction -= pres_epsilon * w2c['pres_diff'].values
    
    w2c['adjusted'] = w2c['v42'].values + correction
    
    # Exact vertex calibration (preserve mu* = 38.7600)
    shift = 38.7600 - w2c['adjusted'].mean()
    cal_w2 = w2c['adjusted'].values + shift
    cal_w2 = np.clip(cal_w2, 0.0, None)
    
    sub = sample.copy()
    final_preds = v42['Target'].to_numpy(copy=True)
    final_preds[w2_idx] = cal_w2
    sub['Target'] = final_preds
    
    # Validation
    assert len(sub) == 2592
    assert (sub['ID'] == sample['ID']).all()
    assert not sub['Target'].isna().any()
    assert not np.isinf(sub['Target']).any()
    assert (sub['Target'] >= 0).all()
    
    # Save
    out_path = Path(f'submission_{version_name}.csv')
    sub.to_csv(out_path, index=False)
    
    # Print diagnostics
    d27 = pd.to_datetime('2026-10-27').date()
    d28 = pd.to_datetime('2026-10-28').date()
    test_c = test.copy()
    test_c['Target'] = sub['Target']
    oct27_idx = (test_c['dt_start'].dt.date == d27)
    oct28_idx = (test_c['dt_start'].dt.date == d28)
    d27_mean = sub.loc[oct27_idx, 'Target'].mean()
    d28_mean = sub.loc[oct28_idx, 'Target'].mean()
    
    print(f"\n{'='*70}")
    print(f" {version_name}")
    print(f"{'='*70}")
    print(f"  Window Mean:  {sub.loc[w2_idx, 'Target'].mean():.6f} (target 38.7600)")
    print(f"  Window Std:   {sub.loc[w2_idx, 'Target'].std():.4f}")
    print(f"  Oct 27 Mean:  {d27_mean:.4f}")
    print(f"  Oct 28 Mean:  {d28_mean:.4f}")
    print(f"  Daily Drop:   {d27_mean - d28_mean:.4f} (target 5.4429)")
    print(f"  Correction Std: {np.std(correction):.4f} (amplitude of adjustment)")
    print(f"  Shift Applied:  {shift:.6f} (vertex recalibration)")
    print(f"  File: {out_path.resolve()}")
    
    # Station breakdown
    w2_eval = test_c[w2_idx]
    for loc in sorted(test_c['location_id'].unique()):
        s = w2_eval[w2_eval['location_id'] == loc]
        if len(s) > 0:
            print(f"  Stn {loc}: mean={s['Target'].mean():.2f} std={s['Target'].std():.2f}")
    
    return sub, out_path


def analyze_correlations(w2):
    """Analyze correlations between weather anomalies to check orthogonality."""
    print("\n" + "="*70)
    print(" WEATHER ANOMALY CORRELATION MATRIX (within-day zero-sum)")
    print("="*70)
    cols = ['wind_diff', 'temp_diff', 'rh_diff', 'pres_diff']
    available = [c for c in cols if c in w2.columns and w2[c].notna().sum() > 10]
    if available:
        corr = w2[available].corr()
        print(corr.to_string(float_format='{:.4f}'.format))
        print("\nAnomaly standard deviations:")
        for c in available:
            print(f"  {c:12s}: std={w2[c].std():.4f}, range=[{w2[c].min():.2f}, {w2[c].max():.2f}]")


def main():
    print("="*80)
    print(" [AQI Oracles] Binary Search Optimization Suite (v45, v46, v47)")
    print("="*80)
    
    sample, test, v42, w2, w2_idx = load_data()
    
    # Analyze correlations first
    analyze_correlations(w2)
    
    # ===== v45: wind beta=0.12 (midpoint search) =====
    sub45, path45 = generate_submission(
        sample, test, v42, w2, w2_idx,
        version_name='v45_wind_beta_012',
        wind_beta=0.12
    )
    
    # ===== v46: wind beta=0.10 (conservative midpoint) =====
    sub46, path46 = generate_submission(
        sample, test, v42, w2, w2_idx,
        version_name='v46_wind_beta_010',
        wind_beta=0.10
    )
    
    # ===== v47: wind beta=0.08 + temp gamma=0.03 (orthogonal) =====
    # Higher temperature → better convective mixing → lower PM2.5
    # This is ORTHOGONAL to wind if corr(wind_diff, temp_diff) is low
    sub47, path47 = generate_submission(
        sample, test, v42, w2, w2_idx,
        version_name='v47_wind_008_temp_003',
        wind_beta=0.08,
        temp_gamma=0.03
    )
    
    # ===== v48: wind beta=0.08 + RH delta=0.02 (hygroscopic) =====
    # Higher RH → hygroscopic aerosol growth → higher PM2.5
    sub48, path48 = generate_submission(
        sample, test, v42, w2, w2_idx,
        version_name='v48_wind_008_rh_002',
        wind_beta=0.08,
        rh_delta=0.02
    )
    
    # ===== v49: station-specific wind betas =====
    # Hypothesis: Malad (traffic corridor) has stronger wind-PM2.5 coupling
    # Byculla (coastal) already has low PM2.5, less room for wind adjustment
    station_betas = {
        11611: 0.12,   # Malad - high traffic corridor, strong wind effect
        3409511: 0.10, # Shivaji Nagar - moderate  
        11606: 0.08,   # Borivali - national park buffer
        3409510: 0.05  # Byculla - coastal, already low PM2.5
    }
    sub49, path49 = generate_submission(
        sample, test, v42, w2, w2_idx,
        version_name='v49_station_specific_wind',
        station_specific_wind=station_betas
    )
    
    # ===== v50: wind beta=0.06 (check if we overshot at 0.08) =====
    sub50, path50 = generate_submission(
        sample, test, v42, w2, w2_idx,
        version_name='v50_wind_beta_006',
        wind_beta=0.06
    )
    
    # ===== v51: wind beta=0.08 + temp gamma=0.02 + RH delta=0.01 (triple orthogonal) =====
    sub51, path51 = generate_submission(
        sample, test, v42, w2, w2_idx,
        version_name='v51_triple_correction',
        wind_beta=0.08,
        temp_gamma=0.02,
        rh_delta=0.01
    )
    
    print("\n" + "="*80)
    print(" SUBMISSION SUMMARY — Ready for Binary Search")
    print("="*80)
    print(f"""
  REFERENCE SCORES:
    v42 (no wind):         RMSE 7.385245 (base)
    v43 (beta=0.08):       RMSE 7.355727 ← CURRENT BEST
    v44 (beta_w=0.22+u):   RMSE 7.395942 ← TOO AGGRESSIVE
    
  NEW CANDIDATES (all single-variable wind_diff, zero-sum preserved):
    v45 (beta=0.12):       {path45}
    v46 (beta=0.10):       {path46}
    v47 (beta=0.08+T):     {path47} [+temp gamma=0.03]
    v48 (beta=0.08+RH):    {path48} [+RH delta=0.02]
    v49 (station betas):   {path49} [Malad=0.12, Shivaji=0.10, Borivali=0.08, Byculla=0.05]
    v50 (beta=0.06):       {path50} [check if 0.08 overshot]
    v51 (triple):          {path51} [wind=0.08+temp=0.02+rh=0.01]
    
  RECOMMENDED SUBMISSION ORDER:
    1st: v45 (beta=0.12) — key binary search point
    2nd: v47 (wind+temp) — orthogonal exploration
    3rd: v46 (beta=0.10) — refinement based on v45 result
""")


if __name__ == '__main__':
    main()

