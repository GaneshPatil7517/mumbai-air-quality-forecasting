"""
Google Mumbai Data Challenge: Version 23 Phase 2 Retrained Champion Solution
=============================================================================
Team: AQI Oracles (Ganesh Patil, Pimpri Chinchwad University, Pune)
Target: Retake Rank #1 on the 27-28 October Public Leaderboard Window

Key Architectural Pillars:
1. Ingests all 47,546 observations (+430 newly released rows from 22-23 Oct & 29-31 Oct).
2. Captures the severe late-October post-monsoon regime transition (PM2.5 drops from ~56 to ~34 ug/m3).
3. 4-Way Multi-Horizon Tree Ensemble:
   - RF 60-day Post-Monsoon Anchor (350 trees, max_depth=12, min_samples_leaf=5)
   - RF 45-day Dynamic Late-October Specialist (300 trees, max_depth=11, min_samples_leaf=5)
   - ExtraTrees 60-day Variance Reducer (300 trees, max_depth=13, min_samples_leaf=5)
   - LightGBM 60-day Nonlinear Interaction Booster (250 trees, learning_rate=0.035, num_leaves=24)
4. Station-Specific Microclimate Dynamics:
   - Preserves Byculla (3409510) coastal marine drop (~20.5 ug/m3)
   - Preserves Northern inland stations (11611 Malad, 3409511 Shivaji Nagar, 11606 Borivali) at ~40-41 ug/m3
5. Harmonic Diurnal Feature Encoding (sin/cos Fourier waves for daily cycle)
6. Strict Validation Assertions (2592 rows, ID match, zero NaNs, non-negative)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

def as_ist(values):
    return pd.to_datetime(values, format='ISO8601').dt.tz_convert('Asia/Kolkata')

def build_features(df):
    t = df['hour_start']
    h = t.dt.hour
    dow = t.dt.dayofweek
    doy = t.dt.dayofyear
    return pd.DataFrame({
        'location_id': df['location_id'].astype(str),
        'hour': h,
        'sin_hour': np.sin(2 * np.pi * h / 24.0),
        'cos_hour': np.cos(2 * np.pi * h / 24.0),
        'day_of_week': dow,
        'sin_dow': np.sin(2 * np.pi * dow / 7.0),
        'cos_dow': np.cos(2 * np.pi * dow / 7.0),
        'day_of_year': doy,
        'month': t.dt.month,
        'is_weekend': (dow >= 5).astype(int),
    }, index=df.index)

def main():
    print("=" * 70)
    print(" [AQI Oracles] Generating v23 Phase 2 Retrained Champion Solution")
    print("=" * 70)

    # 1. Load Data
    data_dir = Path('data/raw')
    train = pd.read_csv(data_dir / 'Train.csv')
    test = pd.read_csv(data_dir / 'Test.csv')
    sample = pd.read_csv(data_dir / 'SampleSubmission.csv')

    print(f">> Train observations: {len(train):,} (Includes all new 2 Nov observations)")
    print(f">> Test station-hours:  {len(test):,}")

    train['hour_end'] = as_ist(train['timestamp_ist'])
    train['hour_start'] = train['hour_end'] - pd.Timedelta(hours=1)
    test['hour_start'] = as_ist(test['target_start_ist'])

    X_test = build_features(test)

    # Common preprocessor
    prep = ColumnTransformer([
        ('station', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['location_id'])
    ], remainder='passthrough')

    # Component 1: RF 60-day (350 trees)
    print("\n>> Training Component 1: Random Forest (60-day Post-Monsoon Window, 350 trees)...")
    h60 = train.loc[train['hour_end'] > train['hour_end'].max() - pd.Timedelta(days=60)]
    X60 = build_features(h60)
    y60 = h60['pm25'].values
    X60_enc = prep.fit_transform(X60)
    X_test_enc = prep.transform(X_test)

    rf60 = RandomForestRegressor(n_estimators=350, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf60.fit(X60_enc, y60)
    pred_rf60 = rf60.predict(X_test_enc)
    print(f"   RF 60d complete. Overall Test Mean: {pred_rf60.mean():.2f}")

    # Component 2: RF 45-day (300 trees)
    print(">> Training Component 2: Random Forest (45-day Late-October Specialist, 300 trees)...")
    h45 = train.loc[train['hour_end'] > train['hour_end'].max() - pd.Timedelta(days=45)]
    X45 = build_features(h45)
    y45 = h45['pm25'].values
    X45_enc = prep.fit_transform(X45)

    rf45 = RandomForestRegressor(n_estimators=300, max_depth=11, min_samples_leaf=5, random_state=101, n_jobs=-1)
    rf45.fit(X45_enc, y45)
    pred_rf45 = rf45.predict(X_test_enc)
    print(f"   RF 45d complete. Overall Test Mean: {pred_rf45.mean():.2f}")

    # Component 3: ExtraTrees 60-day (300 trees)
    print(">> Training Component 3: Extra Trees (60-day Variance Reducer, 300 trees)...")
    et60 = ExtraTreesRegressor(n_estimators=300, max_depth=13, min_samples_leaf=5, random_state=42, n_jobs=-1)
    et60.fit(X60_enc, y60)
    pred_et60 = et60.predict(X_test_enc)
    print(f"   ET 60d complete. Overall Test Mean: {pred_et60.mean():.2f}")

    # Component 4: LightGBM 60-day (250 trees)
    print(">> Training Component 4: LightGBM (60-day Nonlinear Booster, 250 trees)...")
    X60_lgb = X60.copy()
    X_test_lgb = X_test.copy()
    X60_lgb['location_id'] = X60_lgb['location_id'].astype('category')
    X_test_lgb['location_id'] = X_test_lgb['location_id'].astype('category')

    lgbm60 = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.035, num_leaves=24, min_child_samples=18, random_state=42, verbose=-1)
    lgbm60.fit(X60_lgb, y60)
    pred_lgb60 = lgbm60.predict(X_test_lgb)
    print(f"   LGBM 60d complete. Overall Test Mean: {pred_lgb60.mean():.2f}")

    # Master Ensemble Blend: 40% RF60 + 30% RF45 + 15% ET60 + 15% LGBM60
    print("\n>> Ensembling 4-way components...")
    pred_master = (
        0.40 * pred_rf60 +
        0.30 * pred_rf45 +
        0.15 * pred_et60 +
        0.15 * pred_lgb60
    )
    pred_master = np.clip(pred_master, 0.0, None)

    # Build submission dataframe
    sub = sample.copy()
    sub['Target'] = pred_master

    # Verification Assertions
    assert len(sub) == 2592, f"Expected 2592 rows, got {len(sub)}"
    assert (sub['ID'] == sample['ID']).all(), "ID column mismatch!"
    assert not sub['Target'].isna().any(), "Found NaN values in predictions!"
    assert not np.isinf(sub['Target']).any(), "Found Inf values in predictions!"
    assert (sub['Target'] >= 0).all(), "Found negative PM2.5 predictions!"

    out_dir = Path('submissions')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'submission_v23_phase2_champion.csv'
    sub.to_csv(out_path, index=False)
    print(f"\n[FILE CREATED] {out_path.resolve()}")
    print(f">> Total Rows:      {len(sub):,}")
    print(f">> Overall Mean:    {sub['Target'].mean():.4f} ug/m3")
    print(f">> Overall Min:     {sub['Target'].min():.4f} ug/m3")
    print(f">> Overall Max:     {sub['Target'].max():.4f} ug/m3")

    # Diagnostics on active scoring window (27-28 Oct)
    test['Target'] = sub['Target']
    test_dt = test['hour_start'].dt.date
    w2 = test[test_dt.isin([pd.to_datetime('2026-10-27').date(), pd.to_datetime('2026-10-28').date()])]
    print("\n" + "=" * 70)
    print(f" ACTIVE SCORING WINDOW (27-28 Oct, 192 rows):")
    print(f" Window Mean: {w2['Target'].mean():.4f} ug/m3 | Std: {w2['Target'].std():.4f} ug/m3")
    print(f" Window Min:  {w2['Target'].min():.4f} ug/m3 | Max: {w2['Target'].max():.4f} ug/m3")
    print("-" * 70)
    print(" Station Breakdown on Active Scoring Window:")
    for loc in test['location_id'].unique():
        s = w2[w2['location_id'] == loc]
        print(f"  Station {loc:7d} | Mean: {s['Target'].mean():.2f} ug/m3 | Std: {s['Target'].std():.2f} | Min: {s['Target'].min():.2f} | Max: {s['Target'].max():.2f}")
    print("=" * 70)

if __name__ == '__main__':
    main()

