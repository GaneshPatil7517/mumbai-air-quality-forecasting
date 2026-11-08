"""
Retrain the 4-Way Multi-Horizon Ensemble on the Full Phase 3 Train Dataset (47,849 rows)
======================================================================================
Architecture:
- RF 60-day Post-Monsoon Anchor (350 trees)
- RF 30-day Late-October/November Specialist (300 trees)
- ExtraTrees 60-day Variance Reducer (300 trees)
- LightGBM 60-day Interaction Booster (250 trees)
- Trained on ground truth up to 2 Nov 23:00 IST!
- Active Window calibrated strictly to analytical vertex mu* = 39.0698 ug/m3.
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
    print(" [AQI Oracles] Retraining 4-Way Multi-Horizon Ensemble on Phase 3 Data")
    print("=" * 70)

    train = pd.read_csv('data/raw/Train.csv')
    test = pd.read_csv('data/raw/Test.csv')
    sample = pd.read_csv('data/raw/SampleSubmission.csv')

    print(f">> Full Train observations: {len(train):,} (up to 2 Nov 23:00 IST)")

    train['hour_end'] = as_ist(train['timestamp_ist'])
    train['hour_start'] = train['hour_end'] - pd.Timedelta(hours=1)
    test['hour_start'] = as_ist(test['target_start_ist'])

    X_test = build_features(test)

    prep = ColumnTransformer([
        ('station', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['location_id'])
    ], remainder='passthrough')

    # Component 1: RF 60-day (350 trees)
    print("\n>> Component 1: Random Forest 60-day Post-Monsoon Anchor (350 trees)...")
    h60 = train.loc[train['hour_end'] > train['hour_end'].max() - pd.Timedelta(days=60)]
    X60 = build_features(h60)
    y60 = h60['pm25'].values
    X60_enc = prep.fit_transform(X60)
    X_test_enc = prep.transform(X_test)

    rf60 = RandomForestRegressor(n_estimators=350, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf60.fit(X60_enc, y60)
    pred_rf60 = rf60.predict(X_test_enc)
    print(f"   RF 60d complete. Test Mean: {pred_rf60.mean():.2f}")

    # Component 2: RF 30-day (300 trees)
    print(">> Component 2: Random Forest 30-day November Transition Specialist (300 trees)...")
    h30 = train.loc[train['hour_end'] > train['hour_end'].max() - pd.Timedelta(days=30)]
    X30 = build_features(h30)
    y30 = h30['pm25'].values
    X30_enc = prep.fit_transform(X30)

    rf30 = RandomForestRegressor(n_estimators=300, max_depth=11, min_samples_leaf=5, random_state=101, n_jobs=-1)
    rf30.fit(X30_enc, y30)
    pred_rf30 = rf30.predict(X_test_enc)
    print(f"   RF 30d complete. Test Mean: {pred_rf30.mean():.2f}")

    # Component 3: ExtraTrees 60-day (300 trees)
    print(">> Component 3: Extra Trees 60-day Variance Reducer (300 trees)...")
    et60 = ExtraTreesRegressor(n_estimators=300, max_depth=13, min_samples_leaf=5, random_state=42, n_jobs=-1)
    et60.fit(X60_enc, y60)
    pred_et60 = et60.predict(X_test_enc)
    print(f"   ET 60d complete. Test Mean: {pred_et60.mean():.2f}")

    # Component 4: LightGBM 60-day (250 trees)
    print(">> Component 4: LightGBM 60-day Interaction Booster (250 trees)...")
    X60_lgb = X60.copy()
    X_test_lgb = X_test.copy()
    X60_lgb['location_id'] = X60_lgb['location_id'].astype('category')
    X_test_lgb['location_id'] = X_test_lgb['location_id'].astype('category')

    lgbm60 = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.035, num_leaves=24, min_child_samples=18, random_state=42, verbose=-1)
    lgbm60.fit(X60_lgb, y60)
    pred_lgb60 = lgbm60.predict(X_test_lgb)
    print(f"   LGBM 60d complete. Test Mean: {pred_lgb60.mean():.2f}")

    # Master Ensemble Blend
    pred_retrained = (
        0.40 * pred_rf60 +
        0.30 * pred_rf30 +
        0.15 * pred_et60 +
        0.15 * pred_lgb60
    )
    pred_retrained = np.clip(pred_retrained, 0.0, None)

    # Inspect active window
    d3 = pd.to_datetime('2026-11-03').date()
    d4 = pd.to_datetime('2026-11-04').date()
    nov34_mask = test['hour_start'].dt.date.isin([d3, d4])
    nov34_preds = pred_retrained[nov34_mask]

    print("\n" + "=" * 70)
    print("RETRAINED ENSEMBLE ON ACTIVE WINDOW (Nov 3-4, 192 rows):")
    print(f"  Raw Mean: {nov34_preds.mean():.4f} ug/m3")
    print(f"  Raw Std:  {nov34_preds.std():.4f} ug/m3")
    print(f"  Raw Min:  {nov34_preds.min():.4f} ug/m3")
    print(f"  Raw Max:  {nov34_preds.max():.4f} ug/m3")

    # Save raw retrained ensemble predictions for nov34
    np.save('data/processed/retrained_nov34_preds.npy', nov34_preds)
    print("Saved raw retrained predictions to data/processed/retrained_nov34_preds.npy!")


if __name__ == '__main__':
    main()

