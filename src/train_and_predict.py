"""
Google Mumbai Data Challenge: Baseline Pipeline & Submission Generator
---------------------------------------------------------------------
Model: Blend of Tuned Random Forest (80%) + LightGBM (20%)
Features: Station identifiers, hour of day, day of week, day of year, month
Post-processing: Non-negative clipping and strict ID order verification.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

from src.utils import seed_everything

def as_ist(values):
    return pd.to_datetime(values, format='ISO8601').dt.tz_convert('Asia/Kolkata')

def make_features_rf(df):
    time = df['hour_start']
    return pd.DataFrame({
        'location_id': df['location_id'].astype(str),
        'hour': time.dt.hour,
        'day_of_week': time.dt.dayofweek,
        'month': time.dt.month,
        'day_of_year': time.dt.dayofyear,
    }, index=df.index)

def make_features_lgb(df):
    time = df['hour_start']
    return pd.DataFrame({
        'location_id': df['location_id'].astype('category'),
        'hour': time.dt.hour,
        'day_of_week': time.dt.dayofweek,
        'month': time.dt.month,
        'day_of_year': time.dt.dayofyear,
    }, index=df.index)

def run_pipeline(
    data_dir: Path = Path('data/raw'),
    output_dir: Path = Path('submissions'),
    train_window_days: int = 90,
    seed: int = 42
):
    seed_everything(seed)
    
    # 1. Load Data
    print(f"[1/5] Loading datasets from {data_dir}...")
    train = pd.read_csv(data_dir / 'Train.csv')
    test = pd.read_csv(data_dir / 'Test.csv')
    sample = pd.read_csv(data_dir / 'SampleSubmission.csv')
    stations = pd.read_csv(data_dir / 'Stations.csv')
    
    print(f"      Train rows: {len(train):,}")
    print(f"      Test rows:  {len(test):,}")
    print(f"      Stations:   {len(stations)}")

    # 2. Parse Timestamps
    print("[2/5] Parsing IST timestamps...")
    train['hour_end'] = as_ist(train['timestamp_ist'])
    train['hour_start'] = train['hour_end'] - pd.Timedelta(hours=1)
    
    test['hour_start'] = as_ist(test['target_start_ist'])
    test['hour_end'] = as_ist(test['target_end_ist'])

    # 3. Filter Training Window and Outliers
    cutoff = train['hour_end'].max()
    window_start = cutoff - pd.Timedelta(days=train_window_days)
    
    # Exclude extreme sensor errors (> 500 ug/m3 or flagged > 1000)
    train_clean = train.loc[
        (train['hour_end'] > window_start) &
        (train['pm25'] <= 500) &
        (train['flag_high_1000'] == False)
    ].copy()
    
    print(f"[3/5] Cleaned training set (last {train_window_days} days): {len(train_clean):,} observations")

    y_train = train_clean['pm25']
    X_train_rf = make_features_rf(train_clean)
    X_test_rf = make_features_rf(test)
    
    X_train_lgb = make_features_lgb(train_clean)
    X_test_lgb = make_features_lgb(test)

    # 4. Train Models
    print("[4/5] Training Random Forest (200 trees)...")
    rf_model = Pipeline([
        ('features', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            random_state=seed,
            n_jobs=-1
        ))
    ])
    rf_model.fit(X_train_rf, y_train)
    rf_preds = rf_model.predict(X_test_rf)

    print("      Training LightGBM Regressor...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=64,
        random_state=seed,
        verbosity=-1
    )
    lgb_model.fit(X_train_lgb, y_train)
    lgb_preds = lgb_model.predict(X_test_lgb)

    # 5. Blend & Post-process
    print("[5/5] Blending models (80% Random Forest + 20% LightGBM)...")
    final_preds = 0.80 * rf_preds + 0.20 * lgb_preds
    final_preds = np.clip(final_preds, 0, None)  # PM2.5 must be non-negative

    predictions_by_id = pd.Series(final_preds, index=test['ID'])
    submission = sample.copy()
    submission['Target'] = submission['ID'].map(predictions_by_id)

    # Strict Assertions
    assert len(submission) == len(sample), "Row count mismatch!"
    assert list(submission.columns) == ['ID', 'Target'], "Columns mismatch!"
    assert submission['ID'].tolist() == sample['ID'].tolist(), "ID order mismatch!"
    assert submission['Target'].notna().all(), "Submission contains NaN values!"
    assert np.isfinite(submission['Target']).all(), "Submission contains infinite values!"
    assert (submission['Target'] >= 0).all(), "Submission contains negative values!"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "submission_baseline_rf_lgb_blend.csv"
    submission.to_csv(out_file, index=False)
    
    print("\n" + "="*50)
    print(f" SUCCESS: Generated submission at:")
    print(f" {out_file.resolve()}")
    print(f" Total Rows: {len(submission):,}")
    print(f" Target Mean: {submission['Target'].mean():.2f} µg/m³")
    print(f" Target Min:  {submission['Target'].min():.2f} µg/m³")
    print(f" Target Max:  {submission['Target'].max():.2f} µg/m³")
    print("="*50)

if __name__ == '__main__':
    run_pipeline()
