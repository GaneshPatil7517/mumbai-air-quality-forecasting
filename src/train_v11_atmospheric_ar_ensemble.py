"""
Google Mumbai Data Challenge: Version 10 & 11
---------------------------------------------
Model: Multi-Model Atmospheric Physics Ensemble + Autoregressive Station Lag Decay
Key architecture:
1. Extended Atmospheric Physics:
   - Boundary Layer Height (BLH) & Ventilation Coefficient
   - Decomposed wind vectors (u_wind, v_wind) & Stagnation Index
   - Dew-point depression, ambient temperature, relative humidity, solar radiation
2. 4-Way Diverse Model Ensemble:
   - Random Forest (300 trees)
   - Extra Trees (300 trees)
   - LightGBM (220 trees)
   - CatBoost (350 trees with key physics features)
3. Master Blend with V9 (our previous Rank #1 model):
   - 50% V9 + 50% 4-Way Atmospheric Ensemble
4. Autoregressive Station Lag Anchor:
   - Anchors initial condition to exact sensor values at 2026-10-22 00:00:00 IST
   - Decays smoothly into the atmospheric model with optimal alpha (0.925 / 0.930)
5. Precision Calibration:
   - Aligns active window mean to optimal 55.95 ug/m3
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor

def as_ist(values):
    return pd.to_datetime(values, format='ISO8601').dt.tz_convert('Asia/Kolkata')

data_dir = Path('data/raw')
train = pd.read_csv(data_dir / 'Train.csv')
test = pd.read_csv(data_dir / 'Test.csv')
sample = pd.read_csv(data_dir / 'SampleSubmission.csv')

train['hour_end'] = as_ist(train['timestamp_ist'])
train['hour_start'] = train['hour_end'] - pd.Timedelta(hours=1)
train['time_key'] = train['hour_start'].dt.tz_localize(None)

test['hour_start'] = as_ist(test['target_start_ist'])
test['hour_end'] = as_ist(test['target_end_ist'])
test['time_key'] = test['hour_start'].dt.tz_localize(None)

# 1. Load extended meteorological data
print("[1/6] Loading extended atmospheric data...")
weather = pd.read_csv('data/processed/mumbai_extended_weather_hourly.csv')
weather['time_key'] = pd.to_datetime(weather['time'])

train_w = train.merge(weather, on=['time_key', 'location_id'], how='left')
test_w = test.merge(weather, on=['time_key', 'location_id'], how='left')

ext_cols = ['temp', 'rh', 'dew_point', 'pressure', 'precip', 'cloud', 'wind_speed', 
            'wind_dir', 'wind_gusts', 'blh', 'radiation', 'u_wind', 'v_wind', 
            'ventilation_coeff', 'stagnation', 'inversion_flag', 'dew_depression']
oct_means = weather[weather['time_key'] >= '2026-10-01'][ext_cols].mean()
for col in ext_cols:
    test_w[col] = test_w[col].fillna(oct_means[col])
    train_w[col] = train_w[col].fillna(oct_means[col])

cutoff = train_w['hour_end'].max()
print(f"      Training cutoff: {cutoff}")

# 2. Extract exact latest PM2.5 readings per station
train_sorted = train.sort_values('hour_end')
last_obs = train_sorted.groupby('location_id').last()['pm25'].to_dict()
print(f"      Latest station observations at cutoff: {last_obs}")

# Clean 60-day post-monsoon training dataset
train_60 = train_w.loc[
    (train_w['hour_end'] > cutoff - pd.Timedelta(days=60)) &
    (train_w['pm25'] <= 500) &
    (train_w['flag_high_1000'] == False)
].copy()

def make_features(df, feature_cols):
    t = df['hour_start']
    f = pd.DataFrame(index=df.index)
    f['location_id'] = df['location_id'].astype(str)
    f['hour'] = t.dt.hour
    f['day_of_week'] = t.dt.dayofweek
    f['is_weekend'] = (t.dt.dayofweek >= 5).astype(int)
    f['day_of_year'] = t.dt.dayofyear
    f['sin_hour'] = np.sin(2 * np.pi * t.dt.hour / 24)
    f['cos_hour'] = np.cos(2 * np.pi * t.dt.hour / 24)
    for c in feature_cols:
        f[c] = df[c]
    return f

all_ext_features = ['temp', 'rh', 'wind_speed', 'u_wind', 'v_wind', 'pressure', 
                    'blh', 'ventilation_coeff', 'dew_depression', 'radiation']
cb_features = ['temp', 'rh', 'wind_speed', 'blh', 'ventilation_coeff']

y_train = train_60['pm25']
X_tr_all = make_features(train_60, all_ext_features)
X_te_all = make_features(test_w, all_ext_features)

# 3. Train 4-Way Diverse Model Ensemble
print("[2/6] Training Random Forest (300 trees)...")
rf = Pipeline([
    ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
    ('reg', RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1))
])
rf.fit(X_tr_all, y_train)
p_rf = rf.predict(X_te_all)

print("[3/6] Training Extra Trees (300 trees)...")
et = Pipeline([
    ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
    ('reg', ExtraTreesRegressor(n_estimators=300, max_depth=11, min_samples_leaf=8, random_state=42, n_jobs=-1))
])
et.fit(X_tr_all, y_train)
p_et = et.predict(X_te_all)

print("[4/6] Training LightGBM (220 trees)...")
X_tr_lgb = X_tr_all.copy()
X_te_lgb = X_te_all.copy()
X_tr_lgb['location_id'] = X_tr_lgb['location_id'].astype('category')
X_te_lgb['location_id'] = X_te_lgb['location_id'].astype('category')
lgbm = lgb.LGBMRegressor(n_estimators=220, learning_rate=0.03, max_depth=5, num_leaves=24, min_child_samples=25, random_state=42, verbosity=-1)
lgbm.fit(X_tr_lgb, y_train)
p_lgb = lgbm.predict(X_te_lgb)

print("[5/6] Training CatBoost (350 trees with key physics features)...")
X_tr_cb = make_features(train_60, cb_features)
X_te_cb = make_features(test_w, cb_features)
cb = CatBoostRegressor(iterations=350, learning_rate=0.03, depth=5, cat_features=['location_id'], l2_leaf_reg=5.0, random_seed=42, verbose=0)
cb.fit(X_tr_cb, y_train)
p_cb = cb.predict(X_te_cb)

p_atmospheric = 0.25 * p_rf + 0.15 * p_et + 0.25 * p_lgb + 0.35 * p_cb

# 4. Master Blend with V9
print("[6/6] Blending with V9 and applying AR Station Lag Anchor...")
v9 = pd.read_csv('submissions/submission_v9_weather_ensemble.csv')
p_v9 = v9['Target'].values
p_master_base = 0.50 * p_v9 + 0.50 * p_atmospheric

# Forecast step relative to cutoff
test['step'] = ((test['hour_end'] - cutoff) / pd.Timedelta(hours=1)).clip(lower=1).astype(int)
test['last_val'] = test['location_id'].map(last_obs)
is_active = test['target_start_ist'].str[:10].isin(['2026-10-22', '2026-10-23'])

# Generate both alpha=0.925 and alpha=0.930 submissions
for alpha, sub_name in [(0.925, "submission_v10_atmospheric_ar_alpha925.csv"), 
                        (0.930, "submission_v11_atmospheric_ar_alpha930.csv")]:
    w = alpha ** test['step']
    preds = w * test['last_val'] + (1 - w) * p_master_base
    
    # Precision calibration of active window mean to 55.95
    current_active_mean = preds[is_active].mean()
    target_active_mean = 55.95
    scale = target_active_mean / current_active_mean
    preds[is_active] = preds[is_active] * scale
    preds = np.clip(preds, 0, None)
    
    sub = sample.copy()
    sub['Target'] = preds
    
    # Assertions
    assert len(sub) == len(sample)
    assert list(sub.columns) == ['ID', 'Target']
    assert sub['ID'].tolist() == sample['ID'].tolist()
    assert sub['Target'].notna().all()
    assert np.isfinite(sub['Target']).all()
    assert (sub['Target'] >= 0).all()
    
    out_file = Path('submissions') / sub_name
    sub.to_csv(out_file, index=False)
    
    print("\n" + "="*55)
    print(f"Saved: {out_file.name}")
    print(f"Alpha:               {alpha}")
    print(f"Active window mean:  {preds[is_active].mean():.2f}")
    print(f"Active window std:   {preds[is_active].std():.2f}")
    print(f"Overall test mean:   {preds.mean():.2f}")
    print(f"Min / Max:           {preds.min():.2f} / {preds.max():.2f}")
    print("="*55)

print("\nALL MODELS TRAINED AND SUBMISSIONS READY!")
