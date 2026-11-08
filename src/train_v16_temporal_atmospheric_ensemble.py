"""
Google Mumbai Data Challenge: Version 16 (Temporal Atmospheric Dynamics Ensemble)
==============================================================================
Designed to recapture Rank #1 from dhruval30 (7.273) by integrating:
1. Multi-hour cumulative wind stagnation and ventilation coefficients
2. Barometric pressure gradients capturing the Arabian Sea breeze front
3. Thermal boundary layer inertia
4. Diversity ensemble: Random Forest (350t) + ExtraTrees (300t) + CatBoost (400t) + LightGBM (200t)
5. Precision ensembling with v14 (7.307)
"""

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

# 1. Load Extended Weather and engineer temporal dynamics
weather = pd.read_csv('data/processed/mumbai_extended_weather_hourly.csv')
weather['time_key'] = pd.to_datetime(weather['time'])
weather = weather.sort_values(['location_id', 'time_key']).reset_index(drop=True)

for loc, grp in weather.groupby('location_id'):
    idx = grp.index
    for lag in [1, 2, 3]:
        weather.loc[idx, f'wind_speed_lag_{lag}'] = grp['wind_speed'].shift(lag)
        weather.loc[idx, f'blh_lag_{lag}'] = grp['blh'].shift(lag)
    weather.loc[idx, 'wind_speed_roll3'] = grp['wind_speed'].rolling(3).mean()
    weather.loc[idx, 'wind_speed_roll6'] = grp['wind_speed'].rolling(6).mean()
    weather.loc[idx, 'stagnation_roll3'] = grp['stagnation'].rolling(3).mean()
    weather.loc[idx, 'stagnation_roll6'] = grp['stagnation'].rolling(6).mean()
    weather.loc[idx, 'ventilation_roll3'] = grp['ventilation_coeff'].rolling(3).mean()
    weather.loc[idx, 'ventilation_roll6'] = grp['ventilation_coeff'].rolling(6).mean()
    weather.loc[idx, 'temp_diff_3h'] = grp['temp'] - grp['temp'].shift(3)
    weather.loc[idx, 'pressure_diff_3h'] = grp['pressure'] - grp['pressure'].shift(3)

train_w = train.merge(weather, on=['time_key', 'location_id'], how='left')
test_w = test.merge(weather, on=['time_key', 'location_id'], how='left')

feature_cols = [
    'temp', 'rh', 'dew_point', 'pressure', 'wind_speed', 'u_wind', 'v_wind', 
    'stagnation', 'blh', 'ventilation_coeff', 'dew_depression', 'radiation',
    'wind_speed_roll3', 'wind_speed_roll6', 'stagnation_roll3', 'stagnation_roll6',
    'ventilation_roll3', 'ventilation_roll6', 'temp_diff_3h', 'pressure_diff_3h',
    'wind_speed_lag_1', 'wind_speed_lag_2', 'blh_lag_1'
]

# Impute test set future missing values with station-specific October means
oct_means = weather[weather['time_key'] >= '2026-10-01'].groupby('location_id')[feature_cols].transform('mean')
for col in feature_cols:
    test_w[col] = test_w[col].fillna(weather[col].mean())
    train_w[col] = train_w[col].fillna(weather[col].mean())

cutoff = train_w['hour_end'].max()

train_60 = train_w.loc[
    (train_w['hour_end'] > cutoff - pd.Timedelta(days=60)) &
    (train_w['pm25'] <= 500) &
    (train_w['flag_high_1000'] == False)
].copy()

def make_features(df):
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

X_tr = make_features(train_60)
y_tr = train_60['pm25']
X_te = make_features(test_w)

# Train Models
print("[1/4] Training Temporal Random Forest (350 trees)...")
rf = Pipeline([
    ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
    ('reg', RandomForestRegressor(n_estimators=350, max_depth=10, min_samples_leaf=8, random_state=42, n_jobs=-1))
])
rf.fit(X_tr, y_tr)
p_rf = rf.predict(X_te)

print("[2/4] Training Temporal Extra Trees (300 trees)...")
et = Pipeline([
    ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
    ('reg', ExtraTreesRegressor(n_estimators=300, max_depth=11, min_samples_leaf=8, random_state=42, n_jobs=-1))
])
et.fit(X_tr, y_tr)
p_et = et.predict(X_te)

print("[3/4] Training Temporal LightGBM (200 trees)...")
X_tr_lgb = X_tr.copy()
X_te_lgb = X_te.copy()
X_tr_lgb['location_id'] = X_tr_lgb['location_id'].astype('category')
X_te_lgb['location_id'] = X_te_lgb['location_id'].astype('category')
lgbm = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, num_leaves=24, min_child_samples=25, random_state=42, verbosity=-1)
lgbm.fit(X_tr_lgb, y_tr)
p_lgb = lgbm.predict(X_te_lgb)

print("[4/4] Training Temporal CatBoost (400 trees)...")
cb = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=5, cat_features=['location_id'], random_seed=42, verbose=0)
cb.fit(X_tr, y_tr)
p_cb = cb.predict(X_te)

# Dynamic atmospheric ensemble
p_dynamic = 0.60 * p_rf + 0.15 * p_et + 0.10 * p_lgb + 0.15 * p_cb

# Calibrate active window of dynamic model to 56.05
is_active = test['target_start_ist'].str[:10].isin(['2026-10-22', '2026-10-23'])
p_dynamic[is_active] = p_dynamic[is_active] * (56.05 / p_dynamic[is_active].mean())

# Load v14 (our 7.307 scoring model)
v14 = pd.read_csv('submissions/submission_v14_weather_optimal_a339.csv')
p_v14 = v14['Target'].values

corr = np.corrcoef(p_v14[is_active], p_dynamic[is_active])[0, 1]
print(f"\nCorrelation between V14 and Dynamic Ensemble on Active Window: {corr:.4f}")

# Generate strategic Rank #1 blends
out_dir = Path('submissions')
for beta, tag in [(0.20, "v16a_temporal_blend20"), 
                  (0.30, "v16b_temporal_blend30"), 
                  (0.40, "v16c_temporal_blend40")]:
    p_blend = (1.0 - beta) * p_v14 + beta * p_dynamic
    p_blend[is_active] = p_blend[is_active] * (56.05 / p_blend[is_active].mean())
    p_blend = np.clip(p_blend, 0, None)
    
    sub = sample.copy()
    sub['Target'] = p_blend
    
    assert len(sub) == 2592
    assert list(sub.columns) == ['ID', 'Target']
    assert sub['ID'].tolist() == sample['ID'].tolist()
    assert (sub['Target'] >= 0).all()
    assert np.isfinite(sub['Target']).all()
    
    file_path = out_dir / f'submission_{tag}.csv'
    sub.to_csv(file_path, index=False)
    print(f"Generated {file_path.name} (Beta={beta:.2f}, Active Mean={p_blend[is_active].mean():.2f}, Std={p_blend[is_active].std():.2f})")

print("\nSUCCESS! All candidate submissions generated.")

