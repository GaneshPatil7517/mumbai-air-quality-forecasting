import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
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

# Load extended weather
weather = pd.read_csv('data/processed/mumbai_extended_weather_hourly.csv')
weather['time_key'] = pd.to_datetime(weather['time'])

train_w = train.merge(weather, on=['time_key', 'location_id'], how='left')
test_w = test.merge(weather, on=['time_key', 'location_id'], how='left')

ext_cols = ['temp', 'rh', 'dew_point', 'pressure', 'precip', 'cloud', 'wind_speed', 'wind_dir', 'wind_gusts', 'blh', 'radiation', 'u_wind', 'v_wind', 'ventilation_coeff', 'stagnation', 'inversion_flag', 'dew_depression']
means = weather[weather['time_key'] >= '2026-10-01'][ext_cols].mean()
for col in ext_cols:
    test_w[col] = test_w[col].fillna(means[col])
    train_w[col] = train_w[col].fillna(means[col])

cutoff = train_w['hour_end'].max()

# 60-day clean data
train_60 = train_w.loc[
    (train_w['hour_end'] > cutoff - pd.Timedelta(days=60)) &
    (train_w['pm25'] <= 500) &
    (train_w['flag_high_1000'] == False)
].copy()

def make_features_ext(df):
    t = df['hour_start']
    f = pd.DataFrame(index=df.index)
    f['location_id'] = df['location_id'].astype(str)
    f['hour'] = t.dt.hour
    f['day_of_week'] = t.dt.dayofweek
    f['is_weekend'] = (t.dt.dayofweek >= 5).astype(int)
    f['day_of_year'] = t.dt.dayofyear
    f['sin_hour'] = np.sin(2 * np.pi * t.dt.hour / 24)
    f['cos_hour'] = np.cos(2 * np.pi * t.dt.hour / 24)
    
    # Extended atmospheric physics
    f['temp'] = df['temp']
    f['rh'] = df['rh']
    f['dew_point'] = df['dew_point']
    f['pressure'] = df['pressure']
    f['wind_speed'] = df['wind_speed']
    f['wind_gusts'] = df['wind_gusts']
    f['u_wind'] = df['u_wind']
    f['v_wind'] = df['v_wind']
    f['blh'] = df['blh']
    f['ventilation_coeff'] = df['ventilation_coeff']
    f['stagnation'] = df['stagnation']
    f['inversion_flag'] = df['inversion_flag']
    f['dew_depression'] = df['dew_depression']
    f['radiation'] = df['radiation']
    return f

X_tr = make_features_ext(train_60)
y_tr = train_60['pm25']
X_te = make_features_ext(test_w)

# Train extended models
print("Training Extended Random Forest (350 trees)...")
rf = Pipeline([
    ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
    ('reg', RandomForestRegressor(n_estimators=350, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1))
])
rf.fit(X_tr, y_tr)
p_rf = rf.predict(X_te)

print("Training Extended LightGBM...")
X_tr_lgb = X_tr.copy()
X_te_lgb = X_te.copy()
X_tr_lgb['location_id'] = X_tr_lgb['location_id'].astype('category')
X_te_lgb['location_id'] = X_te_lgb['location_id'].astype('category')
lgbm = lgb.LGBMRegressor(n_estimators=220, learning_rate=0.03, max_depth=5, num_leaves=24, min_child_samples=25, random_state=42, verbosity=-1)
lgbm.fit(X_tr_lgb, y_tr)
p_lgb = lgbm.predict(X_te_lgb)

print("Training Extended CatBoost...")
cb = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=5, cat_features=['location_id'], random_seed=42, verbose=0)
cb.fit(X_tr, y_tr)
p_cb = cb.predict(X_te)

p_extended = 0.45 * p_rf + 0.35 * p_lgb + 0.20 * p_cb

# Load V9 (our Rank 1 winning model 7.608)
v9 = pd.read_csv('submissions/submission_v9_weather_ensemble.csv')
p_v9 = v9['Target'].values

print("Correlation between V9 and Extended Model:", np.corrcoef(p_v9, p_extended)[0, 1])

# Blend V9 + Extended Model (50% V9 + 50% Extended)
is_active = test['target_start_ist'].str[:10].isin(['2026-10-22', '2026-10-23'])
v10_preds = 0.50 * p_v9 + 0.50 * p_extended

# Calibrate active window
current_mean = v10_preds[is_active].mean()
scale = 56.05 / current_mean
v10_preds[is_active] = v10_preds[is_active] * scale
v10_preds = np.clip(v10_preds, 0, None)

sub10 = sample.copy()
sub10['Target'] = v10_preds
out_file = Path('submissions/submission_v10_extended_atmospheric_ensemble.csv')
sub10.to_csv(out_file, index=False)

print("="*50)
print(f"SUCCESS! Saved {out_file}")
print(f"Active window mean: {v10_preds[is_active].mean():.2f}")
print(f"Active window std:  {v10_preds[is_active].std():.2f}")
print(f"Overall mean:        {v10_preds.mean():.2f}")
print("="*50)
