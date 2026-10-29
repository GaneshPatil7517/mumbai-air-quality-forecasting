import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

def as_ist(values):
    return pd.to_datetime(values, format='ISO8601').dt.tz_convert('Asia/Kolkata')

train = pd.read_csv('data/raw/Train.csv')
train['hour_end'] = as_ist(train['timestamp_ist'])
train['hour_start'] = train['hour_end'] - pd.Timedelta(hours=1)
train['time_key'] = train['hour_start'].dt.tz_localize(None)

weather = pd.read_csv('data/processed/mumbai_extended_weather_hourly.csv')
weather['time_key'] = pd.to_datetime(weather['time'])

train_w = train.merge(weather, on=['time_key', 'location_id'], how='left')

val_end = train_w['hour_end'].max() # 2026-10-22 00:00:00
val_start = val_end - pd.Timedelta(days=2) # 2026-10-20 00:00:00

train_clean = train_w.loc[
    (train_w['hour_end'] <= val_start) &
    (train_w['hour_end'] > val_start - pd.Timedelta(days=60)) &
    (train_w['pm25'] <= 500) &
    (train_w['flag_high_1000'] == False)
].copy()

valid = train_w.loc[
    (train_w['hour_end'] > val_start) &
    (train_w['hour_end'] <= val_end)
].copy()

def get_features(df, features_list):
    t = df['hour_start']
    f = pd.DataFrame(index=df.index)
    f['location_id'] = df['location_id'].astype(str)
    f['hour'] = t.dt.hour
    f['day_of_week'] = t.dt.dayofweek
    f['is_weekend'] = (t.dt.dayofweek >= 5).astype(int)
    f['day_of_year'] = t.dt.dayofyear
    f['sin_hour'] = np.sin(2 * np.pi * t.dt.hour / 24)
    f['cos_hour'] = np.cos(2 * np.pi * t.dt.hour / 24)
    for col in features_list:
        f[col] = df[col]
    return f

y_tr = train_clean['pm25']
y_va = valid['pm25']

ext_features = ['temp', 'rh', 'wind_speed', 'u_wind', 'v_wind', 'pressure', 'blh', 'ventilation_coeff', 'dew_depression', 'radiation']
cb_features = ['temp', 'rh', 'wind_speed', 'blh', 'ventilation_coeff']

X_tr_rf = get_features(train_clean, ext_features)
X_va_rf = get_features(valid, ext_features)

rf = Pipeline([
    ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
    ('reg', RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1))
])
rf.fit(X_tr_rf, y_tr)
p_rf = rf.predict(X_va_rf)

et = Pipeline([
    ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
    ('reg', ExtraTreesRegressor(n_estimators=300, max_depth=11, min_samples_leaf=8, random_state=42, n_jobs=-1))
])
et.fit(X_tr_rf, y_tr)
p_et = et.predict(X_va_rf)

X_tr_lgb = X_tr_rf.copy()
X_va_lgb = X_va_rf.copy()
X_tr_lgb['location_id'] = X_tr_lgb['location_id'].astype('category')
X_va_lgb['location_id'] = X_va_lgb['location_id'].astype('category')
lgbm = lgb.LGBMRegressor(n_estimators=220, learning_rate=0.03, max_depth=5, num_leaves=24, min_child_samples=25, random_state=42, verbosity=-1)
lgbm.fit(X_tr_lgb, y_tr)
p_lgb = lgbm.predict(X_va_lgb)

X_tr_cb = get_features(train_clean, cb_features)
X_va_cb = get_features(valid, cb_features)
cb = CatBoostRegressor(iterations=350, learning_rate=0.03, depth=5, cat_features=['location_id'], l2_leaf_reg=5.0, random_seed=42, verbose=0)
cb.fit(X_tr_cb, y_tr)
p_cb = cb.predict(X_va_cb)

p_ens = 0.25 * p_rf + 0.15 * p_et + 0.25 * p_lgb + 0.35 * p_cb

last_obs = train_clean.sort_values('hour_end').groupby('location_id').last()['pm25'].to_dict()
valid = valid.copy()
valid['step'] = ((valid['hour_end'] - val_start) / pd.Timedelta(hours=1)).astype(int)
valid['last_val'] = valid['location_id'].map(last_obs)

true_mean = y_va.mean()
print(f"Validation Ground Truth Mean: {true_mean:.2f}")

for alpha in [0.70, 0.80, 0.85, 0.90, 0.92]:
    w = alpha ** valid['step']
    p_ar = w * valid['last_val'] + (1 - w) * p_ens
    # Raw RMSE
    raw_rmse = np.sqrt(mean_squared_error(y_va, p_ar))
    # Calibrated to true mean
    scale = true_mean / p_ar.mean()
    cal_rmse = np.sqrt(mean_squared_error(y_va, p_ar * scale))
    print(f"Alpha {alpha:.2f} -> Raw RMSE: {raw_rmse:.4f} | Calibrated RMSE: {cal_rmse:.4f}")
