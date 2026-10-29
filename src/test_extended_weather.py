import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
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

# Holdout: Oct 20 to Oct 22
val_end = train_w['hour_end'].max()
val_start = val_end - pd.Timedelta(days=2)

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

def get_features(df, mode="extended"):
    t = df['hour_start']
    f = pd.DataFrame(index=df.index)
    f['location_id'] = df['location_id'].astype(str)
    f['hour'] = t.dt.hour
    f['day_of_week'] = t.dt.dayofweek
    f['is_weekend'] = (t.dt.dayofweek >= 5).astype(int)
    f['day_of_year'] = t.dt.dayofyear
    f['sin_hour'] = np.sin(2 * np.pi * t.dt.hour / 24)
    f['cos_hour'] = np.cos(2 * np.pi * t.dt.hour / 24)
    
    if mode in ["basic", "extended"]:
        f['temp'] = df['temp']
        f['rh'] = df['rh']
        f['wind_speed'] = df['wind_speed']
        f['u_wind'] = df['u_wind']
        f['v_wind'] = df['v_wind']
        f['pressure'] = df['pressure']
        f['stagnation'] = df['stagnation']
        
    if mode == "extended":
        f['blh'] = df['blh']
        f['ventilation_coeff'] = df['ventilation_coeff']
        f['inversion_flag'] = df['inversion_flag']
        f['dew_depression'] = df['dew_depression']
        f['radiation'] = df['radiation']
        f['wind_gusts'] = df['wind_gusts']
        f['cloud'] = df['cloud']
    return f

y_tr = train_clean['pm25']
y_va = valid['pm25']

for mode in ["none", "basic", "extended"]:
    X_tr = get_features(train_clean, mode=mode)
    X_va = get_features(valid, mode=mode)
    
    # LightGBM
    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()
    X_tr_lgb['location_id'] = X_tr_lgb['location_id'].astype('category')
    X_va_lgb['location_id'] = X_va_lgb['location_id'].astype('category')
    
    lgbm = lgb.LGBMRegressor(n_estimators=220, learning_rate=0.03, max_depth=5, num_leaves=24, min_child_samples=25, random_state=42, verbosity=-1)
    lgbm.fit(X_tr_lgb, y_tr)
    p_lgb = lgbm.predict(X_va_lgb)
    
    # Random Forest
    rf = Pipeline([
        ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
        ('reg', RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=12, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict(X_va)
    
    # CatBoost
    cat_cols = ['location_id']
    cb = CatBoostRegressor(iterations=350, learning_rate=0.03, depth=5, cat_features=cat_cols, random_seed=42, verbose=0)
    cb.fit(X_tr, y_tr)
    p_cb = cb.predict(X_va)
    
    blend = 0.40 * p_rf + 0.35 * p_lgb + 0.25 * p_cb
    print(f"[{mode.upper():8s}] RF: {np.sqrt(mean_squared_error(y_va, p_rf)):.4f} | LGB: {np.sqrt(mean_squared_error(y_va, p_lgb)):.4f} | CB: {np.sqrt(mean_squared_error(y_va, p_cb)):.4f} | Blend: {np.sqrt(mean_squared_error(y_va, blend)):.4f}")
