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

# Sort weather by station and time to compute rolling/lag features cleanly
weather = weather.sort_values(['location_id', 'time_key']).reset_index(drop=True)

# Engineer temporal weather dynamics per station
for loc, grp in weather.groupby('location_id'):
    idx = grp.index
    for lag in [1, 2, 3, 6]:
        weather.loc[idx, f'wind_speed_lag_{lag}'] = grp['wind_speed'].shift(lag)
        weather.loc[idx, f'blh_lag_{lag}'] = grp['blh'].shift(lag)
        weather.loc[idx, f'temp_lag_{lag}'] = grp['temp'].shift(lag)
    weather.loc[idx, 'wind_speed_roll3'] = grp['wind_speed'].rolling(3).mean()
    weather.loc[idx, 'wind_speed_roll6'] = grp['wind_speed'].rolling(6).mean()
    weather.loc[idx, 'stagnation_roll3'] = grp['stagnation'].rolling(3).mean()
    weather.loc[idx, 'stagnation_roll6'] = grp['stagnation'].rolling(6).mean()
    weather.loc[idx, 'ventilation_roll3'] = grp['ventilation_coeff'].rolling(3).mean()
    weather.loc[idx, 'ventilation_roll6'] = grp['ventilation_coeff'].rolling(6).mean()
    weather.loc[idx, 'temp_diff_3h'] = grp['temp'] - grp['temp'].shift(3)
    weather.loc[idx, 'pressure_diff_3h'] = grp['pressure'] - grp['pressure'].shift(3)

train_w = train.merge(weather, on=['time_key', 'location_id'], how='left')

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

basic_cols = ['temp', 'rh', 'wind_speed', 'u_wind', 'v_wind', 'pressure', 'stagnation']
dynamic_cols = basic_cols + ['blh', 'ventilation_coeff', 'dew_depression', 'radiation',
                            'wind_speed_roll3', 'wind_speed_roll6', 'stagnation_roll3', 
                            'stagnation_roll6', 'ventilation_roll3', 'temp_diff_3h', 'pressure_diff_3h']

def get_features(df, cols):
    t = df['hour_start']
    f = pd.DataFrame(index=df.index)
    f['location_id'] = df['location_id'].astype(str)
    f['hour'] = t.dt.hour
    f['day_of_week'] = t.dt.dayofweek
    f['is_weekend'] = (t.dt.dayofweek >= 5).astype(int)
    f['day_of_year'] = t.dt.dayofyear
    f['sin_hour'] = np.sin(2 * np.pi * t.dt.hour / 24)
    f['cos_hour'] = np.cos(2 * np.pi * t.dt.hour / 24)
    for c in cols:
        f[c] = df[c]
    return f

y_tr = train_clean['pm25']
y_va = valid['pm25']

# Train with Basic vs Dynamic
for mode_name, cols in [("BASIC (V9 style)", basic_cols), ("DYNAMIC (Temporal Physics)", dynamic_cols)]:
    X_tr = get_features(train_clean, cols)
    X_va = get_features(valid, cols)
    
    # Fill NAs in dynamic features
    for c in cols:
        X_tr[c] = X_tr[c].fillna(X_tr[c].mean())
        X_va[c] = X_va[c].fillna(X_tr[c].mean())

    # Random Forest
    rf = Pipeline([
        ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
        ('reg', RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict(X_va)
    
    # LightGBM
    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()
    X_tr_lgb['location_id'] = X_tr_lgb['location_id'].astype('category')
    X_va_lgb['location_id'] = X_va_lgb['location_id'].astype('category')
    lgbm = lgb.LGBMRegressor(n_estimators=220, learning_rate=0.03, max_depth=5, num_leaves=24, min_child_samples=25, random_state=42, verbosity=-1)
    lgbm.fit(X_tr_lgb, y_tr)
    p_lgb = lgbm.predict(X_va_lgb)
    
    # CatBoost
    cb = CatBoostRegressor(iterations=350, learning_rate=0.03, depth=5, cat_features=['location_id'], random_seed=42, verbose=0)
    cb.fit(X_tr, y_tr)
    p_cb = cb.predict(X_va)
    
    blend = 0.40 * p_rf + 0.35 * p_lgb + 0.25 * p_cb
    scale = y_va.mean() / blend.mean()
    cal_blend = blend * scale
    
    print(f"\n--- {mode_name} ---")
    print(f"RF RMSE:        {np.sqrt(mean_squared_error(y_va, p_rf)):.4f}")
    print(f"LGBM RMSE:      {np.sqrt(mean_squared_error(y_va, p_lgb)):.4f}")
    print(f"CatBoost RMSE:  {np.sqrt(mean_squared_error(y_va, p_cb)):.4f}")
    print(f"Blend Raw RMSE: {np.sqrt(mean_squared_error(y_va, blend)):.4f}")
    print(f"Blend Cal RMSE: {np.sqrt(mean_squared_error(y_va, cal_blend)):.4f}")

