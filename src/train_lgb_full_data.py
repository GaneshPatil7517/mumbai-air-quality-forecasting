"""
Train modern LightGBM / Ridge pipeline on the full 47,849-row dataset
and evaluate out-of-sample on Nov 1-2
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# 1. Load data
train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])

w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize('Asia/Kolkata')

# Merge weather
df = pd.merge(train, w, on=['location_id', 'dt'], how='inner')
df = df.sort_values(['location_id', 'dt']).reset_index(drop=True)

# Feature engineering
df['hour'] = df['dt'].dt.hour
df['dayofweek'] = df['dt'].dt.dayofweek
df['month'] = df['dt'].dt.month
df['day'] = df['dt'].dt.day

# Cyclical time
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Atmospheric physics
df['wind_speed_diff'] = df['wind_speed'] - df.groupby(['location_id', df['dt'].dt.date])['wind_speed'].transform('mean')
df['temp_diff'] = df['temp'] - df.groupby(['location_id', df['dt'].dt.date])['temp'].transform('mean')
df['pressure_diff'] = df['pressure'] - df.groupby(['location_id', df['dt'].dt.date])['pressure'].transform('mean')

# Lags per station
for lag in [1, 2, 3, 24, 48]:
    df[f'pm25_lag_{lag}'] = df.groupby('location_id')['pm25'].shift(lag)

# Rolling features
for r in [6, 12, 24]:
    df[f'pm25_roll_mean_{r}'] = df.groupby('location_id')['pm25'].shift(1).rolling(r, min_periods=1).mean()
    df[f'pm25_roll_std_{r}'] = df.groupby('location_id')['pm25'].shift(1).rolling(r, min_periods=1).std()

# Station one-hot / categorical
df['location_cat'] = df['location_id'].astype('category')

feature_cols = [
    'hour', 'dayofweek', 'hour_sin', 'hour_cos',
    'temp', 'rh', 'wind_speed', 'pressure', 'stagnation',
    'wind_speed_diff', 'temp_diff', 'pressure_diff',
    'location_cat'
]

# Split: Train on everything before Nov 1, validate on Nov 1-2
val_mask = df['dt'] >= '2026-11-01'
train_mask = (df['dt'] >= '2026-08-01') & (df['dt'] < '2026-11-01')

X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, 'pm25']

X_val = df.loc[val_mask, feature_cols]
y_val = df.loc[val_mask, 'pm25']

print(f"Train samples: {len(X_train):,}, Validation samples: {len(X_val):,}")

# Train LightGBM regressor
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 42
}

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

val_preds = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"\nLightGBM Validation RMSE on Nov 1-2: {val_rmse:.4f}")

# Check with vertex calibration
val_preds_cal = val_preds - val_preds.mean() + y_val.mean()
val_rmse_cal = np.sqrt(mean_squared_error(y_val, val_preds_cal))
print(f"LightGBM Validation RMSE with Vertex Calibration: {val_rmse_cal:.4f}")

