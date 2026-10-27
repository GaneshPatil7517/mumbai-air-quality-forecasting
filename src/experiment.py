import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

def as_ist(values):
    return pd.to_datetime(values, format='ISO8601').dt.tz_convert('Asia/Kolkata')

train = pd.read_csv('data/raw/Train.csv')
stations = pd.read_csv('data/raw/Stations.csv')

train['hour_end'] = as_ist(train['timestamp_ist'])
train['hour_start'] = train['hour_end'] - pd.Timedelta(hours=1)
train = train.merge(stations[['location_id', 'latitude', 'longitude']], on='location_id', how='left')
train = train.sort_values(['location_id', 'hour_end']).reset_index(drop=True)

# Validation period: Oct 20 to Oct 22 (the exact 48h before test)
val_end = train['hour_end'].max()
val_start = val_end - pd.Timedelta(days=2)

train_clean = train.loc[
    (train['hour_end'] <= val_start) &
    (train['pm25'] <= 500) &
    (train['flag_high_1000'] == False)
].copy()

valid = train.loc[
    (train['hour_end'] > val_start) &
    (train['hour_end'] <= val_end)
].copy()

print(f"Training on data up to: {train_clean['hour_end'].max()}")
print(f"Validating on: {valid['hour_end'].min()} to {valid['hour_end'].max()} ({len(valid)} rows)")

# Experiment with different feature representations
def make_features(df, train_stats=None):
    t = df['hour_start']
    f = pd.DataFrame(index=df.index)
    f['location_id'] = df['location_id'].astype('category')
    f['latitude'] = df['latitude']
    f['longitude'] = df['longitude']
    
    # Time features
    f['hour'] = t.dt.hour.astype('category')
    f['day_of_week'] = t.dt.dayofweek.astype('category')
    f['is_weekend'] = (t.dt.dayofweek >= 5).astype(int)
    f['day_of_year'] = t.dt.dayofyear
    f['month'] = t.dt.month
    
    # Cyclical representations
    f['sin_hour'] = np.sin(2 * np.pi * t.dt.hour / 24)
    f['cos_hour'] = np.cos(2 * np.pi * t.dt.hour / 24)
    f['sin_dow'] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    f['cos_dow'] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
    
    # Station-hour combined categorical
    f['station_hour'] = (df['location_id'].astype(str) + '_' + t.dt.hour.astype(str)).astype('category')
    
    return f

X_train = make_features(train_clean)
y_train = train_clean['pm25']
X_valid = make_features(valid)
y_valid = valid['pm25']

print("\n--- Testing Models on Oct 20-22 Validation Window ---")

# 1. LightGBM with native categorical features
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)
lgb_model.fit(X_train, y_train)
p_lgb = np.clip(lgb_model.predict(X_valid), 0, None)
print(f"LightGBM RMSE: {np.sqrt(mean_squared_error(y_valid, p_lgb)):.4f}")

# 2. CatBoost with native categorical features
cat_cols = ['location_id', 'hour', 'day_of_week', 'station_hour']
X_tr_cb = X_train.copy()
X_va_cb = X_valid.copy()
for c in cat_cols:
    X_tr_cb[c] = X_tr_cb[c].astype(str)
    X_va_cb[c] = X_va_cb[c].astype(str)

cb_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    cat_features=cat_cols,
    random_seed=42,
    verbose=0
)
cb_model.fit(X_tr_cb, y_train)
p_cb = np.clip(cb_model.predict(X_va_cb), 0, None)
print(f"CatBoost RMSE: {np.sqrt(mean_squared_error(y_valid, p_cb)):.4f}")

# 3. Test different training history subsets (e.g., 60 days vs 90 days vs all post-monsoon)
for days in [45, 60, 75, 90]:
    cutoff = train_clean['hour_end'].max()
    sub_tr = train_clean[train_clean['hour_end'] > cutoff - pd.Timedelta(days=days)]
    X_sub = make_features(sub_tr)
    y_sub = sub_tr['pm25']
    
    m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=25, random_state=42, verbosity=-1)
    m.fit(X_sub, y_sub)
    p = np.clip(m.predict(X_valid), 0, None)
    print(f"LightGBM (last {days}d) RMSE: {np.sqrt(mean_squared_error(y_valid, p)):.4f}")
