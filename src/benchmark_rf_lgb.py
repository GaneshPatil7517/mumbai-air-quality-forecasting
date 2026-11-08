import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Load train and test data exactly like starter notebook
raw = pd.read_csv('zindi data/Phase 3 Train.csv')
raw['hour_end'] = pd.to_datetime(raw['timestamp_ist'].str[:19])
raw['hour_start'] = raw['hour_end'] - pd.Timedelta(hours=1)
raw['hour'] = raw['hour_start'].dt.hour
raw['day_of_week'] = raw['hour_start'].dt.dayofweek
raw['day_of_year'] = raw['hour_start'].dt.dayofyear
raw['month'] = raw['hour_start'].dt.month
raw = raw.sort_values(['location_id', 'hour_end']).reset_index(drop=True)

# Test on validation: Oct 31 00:00 to Nov 02 23:00 (last 2-3 days of train)
val_mask = (raw['hour_start'] >= '2026-10-31') & (raw['hour_start'] < '2026-11-03')
train_mask = (raw['hour_start'] < '2026-10-31')

# Test different training window lengths: 14 days, 30 days, 45 days, 60 days, 90 days
print('=== VALIDATION OF GBDT / RF MODELS ON LAST 3 DAYS OF TRAIN ===')

y_val = raw.loc[val_mask, 'pm25'].values
val_df = raw.loc[val_mask].copy()

for window_days in [14, 21, 30, 45, 60, 90]:
    t_start = pd.Timestamp('2026-10-31') - pd.Timedelta(days=window_days)
    tr_sub = raw.loc[train_mask & (raw['hour_start'] >= t_start)].copy()
    
    # Features:
    # 1. location_id (categorical)
    # 2. hour
    # 3. day_of_week
    # 4. station-level recent mean
    stn_means = tr_sub.groupby('location_id')['pm25'].mean().to_dict()
    stn_hourly = tr_sub.groupby(['location_id', 'hour'])['pm25'].mean().to_dict()
    
    tr_sub['stn_mean'] = tr_sub['location_id'].map(stn_means)
    tr_sub['stn_hourly'] = tr_sub.set_index(['location_id', 'hour']).index.map(stn_hourly)
    
    val_sub = val_df.copy()
    val_sub['stn_mean'] = val_sub['location_id'].map(stn_means)
    val_sub['stn_hourly'] = val_sub.set_index(['location_id', 'hour']).index.map(stn_hourly)
    
    # Baseline 1: Station hourly profile directly
    rmse_base = root_mean_squared_error(y_val, val_sub['stn_hourly'])
    
    # Baseline 2: LightGBM
    feats = ['location_id', 'hour', 'day_of_week', 'stn_mean', 'stn_hourly']
    X_tr = tr_sub[feats].copy()
    X_tr['location_id'] = X_tr['location_id'].astype('category')
    X_val = val_sub[feats].copy()
    X_val['location_id'] = X_val['location_id'].astype('category')
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=42)
    model.fit(X_tr, tr_sub['pm25'])
    preds_lgb = model.predict(X_val)
    rmse_lgb = root_mean_squared_error(y_val, preds_lgb)
    
    # Baseline 3: Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    # RF needs dummy encoding
    X_tr_rf = pd.get_dummies(tr_sub[['location_id', 'hour', 'day_of_week', 'stn_mean', 'stn_hourly']], columns=['location_id'])
    X_val_rf = pd.get_dummies(val_sub[['location_id', 'hour', 'day_of_week', 'stn_mean', 'stn_hourly']], columns=['location_id'])
    for col in X_tr_rf.columns:
        if col not in X_val_rf.columns: X_val_rf[col] = 0
    X_val_rf = X_val_rf[X_tr_rf.columns]
    
    rf.fit(X_tr_rf, tr_sub['pm25'])
    preds_rf = rf.predict(X_val_rf)
    rmse_rf = root_mean_squared_error(y_val, preds_rf)
    
    print(f'Window {window_days:2d}d: Hourly-Mean RMSE={rmse_base:.3f}, LightGBM RMSE={rmse_lgb:.3f}, RF RMSE={rmse_rf:.3f}')
