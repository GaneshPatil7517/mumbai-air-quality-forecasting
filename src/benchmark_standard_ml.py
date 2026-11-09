"""
Benchmark standard LightGBM and CatBoost models on Train Phase 4
See what a top competitor ML model predicts for Nov 7-8!
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('Train Phase 4.csv')
test = pd.read_csv('zindi data/Test.csv')
weather = pd.read_csv('data/processed/mumbai_weather_hourly_phase4.csv')

train['dt'] = pd.to_datetime(train['timestamp_ist'].str[:19])
test['dt'] = pd.to_datetime(test['target_end_ist'].str[:19])
weather['dt'] = pd.to_datetime(weather['time'].str[:19])

# Merge weather
train_w = train.merge(weather, on=['dt', 'location_id'], how='left')
test_w = test.merge(weather, on=['dt', 'location_id'], how='left')

for col in ['temp', 'rh', 'wind_speed', 'u_wind', 'v_wind', 'stagnation', 'pressure']:
    st_means = weather.groupby('location_id')[col].mean()
    train_w[col] = train_w[col].fillna(train_w['location_id'].map(st_means))
    test_w[col] = test_w[col].fillna(test_w['location_id'].map(st_means))

def get_feats(df):
    f = pd.DataFrame(index=df.index)
    f['location_id'] = df['location_id'].astype('category')
    f['hour'] = df['dt'].dt.hour
    f['dayofweek'] = df['dt'].dt.dayofweek
    f['is_weekend'] = (df['dt'].dt.dayofweek >= 5).astype(int)
    f['dayofyear'] = df['dt'].dt.dayofyear
    f['temp'] = df['temp']
    f['rh'] = df['rh']
    f['wind_speed'] = df['wind_speed']
    f['u_wind'] = df['u_wind']
    f['v_wind'] = df['v_wind']
    f['stagnation'] = df['stagnation']
    f['pressure'] = df['pressure']
    return f

# Train on clean data (excluding flag_high_1000)
clean_train = train_w[train_w['flag_high_1000'] == False].copy()

# Look at multiple lookback windows:
# Window 1: All post-monsoon (from 1 Sept 2026 onwards)
pm_train = clean_train[clean_train['dt'] >= '2026-09-01'].copy()
X_tr = get_feats(pm_train)
y_tr = pm_train['pm25'].values
X_te = get_feats(test_w)

mask_78 = (test['dt'] >= '2026-11-07') & (test['dt'] < '2026-11-09')

# Model A: LightGBM tuned
m_lgb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42, verbose=-1)
m_lgb.fit(X_tr, y_tr)
p_lgb = m_lgb.predict(X_te)
print(f"LGBM Nov 7-8 Mean: {p_lgb[mask_78].mean():.2f}, Std: {p_lgb[mask_78].std():.2f}, Min: {p_lgb[mask_78].min():.2f}, Max: {p_lgb[mask_78].max():.2f}")

# Model B: CatBoost
cat_features = ['location_id']
m_cat = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0, cat_features=cat_features)
m_cat.fit(X_tr, y_tr)
p_cat = m_cat.predict(X_te)
print(f"CatBoost Nov 7-8 Mean: {p_cat[mask_78].mean():.2f}, Std: {p_cat[mask_78].std():.2f}, Min: {p_cat[mask_78].min():.2f}, Max: {p_cat[mask_78].max():.2f}")

# Model C: Random Forest
prep_rf = pd.get_dummies(X_tr, columns=['location_id'])
prep_te = pd.get_dummies(X_te, columns=['location_id'])
m_rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
m_rf.fit(prep_rf, y_tr)
p_rf = m_rf.predict(prep_te)
print(f"RandomForest Nov 7-8 Mean: {p_rf[mask_78].mean():.2f}, Std: {p_rf[mask_78].std():.2f}, Min: {p_rf[mask_78].min():.2f}, Max: {p_rf[mask_78].max():.2f}")

