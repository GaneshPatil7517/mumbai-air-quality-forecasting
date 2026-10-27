import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def as_ist(values):
    return pd.to_datetime(values, format='ISO8601').dt.tz_convert('Asia/Kolkata')

train = pd.read_csv('data/raw/Train.csv')
train['hour_end'] = as_ist(train['timestamp_ist'])
train['hour_start'] = train['hour_end'] - pd.Timedelta(hours=1)
train = train.sort_values(['location_id', 'hour_end']).reset_index(drop=True)

# Let's test on two different historical 48-hour holdout windows!
# Window A: Oct 20 00:00 to Oct 22 00:00 (Holdout 1)
# Window B: Oct 18 00:00 to Oct 20 00:00 (Holdout 2)

for window_name, end_str in [("Oct 20-22", "2026-10-22 00:00:00+05:30"), ("Oct 18-20", "2026-10-20 00:00:00+05:30")]:
    val_end = pd.to_datetime(end_str)
    val_start = val_end - pd.Timedelta(days=2)
    
    # Use 45-60 days before val_start
    hist = train[
        (train['hour_end'] <= val_start) &
        (train['hour_end'] > val_start - pd.Timedelta(days=60)) &
        (train['pm25'] <= 500) &
        (train['flag_high_1000'] == False)
    ].copy()
    
    valid = train[
        (train['hour_end'] > val_start) &
        (train['hour_end'] <= val_end)
    ].copy()
    
    # Feature extraction
    def get_features(df):
        t = df['hour_start']
        f = pd.DataFrame(index=df.index)
        f['location_id'] = df['location_id'].astype(str)
        f['hour'] = t.dt.hour
        f['day_of_week'] = t.dt.dayofweek
        f['is_weekend'] = (t.dt.dayofweek >= 5).astype(int)
        f['day_of_year'] = t.dt.dayofyear
        
        # Cyclical hour
        f['sin_hour'] = np.sin(2 * np.pi * t.dt.hour / 24)
        f['cos_hour'] = np.cos(2 * np.pi * t.dt.hour / 24)
        return f

    X_tr = get_features(hist)
    y_tr = hist['pm25']
    X_va = get_features(valid)
    y_va = valid['pm25']
    
    # 1. Random Forest
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    
    rf = Pipeline([
        ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
        ('reg', RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=12, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict(X_va)
    
    # 2. LightGBM
    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()
    X_tr_lgb['location_id'] = X_tr_lgb['location_id'].astype('category')
    X_va_lgb['location_id'] = X_va_lgb['location_id'].astype('category')
    
    lgbm = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, num_leaves=24, min_child_samples=25, random_state=42, verbosity=-1)
    lgbm.fit(X_tr_lgb, y_tr)
    p_lgb = lgbm.predict(X_va_lgb)
    
    # 3. CatBoost
    cb = CatBoostRegressor(iterations=350, learning_rate=0.03, depth=5, cat_features=['location_id'], random_seed=42, verbose=0)
    cb.fit(X_tr, y_tr)
    p_cb = cb.predict(X_va)
    
    # Evaluate
    rmse_rf = np.sqrt(mean_squared_error(y_va, p_rf))
    rmse_lgb = np.sqrt(mean_squared_error(y_va, p_lgb))
    rmse_cb = np.sqrt(mean_squared_error(y_va, p_cb))
    
    # Blends
    blend_equal = (p_rf + p_lgb + p_cb) / 3.0
    rmse_blend = np.sqrt(mean_squared_error(y_va, blend_equal))
    
    print(f"=== {window_name} ===")
    print(f"RF: {rmse_rf:.4f} | LGBM: {rmse_lgb:.4f} | CatBoost: {rmse_cb:.4f} | 3-way Blend: {rmse_blend:.4f}")
    print(f"Actual mean: {y_va.mean():.2f} | RF mean: {p_rf.mean():.2f} | Blend mean: {blend_equal.mean():.2f}")
