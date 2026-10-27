import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def as_ist(values):
    return pd.to_datetime(values, format='ISO8601').dt.tz_convert('Asia/Kolkata')

train = pd.read_csv('data/raw/Train.csv')
train['hour_end'] = as_ist(train['timestamp_ist'])
train['hour_start'] = train['hour_end'] - pd.Timedelta(hours=1)
train = train.sort_values(['location_id', 'hour_end']).reset_index(drop=True)

# Holdout: Oct 20 00:00 to Oct 22 00:00
val_end = train['hour_end'].max()
val_start = val_end - pd.Timedelta(days=2)

hist = train[train['hour_end'] <= val_start].copy()
valid = train[(train['hour_end'] > val_start) & (train['hour_end'] <= val_end)].copy()

# 1. Clean history for ML model
hist_clean = hist.loc[
    (hist['hour_end'] > val_start - pd.Timedelta(days=90)) &
    (hist['pm25'] <= 500) &
    (hist['flag_high_1000'] == False)
].copy()

def make_features_rf(df):
    time = df['hour_start']
    return pd.DataFrame({
        'location_id': df['location_id'].astype(str),
        'hour': time.dt.hour,
        'day_of_week': time.dt.dayofweek,
        'month': time.dt.month,
        'day_of_year': time.dt.dayofyear,
    }, index=df.index)

rf = Pipeline([
    ('prep', ColumnTransformer([('station', OneHotEncoder(handle_unknown='ignore'), ['location_id'])], remainder='passthrough')),
    ('reg', RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1))
])
rf.fit(make_features_rf(hist_clean), hist_clean['pm25'])
pred_rf = rf.predict(make_features_rf(valid))

# 2. Autoregressive Diurnal Decay Model
recent_14d = hist[hist['hour_end'] > val_start - pd.Timedelta(days=14)].copy()
recent_14d['hour'] = recent_14d['hour_start'].dt.hour
st_recent_mean = recent_14d.groupby('location_id')['pm25'].mean()
st_hour_mean = recent_14d.groupby(['location_id', 'hour'])['pm25'].mean()
st_diurnal_anomaly = (st_hour_mean - st_recent_mean).to_dict()

last_24h = hist[hist['hour_end'] > val_start - pd.Timedelta(hours=24)]
last_24h_mean = last_24h.groupby('location_id')['pm25'].mean().to_dict()
last_1h = hist.groupby('location_id').last()['pm25'].to_dict()

valid['hour'] = valid['hour_start'].dt.hour
valid['step'] = ((valid['hour_end'] - val_start) / pd.Timedelta(hours=1)).astype(int)

alpha = 0.92
w = alpha ** valid['step']
pred_diurnal = valid.apply(
    lambda r: last_24h_mean.get(r['location_id'], r['pm25']) + st_diurnal_anomaly.get((r['location_id'], r['hour']), 0),
    axis=1
)
pred_ar = w * valid['location_id'].map(last_1h) + (1 - w) * pred_diurnal

print(f"ML (RandomForest) RMSE: {np.sqrt(mean_squared_error(valid['pm25'], pred_rf)):.4f}")
print(f"AR Diurnal Decay (alpha=0.92) RMSE: {np.sqrt(mean_squared_error(valid['pm25'], pred_ar)):.4f}")

# Blend AR Diurnal + ML
for w_ar in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
    blend = w_ar * pred_ar + (1 - w_ar) * pred_rf
    print(f"Weight AR={w_ar:.2f}, RF={1-w_ar:.2f} -> RMSE: {np.sqrt(mean_squared_error(valid['pm25'], blend)):.4f}")
