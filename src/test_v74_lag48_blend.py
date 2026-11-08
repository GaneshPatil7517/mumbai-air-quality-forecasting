"""
Test blending v74 (smooth vertex model) with Lag-48 (empirical micro-dynamics)
"""
import pandas as pd
import numpy as np

test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

v74 = pd.read_csv('submission_v74_phase3_vertex_apex.csv')
v74_act = v74.loc[nov34_mask, 'Target'].values

# Load lag48
from test_lag48_persistence import merged
lag48_act = merged['pred_calibrated'].values

print("=" * 70)
print("v74 vs Lag-48 Analysis on Nov 3-4:")
print("=" * 70)
print(f"v74:    mean={v74_act.mean():.4f}, std={v74_act.std():.4f}")
print(f"Lag-48: mean={lag48_act.mean():.4f}, std={lag48_act.std():.4f}")
corr = np.corrcoef(v74_act, lag48_act)[0, 1]
print(f"Correlation between v74 and Lag-48: {corr:.4f}")

# Test blend weights
for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    blend = w * v74_act + (1 - w) * lag48_act
    print(f"Weight v74={w:.2f}, Lag-48={1-w:.2f} -> Blend Mean={blend.mean():.4f}, Std={blend.std():.4f}")
print("=" * 70)

