"""
Analyze the variance of v72 shape vs v42 shape centered at mu* = 39.0698
"""
import pandas as pd
import numpy as np

test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

v72 = pd.read_csv('submission_v72_phase3_champion.csv')
v74 = pd.read_csv('submission_v74_phase3_vertex_apex.csv')

v72_active = v72.loc[nov34_mask, 'Target'].values
v74_active = v74.loc[nov34_mask, 'Target'].values

print(f"v72 active mean: {v72_active.mean():.4f}, std: {v72_active.std():.4f}")
print(f"v74 active mean: {v74_active.mean():.4f}, std: {v74_active.std():.4f}")

# What if we center v72 to mu* = 39.0698?
v72_centered = v72_active - v72_active.mean() + 39.0698
print(f"v72 centered mean: {v72_centered.mean():.4f}, std: {v72_centered.std():.4f}")

# In v72, score was 10.3053 at mean 32.8589
mse_v72 = 10.30532911**2
bias_v72 = (32.8589 - 39.0698)**2
var_v72 = mse_v72 - bias_v72
print(f"v72 total MSE:       {mse_v72:.4f}")
print(f"v72 squared bias:    {bias_v72:.4f} (due to mean being 32.86 instead of 39.07)")
print(f"v72 residual Var:    {var_v72:.4f}")
print(f"Predicted RMSE of v72 centered at mu*=39.07: {np.sqrt(max(0, var_v72)):.4f} ug/m3!")

