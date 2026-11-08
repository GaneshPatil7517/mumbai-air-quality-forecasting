"""
Forensic Analysis: Direction v75 - v74
MSE(alpha) = MSE(v74 + alpha * (v75 - v74))
We have:
alpha = 0.0 -> MSE = 7.550221357^2 = 57.0058
alpha = 1.0 -> MSE = 7.736204251^2 = 59.8489
"""
import numpy as np

mse_0 = 7.550221357**2
mse_1 = 7.736204251**2

dmse = mse_1 - mse_0
print(f"MSE at alpha=0.0 (v74): {mse_0:.4f}")
print(f"MSE at alpha=1.0 (v75): {mse_1:.4f}")
print(f"Delta MSE:              {dmse:+.4f}")

# MSE(alpha) = c + b*alpha + a*alpha^2
# If a > 0 (convex parabola):
# b + a = dmse = +2.8431
# If the optimal alpha* is negative (i.e. moving in the OPPOSITE direction of v75),
# or if alpha* is slightly positive?
# Let's inspect the covariance of (v75 - v74)
import pandas as pd
v74 = pd.read_csv('submission_v74_phase3_vertex_apex.csv')
v75 = pd.read_csv('submission_v75_phase3_station_rebalanced_apex.csv')
test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

diff = v75.loc[nov34_mask, 'Target'].values - v74.loc[nov34_mask, 'Target'].values
var_diff = np.var(diff)
print(f"Variance of direction (v75 - v74): {var_diff:.4f}")
# In MSE(alpha) = MSE(0) - 2*alpha*Cov(err0, diff) + alpha^2 * Var(diff)
# Here a = Var(diff) = {var_diff:.4f}
a = var_diff
b = dmse - a
alpha_star = -b / (2 * a)
min_mse = mse_0 - b**2 / (4 * a)

print(f"Quadratic coefficient a = Var(diff): {a:.4f}")
print(f"Linear coefficient b:                 {b:+.4f}")
print(f"Optimal step alpha*:                 {alpha_star:+.4f}")
print(f"Minimum predicted MSE at alpha*:     {min_mse:.4f}")
print(f"Minimum predicted RMSE at alpha*:    {np.sqrt(max(0, min_mse)):.4f} ug/m3!")

for al in [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]:
    pred_m = mse_0 + b*al + a*(al**2)
    print(f"  alpha = {al:+5.2f} -> Predicted RMSE = {np.sqrt(pred_m):.4f}")

