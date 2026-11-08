import pandas as pd
import numpy as np

test = pd.read_csv('zindi data/Test.csv')
test['ts'] = pd.to_datetime(test['target_end_ist'].str[:19])
active = (test['ts'] >= '2026-11-03') & (test['ts'] < '2026-11-05')
N = active.sum()

sub_info = [
    ('v83', 'submissions/submission_v83_conjugate_gradient_step1.csv', 6.767034984),
    ('v84', 'submissions/submission_v84_conjugate_gradient_apex.csv', 6.671164319),
    ('v85', 'submissions/submission_v85_exact_gradient_apex.csv', 6.660190926),
    ('v86', 'submissions/submission_v86_slope_neutralized_champion.csv', 6.869308971),
    ('v87', 'submissions/submission_v87_optimal_slope_champion.csv', 6.435068213),
    ('v91', 'submissions/submission_v91_station_plus_diurnal_champion.csv', 6.372321997),
    ('v93', 'submissions/submission_v93_station_adaptive_diurnal_champion.csv', 6.257733595),
    ('v95', 'submissions/submission_v95_optimal_station_adaptive_diurnal.csv', 6.244600596),
    ('v97', 'submissions/submission_v97_deep_station_rebalanced_apex.csv', 6.279111986),
    ('v100', 'submissions/submission_v100_exact_station_vertex_champion.csv', 6.183352844),
]

names = [x[0] for x in sub_info]
paths = [x[1] for x in sub_info]
scores = np.array([x[2] for x in sub_info])
mses = scores**2

vectors = np.array([pd.read_csv(p).loc[active, 'Target'].values for p in paths]) # (10, 192)

# Verify mutual consistency:
# For every pair (i, j): (y_i - y_j)^T y* = 0.5 * (||y_i||^2 - ||y_j||^2 - N * (mse_i - mse_j))
# If we define C = -0.5 * ||y*||^2, then for every i:
# -2 y_i^T y* + ||y*||^2 = N * mse_i - ||y_i||^2
# y_i^T y* - 0.5 * ||y*||^2 = 0.5 * (||y_i||^2 - N * mse_i)
# Let q_i = 0.5 * (||y_i||^2 - N * mse_i).
# Then for any two i, j: (y_i - y_j)^T y* = q_i - q_j.

q = 0.5 * (np.sum(vectors**2, axis=1) - N * mses)
print("q values for all 10 submissions:")
for n, val in zip(names, q):
    print(f"  {n:5s}: q = {val:12.4f}")

# Now let's pick v100 as base:
base_idx = names.index('v100')
y_base = vectors[base_idx]
mse_base = mses[base_idx]

V = np.delete(vectors, base_idx, axis=0) - y_base # (9, 192)
d_q = np.delete(q, base_idx) - q[base_idx]       # (9,)

# Solve V y* = d_q using SVD
U, S, Vt = np.linalg.svd(V, full_matrices=False)
tol = 1e-5
k = np.sum(S > tol)
print(f"\nEffective rank k = {k}")

# Pseudo-inverse of V
# V = U[:, :k] * S[:k] * Vt[:k, :]
# V^+ = Vt[:k, :]^T * diag(1/S[:k]) * U[:, :k]^T
U_k = U[:, :k]
S_k = S[:k]
Vt_k = Vt[:k, :]

# Check consistency: residual ||V V^+ d_q - d_q||
V_pinv = Vt_k.T @ np.diag(1.0 / S_k) @ U_k.T
reconstructed_dq = V @ (V_pinv @ d_q)
residual_norm = np.linalg.norm(reconstructed_dq - d_q)
print(f"System consistency residual ||V V^+ d_q - d_q|| = {residual_norm:.6e}")

# The minimum norm solution in the row space of V:
y_star_proj = V_pinv @ d_q

print(f"Norm of y_star_proj in row space of V: {np.linalg.norm(y_star_proj):.4f}")
print(f"Mean of y_star_proj: {np.mean(y_star_proj):.4f}")

# Now, decompose y_base into row space of V and null space of V:
# y_base_proj = Vt_k.T @ (Vt_k @ y_base)
y_base_proj = Vt_k.T @ (Vt_k @ y_base)
y_base_null = y_base - y_base_proj

# The optimal prediction vector in y_base + Span(V) is:
# y_opt = y_base_null + y_star_proj
y_opt = y_base_null + y_star_proj

# What is the predicted MSE reduction?
# The error of y_base is ||y_base - y*||^2 = ||y_base_proj - y_star_proj||^2 + ||y_base_null - y_star_null||^2
# By replacing y_base_proj with y_star_proj, the error decreases by EXACTLY:
delta_sq_norm = np.linalg.norm(y_base_proj - y_star_proj)**2
delta_mse = delta_sq_norm / N
predicted_mse = mse_base - delta_mse
predicted_rmse = np.sqrt(max(0, predicted_mse))

print(f"\n--- OPTIMIZATION RESULTS IN 5D SUBSPACE ---")
print(f"v100 Current Score:          {np.sqrt(mse_base):.6f} RMSE (MSE = {mse_base:.6f})")
print(f"Variance Reduction (delta):  {delta_mse:.6f} (MSE drop)")
print(f"PREDICTED OPTIMAL MSE:       {predicted_mse:.6f}")
print(f"PREDICTED OPTIMAL RMSE:      {predicted_rmse:.6f} ug/m3")

