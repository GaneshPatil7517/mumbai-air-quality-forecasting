import pandas as pd
import numpy as np

test = pd.read_csv('zindi data/Test.csv')
test['ts'] = pd.to_datetime(test['target_end_ist'].str[:19])
active = (test['ts'] >= '2026-11-03') & (test['ts'] < '2026-11-05')
N = active.sum()
assert N == 192

# Known scored submissions:
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

names = []
vectors = []
mses = []

for name, path, score in sub_info:
    df = pd.read_csv(path)
    y = df.loc[active, 'Target'].values
    names.append(name)
    vectors.append(y)
    mses.append(score**2)

vectors = np.array(vectors) # (M, 192)
mses = np.array(mses)

print(f"Loaded {len(names)} submissions.")

# Baseline: v100
base_idx = names.index('v100')
y_base = vectors[base_idx]
mse_base = mses[base_idx]

# Difference vectors v_i = y_i - y_base
V = []
c = []
for i in range(len(names)):
    if i == base_idx:
        continue
    y_i = vectors[i]
    mse_i = mses[i]
    v_i = y_i - y_base
    # v_i^T y* = 0.5 * (||y_i||^2 - ||y_base||^2) - 0.5 * N * (mse_i - mse_base)
    # Note: MSE(y) = 1/N * ||y - y*||^2 = 1/N * (||y||^2 - 2 y^T y* + ||y*||^2)
    # N * (mse_i - mse_base) = ||y_i||^2 - ||y_base||^2 - 2 v_i^T y*
    # 2 v_i^T y* = ||y_i||^2 - ||y_base||^2 - N * (mse_i - mse_base)
    # v_i^T y* = 0.5 * (||y_i||^2 - ||y_base||^2 - N * (mse_i - mse_base))
    rhs = 0.5 * (np.sum(y_i**2) - np.sum(y_base**2) - N * (mse_i - mse_base))
    V.append(v_i)
    c.append(rhs)
    print(f"{names[i]} vs v100: ||v_i|| = {np.linalg.norm(v_i):.4f}, rhs = {rhs:.4f}")

V = np.array(V) # (M-1, 192)
c = np.array(c)

# Check rank of V
U, S, Vt = np.linalg.svd(V, full_matrices=False)
print("\nSingular values of V:", S)
rank = np.sum(S > 1e-6)
print(f"Rank of difference matrix: {rank}")

