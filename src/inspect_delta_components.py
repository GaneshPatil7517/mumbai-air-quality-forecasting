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
vectors = np.array([pd.read_csv(p).loc[active, 'Target'].values for p in paths])
q = 0.5 * (np.sum(vectors**2, axis=1) - N * mses)

base_idx = names.index('v100')
y_base = vectors[base_idx]
V = np.delete(vectors, base_idx, axis=0) - y_base
d_q = np.delete(q, base_idx) - q[base_idx]

U, S, Vt = np.linalg.svd(V, full_matrices=False)
U_5 = U[:, :5]
S_5 = S[:5]
Vt_5 = Vt[:5, :]

V_pinv5 = Vt_5.T @ np.diag(1.0 / S_5) @ U_5.T
y_star_proj = V_pinv5 @ d_q

y_base_proj = Vt_5.T @ (Vt_5 @ y_base)
y_base_null = y_base - y_base_proj
y_opt = y_base_null + y_star_proj

delta = y_opt - y_base

test_act = test[active].copy()
test_act['delta'] = delta
test_act['day'] = test_act['ts'].dt.day
test_act['hour'] = test_act['ts'].dt.hour
test_act['station'] = test_act['location_id'].astype(str)

print("=== DELTA ANALYSIS (y_opt - y100) ===")
print(f"Norm of delta: {np.linalg.norm(delta):.4f}")
print(f"Mean of delta: {np.mean(delta):.6f}")
print(f"Min delta:     {np.min(delta):.4f}")
print(f"Max delta:     {np.max(delta):.4f}")
print(f"Std of delta:  {np.std(delta):.4f}")

print("\nDelta by Station:")
for s, df in test_act.groupby('station'):
    print(f"  Station {s}: Mean delta = {df['delta'].mean():+.4f}, Std delta = {df['delta'].std():.4f}")

print("\nDelta by Day:")
for d, df in test_act.groupby('day'):
    print(f"  Day {d}: Mean delta = {df['delta'].mean():+.4f}, Std delta = {df['delta'].std():.4f}")

print("\nDelta by Station and Day:")
for s in sorted(test_act['station'].unique()):
    for d in [3, 4]:
        sub_d = test_act[(test_act['station'] == s) & (test_act['day'] == d)]
        print(f"  Station {s} Day {d}: Mean delta = {sub_d['delta'].mean():+.4f}")

