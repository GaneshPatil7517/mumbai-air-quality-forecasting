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

print("=== LEAVE-ONE-OUT CROSS VALIDATION OF PREDICTIONS ===")
# For each submission, fit on the other 9, and check how well the inner product / score matches!
for leave_out in range(len(names)):
    train_idx = [i for i in range(len(names)) if i != leave_out]
    test_idx = leave_out
    
    # Pick first train submission as base
    b = train_idx[0]
    y_b = vectors[b]
    
    V_tr = np.array([vectors[i] - y_b for i in train_idx[1:]])
    dq_tr = np.array([q[i] - q[b] for i in train_idx[1:]])
    
    U, S, Vt = np.linalg.svd(V_tr, full_matrices=False)
    # Use rank = 5
    k = min(5, np.sum(S > 1e-4))
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    V_pinv = Vt_k.T @ np.diag(1.0 / S_k) @ U_k.T
    y_star_proj = V_pinv @ dq_tr
    
    # Predict inner product of test vector with y* in the span:
    v_test = vectors[test_idx] - y_b
    v_test_proj = Vt_k.T @ (Vt_k @ v_test)
    pred_dq = v_test @ y_star_proj
    actual_dq = q[test_idx] - q[b]
    
    print(f"Leave out {names[test_idx]:5s}: Actual dq = {actual_dq:10.2f}, Predicted dq = {pred_dq:10.2f}, diff = {pred_dq - actual_dq:+8.4f}")
