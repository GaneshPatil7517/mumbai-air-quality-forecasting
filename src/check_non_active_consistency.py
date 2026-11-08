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

# Let's check non-active rows: are all submissions 100% identical on non-active rows?
sample = pd.read_csv('zindi data/SampleSubmission.csv')
for name, path, score in sub_info:
    df = pd.read_csv(path)
    diff_non_active = np.max(np.abs(df.loc[~active, 'Target'].values - sample.loc[~active, 'Target'].values))
    print(f"{name}: diff on non-active rows vs sample = {diff_non_active:.6f}")

