"""
Inspect retrained ensemble calibrated to mu* = 39.0698 vs v74
"""
import pandas as pd
import numpy as np

test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])
nov34 = test[nov34_mask].copy()

retrained_raw = np.load('data/processed/retrained_nov34_preds.npy')
shift = 39.0698 - retrained_raw.mean()
nov34['retrained_cal'] = retrained_raw + shift

v74 = pd.read_csv('submission_v74_phase3_vertex_apex.csv')
nov34['v74'] = v74.loc[nov34_mask, 'Target'].values

print("=" * 70)
print("RETRAINED ENSEMBLE (CALIBRATED) vs v74 ON NOV 3-4:")
print("=" * 70)
print(f"Retrained: Mean={nov34['retrained_cal'].mean():.4f}, Std={nov34['retrained_cal'].std():.4f}")
print(f"v74:       Mean={nov34['v74'].mean():.4f}, Std={nov34['v74'].std():.4f}")

corr = np.corrcoef(nov34['retrained_cal'], nov34['v74'])[0, 1]
print(f"Correlation: {corr:.4f}")

print("\nStation Breakdown Comparison:")
for loc, grp in nov34.groupby('location_id'):
    print(f"Station {loc}:")
    print(f"  Retrained: Mean={grp['retrained_cal'].mean():.2f}, Std={grp['retrained_cal'].std():.2f}, Min={grp['retrained_cal'].min():.2f}, Max={grp['retrained_cal'].max():.2f}")
    print(f"  v74:       Mean={grp['v74'].mean():.2f}, Std={grp['v74'].std():.2f}, Min={grp['v74'].min():.2f}, Max={grp['v74'].max():.2f}")

diff = nov34['retrained_cal'] - nov34['v74']
print(f"\nRMS difference between Retrained and v74: {np.sqrt(np.mean(diff**2)):.4f}")

