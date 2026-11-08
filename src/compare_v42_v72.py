import pandas as pd
import numpy as np

v42 = pd.read_csv('submissions/submission_v42_apex_pinnacle_champion.csv')
v72 = pd.read_csv('submission_v72_phase3_champion.csv')
test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])

d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

v42_w = v42.loc[nov34_mask, 'Target']
v72_w = v72.loc[nov34_mask, 'Target']

print("=" * 70)
print("v42 (scored 8.8798) vs v72 (scored 10.3053) ON NOV 3-4")
print("=" * 70)
print(f"v42: Mean={v42_w.mean():.4f}, Std={v42_w.std():.4f}, Min={v42_w.min():.4f}, Max={v42_w.max():.4f}")
print(f"v72: Mean={v72_w.mean():.4f}, Std={v72_w.std():.4f}, Min={v72_w.min():.4f}, Max={v72_w.max():.4f}")

test_w = test[nov34_mask].copy()
test_w['v42'] = v42_w.values
test_w['v72'] = v72_w.values

print("\nStation breakdown:")
for loc, grp in test_w.groupby('location_id'):
    name = grp['station_name'].iloc[0] if 'station_name' in grp.columns else str(loc)
    print(f"Station {loc} ({name:25s}):")
    print(f"  v42: Mean={grp['v42'].mean():.2f}, Std={grp['v42'].std():.2f}")
    print(f"  v72: Mean={grp['v72'].mean():.2f}, Std={grp['v72'].std():.2f}")

diff = v72_w - v42_w
print(f"\nDifference (v72 - v42): Mean={diff.mean():.2f}, Std={diff.std():.2f}")
print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.4f}")
print("=" * 70)

