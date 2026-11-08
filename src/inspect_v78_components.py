"""
Inspect components of the 7.1681 model predictions on Nov 3-4
"""
import pandas as pd
import numpy as np

sub = pd.read_csv('submissions/submission_v78_vertex_calibrated_champion.csv')
test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
test['dt_end'] = pd.to_datetime(test['target_end_ist'])

d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

nov34 = test[nov34_mask].copy()
nov34['pred'] = sub.loc[nov34_mask, 'Target'].values
nov34['hour'] = nov34['dt_start'].dt.hour
nov34['day'] = nov34['dt_start'].dt.day
nov34['date'] = nov34['dt_start'].dt.date

print("=" * 70)
print("DEEP ANALYSIS OF v78 PREDICTIONS (NOV 3-4, 192 ROWS):")
print("=" * 70)
print(f"Overall Mean: {nov34['pred'].mean():.4f}")
print(f"Overall Std:  {nov34['pred'].std():.4f}")
print(f"Min: {nov34['pred'].min():.2f}, Max: {nov34['pred'].max():.2f}")

d3_mean = nov34[nov34['date'] == d3]['pred'].mean()
d4_mean = nov34[nov34['date'] == d4]['pred'].mean()
print(f"\nNov 3 Mean: {d3_mean:.4f}")
print(f"Nov 4 Mean: {d4_mean:.4f}")
print(f"Slope (Nov 4 - Nov 3): {d4_mean - d3_mean:+.4f} ug/m3")

print("\nStation Breakdown:")
for loc, grp in nov34.groupby('location_id'):
    d3_st = grp[grp['date'] == d3]['pred'].mean()
    d4_st = grp[grp['date'] == d4]['pred'].mean()
    print(f"Station {loc:7d}:")
    print(f"  Overall Mean: {grp['pred'].mean():.2f}, Std: {grp['pred'].std():.2f}")
    print(f"  Nov 3 Mean:   {d3_st:.2f}, Nov 4 Mean: {d4_st:.2f} (diff = {d4_st - d3_st:+.2f})")
    print(f"  Range: [{grp['pred'].min():.2f}, {grp['pred'].max():.2f}]")
    print(f"  Hourly Std:   {grp.groupby('hour')['pred'].mean().std():.2f}")
print("=" * 70)

