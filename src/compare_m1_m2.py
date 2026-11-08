"""
Analyze Model 1 (7.1963) vs Model 2 (LightGBM multi-cutoff)
"""
import pandas as pd
import numpy as np

sub1 = pd.read_csv('submission_719_reproduced.csv')
sub2 = pd.read_csv('submissions/submission_lgb_solution2.csv')

test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])

m1 = sub1.loc[nov34_mask, 'Target'].values
m2 = sub2.loc[nov34_mask, 'Target'].values

print("=" * 70)
print("MODEL 1 (7.1963) vs MODEL 2 (LightGBM Solution) on Nov 3-4:")
print("=" * 70)
print(f"Model 1: Mean = {m1.mean():.4f}, Std = {m1.std():.4f}")
print(f"Model 2: Mean = {m2.mean():.4f}, Std = {m2.std():.4f}")

corr = np.corrcoef(m1, m2)[0, 1]
print(f"Correlation: {corr:.4f}")

# Calibrate both to exact vertex mu* = 39.0698
target_mu_star = 39.0698
m1_cal = m1 + (target_mu_star - m1.mean())
m2_cal = m2 + (target_mu_star - m2.mean())

for w1 in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    blend = w1 * m1_cal + (1 - w1) * m2_cal
    print(f"Weight M1={w1:.2f}, M2={1-w1:.2f} -> Mean = {blend.mean():.4f}, Std = {blend.std():.4f}")

