import pandas as pd
import numpy as np

sub85 = pd.read_csv('submissions/submission_v85_exact_gradient_apex.csv')
sub86 = pd.read_csv('submissions/submission_v86_slope_neutralized_champion.csv')
sub87 = pd.read_csv('submissions/submission_v87_optimal_slope_champion.csv')

test = pd.read_csv('zindi data/Test.csv')
test['ts'] = pd.to_datetime(test['target_end_ist'].str[:19])
active = (test['ts'] >= '2026-11-03') & (test['ts'] < '2026-11-05')

y85 = sub85.loc[active, 'Target'].values
y86 = sub86.loc[active, 'Target'].values
y87 = sub87.loc[active, 'Target'].values

print("Diff (y86 - y85): min =", np.min(y86 - y85), "max =", np.max(y86 - y85))
print("Diff (y87 - y85): min =", np.min(y87 - y85), "max =", np.max(y87 - y85))

test_act = test[active].copy()
test_act['d86_85'] = y86 - y85
test_act['d87_85'] = y87 - y85
test_act['day'] = test_act['ts'].dt.day
print("\nDaily mean shifts (y86 - y85):")
print(test_act.groupby('day')['d86_85'].mean())
print("\nDaily mean shifts (y87 - y85):")
print(test_act.groupby('day')['d87_85'].mean())
