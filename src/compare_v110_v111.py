import pandas as pd
import numpy as np

sub106 = pd.read_csv('submissions/submission_v106_exact_subspace_vertex_champion.csv')
sub110 = pd.read_csv('submissions/submission_v110_phase4_full_weather_ensemble.csv')
sub111 = pd.read_csv('submissions/submission_v111_full_diwali_physics_champion.csv')

test = pd.read_csv('zindi data/Test.csv')
test['ts'] = pd.to_datetime(test['target_end_ist'].str[:19])
mask_78 = (test['ts'] >= '2026-11-07') & (test['ts'] < '2026-11-09')

y106 = sub106.loc[mask_78, 'Target'].values
y110 = sub110.loc[mask_78, 'Target'].values
y111 = sub111.loc[mask_78, 'Target'].values

print(f"Correlation (y110, y111): {np.corrcoef(y110, y111)[0, 1]:.4f}")
print(f"Correlation (y106, y111): {np.corrcoef(y106, y111)[0, 1]:.4f}")
print(f"Correlation (y106, y110): {np.corrcoef(y106, y110)[0, 1]:.4f}")

print(f"\nMean y106: {y106.mean():.2f}, Std: {y106.std():.2f}")
print(f"Mean y110: {y110.mean():.2f}, Std: {y110.std():.2f}")
print(f"Mean y111: {y111.mean():.2f}, Std: {y111.std():.2f}")

# Look at combinations:
# In v111, nighttime peaks at 550, daytime is 50.
# In v110, daytime is 112, nighttime is 125.
# If we combine v110 (which captures elevated daytime festival stagnation)
# with the night spike of v111 (which captures the late firecrackers),
# what does the profile look like?
test_78 = test[mask_78].copy()
test_78['y106'] = y106
test_78['y110'] = y110
test_78['y111'] = y111
test_78['day'] = test_78['ts'].dt.day
test_78['hour'] = test_78['ts'].dt.hour
test_78['station'] = test_78['location_id'].astype(str)

# Hybrid: y_hybrid = y110 for daytime, and max(y110, y111 * scale) for nighttime!
for blend_w in [0.0, 0.3, 0.5, 0.7, 1.0]:
    y_blend = blend_w * y110 + (1 - blend_w) * y111
    print(f"Blend {blend_w*100:.0f}% v110 + {(1-blend_w)*100:.0f}% v111: Mean = {y_blend.mean():.2f}, Std = {y_blend.std():.2f}, Min = {y_blend.min():.2f}, Max = {y_blend.max():.2f}")
