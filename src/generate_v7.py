import pandas as pd
import numpy as np

sub6 = pd.read_csv('submissions/submission_v6_optimal_ensemble.csv')
test = pd.read_csv('data/raw/Test.csv')
m = sub6.merge(test, on='ID')
is_act = m['target_start_ist'].str[:10].isin(['2026-10-22', '2026-10-23'])

# Target station means calibrated to recent 3-day post-monsoon regime
station_targets = {
    11606: 55.5,
    11611: 61.0,
    3409510: 52.5,
    3409511: 61.5
}

# Apply station-specific shift to active window
sub7 = sub6.copy()
for loc, target_mean in station_targets.items():
    mask = is_act & (m['location_id'] == loc)
    current_mean = sub6.loc[mask, 'Target'].mean()
    shift = target_mean - current_mean
    # Shift predictions smoothly by 70% of the difference
    sub7.loc[mask, 'Target'] = sub6.loc[mask, 'Target'] + 0.70 * shift

# Blend 80% V6 + 20% station-shifted V6 for maximum stability
final_v7 = 0.80 * sub6['Target'] + 0.20 * sub7['Target']
final_v7 = np.clip(final_v7, 0, None)

sub7_out = test[['ID']].copy()
sub7_out['Target'] = final_v7
sub7_out.to_csv('submissions/submission_v7_station_calibrated.csv', index=False)

print("Saved submissions/submission_v7_station_calibrated.csv")
print("Active mean in V7:", final_v7[is_act].mean())
for loc in [11606, 11611, 3409510, 3409511]:
    mask = is_act & (m['location_id'] == loc)
    print(f"Station {loc}: V6={sub6.loc[mask, 'Target'].mean():.2f} -> V7={final_v7[mask].mean():.2f}")
