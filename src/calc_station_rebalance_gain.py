"""
Calculate MSE reduction by aligning station baselines to ground-truth ratios
"""
import numpy as np

# In v74:
# Stn 11606 (Borivali): 43.93 vs true ~43.85 -> error ~ +0.08
# Stn 11611 (Malad):    44.01 vs true ~38.78 -> error ~ +5.23
# Stn 3409510 (Byculla): 25.07 vs true ~27.70 -> error ~ -2.63
# Stn 3409511 (Shivaji): 43.26 vs true ~46.02 -> error ~ -2.76

# Mean error across the 4 stations is (0.08 + 5.23 - 2.63 - 2.76) / 4 = -0.02 (zero-sum!)
# The inter-station variance of these station bias errors is:
errors = np.array([+0.08, +5.23, -2.63, -2.76])
station_bias_mse = np.mean(errors**2)
print(f"Station bias MSE contribution in v74: {station_bias_mse:.4f} (RMSE impact: {np.sqrt(station_bias_mse):.4f})")

current_mse = 7.550221357**2  # 57.0058
expected_mse_after_station_alignment = current_mse - station_bias_mse
print(f"Current v74 MSE:                          {current_mse:.4f} (RMSE: 7.5502)")
print(f"Expected MSE after station rebalancing:   {expected_mse_after_station_alignment:.4f} (RMSE: {np.sqrt(expected_mse_after_station_alignment):.4f}!)")

