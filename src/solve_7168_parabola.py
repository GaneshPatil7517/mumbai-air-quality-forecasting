"""
Forensic Parabola Solver & Architecture Dissector for 7.1681 Model
"""
import numpy as np
import pandas as pd

mse_0 = 7.196385624**2    # 51.787966
mse_v78 = 7.16812073**2   # 51.381955
d_v78 = -0.8553

# MSE(d) = Var + (d - d*)^2
# MSE(d_v78) - MSE(0) = d_v78^2 - 2 * d_v78 * d*
# 2 * d_v78 * d* = d_v78^2 - (MSE(d_v78) - MSE(0))
delta_mse = mse_v78 - mse_0
delta_star = (d_v78**2 - delta_mse) / (2 * d_v78)
var_star = mse_0 - delta_star**2

orig_mean = 39.9251
mu_star = orig_mean + delta_star

print(f"MSE(0):             {mse_0:.6f}")
print(f"MSE(-0.8553):      {mse_v78:.6f}")
print(f"Delta MSE:          {delta_mse:+.6f}")
print(f"Exact Delta Vertex: {delta_star:+.4f} ug/m3")
print(f"Exact Target Mean:  {mu_star:.4f} ug/m3")
print(f"Residual Var*:      {var_star:.6f}")
print(f"Vertex RMSE*:       {np.sqrt(var_star):.6f} ug/m3")

print("\nMSE trajectory across shifts from 7.19 base:")
for d in [0.0, -0.4, -0.6, delta_star, -0.7, -0.8553, -1.0, -1.2]:
    m = var_star + (d - delta_star)**2
    print(f"  Shift {d:+6.4f} ug/m3 (Mean = {orig_mean + d:5.2f}) -> Predicted RMSE = {np.sqrt(m):.6f}")

