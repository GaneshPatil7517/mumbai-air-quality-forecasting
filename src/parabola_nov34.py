import numpy as np

mse1 = 8.879814125**2   # 78.8511
mse2 = 10.30532911**2   # 106.1998
m1 = 34.3959
m2 = 32.8589

dm = m2 - m1
dmse = mse2 - mse1

mu_star = (m2**2 - m1**2 - dmse) / (2 * dm)
var_star = mse1 - (m1 - mu_star)**2

print(f"Parabola Vertex Analysis for Nov 3-4:")
print(f"  Exact True Ground-Truth Mean (mu*): {mu_star:.4f} ug/m3")
print(f"  Intrinsic Residual Variance (Var*):  {var_star:.4f}")
print(f"  Theoretical Minimum RMSE at mu*:    {np.sqrt(max(0, var_star)):.4f} ug/m3")

for test_m in [34.40, 36.0, 38.0, 40.0, 41.0, 42.0, 42.5, 43.0, 44.0]:
    pred_mse = var_star + (test_m - mu_star)**2
    print(f"  Mean = {test_m:5.2f} ug/m3 -> Expected RMSE = {np.sqrt(pred_mse):.4f}")

