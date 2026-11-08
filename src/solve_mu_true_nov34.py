"""
Solve for true ground truth mean on Nov 3-4 from v42 vs v72 MSE
"""
import numpy as np

# MSE = Var + (mean - mu_true)^2
# MSE2 - MSE1 = (mean2 - mu_true)^2 - (mean1 - mu_true)^2
#             = mean2^2 - 2*mean2*mu_true - (mean1^2 - 2*mean1*mu_true)
#             = mean2^2 - mean1^2 - 2 * mu_true * (mean2 - mean1)
# Therefore:
# 2 * mu_true * (mean2 - mean1) = mean2^2 - mean1^2 - (MSE2 - MSE1)
# mu_true = [mean2^2 - mean1^2 - (MSE2 - MSE1)] / [2 * (mean2 - mean1)]

mse1 = 8.879814125**2
mse2 = 10.30532911**2

m1 = 34.3959
m2 = 32.8589

dm = m2 - m1
dmse = mse2 - mse1

mu_true_approx = (m2**2 - m1**2 - dmse) / (2 * dm)
print(f"MSE1 (v42/v71): {mse1:.4f}")
print(f"MSE2 (v72):     {mse2:.4f}")
print(f"Delta MSE:      {dmse:+.4f}")
print(f"Delta Mean:     {dm:+.4f}")
print(f"Estimated True Ground Truth Mean on Nov 3-4: {mu_true_approx:.2f} ug/m3!")

