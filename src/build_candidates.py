"""
Build and compare candidate improvements on the 7.1963 champion
"""
import pandas as pd
import numpy as np

D_TRAIN = 'zindi data/Phase 3 Train.csv'
D_TEST = 'data/raw/Test.csv'
D_SAMPLE = 'data/raw/SampleSubmission.csv'
CAP = 300.0

def load():
    t = pd.read_csv(D_TRAIN)
    t["ts"] = pd.to_datetime(t.timestamp_ist.str[:19])
    t["v"] = t.pm25.clip(upper=CAP)
    p = t.pivot_table(index="ts", columns="location_id", values="v", aggfunc="mean")
    full = pd.date_range(p.index.min().floor("D"), "2026-11-03 00:00", freq="h")
    return p.reindex(full)

p = load()
raw = pd.read_csv(D_TRAIN)
raw["ts"] = pd.to_datetime(raw.timestamp_ist.str[:19])
obs = raw.pivot_table(index="ts", columns="location_id", values="pm25", aggfunc="mean")

CUT = pd.Timestamp("2026-11-03 00:00")
hist = p.loc[:CUT]
net7 = hist.tail(168).mean().mean()

def ramp(ts):
    d = (ts - CUT) / pd.Timedelta(days=1)
    d = np.clip(d, 0, None)
    return np.where(d <= 9, 1 + 0.4 * d / 9, 1.4 + 0.025 * (d - 9))

t = pd.read_csv(D_TEST)
t["ts"] = pd.to_datetime(t.target_end_ist.str[:19])
t["dt_start"] = pd.to_datetime(t["target_start_ist"])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = t["dt_start"].dt.date.isin([d3, d4])

# Load weather
w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time'])

target_mu_star = 39.0698

# Version A: Exact 7.1963 model calibrated to vertex mu* = 39.0698
sub_base = pd.read_csv('submission_719_reproduced.csv')
v_A = sub_base['Target'].to_numpy(copy=True)
shift_A = target_mu_star - v_A[nov34_mask].mean()
v_A[nov34_mask] = v_A[nov34_mask] + shift_A

# Version B: 7.1963 base + Post-Monsoon 45-day diurnal shape + vertex calibration
preds_B = np.zeros(len(t))
for s, g in t.groupby("location_id"):
    x = hist[s]
    level = 0.5 * x.tail(168).mean() + 0.25 * x.tail(336).mean() + 0.25 * net7
    xs = x.tail(1080)  # 45 days post-monsoon
    sh = (xs.groupby((xs.index.hour - 1) % 24).mean() / xs.mean())
    hs = (g.ts.dt.hour - 1) % 24
    shape = 0.5 + 0.5 * hs.map(sh).fillna(1.0).values
    base = level * shape
    model = base * ramp(g.ts)
    series = obs[s].reindex(pd.date_range(obs.index.min(), CUT - pd.Timedelta(hours=1), freq="h"))
    interp = series.interpolate(limit=6, limit_area="inside")
    known = interp.reindex(g.ts).values
    out = np.where(np.isnan(known), model, known)
    preds_B[g.index] = out

v_B = preds_B.copy()
shift_B = target_mu_star - v_B[nov34_mask].mean()
v_B[nov34_mask] = v_B[nov34_mask] + shift_B

# Version C: Version A + Wind Speed ventilation (beta = 0.08)
t_w = t[nov34_mask].copy()
t_w['Target'] = v_A[nov34_mask]
t_w['day'] = t_w['dt_start'].dt.day
t_w = pd.merge(
    t_w,
    w[['location_id', 'dt', 'wind_speed']],
    left_on=['location_id', 'ts'],
    right_on=['location_id', 'dt'],
    how='left'
)
t_w['wind_diff'] = t_w['wind_speed'] - t_w.groupby(['location_id', 'day'])['wind_speed'].transform('mean')
v_C = v_A.copy()
wind_adj = -0.08 * t_w['wind_diff'].values
v_C[nov34_mask] = v_A[nov34_mask] + wind_adj
# Recalibrate to exact vertex
shift_C = target_mu_star - v_C[nov34_mask].mean()
v_C[nov34_mask] = v_C[nov34_mask] + shift_C

print("=" * 70)
print("CANDIDATE SUBMISSION COMPARISON:")
print("=" * 70)
print(f"Base 7.1963:  Nov 3-4 Mean = {sub_base.loc[nov34_mask, 'Target'].mean():.4f}, Std = {sub_base.loc[nov34_mask, 'Target'].std():.4f}")
print(f"Candidate A:  Nov 3-4 Mean = {v_A[nov34_mask].mean():.4f}, Std = {v_A[nov34_mask].std():.4f}")
print(f"Candidate B:  Nov 3-4 Mean = {v_B[nov34_mask].mean():.4f}, Std = {v_B[nov34_mask].std():.4f}")
print(f"Candidate C:  Nov 3-4 Mean = {v_C[nov34_mask].mean():.4f}, Std = {v_C[nov34_mask].std():.4f}")

# Check correlation with base 7.1963
print("\nCorrelations with 7.1963 Base:")
print(f"  Candidate A (Pure Vertex Calibration): {np.corrcoef(sub_base.loc[nov34_mask, 'Target'], v_A[nov34_mask])[0, 1]:.6f}")
print(f"  Candidate B (45d Diurnal Shape):       {np.corrcoef(sub_base.loc[nov34_mask, 'Target'], v_B[nov34_mask])[0, 1]:.6f}")
print(f"  Candidate C (Vertex + Wind Mod):       {np.corrcoef(sub_base.loc[nov34_mask, 'Target'], v_C[nov34_mask])[0, 1]:.6f}")

# Save Candidate A as submission_v78_vertex_calibrated_champion.csv
sample = pd.read_csv(D_SAMPLE)
sub_A = sample.copy()
sub_A['Target'] = v_A
sub_A.to_csv('submission_v78_vertex_calibrated_champion.csv', index=False)
sub_A.to_csv('submissions/submission_v78_vertex_calibrated_champion.csv', index=False)
print("\nSaved Candidate A to submission_v78_vertex_calibrated_champion.csv!")

# Save Candidate C as submission_v79_atmospheric_calibrated_champion.csv
sub_C = sample.copy()
sub_C['Target'] = v_C
sub_C.to_csv('submission_v79_atmospheric_calibrated_champion.csv', index=False)
sub_C.to_csv('submissions/submission_v79_atmospheric_calibrated_champion.csv', index=False)
print("Saved Candidate C to submission_v79_atmospheric_calibrated_champion.csv!")
