import sys
from pathlib import Path
import numpy as np
import pandas as pd

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
preds = np.zeros(len(t))
for s, g in t.groupby("location_id"):
    x = hist[s]
    level = 0.5 * x.tail(168).mean() + 0.25 * x.tail(336).mean() + 0.25 * net7
    xs = x.tail(4320)
    sh = (xs.groupby((xs.index.hour - 1) % 24).mean() / xs.mean())
    hs = (g.ts.dt.hour - 1) % 24
    shape = 0.5 + 0.5 * hs.map(sh).fillna(1.0).values
    base = level * shape
    model = base * ramp(g.ts)
    series = obs[s].reindex(pd.date_range(obs.index.min(), CUT - pd.Timedelta(hours=1), freq="h"))
    interp = series.interpolate(limit=6, limit_area="inside")
    known = interp.reindex(g.ts).values
    out = np.where(np.isnan(known), model, known)
    preds[g.index] = out

t["Target"] = np.clip(preds, 0, None)
sub = t[["ID", "Target"]]
sample = pd.read_csv(D_SAMPLE)
assert (sample.ID.values == sub.ID.values).all() and np.isfinite(sub.Target).all()

out_path = 'submissions/submission_719_reproduced.csv'
sub.to_csv(out_path, index=False)
sub.to_csv('submission_719_reproduced.csv', index=False)

print("=" * 70)
print("REPRODUCED 7.1963 MODEL PREDICTION ANALYSIS")
print("=" * 70)
print(sub.Target.describe())

d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
t["dt_start"] = pd.to_datetime(t["target_start_ist"])
nov34_mask = t["dt_start"].dt.date.isin([d3, d4])
nov34 = t[nov34_mask]

print("\nACTIVE EVALUATION WINDOW (Nov 3-4, 192 rows):")
print(f"  Mean: {nov34['Target'].mean():.4f} ug/m3")
print(f"  Std:  {nov34['Target'].std():.4f} ug/m3")
print(f"  Min:  {nov34['Target'].min():.4f} ug/m3")
print(f"  Max:  {nov34['Target'].max():.4f} ug/m3")

d3_m = nov34[nov34['dt_start'].dt.date == d3]['Target'].mean()
d4_m = nov34[nov34['dt_start'].dt.date == d4]['Target'].mean()
print(f"  Nov 3 Mean: {d3_m:.4f} ug/m3")
print(f"  Nov 4 Mean: {d4_m:.4f} ug/m3")
print(f"  Daily Drop: {d3_m - d4_m:+.4f} ug/m3")

print("\nStation Breakdown on Nov 3-4:")
for loc, grp in nov34.groupby('location_id'):
    print(f"  Station {loc:7d}: Mean={grp['Target'].mean():.2f}, Std={grp['Target'].std():.2f}, Min={grp['Target'].min():.2f}, Max={grp['Target'].max():.2f}")
print("=" * 70)

