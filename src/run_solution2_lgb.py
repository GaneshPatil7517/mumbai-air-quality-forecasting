"""
Run solution.py predict() pipeline and inspect predictions
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

D_TRAIN = 'zindi data/Phase 3 Train.csv'
D_TEST = 'data/raw/Test.csv'
D_SAMPLE = 'data/raw/SampleSubmission.csv'
CAP = 300.0
HMAX = 19

def load():
    t = pd.read_csv(D_TRAIN)
    t["ts"] = pd.to_datetime(t.timestamp_ist.str[:19])
    t["v"] = t.pm25.clip(upper=CAP)
    p = t.pivot_table(index="ts", columns="location_id", values="v", aggfunc="mean")
    full = pd.date_range(p.index.min().floor("D"), "2026-11-03 00:00", freq="h")
    return p.reindex(full)

def build(p, cutoffs, hmax=HMAX, with_target=True):
    cols = list(p.columns)
    rows = []
    for c in cutoffs:
        hist = p.loc[:c]
        f = {}
        for s in cols:
            x = hist[s]
            d = {
                "m1": x.tail(24).mean(), "m3": x.tail(72).mean(), "m7": x.tail(168).mean(),
                "m14": x.tail(336).mean(), "m30": x.tail(720).mean(), "m90": x.tail(2160).mean(),
                "med7": x.tail(168).median(), "std7": x.tail(168).std(),
            }
            prof = x.tail(336).groupby(x.tail(336).index.hour).mean()
            d["prof"] = prof
            f[s] = d
        net7 = np.nanmean([f[s]["m7"] for s in cols])
        net3 = np.nanmean([f[s]["m3"] for s in cols])
        net1 = np.nanmean([f[s]["m1"] for s in cols])
        for s in cols:
            d = f[s]
            idx = pd.date_range(c + pd.Timedelta(hours=1), periods=24 * hmax, freq="h")
            df = pd.DataFrame({
                "cutoff": c, "station": s, "ts": idx,
                "hour": (idx.hour - 1) % 24,
                "dow": idx.dayofweek,
                "doy": idx.dayofyear,
                "hz": ((idx - c) / pd.Timedelta(hours=1)).astype(int),
            })
            df["hd"] = (df.hz - 1) // 24 + 1
            for k in ["m1", "m3", "m7", "m14", "m30", "m90", "med7", "std7"]:
                df[k] = d[k]
            df["prof_h"] = df.hour.map(lambda h: d["prof"].get((h + 1) % 24, np.nan))
            df["net1"], df["net3"], df["net7"] = net1, net3, net7
            if with_target:
                df["y"] = p[s].reindex(idx).values
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["sid"] = out.station.astype("category").cat.codes
    return out

FEATS = ["sid", "hour", "dow", "hz", "m1", "m3", "m7", "m14", "m30", "m90", "med7", "std7",
         "prof_h", "net1", "net3", "net7"]

print("Loading data and building training set...")
p = load()
last = pd.Timestamp("2026-11-03 00:00")
# Sample cutoffs every 3 days to speed up training
cutoffs = pd.date_range("2025-05-01", "2026-10-31", freq="2D")
data = build(p, cutoffs)
data = data[data.y.notna() & data.m30.notna()]
print(f"Training samples: {len(data):,}")

params = dict(num_leaves=15, learning_rate=0.03, n_estimators=300, min_child_samples=500,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1)

fut = build(p, [last], with_target=False)
pr = np.mean([lgb.LGBMRegressor(**params, random_state=s).fit(data[FEATS], data.y).predict(fut[FEATS])
              for s in range(5)], axis=0)
fut["pred"] = pr

t = pd.read_csv(D_TEST)
t["ts"] = pd.to_datetime(t.target_end_ist.str[:19])
t["station"] = t.location_id
fut["station"] = fut.station.astype(int)
m = t.merge(fut[["station", "ts", "pred"]], on=["station", "ts"], how="left")

obs = p.stack().rename("obs").reset_index()
obs.columns = ["ts", "station", "obs"]
m = m.merge(obs, on=["ts", "station"], how="left")

# Output prediction: if obs notna use obs, else pred
m["Target"] = np.where(m["obs"].notna(), m["obs"], m["pred"])

sub = m[["ID", "Target"]]
sub.to_csv("submissions/submission_lgb_solution2.csv", index=False)

print("\nSOLUTION 2 LIGHTGBM MODEL ON NOV 3-4 (192 rows):")
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
m["dt_start"] = pd.to_datetime(m["target_start_ist"])
nov34 = m[m["dt_start"].dt.date.isin([d3, d4])]
print(f"  Mean: {nov34['Target'].mean():.4f}")
print(f"  Std:  {nov34['Target'].std():.4f}")
print(f"  Min:  {nov34['Target'].min():.4f}")
print(f"  Max:  {nov34['Target'].max():.4f}")

for loc, grp in nov34.groupby('location_id'):
    print(f"  Station {loc:7d}: Mean={grp['Target'].mean():.2f}, Std={grp['Target'].std():.2f}")

