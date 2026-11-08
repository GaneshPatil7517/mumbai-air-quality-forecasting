import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

D = r"zindi data"
CAP = 300.0
HMAX = 19  # forecast horizon in days (cutoff end of day -> Nov 20)


def load():
    t = pd.read_csv(f"{D}\\Phase 3 Train.csv")
    t["ts"] = pd.to_datetime(t.timestamp_ist.str[:19])
    t["v"] = t.pm25.clip(upper=CAP)
    p = t.pivot_table(index="ts", columns="location_id", values="v", aggfunc="mean")
    full = pd.date_range(p.index.min().floor("D"), "2026-11-03 00:00", freq="h")
    return p.reindex(full)  # index = hour-end timestamps


def build(p, cutoffs, hmax=HMAX, with_target=True):
    """One row per (cutoff, station, target hour). Features use data <= cutoff only."""
    cols = list(p.columns)
    daily = p.resample("D").mean()
    rows = []
    allmean = p.mean(axis=1)
    for c in cutoffs:  # c = midnight starting the day after the last known day
        hist = p.loc[:c]  # hour-end stamps <= c
        h24 = hist.tail(24)
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
            n = len(idx)
            df = pd.DataFrame({
                "cutoff": c, "station": s, "ts": idx,
                "hour": (idx.hour - 1) % 24,  # hour-end -> hour start
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


def rmse(a, b):
    m = ~np.isnan(a)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def backtest():
    p = load()
    cutoffs = pd.date_range("2025-03-25", "2026-10-31", freq="D")
    data = build(p, cutoffs)
    data = data[data.y.notna() & data.m30.notna()].copy()
    data["end"] = data.ts
    print(data.shape)
    folds = [("2026-03-01", "2026-04-15"), ("2026-04-15", "2026-06-15"), ("2026-06-15", "2026-08-15"),
             ("2026-08-15", "2026-09-30"), ("2026-09-30", "2026-11-03")]
    res = []
    for a, b in folds:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        te = data[(data.cutoff >= a) & (data.cutoff < b) & (data.ts <= pd.Timestamp("2026-11-03"))]
        te = te[(te.hd >= 1) & (te.hd <= 2)]  # Let's inspect horizon 1-2 days (Phase 3 active window!)
        tr = data[data.ts <= a]
        out = {"fold": str(a.date()), "n": len(te)}
        out["B_m7"] = rmse(te.y.values, te.m7.fillna(te.m30).values)
        out["B_m14"] = rmse(te.y.values, te.m14.fillna(te.m30).values)
        out["B_m30"] = rmse(te.y.values, te.m30.values)
        out["B_prof"] = rmse(te.y.values, te.prof_h.fillna(te.m14).values)
        out["B_blend"] = rmse(te.y.values, (0.5 * te.prof_h + 0.25 * te.m7 + 0.25 * te.m30).values)
        for name, params in [("lgb", dict(num_leaves=15, learning_rate=0.03, n_estimators=300,
                                           min_child_samples=500, subsample=0.7, subsample_freq=1,
                                           colsample_bytree=0.8, reg_lambda=10, verbose=-1))]:
            m = lgb.LGBMRegressor(**params).fit(tr[FEATS], tr.y)
            out[name] = rmse(te.y.values, m.predict(te[FEATS]))
            m2 = lgb.LGBMRegressor(**{**params, "objective": "huber", "alpha": 30}).fit(tr[FEATS], tr.y)
            out[name + "_huber"] = rmse(te.y.values, m2.predict(te[FEATS]))
        res.append(out)
        print(out, flush=True)
    r = pd.DataFrame(res)
    print(r.drop(columns=["fold", "n"]).mul(r.n, axis=0).sum() / r.n.sum())


def predict():
    p = load()
    last = pd.Timestamp("2026-11-03 00:00")
    cutoffs = pd.date_range("2025-03-25", "2026-10-31", freq="D")
    data = build(p, cutoffs)
    data = data[data.y.notna() & data.m30.notna()]
    params = dict(num_leaves=15, learning_rate=0.03, n_estimators=300, min_child_samples=500,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1)
    fut = build(p, [last], with_target=False)
    pr = np.mean([lgb.LGBMRegressor(**params, random_state=s).fit(data[FEATS], data.y).predict(fut[FEATS])
                  for s in range(5)], axis=0)
    fut["pred"] = pr
    t = pd.read_csv(f"{D}\\Test.csv")
    t["ts"] = pd.to_datetime(t.target_end_ist.str[:19])
    t["station"] = t.location_id
    fut["station"] = fut.station.astype(int)
    m = t.merge(fut[["station", "ts", "pred"]], on=["station", "ts"], how="left")
    
    # known observations (train) override model output; gaps inside known days use model/local mean
    raw = pd.read_csv(f"{D}\\Phase 3 Train.csv")
    raw["ts"] = pd.to_datetime(raw.timestamp_ist.str[:19])
    obs = raw.pivot_table(index="ts", columns="location_id", values="pm25", aggfunc="mean")
    
    preds = np.zeros(len(m))
    for s, g in m.groupby("location_id"):
        series = obs[s].reindex(pd.date_range(obs.index.min(), last - pd.Timedelta(hours=1), freq="h"))
        interp = series.interpolate(limit=6, limit_area="inside")
        known = interp.reindex(g.ts).values
        out = np.where(np.isnan(known), g.pred.values, known)
        preds[g.index] = out
        
    m["Target"] = np.clip(preds, 0, None)
    sub = m[["ID", "Target"]]
    sample = pd.read_csv(f"{D}\\SampleSubmission.csv")
    assert (sample.ID.values == sub.ID.values).all() and np.isfinite(sub.Target).all()
    sub.to_csv("submissions/submission_lgb_solution.csv", index=False)
    print("Saved submissions/submission_lgb_solution.csv")
    active = m[(m.ts >= '2026-11-03') & (m.ts < '2026-11-05')]
    print("LGB Active Mean:", active.Target.mean())
    return m


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bt":
        backtest()
    elif len(sys.argv) > 1 and sys.argv[1] == "pred":
        predict()
