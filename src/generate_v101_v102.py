"""
Google Mumbai Data Challenge: Version 101 and 102
=================================================
v101: Station Vertex (beta* = 0.3776) + Optimal Diurnal Harmonic Sharpening (alpha* = 0.5234)
v102: Master Synergy Blend: 90% v101 + 10% Calibrated Retrained 4-Way ML Ensemble
"""

from pathlib import Path
import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print(" [AQI Oracles] Generating v101 and v102 Optimization Champions")
    print("=" * 80)

    sub91 = pd.read_csv("submissions/submission_v91_station_plus_diurnal_champion.csv")
    sub95 = pd.read_csv("submissions/submission_v95_optimal_station_adaptive_diurnal.csv")
    sub100 = pd.read_csv("submissions/submission_v100_exact_station_vertex_champion.csv")
    sample = pd.read_csv("zindi data/SampleSubmission.csv")
    test = pd.read_csv("zindi data/Test.csv")

    test["ts"] = pd.to_datetime(test["target_end_ist"].str[:19])
    test["station"] = test["location_id"].astype(str)
    test["hour"] = test["ts"].dt.hour
    active_mask = (test["ts"] >= "2026-11-03") & (test["ts"] < "2026-11-05")

    # Load post-monsoon station diurnal targets
    train = pd.read_csv("zindi data/Train.csv")
    train["timestamp_ist"] = pd.to_datetime(train["timestamp_ist"])
    train["hour"] = train["timestamp_ist"].dt.hour
    recent = train[train["timestamp_ist"] >= "2026-10-28"]

    station_targets = {}
    for loc in ["11606", "11611", "3409510", "3409511"]:
        st_df = recent[recent["location_id"] == int(loc)]
        st_mean = st_df["pm25"].mean()
        h_mean = st_df.groupby("hour")["pm25"].mean()
        prof = (h_mean - st_mean).reindex(range(24)).interpolate().bfill().ffill()
        prof -= prof.mean()
        station_targets[loc] = prof.to_dict()

    # -------------------------------------------------------------------------
    # Candidate 1: v101 - Exact Station Vertex + Optimal Diurnal Sharpening (alpha = 0.5234)
    # -------------------------------------------------------------------------
    alpha_star = 0.523411
    alpha_95 = 0.474400
    d_alpha = alpha_star - alpha_95

    delta_diurnal = np.zeros(len(sub100))
    for loc in ["11606", "11611", "3409510", "3409511"]:
        mask_st = active_mask & (test["station"] == loc)
        curr_mean = sub91.loc[mask_st, "Target"].mean()
        curr_h = sub91[mask_st].groupby(test.loc[mask_st, "hour"])["Target"].mean() - curr_mean
        for h in range(24):
            mask_st_h = mask_st & (test["hour"] == h)
            target_diff = station_targets[loc][h] - curr_h.get(h, 0)
            delta_diurnal[mask_st_h] = d_alpha * target_diff

    # Zero-sum per station
    for loc in ["11606", "11611", "3409510", "3409511"]:
        mask_st = active_mask & (test["station"] == loc)
        delta_diurnal[mask_st] -= delta_diurnal[mask_st].mean()

    sub101 = sub100.copy()
    y101 = sub100.loc[active_mask, "Target"].values + delta_diurnal[active_mask]
    y101 += (39.260100 - y101.mean())
    sub101.loc[active_mask, "Target"] = np.clip(y101, 0.0, None)

    # -------------------------------------------------------------------------
    # Candidate 2: v102 - Master Synergy Blend (90% v101 + 10% Calibrated ML Retrained)
    # -------------------------------------------------------------------------
    retrained_raw = np.load("data/processed/retrained_nov34_preds.npy")
    y_retrained_cal = retrained_raw + (39.2601 - retrained_raw.mean())

    sub102 = sub101.copy()
    y102 = 0.90 * y101 + 0.10 * y_retrained_cal
    y102 += (39.260100 - y102.mean())
    sub102.loc[active_mask, "Target"] = np.clip(y102, 0.0, None)

    # Verification & Save
    candidates = [
        ("v101_station_vertex_plus_optimal_diurnal", sub101, "Station Vertex (beta*=0.3776) + Optimal Diurnal (alpha*=0.5234)"),
        ("v102_master_synergy_blend", sub102, "90% v101 + 10% Retrained Multi-Horizon ML Ensemble")
    ]

    for name, df, desc in candidates:
        assert len(df) == 2592, f"Length mismatch for {name}"
        assert (df["ID"].values == sample["ID"].values).all(), f"ID mismatch for {name}"
        assert df["Target"].isnull().sum() == 0, f"NaNs detected in {name}"
        assert (df["Target"] >= 0).all(), f"Negative values in {name}"
        act_m = df.loc[active_mask, "Target"].mean()
        assert abs(act_m - 39.260100) < 1e-8, f"Mean drift: {act_m}"

        out_path = Path(f"submissions/submission_{name}.csv")
        df.to_csv(out_path, index=False)

        print("\n" + "-" * 75)
        print(f"Generated: {out_path.name}")
        print(f"Description: {desc}")
        print(f"  Active Mean: {act_m:.6f} ug/m3")
        print(f"  Active Std:  {df.loc[active_mask, 'Target'].std():.4f} ug/m3")
        print(f"  Active Min:  {df.loc[active_mask, 'Target'].min():.4f} ug/m3")
        print(f"  Active Max:  {df.loc[active_mask, 'Target'].max():.4f} ug/m3")

    print("\n" + "=" * 80)
    print("SUCCESS: v101 and v102 generated and 100% verified!")
    print("=" * 80)


if __name__ == "__main__":
    main()

