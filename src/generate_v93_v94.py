"""
Script to generate submission v93 and v94:
Station-Adaptive Diurnal Harmonics Champion (alpha = 0.35 and alpha = 0.22)
Preserves active mean mu* = 39.2601 ug/m3 and zero-sum balance per station.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    sub91_path = Path("submissions/submission_v91_station_plus_diurnal_champion.csv")
    assert sub91_path.exists(), f"Missing {sub91_path}"
    
    sub91 = pd.read_csv(sub91_path)
    date_col = sub91["ID"].apply(lambda x: x.split("_")[2][:8])
    hour_col = sub91["ID"].apply(lambda x: int(x.split("_")[2][9:11]))
    station_col = sub91["ID"].apply(lambda x: x.split("_")[1])
    eval_mask = date_col.isin(["20261103", "20261104"])

    # Load Train.csv for post-monsoon ground-truth diurnal harmonics
    train = pd.read_csv("Train.csv")
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

    for name, alpha in [("v93_station_adaptive_diurnal_champion", 0.35), ("v94_conservative_station_adaptive_diurnal", 0.22)]:
        df = sub91[["ID", "Target"]].copy()
        delta_col = np.zeros(len(df))
        for loc in ["11606", "11611", "3409510", "3409511"]:
            mask_st = eval_mask & (station_col == loc)
            curr_mean = sub91.loc[mask_st, "Target"].mean()
            curr_h = sub91[mask_st].groupby(hour_col)["Target"].mean() - curr_mean
            for h in range(24):
                mask_st_h = mask_st & (hour_col == h)
                target_diff = station_targets[loc][h] - curr_h.get(h, 0)
                delta_col[mask_st_h] = alpha * target_diff
                
        # Zero-sum per station
        for loc in ["11606", "11611", "3409510", "3409511"]:
            mask_st = eval_mask & (station_col == loc)
            mean_d = delta_col[mask_st].mean()
            delta_col[mask_st] -= mean_d
            
        df["Target"] += delta_col
        
        assert df["Target"].isnull().sum() == 0
        assert len(df) == 2592
        diff_eval = df.loc[eval_mask, "Target"].mean() - sub91.loc[eval_mask, "Target"].mean()
        assert abs(diff_eval) < 1e-10, f"Mean shifted: {diff_eval}"
        
        out_path = Path(f"submissions/submission_{name}.csv")
        df.to_csv(out_path, index=False)
        m_min = float(df["Target"].min())
        m_mean = float(df["Target"].mean())
        m_max = float(df["Target"].max())
        print(f"Created {out_path}: min={m_min:.4f}, mean={m_mean:.4f}, max={m_max:.4f}, max_shift={np.max(np.abs(delta_col)):.4f}")

if __name__ == "__main__":
    main()

