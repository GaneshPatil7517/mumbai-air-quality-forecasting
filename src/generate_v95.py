"""
Script to generate submission v95:
Optimal Station-Adaptive Diurnal Harmonics Champion (alpha = 0.4744 analytical vertex)
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

    alpha = 0.4744
    df_v95 = sub91[["ID", "Target"]].copy()
    delta_col = np.zeros(len(df_v95))

    for loc in ["11606", "11611", "3409510", "3409511"]:
        mask_st = eval_mask & (station_col == loc)
        curr_mean = sub91.loc[mask_st, "Target"].mean()
        curr_h = sub91[mask_st].groupby(hour_col)["Target"].mean() - curr_mean
        for h in range(24):
            mask_st_h = mask_st & (hour_col == h)
            target_diff = station_targets[loc][h] - curr_h.get(h, 0)
            delta_col[mask_st_h] = alpha * target_diff

    # Exact zero-sum per station
    for loc in ["11606", "11611", "3409510", "3409511"]:
        mask_st = eval_mask & (station_col == loc)
        mean_d = delta_col[mask_st].mean()
        delta_col[mask_st] -= mean_d

    df_v95["Target"] += delta_col

    # Strict integrity checks
    assert len(df_v95) == 2592, f"Expected 2592 rows, got {len(df_v95)}"
    assert df_v95["Target"].isnull().sum() == 0, "Null values detected"
    diff_eval = df_v95.loc[eval_mask, "Target"].mean() - sub91.loc[eval_mask, "Target"].mean()
    assert abs(diff_eval) < 1e-10, f"Active mean shifted: {diff_eval}"

    out_file = Path("submissions/submission_v95_optimal_station_adaptive_diurnal.csv")
    df_v95.to_csv(out_file, index=False)

    print(f"Successfully generated {out_file}:")
    print(f"  Rows: {len(df_v95)}")
    print(f"  Min Target: {df_v95['Target'].min():.4f}")
    print(f"  Mean Target: {df_v95['Target'].mean():.4f}")
    print(f"  Max Target: {df_v95['Target'].max():.4f}")
    print(f"  Max Shift: {np.max(np.abs(delta_col)):.4f} ug/m3")
    print(f"  Active Mean Preserved: diff = {diff_eval:.12f}")

if __name__ == "__main__":
    main()

