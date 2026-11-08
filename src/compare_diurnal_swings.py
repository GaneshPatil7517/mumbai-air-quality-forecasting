"""
Compare the diurnal shape of Nov 1-2 vs the 180-day monsoon-diluted shape in v78
"""
import pandas as pd
import numpy as np

D_TRAIN = 'zindi data/Phase 3 Train.csv'
D_TEST = 'data/raw/Test.csv'
CAP = 300.0

t = pd.read_csv(D_TRAIN)
t["ts"] = pd.to_datetime(t.timestamp_ist.str[:19])
t["v"] = t.pm25.clip(upper=CAP)
p = t.pivot_table(index="ts", columns="location_id", values="v", aggfunc="mean")
full = pd.date_range(p.index.min().floor("D"), "2026-11-03 00:00", freq="h")
p = p.reindex(full)

CUT = pd.Timestamp("2026-11-03 00:00")
hist = p.loc[:CUT]

print("=" * 70)
print("DIURNAL AMPLITUDE COMPARISON: 180-DAY (v78) vs POST-MONSOON (LATE OCT / EARLY NOV)")
print("=" * 70)

for s in p.columns:
    x = hist[s]
    # v78 shape (180 days)
    xs_180 = x.tail(4320)
    sh_180 = (xs_180.groupby((xs_180.index.hour - 1) % 24).mean() / xs_180.mean())
    damped_180 = 0.5 + 0.5 * sh_180
    
    # Post-monsoon shape (last 14 days, Oct 20 - Nov 2)
    xs_14 = x.tail(336)
    sh_14 = (xs_14.groupby((xs_14.index.hour - 1) % 24).mean() / xs_14.mean())
    
    # Full post-monsoon un-damped vs v78 damped
    print(f"\nStation {s}:")
    print(f"  180-day damped (v78) Amplitude: [{damped_180.min():.3f}, {damped_180.max():.3f}] (Swing: {damped_180.max() - damped_180.min():.3f})")
    print(f"  14-day post-monsoon Amplitude:  [{sh_14.min():.3f}, {sh_14.max():.3f}] (Swing: {sh_14.max() - sh_14.min():.3f})")
    print(f"  Ratio of Real Winter Swing to v78 Swing: {(sh_14.max() - sh_14.min()) / (damped_180.max() - damped_180.min()):.2f}x!")
print("=" * 70)

