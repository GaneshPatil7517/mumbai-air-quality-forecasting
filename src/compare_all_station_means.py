import pandas as pd
import numpy as np

test = pd.read_csv('zindi data/Test.csv')
test['ts'] = pd.to_datetime(test['target_end_ist'].str[:19])
active = (test['ts'] >= '2026-11-03') & (test['ts'] < '2026-11-05')
test['station'] = test['location_id'].astype(str)

versions = ['v85', 'v87', 'v91', 'v93', 'v95', 'v97', 'v100']
for v in versions:
    path = f'submissions/submission_{v}'
    import glob
    matches = glob.glob(f'submissions/*{v}*.csv')
    if matches:
        sub = pd.read_csv(matches[0])
        test['pred'] = sub['Target']
        st_means = test[active].groupby('station')['pred'].mean()
        overall = test.loc[active, 'pred'].mean()
        print(f"--- {v} ({matches[0].split('/')[-1].split(chr(92))[-1]}) | Overall Mean: {overall:.4f} ---")
        for s, m in st_means.items():
            print(f"  Station {s}: {m:.2f}")

