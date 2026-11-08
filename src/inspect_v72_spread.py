import pandas as pd

test = pd.read_csv('data/raw/Test.csv')
test['dt_start'] = pd.to_datetime(test['target_start_ist'])
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()
nov34_mask = test['dt_start'].dt.date.isin([d3, d4])
sub = pd.read_csv('submission_v72_phase3_champion.csv')
test['pred'] = sub['Target']

nov34 = test[nov34_mask]
for loc, grp in nov34.groupby('location_id'):
    print(f"Station {loc}: range=[{grp['pred'].min():.2f}, {grp['pred'].max():.2f}], std={grp['pred'].std():.2f}")
    print("  First 6 hours:", [round(x, 2) for x in grp['pred'].iloc[:6].values])
    print("  Afternoon 12-18:", [round(x, 2) for x in grp['pred'].iloc[12:18].values])
    print("  Evening 19-23:", [round(x, 2) for x in grp['pred'].iloc[19:24].values])

