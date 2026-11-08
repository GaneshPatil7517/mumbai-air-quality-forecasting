import pandas as pd
import numpy as np

sub100 = pd.read_csv('submissions/submission_v100_exact_station_vertex_champion.csv')
test = pd.read_csv('zindi data/Test.csv')
test['ts'] = pd.to_datetime(test['target_end_ist'].str[:19])
active = (test['ts'] >= '2026-11-03') & (test['ts'] < '2026-11-05')

test_act = test[active].copy()
test_act['Target'] = sub100.loc[active, 'Target'].values
test_act['day'] = test_act['ts'].dt.day
test_act['hour'] = test_act['ts'].dt.hour
test_act['station'] = test_act['location_id'].astype(str)

print('Overall Active Mean:', test_act['Target'].mean())
print('Overall Active Std:', test_act['Target'].std())

print('\nStation Means and Stds in v100:')
for s, df in test_act.groupby('station'):
    m = df['Target'].mean()
    sd = df['Target'].std()
    m3 = df[df['day']==3]['Target'].mean()
    m4 = df[df['day']==4]['Target'].mean()
    print(f'Station {s}: Mean={m:.2f}, Std={sd:.2f}, Nov3={m3:.2f}, Nov4={m4:.2f}')

print('\nDay Means in v100:')
for d, df in test_act.groupby('day'):
    print(f'Day {d}: Mean={df["Target"].mean():.2f}, Std={df["Target"].std():.2f}')

