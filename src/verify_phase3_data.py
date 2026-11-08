import pandas as pd
import numpy as np
import shutil

src_path = 'zindi data/Phase 3 Train.csv'
df = pd.read_csv(src_path)
print('=' * 70)
print('PHASE 3 TRAIN FILE VERIFICATION REPORT')
print('=' * 70)
print(f'Total Rows: {len(df):,}')
print(f'Columns: {list(df.columns)}')
df['dt'] = pd.to_datetime(df['timestamp_ist'])
print(f'Date range: {df["dt"].min()} to {df["dt"].max()}')

print('-' * 70)
print('Date breakdown for late Oct / early Nov:')
dates = df['dt'].dt.date.value_counts().sort_index()
print(dates.tail(15))

print('-' * 70)
print('Station breakdown:')
for loc, grp in df.groupby('location_id'):
    print(f'  Station {loc}: {len(grp)} rows, max date: {grp["dt"].max()}, pm25 mean: {grp["pm25"].mean():.2f}')

print('-' * 70)
print(f'Missing pm25: {df["pm25"].isna().sum()}')

if len(df) == 47849:
    print('>>> SUCCESS: Exact row count matches Zindi announcement (47,849 rows)! <<<')
    shutil.copyfile(src_path, 'data/raw/Train.csv')
    shutil.copyfile(src_path, 'data/raw/Train_phase3_nov5_release.csv')
    shutil.copyfile(src_path, 'Train.csv')
    print('Successfully copied to data/raw/Train.csv, data/raw/Train_phase3_nov5_release.csv, and Train.csv!')
else:
    print(f'Row count is {len(df)}, checking against expected 47,849.')

