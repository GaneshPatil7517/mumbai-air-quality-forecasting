import pandas as pd

train = pd.read_csv('data/raw/Train.csv')
train['dt'] = pd.to_datetime(train['timestamp_ist'])
nov2_night = train[(train['dt'] >= '2026-11-02 20:00:00') & (train['dt'] <= '2026-11-02 23:00:00')]

print("Station observations on Nov 2 evening/night:")
for loc, grp in nov2_night.groupby('location_id'):
    print(f"\nStation {loc} ({grp['station_name'].iloc[0]}):")
    for _, r in grp.sort_values('dt').iterrows():
        print(f"   {r['dt']} : {r['pm25']:.2f}")

