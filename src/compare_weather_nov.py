import pandas as pd
import numpy as np

w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time'])

w_nov = w[(w['dt'] >= '2026-11-01') & (w['dt'] <= '2026-11-04 23:59:59')].copy()
w_nov['day'] = w_nov['dt'].dt.date

print("=" * 70)
print("WEATHER CONDITIONS: NOV 1-2 (KNOWN) vs NOV 3-4 (TEST TARGET)")
print("=" * 70)
for d, grp in w_nov.groupby('day'):
    print(f"Date {d}:")
    print(f"  Wind Speed: mean={grp['wind_speed'].mean():.2f} km/h (min={grp['wind_speed'].min():.2f}, max={grp['wind_speed'].max():.2f})")
    print(f"  Temperature: mean={grp['temp'].mean():.2f} C (min={grp['temp'].min():.2f}, max={grp['temp'].max():.2f})")
    print(f"  Pressure:   mean={grp['pressure'].mean():.2f} hPa")
    print(f"  Stagnation: mean={grp['stagnation'].mean():.3f}")
print("=" * 70)

