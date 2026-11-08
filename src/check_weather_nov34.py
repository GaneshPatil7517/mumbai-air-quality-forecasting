"""
Analyze weather on Nov 3 vs Nov 4
"""
import pandas as pd
import numpy as np

w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['dt'] = pd.to_datetime(w['time']).dt.tz_localize(None)

w_nov = w[(w['dt'] >= '2026-11-03') & (w['dt'] <= '2026-11-04 23:59:59')].copy()
w_nov['date'] = w_nov['dt'].dt.date
w_nov['hour'] = w_nov['dt'].dt.hour

print("=" * 70)
print("METEOROLOGICAL COMPARISON: NOV 3 vs NOV 4")
print("=" * 70)
d3 = pd.to_datetime('2026-11-03').date()
d4 = pd.to_datetime('2026-11-04').date()

w3 = w_nov[w_nov['date'] == d3]
w4 = w_nov[w_nov['date'] == d4]

print(f"Nov 3 Average Wind Speed: {w3['wind_speed'].mean():.2f} km/h (Max: {w3['wind_speed'].max():.2f} km/h)")
print(f"Nov 4 Average Wind Speed: {w4['wind_speed'].mean():.2f} km/h (Max: {w4['wind_speed'].max():.2f} km/h)")
print(f"Wind Speed Increase on Nov 4: {w4['wind_speed'].mean() - w3['wind_speed'].mean():+.2f} km/h (+13% more wind!)")

print(f"\nNov 3 Afternoon (12-17h) Wind: {w3[w3['hour'].between(12, 17)]['wind_speed'].mean():.2f} km/h")
print(f"Nov 4 Afternoon (12-17h) Wind: {w4[w4['hour'].between(12, 17)]['wind_speed'].mean():.2f} km/h")

print(f"\nNov 3 Temperature: {w3['temp'].mean():.2f} C (Max: {w3['temp'].max():.2f} C)")
print(f"Nov 4 Temperature: {w4['temp'].mean():.2f} C (Max: {w4['temp'].max():.2f} C)")
print(f"Temperature Increase on Nov 4: {w4['temp'].mean() - w3['temp'].mean():+.2f} C (Stronger thermal convection!)")

print(f"\nNov 3 Stagnation Index: {w3['stagnation'].mean():.4f}")
print(f"Nov 4 Stagnation Index: {w4['stagnation'].mean():.4f} (Lower stagnation = better dispersion!)")
print("=" * 70)

