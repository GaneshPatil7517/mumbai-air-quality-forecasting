import pandas as pd
import numpy as np

w = pd.read_csv('data/processed/mumbai_weather_hourly_updated.csv')
w['ts'] = pd.to_datetime(w['time'])
w_nov34 = w[(w['ts'] >= '2026-11-03') & (w['ts'] < '2026-11-05')].copy()
w_nov34['day'] = w_nov34['ts'].dt.day
w_nov34['hour'] = w_nov34['ts'].dt.hour

print("--- Nov 3 vs Nov 4 Weather Summary ---")
for col in ['temp', 'rh', 'wind_speed', 'u_wind', 'v_wind', 'stagnation', 'pressure']:
    m3 = w_nov34[w_nov34['day'] == 3][col].mean()
    m4 = w_nov34[w_nov34['day'] == 4][col].mean()
    print(f"{col:12s}: Nov 3 = {m3:8.2f} | Nov 4 = {m4:8.2f} (diff = {m4-m3:+8.2f})")

print("\n--- Nov 3 vs Nov 4 Wind Speed by Afternoon Hours (12:00 - 18:00) ---")
for h in range(11, 20):
    ws3 = w_nov34[(w_nov34['day'] == 3) & (w_nov34['hour'] == h)]['wind_speed'].mean()
    ws4 = w_nov34[(w_nov34['day'] == 4) & (w_nov34['hour'] == h)]['wind_speed'].mean()
    print(f"Hour {h:02d}:00: Nov 3 = {ws3:5.2f} km/h | Nov 4 = {ws4:5.2f} km/h (diff = {ws4-ws3:+5.2f} km/h, {(ws4-ws3)/ws3*100:+5.1f}%)")
