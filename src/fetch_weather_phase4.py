"""
Fetch complete hourly weather from Open-Meteo for all 4 Mumbai stations
Covering Oct 25 to Nov 14 (covering Nov 7-8 public test and Nov 11-12 private test)
"""

import json
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

stations = pd.read_csv('data/raw/Stations.csv')
output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

all_weather = []

for _, st in stations.iterrows():
    loc_id = st['location_id']
    lat = st['latitude']
    lon = st['longitude']
    name = st['station_name']
    
    print(f"Fetching weather for Station {loc_id} ({name}) at {lat}, {lon}...")
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"past_days=14&forecast_days=7&"
        f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure&"
        f"timezone=Asia%2FKolkata"
    )
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Forecast API failed: {e}. Trying alternative...")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date=2026-10-25&end_date=2026-11-13&"
            f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure&"
            f"timezone=Asia%2FKolkata"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())

    hourly = data['hourly']
    df = pd.DataFrame({
        'time': pd.to_datetime(hourly['time']),
        'location_id': loc_id,
        'temp': hourly['temperature_2m'],
        'rh': hourly['relative_humidity_2m'],
        'wind_speed': hourly['wind_speed_10m'],
        'wind_dir': hourly['wind_direction_10m'],
        'pressure': hourly['surface_pressure'],
    })
    
    wind_rad = np.radians(df['wind_dir'])
    df['u_wind'] = -df['wind_speed'] * np.sin(wind_rad)
    df['v_wind'] = -df['wind_speed'] * np.cos(wind_rad)
    df['stagnation'] = 1.0 / (df['wind_speed'] + 0.5)
    
    all_weather.append(df)

weather_df = pd.concat(all_weather, ignore_index=True)
out_file = output_dir / 'mumbai_weather_hourly_phase4.csv'
weather_df.to_csv(out_file, index=False)
print(f"Saved {len(weather_df)} weather rows to {out_file}")
print(f"Date range: {weather_df['time'].min()} to {weather_df['time'].max()}")

