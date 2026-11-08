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
    
    # Try forecast API with past_days=7 and forecast_days=3 to cover Nov 1 to Nov 5 seamlessly
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"past_days=7&forecast_days=3&"
        f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure&"
        f"timezone=Asia%2FKolkata"
    )
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Forecast API failed: {e}. Trying archive API...")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date=2026-10-25&end_date=2026-11-05&"
            f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure&"
            f"timezone=Asia%2FKolkata"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
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

new_weather_df = pd.concat(all_weather, ignore_index=True)
print(f"Fetched {len(new_weather_df):,} fresh weather rows covering {new_weather_df['time'].min()} to {new_weather_df['time'].max()}.")

# Now merge with existing weather up to 2026-10-28 to avoid missing historical data
old_weather_file = output_dir / 'mumbai_weather_hourly_updated.csv'
if old_weather_file.exists():
    old_weather = pd.read_csv(old_weather_file)
    old_weather['time'] = pd.to_datetime(old_weather['time'])
    
    # Filter old weather before the start of new weather
    cutoff = new_weather_df['time'].min()
    combined = pd.concat([old_weather[old_weather['time'] < cutoff], new_weather_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['time', 'location_id']).sort_values(['location_id', 'time']).reset_index(drop=True)
else:
    combined = new_weather_df

combined.to_csv(old_weather_file, index=False)
print("=" * 60)
print(f"SUCCESS! Updated weather database: {old_weather_file}")
print(f"Total Rows: {len(combined):,}")
print(f"Date Range: {combined['time'].min()} to {combined['time'].max()}")
print("=" * 60)

