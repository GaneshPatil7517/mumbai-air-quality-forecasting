"""
Fetch Planetary Boundary Layer (PBL) 100m wind and soil thermodynamics features from Open-Meteo:
1. wind_speed_100m, wind_direction_100m (upper planetary boundary layer flow)
2. vapour_pressure_deficit (drying / aerosol growth dynamics)
3. soil_temperature_0_to_7cm, soil_moisture_0_to_7cm (surface latent heat & dust resuspension)
4. Derived: u_wind_100m, v_wind_100m, vertical_wind_shear, ventilation_100m
"""

import json
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

def fetch_pbl_weather():
    stations = pd.read_csv('data/raw/Stations.csv')
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_weather = []
    
    for _, st in stations.iterrows():
        loc_id = st['location_id']
        lat = st['latitude']
        lon = st['longitude']
        name = st['station_name']
        
        print(f"Fetching PBL 100m atmospheric data for Station {loc_id} ({name})...")
        
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date=2026-08-01&end_date=2026-10-24&"
            f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,"
            f"boundary_layer_height,wind_speed_100m,wind_direction_100m,vapour_pressure_deficit,"
            f"soil_temperature_0_to_7cm,soil_moisture_0_to_7cm&"
            f"timezone=Asia%2FKolkata"
        )
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode())
            hourly = data['hourly']
            
            df = pd.DataFrame({
                'time': pd.to_datetime(hourly['time']),
                'location_id': loc_id,
                'temp': hourly['temperature_2m'],
                'rh': hourly['relative_humidity_2m'],
                'pressure': hourly['surface_pressure'],
                'blh': hourly['boundary_layer_height'],
                'wind_speed_10m': hourly['wind_speed_10m'],
                'wind_dir_10m': hourly['wind_direction_10m'],
                'wind_speed_100m': hourly['wind_speed_100m'],
                'wind_dir_100m': hourly['wind_direction_100m'],
                'vpd': hourly['vapour_pressure_deficit'],
                'soil_temp': hourly['soil_temperature_0_to_7cm'],
                'soil_moisture': hourly['soil_moisture_0_to_7cm'],
            })
            
            # Surface 10m wind vectors
            rad_10 = np.radians(df['wind_dir_10m'])
            df['u_wind_10m'] = -df['wind_speed_10m'] * np.sin(rad_10)
            df['v_wind_10m'] = -df['wind_speed_10m'] * np.cos(rad_10)
            
            # Planetary boundary layer 100m wind vectors
            rad_100 = np.radians(df['wind_dir_100m'])
            df['u_wind_100m'] = -df['wind_speed_100m'] * np.sin(rad_100)
            df['v_wind_100m'] = -df['wind_speed_100m'] * np.cos(rad_100)
            
            # Vertical wind shear (turbulent mechanical mixing)
            df['vertical_shear'] = df['wind_speed_100m'] - df['wind_speed_10m']
            
            # True PBL ventilation (volume flux per unit width)
            df['ventilation_100m'] = df['wind_speed_100m'] * df['blh']
            
            # Ground stagnation
            df['stagnation_10m'] = 1.0 / (df['wind_speed_10m'] + 0.5)
            df['stagnation_100m'] = 1.0 / (df['wind_speed_100m'] + 0.5)
            
            all_weather.append(df)
            
    pbl_df = pd.concat(all_weather, ignore_index=True)
    out_file = output_dir / 'mumbai_pbl_100m_weather_hourly.csv'
    pbl_df.to_csv(out_file, index=False)
    print("=" * 60)
    print(f"SUCCESS! Saved PBL weather dataset: {out_file} ({len(pbl_df):,} rows)")
    print(pbl_df.head())
    print("=" * 60)

if __name__ == '__main__':
    fetch_pbl_weather()

