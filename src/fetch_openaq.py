"""
Fetch OpenAQ sensor measurements for Mumbai target stations:
- Station 11606 (Borivali East): sensor 12236418
- Station 11611 (Malad West): sensor 12236426
- Station 3409510 (Byculla): sensor 12238567
- Station 3409511 (Shivaji Nagar): sensor 12238576
"""

import urllib.request
import json
import pandas as pd
from datetime import datetime

sensors = {
    11606: 12236418,
    11611: 12236426,
    3409510: 12238567,
    3409511: 12238576
}

# Test OpenAQ v3 API or v2 API
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

for loc_id, sensor_id in sensors.items():
    print(f"Checking Sensor {sensor_id} for Station {loc_id}...")
    # OpenAQ v3 API
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements?datetime_from=2026-10-26T18:30:00Z&datetime_to=2026-10-28T18:30:00Z&limit=1000"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            results = data.get('results', [])
            print(f"  v3 Success! Got {len(results)} records.")
            if results:
                print(f"  First sample: {results[0]}")
    except Exception as e:
        print(f"  v3 failed: {e}")
        # Try v2 or location measurements
        try:
            url_v2 = f"https://api.openaq.org/v2/measurements?location_id={loc_id}&parameter=pm25&date_from=2026-10-26T18:30:00Z&date_to=2026-10-28T18:30:00Z&limit=1000"
            req2 = urllib.request.Request(url_v2, headers=headers)
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                data2 = json.loads(resp2.read().decode())
                res2 = data2.get('results', [])
                print(f"  v2 Success! Got {len(res2)} records.")
        except Exception as e2:
            print(f"  v2 failed: {e2}")

