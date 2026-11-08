# 🌊 Mumbai BreatheWatch AI
## Hyperlocal Air Quality & Coastal Ventilation Intelligence System
### Submitted for the Google Mumbai Data Challenge ($500 Creative Award)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Author](https://img.shields.io/badge/Author-Ganesh%20Patil-blue.svg)](https://zindi.world)
[![Team](https://img.shields.io/badge/Team-AQI%20Oracles-orange.svg)](https://zindi.world)
[![University](https://img.shields.io/badge/University-Pimpri%20Chinchwad%20University%2C%20Pune-green.svg)](https://pcu.edu.in)
[![Champion Score](https://img.shields.io/badge/RMSE-6.2446-brightgreen.svg)](https://zindi.world)

---

## 🌟 Executive Summary
**Mumbai BreatheWatch AI** is an interactive, real-time air quality forecasting and citizen exposure advisory dashboard developed for the **Google Mumbai Data Challenge: Air Quality Mapping & Forecasting** ($6,000 USD Prize Pool + $500 Creative Application Award).

Powered by **Team AQI Oracles' Champion Model** (`v95 Optimal Station-Adaptive Diurnal Harmonics`, achieving **`6.24460` RMSE** — Rank #1 in India & Worldwide), the application bridges complex atmospheric physics with actionable citizen decision-making.

---

## 🚀 Key Features

### 1. 🗺️ Hyperlocal Geospatial Station Map
- Pinpoints all 4 CPCB air quality monitoring stations across Mumbai with exact GPS coordinates:
  - **Malad West (`11611`):** Western Highway traffic corridor
  - **Shivaji Nagar (`3409511`):** Dense urban nocturnal boundary trapping
  - **Borivali East (`11606`):** National Park vegetative buffer
  - **Colaba / Byculla (`3409510`):** Arabian Sea marine boundary layer anchor
- Real-time CPCB Air Quality Index color coding (*Good, Satisfactory, Moderate, Poor, Very Poor*).

### 2. 🕒 48-Hour Continuous Timeline Slider
- Allows citizens and city planners to scrub forward across the 48-hour forecast horizon (Nov 3 – Nov 4).
- Dynamic bubble sizing and spatial heat distribution updates in real time.

### 3. 🌊 "Best Time to Breathe" Coastal Ventilation Intelligence
- Identifies Mumbai's critical diurnal atmospheric phases:
  - **Morning Rush-Hour Thermal Inversion (08:00 – 10:00 IST):** Traps traffic exhaust along major highways $\rightarrow$ *Warning Advisory: Avoid outdoor cardio workouts*.
  - **Arabian Sea-Breeze Window (16:00 – 19:00 IST):** Marine winds ventilate the city, reducing $\text{PM}_{2.5}$ by up to 25% $\rightarrow$ *Safe Window: Ideal time for outdoor runs, cycling, and natural home ventilation*.

### 4. ⚠️ CPCB Exceedance Probability Risk Gauge ($P(\text{PM}_{2.5} > 60\ \mu\text{g/m}^3)$)
- Implements the critical risk probability metric highlighted by competition lead mentor Simanga during the official webinar.
- Flags when vulnerable demographics (children, elderly, asthmatics) must take precautionary measures.

### 5. 📥 Exportable City Planning & Exposure Report
- One-click CSV export of hourly forecasted exposure across all Mumbai zones for researchers and environmental policymakers.

---

## 💻 How to Run Locally

```bash
# 1. Install dependencies
pip install -r app/requirements.txt

# 2. Launch Streamlit Application
streamlit run app/app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`!*

---

## ☁️ 1-Click Cloud Deployment
This app is ready to deploy directly to:
1. **Streamlit Community Cloud:** Connect your GitHub repo `GaneshPatil7517/mumbai-air-quality-forecasting` and set file path to `app/app.py`.
2. **Hugging Face Spaces:** Create a new Streamlit Space and push the `app/` folder.

