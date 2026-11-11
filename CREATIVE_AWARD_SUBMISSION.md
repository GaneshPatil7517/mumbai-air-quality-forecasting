# 🌊 Mumbai BreatheWatch AI
## Hyperlocal Air Quality & Coastal Ventilation Intelligence System
### Official Submission for the Google Mumbai Data Challenge — Most Creative Submission Prize ($500 USD)

[![Platform: Zindi](https://img.shields.io/badge/Platform-Zindi-blue.svg)](https://zindi.world/competitions/google-mumbai-data-challenge)
[![Host: Google](https://img.shields.io/badge/Host-Google-red.svg)](https://google.com)
[![Category: Creative Award](https://img.shields.io/badge/Award-Most%20Creative%20Submission%20($500)-gold.svg)](https://zindi.world)
[![Team](https://img.shields.io/badge/Team-AQI%20Oracles-orange.svg)](https://zindi.world)
[![Affiliation](https://img.shields.io/badge/University-Pimpri%20Chinchwad%20University%2C%20Pune-green.svg)](https://pcu.edu.in)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

---

## 🌟 Executive Summary & Motivation

Mumbai, home to over 21 million citizens, faces severe, non-stationary air pollution dynamics driven by the interplay between industrial emissions, heavy vehicular transit, and coastal marine meteorology. 

Standard air quality monitoring apps present citywide averages that fail Mumbai citizens in two critical ways:
1. **Spatial Blindness:** They ignore the massive microclimate gap between marine buffer zones (Colaba/Byculla: ~30 µg/m³) and highway inversion corridors (Malad West / Shivaji Nagar: >100 µg/m³).
2. **Temporal Passivity:** They report retrospective historical measurements rather than actionable forward-looking forecast windows.

**Mumbai BreatheWatch AI** bridges this gap. Developed by **Team AQI Oracles** (Pimpri Chinchwad University, Pune) for the **Google Mumbai Data Challenge**, the platform translates cutting-edge atmospheric machine learning into an interactive, citizen-centric public health and municipal planning decision-support suite.

---

## 🚀 Key Features & Novel Functionality

```
                  ┌────────────────────────────────────────┐
                  │        Mumbai BreatheWatch AI          │
                  └───────────────────┬────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  CleanCommute    │        │  Google Gemini   │        │ Municipal Policy │
│  Route Exposure  │        │  AI Health Agent │        │ Action Sandbox   │
│  Minimizer       │        │  (Hyperlocal)    │        │ (BMC / MPCB)     │
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

### 1. 🚗 CleanCommute Mumbai: Hyperlocal Route Exposure Calculator
- **The Problem:** Commuters travel along high-exposure corridors (Western Express Highway, Eastern Freeway) during peak morning hours without knowing their cumulative inhaled dose.
- **The Breakthrough:** CleanCommute allows citizens to select origin and destination stations (e.g., Borivali East → Byculla). By comparing morning rush hour (09:00 IST) against the afternoon sea-breeze clearance window (17:00 IST), the engine demonstrates **up to a 34% reduction in inhaled particulate exposure** by shifting departure times.

### 2. 🤖 Google Gemini AI Citizen Health Assistant (Google Tools Bonus)
- Directly integrates Google Gemini prompt-engineering to synthesize raw numerical PM2.5 forecasts into empathetic, actionable health advisories.
- **Role-Based Dynamic Guidance:**
  - **Outdoor Runners / Athletes:** Recommends optimal training windows when marine sea-breezes dilute ground stagnation.
  - **Pediatric & Geriatric Care:** Alerts sensitive respiratory populations during nocturnal thermal inversions.
  - **Masking Protocols:** Dynamic N95/FFP2 recommendation based on National Ambient Air Quality Standards (NAAQS).

### 3. 🏛️ Municipal Clean Air Action Sandbox (BMC / MPCB Decision Support)
- An interactive simulation sandbox for city administrators (Brihanmumbai Municipal Corporation).
- **Simulates Real-World Interventions:**
  - *Emergency Smog Guns & Anti-Smog Misting:* Models localized particulate settling (-10% to -15% PM2.5).
  - *Heavy Freight & Diesel Truck Diversions:* Models peak rush-hour emission reductions along the Western Express Highway (-15% to -20% PM2.5).
  - *Construction Dust Suppressants:* Calculates real-time CPCB compliance gains across northern suburban corridors.

### 4. 🗺️ Interactive Geospatial Cartography & 48-Hour Timeline Scrubber
- High-contrast, dark-matter Google Maps cartography pinning all 4 CPCB monitoring stations with live GPS accuracy.
- Continuous 48-hour timeline slider spanning the **11–12 November 2026** private evaluation window.
- Dynamic color-coded CPCB Air Quality Index (AQI) bands (*Good, Satisfactory, Moderate, Poor, Very Poor*).

### 5. 🌊 Coastal Marine Ventilation & Diurnal Rhythm Explorer
- Visualizes the physical diurnal cycle across Mumbai:
  - **Morning Thermal Inversion Trap (08:00 – 10:00 IST):** Trapping commuter vehicular exhaust.
  - **Afternoon Sea-Breeze Clearance Window (14:00 – 18:00 IST):** Marine boundary layer expansion flushing urban pollutants.
  - **Nocturnal Boundary Layer Collapse (22:00 – 02:00 IST):** Ground cooling trapping emissions under a low mixing height.

---

## 🛠️ Google Ecosystem Integration

| Google Tool | Implementation in BreatheWatch AI | Impact |
| :--- | :--- | :--- |
| **Google Gemini AI** | Natural language citizen health reasoning engine | Converts complex $\mu\text{g/m}^3$ numbers into tailored lifestyle recommendations |
| **Google Maps Cartography** | Dark-matter thematic cartography & spatial coordinates | Visualizes urban microclimate contrast across coastal vs inland zones |
| **Google Colab Cloud** | 1-Click reproducible verification notebook | Enables instant evaluation and replication by Google and Zindi judges |

---

## 📁 Repository Structure & Deliverables

```
├── app/
│   ├── app.py                     # Streamlit Interactive Web Application (670 lines)
│   ├── README.md                  # Detailed App Documentation & Feature Architecture
│   ├── requirements.txt           # Deployment Dependencies (Streamlit, Plotly, ReportLab)
│   └── zindi_logo_white.svg       # Official Vector Branding
├── docs/
│   ├── Mumbai_BreatheWatch_AI_Google_Submission.pdf  # Official 5-Page Publication PDF Whitepaper
│   └── Google_Mumbai_AQI_Champion_Presentation.md     # 10-Slide Finalist Defense Deck with Speaker Script
├── src/
│   ├── build_final_champion_pipeline.py              # Reproducible End-to-End ML Pipeline
│   └── generate_submission_pdf.py                    # PDF Generation Engine (ReportLab)
├── submissions/
│   ├── submission_final_phase_master_champion.csv    # Selection 1 (ID: VTjUBvJm, Mean 73.02 µg/m³)
│   └── submission_final_phase_robust_seasonal_hybrid.csv # Selection 2 (ID: tpViw7Sg, Mean 64.22 µg/m³)
└── CREATIVE_AWARD_SUBMISSION.md   # This Master Submission Document
```

---

## 🚀 How to Run the Application Locally

```bash
# 1. Clone repository
git clone https://github.com/GaneshPatil7517/mumbai-air-quality-forecasting.git
cd mumbai-air-quality-forecasting

# 2. Install dependencies
pip install -r app/requirements.txt

# 3. Launch the Streamlit Web Application
streamlit run app/app.py
```
*The app will automatically launch in your browser at `http://localhost:8501`.*

---

## 👥 Team & Institutional Credentials
- **Team:** AQI Oracles  
- **Team Lead:** Ganesh Patil (`ganeshpatil191205@gmail.com`)  
- **Teammate:** Shankar  
- **Institution:** Pimpri Chinchwad University (PCU), Pune, Maharashtra  
- **Event Eligibility:** Formally confirmed resident and student of Maharashtra, 100% available to attend and present at the **Google & Zindi Mumbai Grand Finale on 20 November 2026**.

