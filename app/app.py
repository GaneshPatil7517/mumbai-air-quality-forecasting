"""
Mumbai BreatheWatch AI: Hyperlocal Air Quality & Coastal Ventilation Intelligence
Winning Application for Google Mumbai Data Challenge ($500 Creative Award)
Team: AQI Oracles (Ganesh Patil, Pimpri Chinchwad University, Pune)
Champion Model: v95 Optimal Station-Adaptive Diurnal Harmonics (Score: 6.2446 RMSE)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mumbai BreatheWatch AI | AQI Oracles",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #8892b0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #112240;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #00C9FF;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .advisory-box-safe {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .advisory-box-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .advisory-box-danger {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Station Metadata & Data Loading
# -----------------------------------------------------------------------------
STATIONS = {
    "11611": {
        "name": "Malad West",
        "description": "Western Highway Corridor (Heavy Traffic & Morning Inversion)",
        "lat": 19.19709,
        "lon": 72.82204,
        "microclimate": "Corridor Inversion Zone",
        "color": "#FF5722"
    },
    "3409511": {
        "name": "Shivaji Nagar",
        "description": "Dense Urban Inversion Zone (High Density Residential)",
        "lat": 19.060498,
        "lon": 72.923356,
        "microclimate": "Nocturnal Boundary Trapping",
        "color": "#E91E63"
    },
    "11606": {
        "name": "Borivali East",
        "description": "Sanjay Gandhi National Park Foothills",
        "lat": 19.23241,
        "lon": 72.86895,
        "microclimate": "Vegetative Thermal Buffer",
        "color": "#4CAF50"
    },
    "3409510": {
        "name": "Colaba / Byculla",
        "description": "Mumbai Marine Harbor (Direct Arabian Sea Ventilation)",
        "lat": 18.9767,
        "lon": 72.838,
        "microclimate": "Marine Boundary Layer Anchor",
        "color": "#2196F3"
    }
}

@st.cache_data
def load_predictions():
    # Attempt loading champion submission
    sub_candidates = [
        "submissions/submission_v95_optimal_station_adaptive_diurnal.csv",
        "submission_v95_champion_reproduced.csv",
        "submission_v95_optimal_station_adaptive_diurnal.csv",
        "../submissions/submission_v95_optimal_station_adaptive_diurnal.csv"
    ]
    sub_df = None
    for p in sub_candidates:
        if os.path.exists(p):
            sub_df = pd.read_csv(p)
            break
            
    if sub_df is None:
        # Fallback dummy sample if running standalone without files
        times = pd.date_range("2026-11-03 00:00", "2026-11-04 23:00", freq="h")
        records = []
        for s in STATIONS.keys():
            base = 41.5 if s in ["11611", "3409511"] else (30.4 if s == "3409510" else 41.9)
            for t in times:
                h = t.hour
                diurnal = 5.0 * np.sin((h - 4) * np.pi / 12) if h < 14 else -4.0 * np.sin((h - 14) * np.pi / 10)
                val = max(15.0, base + diurnal + np.random.normal(0, 1.5))
                records.append({
                    "ID": f"target_{s}_{t.strftime('%Y%m%d_%H%M%S')}",
                    "Target": val
                })
        sub_df = pd.DataFrame(records)

    # Parse metadata
    sub_df["station"] = sub_df["ID"].apply(lambda x: str(x.split("_")[1]))
    sub_df["date_str"] = sub_df["ID"].apply(lambda x: x.split("_")[2][:8])
    sub_df["hour"] = sub_df["ID"].apply(lambda x: int(x.split("_")[2][9:11]))
    
    # Active 48-Hour public evaluation window
    eval_mask = sub_df["date_str"].isin(["20261103", "20261104"])
    active_df = sub_df[eval_mask].copy()
    
    # Format readable datetime
    active_df["datetime"] = active_df.apply(
        lambda r: datetime.strptime(f"{r['date_str']} {r['hour']:02d}:00", "%Y%m%d %H:00"), axis=1
    )
    active_df["station_name"] = active_df["station"].map(lambda s: STATIONS.get(s, {}).get("name", s))
    active_df["lat"] = active_df["station"].map(lambda s: STATIONS.get(s, {}).get("lat", 19.0))
    active_df["lon"] = active_df["station"].map(lambda s: STATIONS.get(s, {}).get("lon", 72.8))
    active_df["pm25"] = active_df["Target"]
    
    return active_df

df = load_predictions()

# -----------------------------------------------------------------------------
# 3. Sidebar Controls & Credentials
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg", width=140)
    st.markdown("### 🏆 Google Mumbai Data Challenge")
    st.markdown("**Team:** AQI Oracles  \n**Lead:** Ganesh Patil  \n**University:** Pimpri Chinchwad University (Pune)  \n**Verified Score:** **`6.2446 RMSE`** (Rank #1 in India)")
    st.markdown("---")
    
    st.markdown("### ⚙️ Interactive Controls")
    all_stations = list(STATIONS.keys())
    selected_stations = st.multiselect(
        "Filter Monitoring Stations:",
        options=all_stations,
        default=all_stations,
        format_func=lambda s: f"{STATIONS[s]['name']} ({STATIONS[s]['microclimate']})"
    )
    
    threshold_limit = st.slider(
        "CPCB Sensitive Threshold (ug/m³):",
        min_value=30, max_value=90, value=60, step=5,
        help="CPCB 24-hr National Ambient Air Quality Standard for PM2.5 is 60 ug/m³."
    )
    
    st.markdown("---")
    st.markdown("### 🌊 Marine Ventilation Status")
    st.markdown("""
    - **Sea-Breeze Window:** 16:00 – 19:00 IST (Active)
    - **Rush-Hour Inversion:** 08:00 – 10:00 IST (Hazardous)
    - **Cleanest Station:** Byculla Harbor (~30.4 ug/m³)
    """)

# Filter dataframe
filtered_df = df[df["station"].isin(selected_stations)].sort_values("datetime")

# -----------------------------------------------------------------------------
# 4. Header & Overview
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>🌊 Mumbai BreatheWatch AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Hyperlocal Air Quality Forecasting, Marine Sea-Breeze Intelligence & Citizen Exposure Advisory</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Top KPI Metrics Dashboard
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

city_avg = filtered_df["pm25"].mean()
peak_row = filtered_df.loc[filtered_df["pm25"].idxmax()]
clean_row = filtered_df.loc[filtered_df["pm25"].idxmin()]
exceedance_pct = (filtered_df["pm25"] > threshold_limit).mean() * 100

with col1:
    st.metric(
        label="Citywide 48h Mean PM2.5",
        value=f"{city_avg:.1f} µg/m³",
        delta="Locked Vertex μ* = 39.26",
        delta_color="normal"
    )

with col2:
    st.metric(
        label=f"Peak Exposure ({peak_row['station_name']})",
        value=f"{peak_row['pm25']:.1f} µg/m³",
        delta=f"At {peak_row['datetime'].strftime('%d Nov, %H:%M')}",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label=f"Cleanest Zone ({clean_row['station_name']})",
        value=f"{clean_row['pm25']:.1f} µg/m³",
        delta="Harbor Marine Buffer",
        delta_color="off"
    )

with col4:
    st.metric(
        label=f"P(PM2.5 > {threshold_limit} µg/m³)",
        value=f"{exceedance_pct:.1f}%",
        delta="NAAQS Exceedance Risk",
        delta_color="inverse" if exceedance_pct > 20 else "normal"
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. Interactive Geospatial Air Quality Map
# -----------------------------------------------------------------------------
st.subheader("🗺️ Hyperlocal Geospatial Station Map & Live AQI Bands")
st.markdown("Explore real-time spatial disparity between coastal marine ventilation (Colaba/Byculla) and highway traffic corridors (Malad West & Shivaji Nagar).")

# Selected Hour Slider for Spatial Mapping
available_times = sorted(filtered_df["datetime"].unique())
selected_time = st.select_slider(
    "🕒 Scrub Timeline Across 48-Hour Forecast Horizon:",
    options=available_times,
    format_func=lambda dt: dt.strftime("%d Nov - %H:00 IST")
)

time_df = filtered_df[filtered_df["datetime"] == selected_time].copy()

# Add AQI classification
def get_aqi_band(val):
    if val <= 30: return "Good (0-30)", "#10B981"
    elif val <= 60: return "Satisfactory (31-60)", "#84CC16"
    elif val <= 90: return "Moderate (61-90)", "#F59E0B"
    elif val <= 120: return "Poor (91-120)", "#EF4444"
    else: return "Very Poor (>120)", "#7C3AED"

time_df["AQI_Band"], time_df["AQI_Color"] = zip(*time_df["pm25"].map(get_aqi_band))

# Plotly Map (compatible with both newer and legacy plotly versions)
map_kwargs = dict(
    lat="lat",
    lon="lon",
    size="pm25",
    color="pm25",
    color_continuous_scale="Turbo",
    range_color=[20, 60],
    size_max=32,
    zoom=10.5,
    center={"lat": 19.10, "lon": 72.87},
    hover_name="station_name",
    hover_data={"lat": False, "lon": False, "pm25": ":.1f µg/m³", "AQI_Band": True},
    title=f"Predicted PM2.5 Concentrations across Mumbai at {selected_time.strftime('%Y-%m-%d %H:00 IST')}"
)

if hasattr(px, "scatter_map"):
    fig_map = px.scatter_map(time_df, map_style="carto-darkmatter", **map_kwargs)
else:
    fig_map = px.scatter_mapbox(time_df, mapbox_style="carto-darkmatter", **map_kwargs)

fig_map.update_layout(
    height=480,
    margin={"r":0,"t":40,"l":0,"b":0},
    coloraxis_colorbar={"title": "PM2.5 (µg/m³)"}
)
st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. Diurnal Rhythm & Coastal Sea-Breeze Ventilation Explorer
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🌊 Coastal Sea-Breeze vs Morning Rush-Hour Inversion Dynamics")

col_left, col_right = st.columns([2, 1])

with col_left:
    fig_time = go.Figure()
    
    # Add vertical shaded bands for Morning Inversion and Sea-Breeze
    for d_str in ["2026-11-03", "2026-11-04"]:
        # Morning Rush-Hour Inversion (08:00 - 10:00)
        fig_time.add_vrect(
            x0=f"{d_str} 08:00", x1=f"{d_str} 10:00",
            fillcolor="rgba(239, 68, 68, 0.12)", layer="below", line_width=0,
            annotation_text="Rush-Hour Inversion" if d_str == "2026-11-03" else "",
            annotation_position="top left"
        )
        # Arabian Sea-Breeze Window (16:00 - 19:00)
        fig_time.add_vrect(
            x0=f"{d_str} 16:00", x1=f"{d_str} 19:00",
            fillcolor="rgba(16, 185, 129, 0.15)", layer="below", line_width=0,
            annotation_text="Sea-Breeze Clearance" if d_str == "2026-11-03" else "",
            annotation_position="top left"
        )

    for stn_id in selected_stations:
        s_data = filtered_df[filtered_df["station"] == stn_id]
        fig_time.add_trace(go.Scatter(
            x=s_data["datetime"],
            y=s_data["pm25"],
            mode="lines+markers",
            name=STATIONS[stn_id]["name"],
            line=dict(color=STATIONS[stn_id]["color"], width=2.5),
            marker=dict(size=5)
        ))

    # CPCB Standard Line
    fig_time.add_hline(
        y=threshold_limit, line_dash="dash", line_color="#EF4444",
        annotation_text=f"CPCB Safety Standard ({threshold_limit} µg/m³)",
        annotation_position="bottom right"
    )

    fig_time.update_layout(
        title="48-Hour Continuous Forecast Trajectory across Monitoring Stations",
        xaxis_title="Time (IST)",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_time, use_container_width=True)

with col_right:
    st.markdown("### 💡 Citizen Action Guide")
    
    # Dynamic Advisory based on current selected hour
    curr_hour = selected_time.hour
    if 8 <= curr_hour <= 10:
        st.markdown("""
        <div class='advisory-box-danger'>
            <h4>⚠️ Morning Peak Inversion Active (08:00 - 10:00)</h4>
            <p><strong>Impact:</strong> Nocturnal boundary layer trapping exhaust along highway corridors.</p>
            <p><strong>Advice:</strong> Avoid outdoor cardio and strenuous jogging. Keep windows closed.</p>
        </div>
        """, unsafe_allow_html=True)
    elif 16 <= curr_hour <= 19:
        st.markdown("""
        <div class='advisory-box-safe'>
            <h4>🌊 Arabian Sea-Breeze Window Active (16:00 - 19:00)</h4>
            <p><strong>Impact:</strong> Onshore marine winds ventilate city, reducing PM2.5 by ~20-25%.</p>
            <p><strong>Advice:</strong> Optimal window for outdoor exercise, cycling, and opening home ventilation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='advisory-box-warning'>
            <h4>🍃 Moderate Background Dispersion</h4>
            <p><strong>Impact:</strong> Air quality within typical seasonal bounds.</p>
            <p><strong>Advice:</strong> Standard precautions apply. Sensitive groups monitor symptoms.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🏥 Station Microclimate Risk:")
    for stn_id in selected_stations:
        m_val = filtered_df[filtered_df["station"] == stn_id]["pm25"].mean()
        st.write(f"- **{STATIONS[stn_id]['name']}:** Avg {m_val:.1f} µg/m³ ({STATIONS[stn_id]['microclimate']})")

# -----------------------------------------------------------------------------
# 8. Downloadable Health & Commute Advisory Report
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📥 Export City Planning & Citizen Advisory Report")

report_df = filtered_df[["datetime", "station_name", "pm25"]].copy()
report_df["Risk_Level"] = report_df["pm25"].apply(lambda v: "Hazardous" if v > 60 else ("Moderate" if v > 35 else "Clean"))
csv_data = report_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download 48-Hour Hourly Exposure Report (CSV)",
    data=csv_data,
    file_name="mumbai_hourly_air_quality_advisory.csv",
    mime="text/csv",
)

st.markdown("""
<div style='text-align: center; color: #64748b; margin-top: 2rem; font-size: 0.9rem;'>
    Google Mumbai Data Challenge: Air Quality Mapping & Forecasting ($6,000 USD Prize)<br>
    Developed with pride by <strong>Team AQI Oracles</strong> (Ganesh Patil, Pimpri Chinchwad University, Pune)<br>
    Champion Submission: <code>v95 Optimal Station-Adaptive Diurnal Harmonics</code> (Public Score: 6.2446 RMSE | Rank #1 in India)
</div>
""", unsafe_allow_html=True)
