import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# -----------------------------------------------------
# 1. PAGE CONFIG & CYBER-TECH CSS
# -----------------------------------------------------
st.set_page_config(page_title="Rocket Mission Control", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* Glowing Metric Cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 2px solid #00f2ff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.3);
        transition: transform 0.3s ease;
    }
    [data-testid="stMetric"]:hover { transform: scale(1.02); box-shadow: 0 0 25px rgba(0, 242, 255, 0.5); }
    [data-testid="stMetricLabel"] { color: #00f2ff !important; font-weight: bold; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #ffffff !important; text-shadow: 0 0 5px #00f2ff; font-family: 'Courier New', monospace; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #11141a; border-right: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Rocket Path Visualization & Mission Analytics")

# -----------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------
@st.cache_data
def load_data():
    # Make sure 'space_missions_dataset.csv' is in your script folder
    df = pd.read_csv("space_missions_dataset.csv")
    df["Launch Date"] = pd.to_datetime(df["Launch Date"], errors="coerce")

    numeric_cols = [
        "Distance from Earth (light-years)", "Mission Duration (years)",
        "Mission Cost (billion USD)", "Scientific Yield (points)",
        "Crew Size", "Mission Success (%)",
        "Fuel Consumption (tons)", "Payload Weight (tons)"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=['Mission Name'])
    
    df["Mission Label"] = df.apply(
        lambda row: f"{row['Mission Name']} ({row['Mission Success (%)']}%)",
        axis=1
    )
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# -----------------------------------------------------
# 3. SIDEBAR: FILTERS & CONTROL MODES
# -----------------------------------------------------
st.sidebar.header("🔎 Dashboard Filters")
all_types = df["Mission Type"].unique().tolist()
all_vehicles = df["Launch Vehicle"].unique().tolist()

selected_types = st.sidebar.multiselect("Mission Type", all_types, default=all_types)
selected_vehicles = st.sidebar.multiselect("Launch Vehicle", all_vehicles, default=all_vehicles)
success_filter = st.sidebar.slider("Min Success Rate (%)", 0, 100, 60)

# Create Filtered DF for Analytics Section
filtered_df = df[
    (df["Mission Type"].isin(selected_types)) & 
    (df["Launch Vehicle"].isin(selected_vehicles)) & 
    (df["Mission Success (%)"] >= success_filter)
].copy()

st.sidebar.divider()
st.sidebar.header("🕹️ Simulator Control")
mode = st.sidebar.radio("Input Mode:", ["Historical Mission", "Manual Design"])

if mode == "Historical Mission":
    st.sidebar.subheader("🚀 Mission Selection")
    mission_label = st.sidebar.selectbox("Choose Mission", df["Mission Label"])
    mission_name = mission_label.split(" (")[0]
    m_row = df[df["Mission Name"] == mission_name].iloc[0]
    
    # Auto-load mission specs
    payload_kg = m_row["Payload Weight (tons)"] * 1000
    fuel_kg = m_row["Fuel Consumption (tons)"] * 1000
    thrust_n = fuel_kg * 35 # Derived thrust approximation
    target_name = m_row["Target Name"]
    distance_ly = m_row["Distance from Earth (light-years)"]
    hist_success = m_row["Mission Success (%)"]
else:
    st.sidebar.subheader("🛠️ Custom Rocket Specs")
    target_name = st.sidebar.text_input("Target Planet Name", "Proxima b")
    payload_kg = st.sidebar.slider("Payload (kg)", 1000, 150000, 50000)
    fuel_kg = st.sidebar.slider("Fuel Mass (kg)", 50000, 1000000, 400000)
    thrust_n = st.sidebar.slider("Thrust (N)", 1000000, 20000000, 8000000)
    distance_ly = 1.0 # Default for manual
    hist_success = 100

# -------------------------------------------------
# 4. DYNAMIC PHYSICS ENGINE
# -------------------------------------------------
def run_simulation(p_load, f_mass, t_force):
    g, air_density, area, drag_coeff, dt = 9.81, 1.225, 12, 0.45, 1
    v, alt, f_rem = 0, 0, f_mass
    sim_data = []
    
    for t in range(200):
        current_thrust = t_force if f_rem > 0 else 0
        f_rem -= f_mass / 100 if f_rem > 0 else 0 # Fuel depletion
        
        m_total = 400000 + f_rem + p_load
        # Drag reduces as altitude increases
        drag = 0.5 * air_density * (v**2) * drag_coeff * area if alt < 60000 else 0
        accel = (current_thrust - (m_total * g) - drag) / m_total
        
        v += accel * dt
        alt += v * dt
        if alt < 0: alt, v = 0, 0
        
        sim_data.append([t, m_total, accel, v, alt])
    return pd.DataFrame(sim_data, columns=["Time", "Mass", "Acc", "Vel", "Alt"])

# Recalculate simulation instantly based on current Sidebar state
sim_df = run_simulation(payload_kg, fuel_kg, thrust_n)

# -----------------------------------------------------
# 5. ANALYTICS SECTION (Vibrant Plotly Charts)
# -----------------------------------------------------
st.header("📊 Mission Analytics Dashboard")
if filtered_df.empty:
    st.warning("No data matches current filters.")
else:
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(filtered_df, x="Payload Weight (tons)", y="Fuel Consumption (tons)", 
                         color="Mission Type", size="Mission Cost (billion USD)", 
                         hover_name="Mission Name", title="Payload vs Fuel Efficiency", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        line_data = filtered_df.copy()
        line_data['Year'] = line_data['Launch Date'].dt.year
        success_trend = line_data.groupby(['Year', 'Mission Type'])['Mission Success (%)'].mean().reset_index()
        fig2 = px.line(success_trend, x="Year", y="Mission Success (%)", color="Mission Type",
                      title="Success Trends Over Time", template="plotly_dark", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# 6. SIMULATOR SECTION
# -------------------------------------------------
st.divider()
st.header(f"🛰️ Rocket Simulator: Destination {target_name}")

# Metrics with Cyber-Glow styling
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Payload", f"{payload_kg:,.0f} kg")
m2.metric("Total Fuel", f"{fuel_kg:,.0f} kg")
m3.metric("Peak Velocity", f"{sim_df['Vel'].max():,.1f} m/s")
m4.metric("Success Record", f"{hist_success}%")

col3, col4 = st.columns(2)
with col3:
    fig_alt = px.area(sim_results:=sim_df, x="Time", y="Alt", title="Altitude Over Time (m)", 
                     template="plotly_dark", color_discrete_sequence=['#00f2ff'])
    st.plotly_chart(fig_alt, use_container_width=True)

with col4:
    fig_vel = px.line(sim_df, x="Time", y="Vel", title="Velocity Profile (m/s)", 
                     template="plotly_dark", color_discrete_sequence=['#ff00ff'])
    st.plotly_chart(fig_vel, use_container_width=True)

# -------------------------------------------------
# 7. ANIMATED LAUNCH TRAJECTORY
# -------------------------------------------------
if st.button("🚀 INITIATE LAUNCH SEQUENCE"):
    ani_placeholder = st.empty()
    # Dynamic planet distance based on data
    planet_dist = 30000 + (distance_ly * 5000)
    theta = np.linspace(0, 2*np.pi, 100)
    earth_x, earth_y = 6371 * np.cos(theta), 6371 * np.sin(theta)

    for i in range(0, len(sim_df), 4):
        fig_sim = go.Figure()
        
        # Earth
        fig_sim.add_trace(go.Scatter(x=earth_x, y=earth_y, fill="toself", name="Earth", line=dict(color="#1f77b4")))
        # Target Planet
        fig_sim.add_trace(go.Scatter(x=[planet_dist], y=[0], mode="markers+text", 
                                     marker=dict(size=30, color="orange"), text=[target_name], textposition="top center"))
        # Rocket position
        prog = i / len(sim_df)
        rx = prog * planet_dist
        ry = (sim_df['Alt'].iloc[i] / 10) if prog < 0.2 else (np.sin(prog * np.pi) * 8000)
        
        fig_sim.add_trace(go.Scatter(x=[rx], y=[ry], mode="markers+text", 
                                     marker=dict(size=18, color="#00f2ff", symbol="triangle-up", line=dict(width=2, color="white")),
                                     text=["🚀"], textposition="top center"))

        fig_sim.update_layout(
            template="plotly_dark", height=500, showlegend=False,
            xaxis=dict(range=[-10000, planet_dist + 15000], showgrid=False, zeroline=False),
            yaxis=dict(range=[-15000, 25000], showgrid=False, zeroline=False),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        ani_placeholder.plotly_chart(fig_sim, use_container_width=True, key=f"sim_frame_{i}")
        time.sleep(0.01)

    if sim_df['Alt'].iloc[-1] > 10000:
        st.success(f"STATIONARY ORBIT ACHIEVED: Mission to {target_name} is a Success!")
    else:
        st.error("MISSION CRITICAL FAILURE: Rocket failed to reach exit velocity.")
