import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Rocket Launch Dashboard", layout="wide")

# Custom CSS for Vibrant, Glowing Metric Cards
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Glowing Metric Cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 2px solid #00f2ff; /* Neon Cyan Border */
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.4); /* Outer Glow */
        transition: transform 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(0, 242, 255, 0.6);
    }

    /* Target the Label (Payload, Fuel, etc.) */
    [data-testid="stMetricLabel"] {
        color: #00f2ff !important;
        font-weight: bold !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Target the Value (The big numbers) */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Rocket Path Visualization & Mission Analytics")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    # Ensure 'space_missions_dataset.csv' is in the same directory
    df = pd.read_csv("space_missions_dataset.csv")
    df["Launch Date"] = pd.to_datetime(df["Launch Date"], errors="coerce")

    numeric_cols = [
        "Distance from Earth (light-years)", "Mission Duration (years)",
        "Mission Cost (billion USD)", "Scientific Yield (points)",
        "Crew Size", "Mission Success (%)",
        "Fuel Consumption (tons)", "Payload Weight (tons)"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    
    # Create labels for the simulator dropdown
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
# SIDEBAR FILTERS (Defined BEFORE they are used)
# -----------------------------------------------------
st.sidebar.header("🔎 Analytics Filters")

# Provide safe defaults to prevent empty dataframes
all_mission_types = df["Mission Type"].unique().tolist()
all_vehicles = df["Launch Vehicle"].unique().tolist()

mission_type = st.sidebar.multiselect(
    "Mission Type",
    options=all_mission_types,
    default=all_mission_types
)

vehicle = st.sidebar.multiselect(
    "Launch Vehicle",
    options=all_vehicles,
    default=all_vehicles
)

success_filter = st.sidebar.slider("Min Success Rate (%)", 0, 100, 70)

# -----------------------------------------------------
# CREATE FILTERED DATAFRAME
# -----------------------------------------------------
# We define filtered_df here so it's available for all charts below
filtered_df = df[
    (df["Mission Type"].isin(mission_type)) &
    (df["Launch Vehicle"].isin(vehicle)) &
    (df["Mission Success (%)"] >= success_filter)
].copy()

# Fallback check: if the user filters everything out, show a warning instead of a NameError
if filtered_df.empty:
    st.warning("No missions match these filters. Please adjust the sidebar settings.")
    st.stop()
# -------------------------------------------------
# 3. SIDEBAR: MODE SELECTION & CONTROLS
# -------------------------------------------------
st.sidebar.header("🕹️ Control Mode")
mode = st.sidebar.radio("Select Input Mode:", ["Historical Mission", "Manual Design"])

if mode == "Historical Mission":
    st.sidebar.subheader("🚀 Mission Selection")
    mission_label = st.sidebar.selectbox("Choose Mission", df["Mission Label"])
    mission_name = mission_label.split(" (")[0]
    m_row = df[df["Mission Name"] == mission_name].iloc[0]
    
    # Lock values to mission data
    payload = m_row["Payload Weight (tons)"] * 1000
    fuel = m_row["Fuel Consumption (tons)"] * 1000
    thrust = fuel * 30 
    target_name = "Target Planet" 
    distance = m_row["Distance from Earth (light-years)"]
    success_rate = m_row["Mission Success (%)"]

else:
    st.sidebar.subheader("🛠️ Custom Rocket Specs")
    target_name = st.sidebar.text_input("Target Planet Name", "Mars")
    payload = st.sidebar.slider("Payload (kg)", 1000, 150000, 50000)
    fuel = st.sidebar.slider("Fuel Mass (kg)", 50000, 1000000, 300000)
    thrust = st.sidebar.slider("Thrust (N)", 1000000, 20000000, 7000000)
    # Manual mode distance/success defaults
    distance = 0.5 
    success_rate = 100

# -------------------------------------------------
# 4. DYNAMIC PHYSICS CALCULATION
# -------------------------------------------------
# This block now runs every time a slider moves
base_mass = 400000
g, air_density, area, drag_coeff, dt = 9.81, 1.225, 10, 0.5, 1
time_steps = 200

def run_simulation(p_load, f_mass, t_force):
    v, alt, f_rem = 0, 0, f_mass
    sim_data = []
    for t in range(time_steps):
        current_thrust = t_force if f_rem > 0 else 0
        f_rem -= f_mass / 100 if f_rem > 0 else 0
        
        m_total = base_mass + f_rem + p_load
        drag = 0.5 * air_density * (v**2) * drag_coeff * area
        accel = (current_thrust - (m_total * g) - drag) / m_total
        
        v += accel * dt
        alt += v * dt
        if alt < 0: alt, v = 0, 0
        sim_data.append([t, m_total, accel, v, alt])
    return pd.DataFrame(sim_data, columns=["Time", "Mass", "Acc", "Vel", "Alt"])

# Generate fresh data based on current UI state
sim_df = run_simulation(payload, fuel, thrust)

# -------------------------------------------------
# 5. UPDATED VISUALS
# -------------------------------------------------
# Now fig_alt and fig_vel will automatically use the NEW sim_df
col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(px.line(sim_df, x="Time", y="Alt", title=f"Altitude to {target_name}", 
                            template="plotly_dark", color_discrete_sequence=['#00f2ff']), use_container_width=True)
with col6:
    st.plotly_chart(px.line(sim_df, x="Time", y="Vel", title="Velocity Profile", 
                            template="plotly_dark", color_discrete_sequence=['#ff00ff']), use_container_width=True)

# -----------------------------------------------------
# ANALYTICS SECTION
# -----------------------------------------------------
st.header("📊 Mission Visualizations")

col1, col2 = st.columns(2)

with col1:
    # 1. Payload vs Fuel (Scatter)
    fig1 = px.scatter(
        filtered_df,
        x="Payload Weight (tons)",
        y="Fuel Consumption (tons)",
        color="Mission Type",
        hover_data=["Mission Name"],
        title="Payload vs Fuel Consumption",
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # 2. Success Over Years (Line - Replacing Seaborn)
    # Grouping by year and mission type for a cleaner line plot
    line_data = filtered_df.copy()
    line_data['Year'] = line_data['Launch Date'].dt.year
    line_data = line_data.groupby(['Year', 'Mission Type'])['Mission Success (%)'].mean().reset_index()
    
    fig2 = px.line(
        line_data,
        x="Year",
        y="Mission Success (%)",
        color="Mission Type",
        title="Average Mission Success Over Years",
        template="plotly_white",
        markers=True
    )
    st.plotly_chart(fig2, use_container_width=True)

# 3. Avg Cost by Vehicle (Bar)
avg_cost = filtered_df.groupby("Launch Vehicle")["Mission Cost (billion USD)"].mean().reset_index()
fig3 = px.bar(
    avg_cost,
    x="Launch Vehicle",
    y="Mission Cost (billion USD)",
    title="Average Mission Cost by Launch Vehicle",
    template="plotly_white",
    color_discrete_sequence=['#1f77b4']
)
st.plotly_chart(fig3, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # 4. Mission Cost Distribution (Boxplot - Replacing Seaborn)
    fig4 = px.box(
        filtered_df,
        x="Mission Type",
        y="Mission Cost (billion USD)",
        color="Mission Type",
        title="Mission Cost Distribution by Type",
        template="plotly_white"
    )
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    # 5. Distance vs Duration (Scatter)
    fig5 = px.scatter(
        filtered_df,
        x="Distance from Earth (light-years)",
        y="Mission Duration (years)",
        color="Mission Type",
        size="Mission Cost (billion USD)",
        title="Distance vs Mission Duration (Size=Cost)",
        template="plotly_white"
    )
    st.plotly_chart(fig5, use_container_width=True)

# -----------------------------------------------------
# ROCKET SIMULATION SECTION
# -----------------------------------------------------
st.divider()
st.header("🛰 Data-Driven Rocket Mission Simulator")

# Mission Selection for Simulator
st.sidebar.markdown("---")
st.sidebar.header("🚀 Simulator Selection")
mission_label = st.sidebar.selectbox("Choose Mission Profile", df["Mission Label"])
mission_name = mission_label.split(" (")[0]
mission_row = df[df["Mission Name"] == mission_name].iloc[0]

# This part of your code remains the same but will now look much different!
m1, m2, m3 = st.columns(3)
m1.metric("Payload (kg)", f"{mission_row['Payload Weight (tons)']*1000:,.0f}")
m2.metric("Fuel (kg)", f"{mission_row['Fuel Consumption (tons)']*1000:,.0f}")
m3.metric("Success Record", f"{mission_row['Mission Success (%)']}%")

# Physics Simulation Logic
payload = mission_row["Payload Weight (tons)"] * 1000
fuel_mass = mission_row["Fuel Consumption (tons)"] * 1000
base_mass = 400000
thrust = fuel_mass * 30 
g, air_density, area, drag_coeff, dt = 9.81, 1.225, 10, 0.5, 1
time_steps = 150

velocity, altitude, fuel = 0, 0, fuel_mass
results = []

for t in range(time_steps):
    thrust_force = thrust if fuel > 0 else 0
    fuel -= fuel_mass / 50 if fuel > 0 else 0
    
    total_mass = base_mass + fuel + payload
    drag = 0.5 * air_density * (velocity**2) * drag_coeff * area
    net_force = thrust_force - (total_mass * g) - drag
    acceleration = net_force / total_mass
    
    velocity += acceleration * dt
    altitude += velocity * dt
    if altitude < 0: altitude, velocity = 0, 0
    
    results.append([t, total_mass, acceleration, velocity, altitude])

sim_df = pd.DataFrame(results, columns=["Time", "Mass", "Acc", "Vel", "Alt"])

# Animation Placeholder
if st.button("▶ Start Mission Simulation"):
    chart_placeholder = st.empty()
    
    # Pre-calculate Earth and Planet Positions
    planet_dist = 25000 + (mission_row["Distance from Earth (light-years)"] * 1000)
    theta = np.linspace(0, 2*np.pi, 100)
    earth_x, earth_y = 6371 * np.cos(theta), 6371 * np.sin(theta)

    for i in range(0, len(sim_df), 2): # Step by 2 for speed
        fig_sim = go.Figure()
        
        # Earth
        fig_sim.add_trace(go.Scatter(x=earth_x, y=earth_y, fill="toself", name="Earth", line=dict(color="#1f77b4")))
        # Target
        fig_sim.add_trace(go.Scatter(x=[planet_dist], y=[0], mode="markers", marker=dict(size=25, color="orange"), name="Target"))
        # Rocket Path
        prog = i / len(sim_df)
        rx = prog * planet_dist
        ry = np.sin(prog * np.pi) * 5000 if prog > 0.1 else sim_df['Alt'].iloc[i]
        
        fig_sim.add_trace(go.Scatter(
    x=[rx], 
    y=[ry], 
    mode="markers+text", 
    marker=dict(
        size=18, 
        color="#00f2ff", # Using that glowing cyan we set earlier
        symbol="triangle-up", # Changed from "rocket" to fix the error
        line=dict(width=2, color="white") # Added a small outline for 'classic' feel
    ),
    text=["🚀"], # We can use the emoji in the text field instead!
    textposition="top center"
))

        fig_sim.update_layout(
            template="plotly_dark", height=500, showlegend=False,
            xaxis=dict(range=[-8000, planet_dist + 5000], showgrid=False, zeroline=False),
            yaxis=dict(range=[-10000, 15000], showgrid=False, zeroline=False),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        chart_placeholder.plotly_chart(fig_sim, use_container_width=True, key=f"sim_{i}")
        time.sleep(0.01)

    # Final Outcome
    if sim_df['Alt'].iloc[-1] > 1000 and mission_row["Mission Success (%)"] > 50:
        st.success("✅ Mission Successful: Objective Reached.")
    else:
        st.error("💥 Mission Failure: Insufficient trajectory or mechanical failure.")

# -----------------------------------------------------
# FINAL PHYSICS PLOTS
# -----------------------------------------------------
col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(px.line(sim_df, x="Time", y="Alt", title="Altitude Profile", template="plotly_white"), use_container_width=True)
with col6:
    st.plotly_chart(px.line(sim_df, x="Time", y="Vel", title="Velocity Profile", template="plotly_white"), use_container_width=True)






