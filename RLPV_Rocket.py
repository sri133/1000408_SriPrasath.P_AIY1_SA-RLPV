import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Rocket Launch Dashboard", layout="wide")
st.title("🚀 Rocket Launch Path Visualization & Mission Analytics")

# -----------------------------------------------------
# LOAD DATA (Stage 2)
# -----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("space_missions_dataset.csv")

    df["Launch Date"] = pd.to_datetime(df["Launch Date"], errors="coerce")

    numeric_cols = [
        "Distance from Earth (light-years)",
        "Mission Duration (years)",
        "Mission Cost (billion USD)",
        "Scientific Yield (points)",
        "Crew Size",
        "Mission Success (%)",
        "Fuel Consumption (tons)",
        "Payload Weight (tons)"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    return df

df = load_data()

# -----------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------
st.sidebar.header("🔎 Filters")

mission_type = st.sidebar.multiselect(
    "Mission Type",
    df["Mission Type"].unique(),
    default=df["Mission Type"].unique()
)

vehicle = st.sidebar.multiselect(
    "Launch Vehicle",
    df["Launch Vehicle"].unique(),
    default=df["Launch Vehicle"].unique()
)

success_filter = st.sidebar.slider(
    "Minimum Success Rate (%)",
    60, 100, 70
)

filtered_df = df[
    (df["Mission Type"].isin(mission_type)) &
    (df["Launch Vehicle"].isin(vehicle)) &
    (df["Mission Success (%)"] >= success_filter)
]

# =====================================================
# STAGE 4 — VISUALIZATIONS
# =====================================================
st.header("📊 Mission Visualizations")

col1, col2 = st.columns(2)

# 1️⃣ Plotly Scatter
with col1:
    fig1 = px.scatter(
        filtered_df,
        x="Payload Weight (tons)",
        y="Fuel Consumption (tons)",
        color="Mission Type",
        hover_data=["Mission Name"],
        title="Payload vs Fuel Consumption"
    )
    st.plotly_chart(fig1, use_container_width=True)

# 2️⃣ Seaborn Lineplot
with col2:
    fig2, ax2 = plt.subplots()
    sns.lineplot(
        data=filtered_df,
        x=filtered_df["Launch Date"].dt.year,
        y="Mission Success (%)",
        hue="Mission Type",
        ax=ax2
    )
    ax2.set_title("Mission Success Over Years")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Mission Success (%)")
    st.pyplot(fig2)

# 3️⃣ Plotly Bar Chart
avg_cost = filtered_df.groupby("Launch Vehicle")["Mission Cost (billion USD)"].mean().reset_index()

fig3 = px.bar(
    avg_cost,
    x="Launch Vehicle",
    y="Mission Cost (billion USD)",
    title="Average Mission Cost by Launch Vehicle"
)
st.plotly_chart(fig3, use_container_width=True)

col3, col4 = st.columns(2)

# 4️⃣ Seaborn Boxplot
with col3:
    fig4, ax4 = plt.subplots()
    sns.boxplot(
        data=filtered_df,
        x="Mission Type",
        y="Mission Cost (billion USD)",
        ax=ax4
    )
    ax4.set_title("Mission Cost Distribution by Type")
    st.pyplot(fig4)

# 5️⃣ Plotly Scatter (Distance vs Duration)
with col4:
    fig5 = px.scatter(
        filtered_df,
        x="Distance from Earth (light-years)",
        y="Mission Duration (years)",
        color="Mission Type",
        title="Distance vs Mission Duration"
    )
    st.plotly_chart(fig5, use_container_width=True)

# =====================================================
# STAGE 3 — ROCKET SIMULATION
# =====================================================
st.header("🛰Rocket Launch Simulation")

st.sidebar.header("⚙ Simulation Controls")

rocket_mass = st.sidebar.slider("Rocket Base Mass (kg)", 100000, 800000, 500000, step=50000)
fuel_mass = st.sidebar.slider("Fuel Mass (kg)", 100000, 600000, 300000, step=50000)
payload = st.sidebar.slider("Payload Weight (kg)", 10000, 100000, 50000, step=5000)
thrust = st.sidebar.slider("Thrust (N)", 1000000, 10000000, 7000000, step=500000)
drag_coeff = st.sidebar.slider("Drag Coefficient", 0.1, 1.0, 0.5)
time_steps = st.sidebar.slider("Simulation Time Steps", 50, 300, 200)

g = 9.81
air_density = 1.225
area = 10
dt = 1

velocity = 0
altitude = 0
fuel = fuel_mass

results = []

for t in range(time_steps):
    if fuel > 0:
        thrust_force = thrust
        fuel -= fuel_mass / time_steps
    else:
        thrust_force = 0

    total_mass = rocket_mass + fuel + payload
    drag = 0.5 * air_density * velocity**2 * drag_coeff * area
    net_force = thrust_force - (total_mass * g) - drag
    acceleration = net_force / total_mass

    velocity += acceleration * dt
    altitude += velocity * dt

    results.append([t, total_mass, acceleration, velocity, altitude])

sim_df = pd.DataFrame(
    results,
    columns=["Time (s)", "Mass (kg)", "Acceleration (m/s²)", "Velocity (m/s)", "Altitude (m)"]
)

col5, col6 = st.columns(2)

with col5:
    fig_alt = px.line(sim_df, x="Time (s)", y="Altitude (m)", title="Altitude Over Time")
    st.plotly_chart(fig_alt, use_container_width=True)

with col6:
    fig_vel = px.line(sim_df, x="Time (s)", y="Velocity (m/s)", title="Velocity Over Time")
    st.plotly_chart(fig_vel, use_container_width=True)

# -----------------------------------------------------
# FINAL METRICS
# -----------------------------------------------------
st.subheader("🚀 Simulation Results")

m1, m2, m3 = st.columns(3)
m1.metric("Final Altitude (m)", f"{sim_df['Altitude (m)'].iloc[-1]:,.2f}")
m2.metric("Max Velocity (m/s)", f"{sim_df['Velocity (m/s)'].max():,.2f}")
m3.metric("Final Mass (kg)", f"{sim_df['Mass (kg)'].iloc[-1]:,.2f}")


#----------------------------------------------------------
#additional features
#----------------------------------------------------------
# =====================================================
# 🌍 2D ORBITAL MECHANICS SIMULATION
# =====================================================
st.header("🌍 Orbital Simulation Around Earth")

import math

# Constants
G = 6.67430e-11
EARTH_MASS = 5.972e24
EARTH_RADIUS = 6.371e6

dt = 1
steps = 5000

# Initial position (on Earth's surface)
x = EARTH_RADIUS
y = 0

# Initial velocities
vx = 0
vy = 7800  # orbital speed approx

positions_x = []
positions_y = []

for _ in range(steps):

    r = math.sqrt(x**2 + y**2)

    # Gravitational acceleration
    ax = -G * EARTH_MASS * x / r**3
    ay = -G * EARTH_MASS * y / r**3

    # Update velocity
    vx += ax * dt
    vy += ay * dt

    # Update position
    x += vx * dt
    y += vy * dt

    positions_x.append(x)
    positions_y.append(y)

    # Stop if crashed
    if r < EARTH_RADIUS:
        break

orbit_df = pd.DataFrame({
    "x": positions_x,
    "y": positions_y
})

# Create Earth circle
theta = np.linspace(0, 2*np.pi, 500)
earth_x = EARTH_RADIUS * np.cos(theta)
earth_y = EARTH_RADIUS * np.sin(theta)

# Destination marker (example)
destination_x = EARTH_RADIUS + 400000  # 400 km altitude
destination_y = 0

# Plot
fig_orbit = go.Figure()

# Earth
fig_orbit.add_trace(
    go.Scatter(
        x=earth_x,
        y=earth_y,
        mode="lines",
        fill="toself",
        name="Earth"
    )
)

# Orbit path
fig_orbit.add_trace(
    go.Scatter(
        x=orbit_df["x"],
        y=orbit_df["y"],
        mode="lines",
        name="Orbit Path"
    )
)

# Rocket marker
fig_orbit.add_trace(
    go.Scatter(
        x=[orbit_df["x"].iloc[0]],
        y=[orbit_df["y"].iloc[0]],
        mode="markers+text",
        text=["🚀"],
        textposition="middle center",
        marker=dict(size=12),
        name="Rocket"
    )
)

# Destination
fig_orbit.add_trace(
    go.Scatter(
        x=[destination_x],
        y=[destination_y],
        mode="markers+text",
        text=["🎯 Destination"],
        textposition="top center",
        marker=dict(size=10),
        name="Target Orbit"
    )
)

# Animate
frames = [
    go.Frame(
        data=[
            go.Scatter(x=earth_x, y=earth_y),
            go.Scatter(x=orbit_df["x"][:k], y=orbit_df["y"][:k]),
            go.Scatter(
                x=[orbit_df["x"].iloc[k]],
                y=[orbit_df["y"].iloc[k]],
                mode="markers+text",
                text=["🚀"]
            ),
            go.Scatter(x=[destination_x], y=[destination_y])
        ]
    )
    for k in range(1, len(orbit_df), 10)
]

fig_orbit.frames = frames

fig_orbit.update_layout(
    xaxis=dict(scaleanchor="y", scaleratio=1),
    height=700,
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "Start Orbit 🚀",
            "method": "animate",
            "args": [None, {"frame": {"duration": 20, "redraw": False}}]
        }]
    }]
)

st.plotly_chart(fig_orbit, use_container_width=True)
