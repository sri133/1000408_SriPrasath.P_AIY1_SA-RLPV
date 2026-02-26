import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="🚀 Rocket Mission Control", layout="wide")
st.title("🚀 Rocket Mission Analytics & Physics Simulator")

# ============================================================
# LOAD DATA
# ============================================================
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
    df["Year"] = df["Launch Date"].dt.year

    df["Mission Label"] = df.apply(
        lambda r: f"{r['Mission Name']} ({int(r['Mission Success (%)'])}%)",
        axis=1
    )

    return df

df = load_data()

# ============================================================
# SIDEBAR - ANALYTICS FILTERS
# ============================================================
st.sidebar.header("📊 Analytics Filters")

mission_types = st.sidebar.multiselect(
    "Mission Type",
    df["Mission Type"].unique(),
    default=df["Mission Type"].unique()
)

vehicles = st.sidebar.multiselect(
    "Launch Vehicle",
    df["Launch Vehicle"].unique(),
    default=df["Launch Vehicle"].unique()
)

min_success = st.sidebar.slider("Minimum Success %", 0, 100, 0)

filtered_df = df[
    (df["Mission Type"].isin(mission_types)) &
    (df["Launch Vehicle"].isin(vehicles)) &
    (df["Mission Success (%)"] >= min_success)
].copy()

if filtered_df.empty:
    st.warning("No missions match selected filters.")
    st.stop()

# ============================================================
# KPI METRICS
# ============================================================
st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Missions", len(filtered_df))
col2.metric("Avg Success Rate", f"{filtered_df['Mission Success (%)'].mean():.1f}%")
col3.metric("Avg Cost", f"${filtered_df['Mission Cost (billion USD)'].mean():.2f}B")
col4.metric("Avg Payload", f"{filtered_df['Payload Weight (tons)'].mean():.1f} tons")

# ============================================================
# ANALYTICS VISUALS
# ============================================================
st.header("📊 Mission Analytics")

# Scatter Plot
fig_scatter = px.scatter(
    filtered_df,
    x="Payload Weight (tons)",
    y="Fuel Consumption (tons)",
    color="Mission Type",
    hover_data=["Mission Name"],
    title="Payload vs Fuel Consumption",
    template="plotly_dark"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Bar Chart
avg_cost = filtered_df.groupby("Launch Vehicle")["Mission Cost (billion USD)"].mean().reset_index()
fig_bar = px.bar(
    avg_cost,
    x="Launch Vehicle",
    y="Mission Cost (billion USD)",
    title="Average Cost by Launch Vehicle",
    template="plotly_dark"
)
st.plotly_chart(fig_bar, use_container_width=True)

# Line Chart
success_trend = filtered_df.groupby(["Year", "Mission Type"])["Mission Success (%)"].mean().reset_index()
fig_line = px.line(
    success_trend,
    x="Year",
    y="Mission Success (%)",
    color="Mission Type",
    markers=True,
    title="Mission Success Over Time",
    template="plotly_dark"
)
st.plotly_chart(fig_line, use_container_width=True)

# Box Plot
fig_box = px.box(
    filtered_df,
    x="Mission Type",
    y="Mission Cost (billion USD)",
    color="Mission Type",
    title="Mission Cost Distribution",
    template="plotly_dark"
)
st.plotly_chart(fig_box, use_container_width=True)

# Correlation Box Plot
filtered_df["Success Category"] = filtered_df["Mission Success (%)"].apply(
    lambda x: "High Success (≥75%)" if x >= 75 else "Low Success (<75%)"
)

fig_corr_box = px.box(
    filtered_df,
    x="Success Category",
    y="Mission Cost (billion USD)",
    color="Success Category",
    title="Cost vs Success Category",
    template="plotly_dark"
)
st.plotly_chart(fig_corr_box, use_container_width=True)

# ============================================================
# SIMULATION SECTION
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🚀 Simulation Mode")

mode = st.sidebar.radio("Choose Mode", ["Historical Mission", "Manual Design"])

if mode == "Historical Mission":
    mission_label = st.sidebar.selectbox("Select Mission", df["Mission Label"])
    mission_name = mission_label.split(" (")[0]
    mission = df[df["Mission Name"] == mission_name].iloc[0]

    payload = mission["Payload Weight (tons)"] * 1000
    fuel_mass = mission["Fuel Consumption (tons)"] * 1000
    thrust = fuel_mass * 30
    mission_success_rate = mission["Mission Success (%)"]

else:
    payload = st.sidebar.slider("Payload (kg)", 1000, 200000, 50000)
    fuel_mass = st.sidebar.slider("Fuel Mass (kg)", 50000, 1000000, 300000)
    thrust = st.sidebar.slider("Thrust (N)", 1000000, 30000000, 8000000)
    mission_success_rate = 100

# ============================================================
# PHYSICS SIMULATION
# ============================================================
st.header("🛰 Rocket Launch Simulation")

def simulate(payload, fuel_mass, thrust):
    g = 9.81
    dt = 1
    steps = 150

    base_mass = 300000
    velocity = 0
    altitude = 0
    fuel = fuel_mass

    altitudes = []
    velocities = []

    for _ in range(steps):
        thrust_force = thrust if fuel > 0 else 0
        fuel -= fuel_mass / 120 if fuel > 0 else 0

        total_mass = base_mass + payload + fuel
        acceleration = (thrust_force - total_mass * g) / total_mass

        velocity += acceleration * dt
        altitude += velocity * dt

        if altitude < 0:
            altitude = 0
            velocity = 0

        altitudes.append(altitude)
        velocities.append(velocity)

    return altitudes, velocities

if st.button("🚀 Launch Mission"):

    altitudes, velocities = simulate(payload, fuel_mass, thrust)

    colA, colB = st.columns(2)

    with colA:
        fig_alt = px.area(
            x=list(range(len(altitudes))),
            y=altitudes,
            labels={"x": "Time (s)", "y": "Altitude (m)"},
            title="Altitude Profile",
            template="plotly_dark"
        )
        st.plotly_chart(fig_alt, use_container_width=True)

    with colB:
        fig_vel = px.line(
            x=list(range(len(velocities))),
            y=velocities,
            labels={"x": "Time (s)", "y": "Velocity (m/s)"},
            title="Velocity Profile",
            template="plotly_dark"
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    final_altitude = altitudes[-1]

    # Final Mission Success Logic
    if mission_success_rate >= 75 and final_altitude > 1000:
        st.success("✅ Mission Successful")
    else:
        st.error("💥 Mission Failed")
