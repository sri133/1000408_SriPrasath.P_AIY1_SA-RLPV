import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Rocket Mission Analytics", layout="wide")
st.title("🚀 Rocket Mission Analytics & Simulator")

# -----------------------------------------------------
# LOAD DATA
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
    df["Year"] = df["Launch Date"].dt.year

    df["Mission Label"] = df.apply(
        lambda r: f"{r['Mission Name']} ({r['Mission Success (%)']}%)",
        axis=1
    )

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

success_filter = st.sidebar.slider("Minimum Success Rate (%)", 0, 100, 50)

filtered_df = df[
    (df["Mission Type"].isin(mission_type)) &
    (df["Launch Vehicle"].isin(vehicle)) &
    (df["Mission Success (%)"] >= success_filter)
].copy()

if filtered_df.empty:
    st.warning("No missions match your filters.")
    st.stop()

# =====================================================
# SECTION 1 — RESOURCE ANALYSIS
# =====================================================
st.header("📊 Resource & Cost Analysis")

col1, col2 = st.columns(2)

# 1️⃣ Scatter Plot
with col1:
    fig_scatter = px.scatter(
        filtered_df,
        x="Payload Weight (tons)",
        y="Fuel Consumption (tons)",
        color="Mission Type",
        hover_data=["Mission Name"],
        template="plotly_dark",
        title="Payload vs Fuel Consumption"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# 2️⃣ Bar Chart
with col2:
    avg_cost = filtered_df.groupby("Launch Vehicle")["Mission Cost (billion USD)"].mean().reset_index()

    fig_bar = px.bar(
        avg_cost,
        x="Launch Vehicle",
        y="Mission Cost (billion USD)",
        template="plotly_dark",
        title="Average Mission Cost by Launch Vehicle"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# 3️⃣ Box Plot
fig_box = px.box(
    filtered_df,
    x="Mission Type",
    y="Mission Cost (billion USD)",
    color="Mission Type",
    template="plotly_dark",
    title="Mission Cost Distribution by Type"
)
st.plotly_chart(fig_box, use_container_width=True)

# =====================================================
# SECTION 2 — PERFORMANCE ANALYSIS
# =====================================================
st.header("📈 Performance & Trends")

col3, col4 = st.columns(2)

# 4️⃣ Line Plot
with col3:
    success_trend = filtered_df.groupby(["Year", "Mission Type"])["Mission Success (%)"].mean().reset_index()

    fig_line = px.line(
        success_trend,
        x="Year",
        y="Mission Success (%)",
        color="Mission Type",
        markers=True,
        template="plotly_dark",
        title="Mission Success Trends Over Time"
    )
    st.plotly_chart(fig_line, use_container_width=True)

# 5️⃣ Extra Scatter
with col4:
    fig_scatter2 = px.scatter(
        filtered_df,
        x="Distance from Earth (light-years)",
        y="Mission Duration (years)",
        color="Mission Type",
        template="plotly_dark",
        title="Distance vs Mission Duration"
    )
    st.plotly_chart(fig_scatter2, use_container_width=True)

# =====================================================
# SECTION 3 — CORRELATION BOX ANALYSIS
# =====================================================
st.header("📦 Correlation Box Analysis")

filtered_df["Success Category"] = filtered_df["Mission Success (%)"].apply(
    lambda x: "High Success (≥75%)" if x >= 75 else "Lower Success (<75%)"
)

fig_corr_box = px.box(
    filtered_df,
    x="Success Category",
    y="Mission Cost (billion USD)",
    color="Success Category",
    template="plotly_dark",
    title="Mission Cost Distribution by Success Category"
)

st.plotly_chart(fig_corr_box, use_container_width=True)

# =====================================================
# SECTION 4 — ROCKET SIMULATOR
# =====================================================
st.header("🛰 Data-Driven Rocket Simulator")

mission_label = st.selectbox("Select Mission for Simulation", df["Mission Label"])
mission_name = mission_label.split(" (")[0]
mission_row = df[df["Mission Name"] == mission_name].iloc[0]

payload = mission_row["Payload Weight (tons)"] * 1000
fuel_mass = mission_row["Fuel Consumption (tons)"] * 1000
success_rate = mission_row["Mission Success (%)"]

base_mass = 400000
thrust = fuel_mass * 25

g = 9.81
dt = 1
steps = 150

velocity = 0
altitude = 0
fuel = fuel_mass

altitudes = []
velocities = []

for i in range(steps):
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

if st.button("🚀 Launch Mission"):

    col5, col6 = st.columns(2)

    with col5:
        fig_alt = px.area(
            x=list(range(steps)),
            y=altitudes,
            template="plotly_dark",
            labels={"x": "Time", "y": "Altitude (m)"},
            title="Altitude Over Time"
        )
        st.plotly_chart(fig_alt, use_container_width=True)

    with col6:
        fig_vel = px.line(
            x=list(range(steps)),
            y=velocities,
            template="plotly_dark",
            labels={"x": "Time", "y": "Velocity (m/s)"},
            title="Velocity Profile"
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    if success_rate >= 75 and altitudes[-1] > 1000:
        st.success("✅ Mission Successful")
    else:
        st.error("💥 Mission Failed")
