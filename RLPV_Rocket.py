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

# -----------------------------------------------------
# CUSTOM DARK UI
# -----------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0E1117; }

[data-testid="stMetric"] {
    background-color: #161b22;
    border: 2px solid #00f2ff;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 15px rgba(0, 242, 255, 0.4);
}

[data-testid="stMetricLabel"] {
    color: #00f2ff !important;
    font-weight: bold !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Rocket Path Visualization & Mission Analytics")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
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

    df["Mission Label"] = df.apply(
        lambda r: f"{r['Mission Name']} ({r['Mission Success (%)']}%)",
        axis=1
    )
    return df

df = load_data()

# -----------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------
st.sidebar.header("🔎 Analytics Filters")

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

success_filter = st.sidebar.slider("Min Success Rate (%)", 0, 100, 50)

filtered_df = df[
    (df["Mission Type"].isin(mission_type)) &
    (df["Launch Vehicle"].isin(vehicle)) &
    (df["Mission Success (%)"] >= success_filter)
]

if filtered_df.empty:
    st.warning("No missions match filters.")
    st.stop()

# -----------------------------------------------------
# ANALYTICS SECTION
# -----------------------------------------------------
st.header("📊 Mission Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        filtered_df,
        x="Payload Weight (tons)",
        y="Fuel Consumption (tons)",
        color="Mission Type",
        hover_data=["Mission Name"],
        template="plotly_dark",
        title="Payload vs Fuel"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    line_data = filtered_df.copy()
    line_data["Year"] = line_data["Launch Date"].dt.year
    line_data = line_data.groupby(["Year", "Mission Type"])["Mission Success (%)"].mean().reset_index()

    fig2 = px.line(
        line_data,
        x="Year",
        y="Mission Success (%)",
        color="Mission Type",
        markers=True,
        template="plotly_dark",
        title="Average Success Over Years"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------
# SIMULATOR SECTION
# -----------------------------------------------------
st.divider()
st.header("🛰 Data-Driven Rocket Mission Simulator")

st.sidebar.markdown("---")
st.sidebar.header("🚀 Simulator")

sim_label = st.sidebar.selectbox(
    "Select Mission",
    df["Mission Label"],
    key="sim_select"
)

sim_name = sim_label.split(" (")[0]
sim_row = df[df["Mission Name"] == sim_name].iloc[0]

payload = sim_row["Payload Weight (tons)"] * 1000
fuel_mass = sim_row["Fuel Consumption (tons)"] * 1000
success_rate = sim_row["Mission Success (%)"]
distance = sim_row["Distance from Earth (light-years)"]

base_mass = 400000
thrust = fuel_mass * 25

m1, m2, m3 = st.columns(3)
m1.metric("Payload (kg)", f"{payload:,.0f}")
m2.metric("Fuel (kg)", f"{fuel_mass:,.0f}")
m3.metric("Historical Success", f"{success_rate}%")

# -----------------------------------------------------
# SIMULATION CALCULATION
# -----------------------------------------------------
g = 9.81
dt = 1
steps = 160

velocity = 0
altitude = 0
fuel = fuel_mass

earth_radius = 6371
orbit_radius = earth_radius + 1000
planet_distance = 25000 + (distance * 1500)

trajectory_x = []
trajectory_y = []
orbit_achieved = False

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

    if altitude < orbit_radius:
        x = 0
        y = earth_radius + altitude

    elif velocity >= 7800:
        orbit_achieved = True
        angle = (i - 40) * 0.08
        x = orbit_radius * np.cos(angle)
        y = orbit_radius * np.sin(angle)

    elif orbit_achieved:
        progress = i / steps
        x = orbit_radius + progress * (planet_distance - orbit_radius)
        y = orbit_radius * np.sin(progress * np.pi)

    else:
        x = 0
        y = earth_radius - 2000

    trajectory_x.append(x)
    trajectory_y.append(y)

mission_failed = (not orbit_achieved) or (np.random.rand() > success_rate / 100)

# -----------------------------------------------------
# ANIMATION
# -----------------------------------------------------
if st.button("▶ Launch Mission"):

    theta = np.linspace(0, 2*np.pi, 200)
    earth_x = earth_radius * np.cos(theta)
    earth_y = earth_radius * np.sin(theta)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=earth_x,
        y=earth_y,
        mode="lines",
        fill="toself"
    ))

    fig.add_trace(go.Scatter(
        x=[planet_distance],
        y=[0],
        mode="markers"
    ))

    fig.add_trace(go.Scatter(
        x=[trajectory_x[0]],
        y=[trajectory_y[0]],
        mode="markers",
        marker=dict(size=14, symbol="triangle-up")
    ))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        showlegend=False,
        xaxis=dict(range=[-planet_distance, planet_distance+5000], visible=False),
        yaxis=dict(range=[-planet_distance, planet_distance], visible=False),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    placeholder = st.empty()

    for i in range(len(trajectory_x)):
        fig.data[2].x = [trajectory_x[i]]
        fig.data[2].y = [trajectory_y[i]]
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.02)

    st.subheader("Mission Outcome")

    if mission_failed:
        st.error("💥 Mission Failed")
    else:
        st.success("🛰 Mission Successful – Orbit Achieved")
