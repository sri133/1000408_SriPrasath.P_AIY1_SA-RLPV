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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Rocket Mission Simulator", layout="wide")
st.title("🚀 Data-Driven Rocket Mission Simulator")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
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

# -------------------------------------------------
# SIDEBAR MISSION SELECTION
# -------------------------------------------------
st.sidebar.header("🛰 Mission Selection")

# Add mission label with success %
df["Mission Label"] = df.apply(
    lambda row: f"{row['Mission Name']} ({row['Mission Success (%)']}%)",
    axis=1
)

mission_label = st.sidebar.selectbox("Choose Mission", df["Mission Label"])
mission_name = mission_label.split(" (")[0]

mission_row = df[df["Mission Name"] == mission_name].iloc[0]

# -------------------------------------------------
# EXTRACT DATA FROM SELECTED MISSION
# -------------------------------------------------
payload = mission_row["Payload Weight (tons)"] * 1000
fuel = mission_row["Fuel Consumption (tons)"] * 1000
success_rate = mission_row["Mission Success (%)"]
distance = mission_row["Distance from Earth (light-years)"]

base_mass = 400000
total_mass = base_mass + payload

# Thrust derived from fuel amount
thrust = fuel * 30
initial_velocity = thrust / total_mass

orbit_velocity_required = 7800
planet_distance = 20000 + (distance * 1200)

# -------------------------------------------------
# DISPLAY MISSION INFO
# -------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Payload (kg)", f"{payload:,.0f}")
col2.metric("Fuel (kg)", f"{fuel:,.0f}")
col3.metric("Success Rate", f"{success_rate}%")

if success_rate >= 70:
    st.success("🟢 Historically Successful Mission")
else:
    st.warning("🔴 Low Historical Success Mission")

# -------------------------------------------------
# SIMULATION LOGIC
# -------------------------------------------------
frames = 200
earth_radius = 6371
orbit_radius = earth_radius + 800

trajectory_x = []
trajectory_y = []

orbit_achieved = False

for i in range(frames):
    t = i / frames

    # Launch Phase
    if t < 0.2:
        x = 0
        y = earth_radius + t * initial_velocity * 2

    # Orbit Phase
    elif t < 0.5:
        if initial_velocity > orbit_velocity_required:
            orbit_achieved = True
            angle = (t - 0.2) * 12
            x = orbit_radius * np.cos(angle)
            y = orbit_radius * np.sin(angle)
        else:
            x = 0
            y = earth_radius - (t - 0.2) * 5000

    # Transfer Phase
    else:
        if orbit_achieved:
            progress = (t - 0.5) / 0.5
            x = orbit_radius + progress * (planet_distance - orbit_radius)
            y = orbit_radius * np.sin(progress * np.pi)
        else:
            x = 0
            y = earth_radius - 2000

    trajectory_x.append(x)
    trajectory_y.append(y)

# -------------------------------------------------
# MISSION SUCCESS LOGIC
# -------------------------------------------------
if not orbit_achieved:
    mission_failed = True
else:
    mission_failed = np.random.rand() > (success_rate / 100)

# -------------------------------------------------
# LAUNCH BUTTON
# -------------------------------------------------
if st.button("Launch Mission"):

    chart = st.empty()

    theta = np.linspace(0, 2*np.pi, 200)
    earth_x = earth_radius * np.cos(theta)
    earth_y = earth_radius * np.sin(theta)

    for i in range(len(trajectory_x)):

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
            mode="markers",
            marker=dict(size=18)
        ))

        fig.add_trace(go.Scatter(
            x=[trajectory_x[i]],
            y=[trajectory_y[i]],
            mode="markers",
            marker=dict(size=10)
        ))

        fig.update_layout(
            xaxis=dict(range=[-planet_distance-5000, planet_distance+5000], visible=False),
            yaxis=dict(range=[-planet_distance-5000, planet_distance+5000], visible=False),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        chart.plotly_chart(fig, use_container_width=True, key=f"sim_{i}")
        time.sleep(0.02)

        chart.empty()  # clear previous frame
    # -------------------------------------------------
    # FINAL STATUS
    # -------------------------------------------------
    st.subheader("🧠 Mission Outcome")

    if mission_failed:
        st.error("💥 Mission Failed")
    else:
        st.success("🛰 Mission Successful – Orbit Achieved & Transfer Complete")

