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
# 🚀 SMOOTH NASA-STYLE ANIMATION (NO FLICKER)
# =====================================================
st.header("🚀 Rocket Launch Simulation (Smooth Animation)")

ORBIT_VELOCITY = 7800

frames = []
orbit_reached = False

max_alt = sim_df["Altitude (m)"].max()
max_time = sim_df["Time (s)"].max()

def atmosphere_color(alt):
    if alt < 10000:
        return "#87CEEB"
    elif alt < 50000:
        return "#4B79A1"
    elif alt < 100000:
        return "#1B3C59"
    else:
        return "#000000"

for i in range(len(sim_df)):
    current_alt = sim_df["Altitude (m)"].iloc[i]
    current_vel = sim_df["Velocity (m/s)"].iloc[i]
    current_time = sim_df["Time (s)"].iloc[i]

    bg_color = atmosphere_color(current_alt)

    flame_visible = sim_df["Mass (kg)"].iloc[i] > rocket_mass + payload

    frame_data = [
        go.Scatter(
            x=sim_df["Time (s)"][:i+1],
            y=sim_df["Altitude (m)"][:i+1],
            mode="lines",
            line=dict(width=3),
        ),
        go.Scatter(
            x=[current_time],
            y=[current_alt],
            mode="text",
            text=["🚀"],
            textfont=dict(size=28),
        )
    ]

    if flame_visible:
        frame_data.append(
            go.Scatter(
                x=[current_time],
                y=[current_alt - max_alt*0.03],
                mode="text",
                text=["🔥"],
                textfont=dict(size=22),
            )
        )

    frames.append(
        go.Frame(
            data=frame_data,
            layout=go.Layout(
                plot_bgcolor=bg_color,
                paper_bgcolor=bg_color
            )
        )
    )

    if current_vel >= ORBIT_VELOCITY:
        orbit_reached = True

fig = go.Figure(
    data=[
        go.Scatter(mode="lines"),
        go.Scatter(mode="text"),
        go.Scatter(mode="text")
    ],
    frames=frames
)

fig.update_layout(
    xaxis=dict(range=[0, max_time], visible=False),
    yaxis=dict(range=[0, max_alt*1.1], visible=False),
    height=600,
    showlegend=False,
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "Launch 🚀",
            "method": "animate",
            "args": [None, {
                "frame": {"duration": 40, "redraw": False},
                "fromcurrent": True
            }]
        }]
    }]
)

st.plotly_chart(fig, use_container_width=True)

if orbit_reached:
    st.success("🛰 ORBIT ACHIEVED! Stable orbital velocity reached.")
else:
    st.warning("⚠ Orbit not achieved. Try increasing thrust or reducing payload.")
