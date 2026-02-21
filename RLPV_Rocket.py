import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Rocket Launch Analytics Dashboard", layout="wide")
st.title("🚀 Rocket Launch Path Visualization & Mission Analytics")

# --------------------------------------------------
# LOAD & CLEAN DATA (STAGE 2)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("space_missions_dataset.csv")

    # Convert Launch Date to datetime
    df["Launch Date"] = pd.to_datetime(df["Launch Date"], errors="coerce")

    # Convert numeric columns properly
    numeric_columns = [
        "Distance from Earth (light-years)",
        "Mission Duration (years)",
        "Mission Cost (billion USD)",
        "Scientific Yield (points)",
        "Crew Size",
        "Mission Success (%)",
        "Fuel Consumption (tons)",
        "Payload Weight (tons)"
    ]

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Remove missing values
    df = df.dropna()

    return df

df = load_data()
# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("🔎 Mission Filters")

mission_type = st.sidebar.multiselect(
    "Mission Type",
    df["Mission Type"].unique(),
    default=df["Mission Type"].unique()
)

launch_vehicle = st.sidebar.multiselect(
    "Launch Vehicle",
    df["Launch Vehicle"].unique(),
    default=df["Launch Vehicle"].unique()
)

success_threshold = st.sidebar.slider(
    "Minimum Mission Success (%)",
    60, 100, 70
)

filtered_df = df[
    (df["Mission Type"].isin(mission_type)) &
    (df["Launch Vehicle"].isin(launch_vehicle)) &
    (df["Mission Success (%)"] >= success_threshold)
]

# ==================================================
# STAGE 2 — INTERACTIVE EDA (PLOTLY)
# ==================================================
st.header("📊 Stage 2: Mission Data Insights")

col1, col2 = st.columns(2)

# Plotly Scatter
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

# Plotly Bar
with col2:
    avg_cost = filtered_df.groupby("Launch Vehicle")["Mission Cost (billion USD)"].mean().reset_index()
    fig2 = px.bar(
        avg_cost,
        x="Launch Vehicle",
        y="Mission Cost (billion USD)",
        title="Average Mission Cost by Launch Vehicle"
    )
    st.plotly_chart(fig2, use_container_width=True)

# Correlation Heatmap (Plotly)
st.subheader("🔬 Correlation Heatmap")

numeric_columns = [
    "Distance from Earth (light-years)",
    "Mission Duration (years)",
    "Mission Cost (billion USD)",
    "Scientific Yield (points)",
    "Crew Size",
    "Mission Success (%)",
    "Fuel Consumption (tons)",
    "Payload Weight (tons)"
]

corr = filtered_df[numeric_columns].corr()

heatmap = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    text=corr.round(2).values,
    texttemplate="%{text}"
))

heatmap.update_layout(title="Correlation Between Mission Factors")
st.plotly_chart(heatmap, use_container_width=True)

# ==================================================
# STAGE 3 — ROCKET SIMULATION
# ==================================================
st.header("🛰 Stage 3: Rocket Launch Simulation")

st.sidebar.header("⚙ Simulation Controls")

rocket_mass = st.sidebar.slider("Rocket Base Mass (kg)", 100000, 800000, 500000, step=50000)
fuel_mass = st.sidebar.slider("Fuel Mass (kg)", 100000, 600000, 300000, step=50000)
payload_weight = st.sidebar.slider("Payload Weight (kg)", 10000, 100000, 50000, step=5000)
thrust = st.sidebar.slider("Thrust (Newtons)", 1000000, 10000000, 7000000, step=500000)
drag_coefficient = st.sidebar.slider("Drag Coefficient", 0.1, 1.0, 0.5)
time_steps = st.sidebar.slider("Simulation Time Steps", 50, 300, 200)

g = 9.81
air_density = 1.225
cross_section_area = 10
time_step = 1

velocity = 0
altitude = 0
current_fuel = fuel_mass

results = []

for step in range(time_steps):
    if current_fuel <= 0:
        thrust_force = 0
    else:
        thrust_force = thrust
        current_fuel -= fuel_mass / time_steps

    total_mass = rocket_mass + current_fuel + payload_weight
    drag_force = 0.5 * air_density * velocity**2 * drag_coefficient * cross_section_area
    net_force = thrust_force - (total_mass * g) - drag_force
    acceleration = net_force / total_mass

    velocity += acceleration * time_step
    altitude += velocity * time_step

    results.append([step, total_mass, acceleration, velocity, altitude])

simulation_df = pd.DataFrame(
    results,
    columns=["Time (s)", "Mass (kg)", "Acceleration (m/s²)", "Velocity (m/s)", "Altitude (m)"]
)

col3, col4 = st.columns(2)

with col3:
    fig_alt = px.line(simulation_df, x="Time (s)", y="Altitude (m)",
                      title="Rocket Altitude Over Time")
    st.plotly_chart(fig_alt, use_container_width=True)

with col4:
    fig_vel = px.line(simulation_df, x="Time (s)", y="Velocity (m/s)",
                      title="Rocket Velocity Over Time")
    st.plotly_chart(fig_vel, use_container_width=True)

# ==================================================
# STAGE 4 — REQUIRED VISUALIZATIONS
# ==================================================
st.header("📈 Stage 4: Required Visualizations")

col5, col6 = st.columns(2)

# 1️⃣ Seaborn Scatter
with col5:
    fig_sns1, ax_sns1 = plt.subplots()
    sns.scatterplot(data=filtered_df,
                    x="Mission Cost (billion USD)",
                    y="Mission Success (%)",
                    hue="Mission Type",
                    ax=ax_sns1)
    ax_sns1.set_title("Cost vs Mission Success")
    st.pyplot(fig_sns1)

# 2️⃣ Matplotlib Line
with col6:
    yearly = filtered_df.groupby(filtered_df["Launch Date"].dt.year)["Mission Success (%)"].mean()
    fig_line, ax_line = plt.subplots()
    ax_line.plot(yearly.index, yearly.values)
    ax_line.set_title("Average Success Rate Over Years")
    ax_line.set_xlabel("Year")
    ax_line.set_ylabel("Mission Success (%)")
    st.pyplot(fig_line)

# 3️⃣ Seaborn Boxplot
fig_box, ax_box = plt.subplots()
sns.boxplot(data=filtered_df,
            x="Mission Type",
            y="Mission Cost (billion USD)",
            ax=ax_box)
ax_box.set_title("Mission Cost Distribution by Type")
st.pyplot(fig_box)

# 4️⃣ Plotly go.Bar
avg_payload = filtered_df.groupby("Launch Vehicle")["Payload Weight (tons)"].mean().reset_index()
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=avg_payload["Launch Vehicle"],
    y=avg_payload["Payload Weight (tons)"]
))
fig_bar.update_layout(title="Average Payload by Launch Vehicle",
                      xaxis_title="Launch Vehicle",
                      yaxis_title="Payload Weight (tons)")
st.plotly_chart(fig_bar, use_container_width=True)

# 5️⃣ Plotly go.Scatter
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=filtered_df["Distance from Earth (light-years)"],
    y=filtered_df["Mission Duration (years)"],
    mode="markers"
))
fig_scatter.update_layout(title="Distance vs Mission Duration",
                          xaxis_title="Distance from Earth (light-years)",
                          yaxis_title="Mission Duration (years)")
st.plotly_chart(fig_scatter, use_container_width=True)

# ==================================================
# FINAL METRICS
# ==================================================
st.subheader("🚀 Final Simulation Metrics")

col7, col8, col9 = st.columns(3)
col7.metric("Final Altitude (m)", f"{simulation_df['Altitude (m)'].iloc[-1]:,.2f}")
col8.metric("Max Velocity (m/s)", f"{simulation_df['Velocity (m/s)'].max():,.2f}")
col9.metric("Final Rocket Mass (kg)", f"{simulation_df['Mass (kg)'].iloc[-1]:,.2f}")
