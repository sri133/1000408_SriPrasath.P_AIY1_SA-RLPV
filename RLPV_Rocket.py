# ==========================================================
# ROCKET LAUNCH PATH VISUALIZATION APP
# Mathematics for AI – Summative Assessment
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Rocket Launch Visualization", layout="wide")

# ==========================================================
# STAGE 1 – PROBLEM CONTEXT
# ==========================================================

st.title("🚀 Rocket Launch Path Visualization Dashboard")

st.markdown("""
This application explores **rocket launch dynamics** using:

• Real-world mission dataset  
• Mathematical simulation using Newton’s Second Law  
• Interactive visualizations  

Rocket motion depends on:
- Thrust (upward force)
- Gravity (downward force)
- Drag (air resistance)
- Changing mass due to fuel burn
""")

# ==========================================================
# STAGE 2 – DATA LOADING & CLEANING
# ==========================================================

DATA_URL = "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_AIY1_SA-RLPV/main/space_missions_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)

    # Convert date column
    if "Launch Date" in df.columns:
        df["Launch Date"] = pd.to_datetime(df["Launch Date"], errors='coerce')

    # Convert numeric columns safely
    numeric_cols = [
        "Distance from Earth",
        "Mission Duration",
        "Mission Cost",
        "Fuel Consumption",
        "Payload Weight",
        "Crew Size",
        "Scientific Yield"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with too many missing values
    df = df.dropna(thresh=5)

    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

st.sidebar.header("Filters")

if "Mission Success" in df.columns:
    success_filter = st.sidebar.selectbox(
        "Mission Success",
        ["All", "Success", "Failure"]
    )

    if success_filter != "All":
        df = df[df["Mission Success"] == success_filter]

# ==========================================================
# STAGE 4 – VISUALIZATIONS (ALL 5 REQUIRED TYPES)
# ==========================================================

st.header("📊 Data Analysis & Visualizations")

col1, col2 = st.columns(2)

# 1️⃣ SCATTER PLOT (Payload vs Fuel)
with col1:
    st.subheader("Payload vs Fuel Consumption")
    fig1 = plt.figure()
    sns.scatterplot(
        data=df,
        x="Payload Weight",
        y="Fuel Consumption",
        hue="Mission Success"
    )
    plt.title("Payload vs Fuel")
    plt.xlabel("Payload Weight (tons)")
    plt.ylabel("Fuel Consumption (tons)")
    st.pyplot(fig1)

# 2️⃣ BAR CHART (Mission Cost vs Success)
with col2:
    st.subheader("Mission Cost by Success")
    fig2 = plt.figure()
    sns.barplot(
        data=df,
        x="Mission Success",
        y="Mission Cost"
    )
    plt.title("Average Cost by Success")
    st.pyplot(fig2)

# 3️⃣ LINE PLOT (Duration vs Distance)
st.subheader("Mission Duration vs Distance")
fig3 = plt.figure()
sns.lineplot(
    data=df,
    x="Distance from Earth",
    y="Mission Duration"
)
plt.title("Duration vs Distance")
st.pyplot(fig3)

# 4️⃣ BOX PLOT (Fuel Consumption Distribution)
st.subheader("Fuel Consumption Distribution")
fig4 = plt.figure()
sns.boxplot(
    data=df,
    y="Fuel Consumption"
)
plt.title("Fuel Consumption Spread")
st.pyplot(fig4)

# 5️⃣ CORRELATION HEATMAP
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=np.number)

fig5 = plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
st.pyplot(fig5)

# ==========================================================
# STAGE 3 – ROCKET SIMULATION (Differential Equation)
# ==========================================================

st.header("🚀 Rocket Launch Simulation")

st.markdown("""
Using Newton’s Second Law:

Acceleration = (Thrust - Gravity - Drag) / Mass
""")

# Sliders
thrust = st.slider("Thrust (Newtons)", 500000, 2000000, 1000000)
payload = st.slider("Payload Mass (kg)", 1000, 20000, 5000)
fuel = st.slider("Fuel Mass (kg)", 50000, 200000, 100000)
drag_coeff = st.slider("Drag Coefficient", 0.0, 0.5, 0.1)

g = 9.81
time_steps = 200
dt = 1

mass = payload + fuel + 10000  # structure mass
velocity = 0
altitude = 0

results = []

for t in range(time_steps):
    drag = drag_coeff * velocity**2
    weight = mass * g
    acceleration = (thrust - weight - drag) / mass

    velocity += acceleration * dt
    altitude += velocity * dt

    fuel -= 300
    if fuel < 0:
        fuel = 0

    mass = payload + fuel + 10000

    results.append([t, altitude, velocity, acceleration])

sim_df = pd.DataFrame(results, columns=["Time", "Altitude", "Velocity", "Acceleration"])

# Plot simulation
fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(x=sim_df["Time"], y=sim_df["Altitude"],
                             mode='lines', name='Altitude'))
fig_sim.update_layout(title="Rocket Altitude Over Time",
                      xaxis_title="Time (s)",
                      yaxis_title="Altitude (m)")

st.plotly_chart(fig_sim, use_container_width=True)

# ==========================================================
# FINAL INSIGHT SECTION
# ==========================================================

st.header("📌 Key Insights")

st.markdown("""
• Higher payload requires more fuel.  
• Increasing thrust increases altitude faster.  
• Drag reduces acceleration significantly at high speeds.  
• Real dataset trends confirm fuel increases with payload weight.  
""")
