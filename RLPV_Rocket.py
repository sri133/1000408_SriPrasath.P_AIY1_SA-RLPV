# ==========================================================
# ROCKET LAUNCH PATH VISUALIZATION
# Mathematics for AI – Summative Assessment
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Rocket Launch Path Visualization", layout="wide")

# ==========================================================
# STAGE 1 – PROBLEM UNDERSTANDING
# ==========================================================

st.title("🚀 Rocket Launch Path Visualization Dashboard")

st.markdown("""
This app explores real-world rocket mission data and simulates rocket motion using Newton’s Second Law.

Rocket motion depends on:
• Thrust (engine force upward)  
• Gravity (pulling downward)  
• Drag (air resistance)  
• Changing mass due to fuel burn  

The goal is to analyse how payload, fuel, cost, and success relate to each other
and compare them with simulated rocket physics.
""")

# ==========================================================
# STAGE 2 – DATA LOADING & CLEANING
# ==========================================================

DATA_URL = "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_AIY1_SA-RLPV/main/space_missions_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert Launch Date
    if "Launch Date" in df.columns:
        df["Launch Date"] = pd.to_datetime(df["Launch Date"], errors="coerce")

    # Convert numeric columns
    numeric_cols = [
        "Distance from Earth",
        "Mission Duration",
        "Mission Cost",
        "Scientific Yield",
        "Crew Size",
        "Fuel Consumption",
        "Payload Weight"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values (drop rows missing key values)
    df = df.dropna(subset=["Fuel Consumption", "Payload Weight"])

    return df

df = load_data()

st.subheader("📄 Dataset Preview")
st.write(df.head())

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Filter Options")

if "Mission Success" in df.columns:
    mission_status = st.sidebar.selectbox(
        "Mission Success Filter",
        ["All"] + list(df["Mission Success"].dropna().unique())
    )

    if mission_status != "All":
        df = df[df["Mission Success"] == mission_status]

# ==========================================================
# STAGE 4 – REQUIRED VISUALIZATIONS
# ==========================================================

st.header("📊 Data Visualizations")

# 1️⃣ Scatter Plot (Payload vs Fuel)
st.subheader("1️⃣ Payload vs Fuel Consumption (Scatter Plot)")
fig1 = plt.figure()
sns.scatterplot(
    data=df,
    x="Payload Weight",
    y="Fuel Consumption",
    hue="Mission Success" if "Mission Success" in df.columns else None
)
plt.title("Payload vs Fuel Consumption")
plt.xlabel("Payload Weight (tons)")
plt.ylabel("Fuel Consumption (tons)")
st.pyplot(fig1)

# 2️⃣ Bar Chart (Mission Cost vs Success)
st.subheader("2️⃣ Average Mission Cost by Success (Bar Chart)")
if "Mission Success" in df.columns:
    fig2 = plt.figure()
    sns.barplot(
        data=df,
        x="Mission Success",
        y="Mission Cost"
    )
    plt.title("Average Mission Cost by Success")
    st.pyplot(fig2)

# 3️⃣ Line Plot (Distance vs Duration)
st.subheader("3️⃣ Distance vs Mission Duration (Line Plot)")
fig3 = plt.figure()
sns.lineplot(
    data=df,
    x="Distance from Earth",
    y="Mission Duration"
)
plt.title("Mission Duration vs Distance")
plt.xlabel("Distance from Earth (km)")
plt.ylabel("Mission Duration (days)")
st.pyplot(fig3)

# 4️⃣ Box Plot (Fuel Consumption Spread)
st.subheader("4️⃣ Fuel Consumption Distribution (Box Plot)")
fig4 = plt.figure()
sns.boxplot(
    data=df,
    y="Fuel Consumption"
)
plt.title("Fuel Consumption Distribution")
st.pyplot(fig4)

# 5️⃣ Correlation Heatmap
st.subheader("5️⃣ Correlation Heatmap")

numeric_df = df.select_dtypes(include=np.number)

fig5 = plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
st.pyplot(fig5)

# ==========================================================
# STAGE 3 – ROCKET PHYSICS SIMULATION
# ==========================================================

st.header("🚀 Rocket Launch Simulation")

st.markdown("""
Using Newton’s Second Law:

Acceleration = (Thrust − Gravity − Drag) / Mass
""")

# User controls
thrust = st.slider("Thrust (Newtons)", 500000, 2000000, 1200000)
payload = st.slider("Payload Mass (kg)", 1000, 20000, 5000)
fuel = st.slider("Fuel Mass (kg)", 50000, 200000, 120000)
drag_coeff = st.slider("Drag Coefficient", 0.0, 0.5, 0.1)

g = 9.81
dt = 1
time_steps = 200

structure_mass = 10000
mass = payload + fuel + structure_mass

velocity = 0
altitude = 0

results = []

for t in range(time_steps):

    drag = drag_coeff * velocity**2
    weight = mass * g
    acceleration = (thrust - weight - drag) / mass

    velocity += acceleration * dt
    altitude += velocity * dt

    # Burn fuel
    fuel -= 300
    if fuel < 0:
        fuel = 0

    mass = payload + fuel + structure_mass

    results.append([t, altitude, velocity, acceleration])

sim_df = pd.DataFrame(results, columns=["Time", "Altitude", "Velocity", "Acceleration"])

# Interactive Plotly chart
fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(x=sim_df["Time"], y=sim_df["Altitude"],
                             mode="lines", name="Altitude"))

fig_sim.update_layout(
    title="Rocket Altitude Over Time",
    xaxis_title="Time (seconds)",
    yaxis_title="Altitude (meters)"
)

st.plotly_chart(fig_sim, use_container_width=True)

# ==========================================================
# FINAL INSIGHTS
# ==========================================================

st.header("📌 Key Insights")

st.markdown("""
• Heavier payload requires more fuel consumption.  
• Missions with higher cost are not always more successful.  
• Distance from Earth strongly affects mission duration.  
• Increasing thrust in simulation increases altitude growth rate.  
• Drag significantly reduces acceleration at higher velocity.  
""")
