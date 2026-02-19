import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Rocket Launch Dashboard", layout="wide")

st.title("🚀 Rocket Launch Path Visualization Dashboard")
st.write("Interactive analysis of space missions and rocket simulation.")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("your_dataset.csv")
    return df

df = load_data()

# ---------------------------
# DATA CLEANING
# ---------------------------
df['Launch Date'] = pd.to_datetime(df['Launch Date'], errors='coerce')

numeric_cols = ['Mission Cost', 'Payload Weight', 'Fuel Consumption',
                'Mission Duration', 'Distance from Earth', 'Crew Size']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.drop_duplicates()
df = df.dropna()

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
st.sidebar.header("Filter Options")

mission_type = st.sidebar.selectbox(
    "Select Mission Type",
    df['Mission Type'].unique()
)

filtered_df = df[df['Mission Type'] == mission_type]

# ---------------------------
# REQUIRED VISUALIZATIONS
# ---------------------------

st.subheader("1. Payload vs Fuel Consumption (Scatter)")
fig1 = sns.scatterplot(
    data=filtered_df,
    x="Payload Weight",
    y="Fuel Consumption",
    hue="Mission Success"
)
st.pyplot(plt.gcf())
plt.clf()

st.subheader("2. Mission Cost: Success vs Failure (Bar)")
cost_data = filtered_df.groupby("Mission Success")["Mission Cost"].mean()
fig2 = plt.figure()
plt.bar(cost_data.index.astype(str), cost_data.values)
plt.xlabel("Mission Success")
plt.ylabel("Average Mission Cost")
st.pyplot(fig2)

st.subheader("3. Mission Duration vs Distance (Line)")
fig3 = sns.lineplot(
    data=filtered_df,
    x="Distance from Earth",
    y="Mission Duration"
)
st.pyplot(plt.gcf())
plt.clf()

st.subheader("4. Crew Size vs Mission Success (Box Plot)")
fig4 = sns.boxplot(
    data=filtered_df,
    x="Mission Success",
    y="Crew Size"
)
st.pyplot(plt.gcf())
plt.clf()

st.subheader("5. Scientific Yield vs Mission Cost")
fig5 = sns.scatterplot(
    data=filtered_df,
    x="Mission Cost",
    y="Scientific Yield"
)
st.pyplot(plt.gcf())
plt.clf()

# ---------------------------
# ROCKET SIMULATION
# ---------------------------

st.header("🚀 Rocket Launch Simulation")

payload = st.slider("Payload Weight (tons)", 1000, 10000, 3000)
fuel = st.slider("Fuel Amount (tons)", 5000, 20000, 10000)
thrust = st.slider("Thrust Force (N)", 1000000, 5000000, 2000000)

mass = payload + fuel
gravity = 9.81
drag_factor = 0.0001

time_steps = 150
velocity = 0
altitude = 0

altitudes = []

for t in range(time_steps):
    drag = drag_factor * velocity**2
    acceleration = (thrust - (mass * gravity) - drag) / mass
    velocity = velocity + acceleration
    altitude = altitude + velocity
    fuel = fuel - 50
    mass = payload + max(fuel, 0)
    altitudes.append(altitude)

sim_df = pd.DataFrame({
    "Time": range(time_steps),
    "Altitude": altitudes
})

fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(
    x=sim_df["Time"],
    y=sim_df["Altitude"],
    mode="lines",
    name="Altitude"
))

st.plotly_chart(fig_sim)
