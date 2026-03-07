# 🚀 Rocket Launch Path Visualization 
Streamlit Web App for Rocket Mission Analytics & Physics Simulation
📌 Project Overview

This project was developed as part of the CRS – Artificial Intelligence course under Mathematics for AI-I.

The application is an interactive Streamlit web dashboard that:

Analyzes real-world space mission data

Visualizes relationships between cost, fuel, payload, and success rate

Simulates rocket launch physics using Newton’s Second Law

Displays animated trajectory visualization

Deploys live on Streamlit Cloud

The system combines mathematical modeling, data analytics, and interactive visualization to help users explore how rocket design parameters influence mission success.

1️⃣ Problem Understanding & Research 

Rocket launches are governed by fundamental physics principles. A rocket moves upward due to thrust, while gravity pulls it downward. As fuel burns, the rocket becomes lighter, which changes acceleration dynamically over time.

Key forces involved:

Thrust (T) – Upward engine force

Gravity (mg) – Downward force

Mass (m) – Changes as fuel burns

Understanding these relationships is essential for aerospace engineers when designing efficient and successful missions.

🧠 Research Insights

Guiding questions explored:

How does increasing payload affect required fuel?

Does higher mission cost increase mission success?

How does thrust-to-mass ratio impact altitude?

Can simulated physics results align with real historical mission data?

The project integrates mathematical modeling with historical space mission data to answer these questions interactively.

2️⃣ Data Preprocessing & Cleaning 
📂 Dataset Overview

The dataset contains:

Mission Name

Launch Date

Mission Type

Launch Vehicle

Payload Weight (tons)

Fuel Consumption (tons)

Mission Cost (billion USD)

Mission Success (%)

Distance from Earth

Mission Duration

Crew Size

Scientific Yield

🧹 Cleaning Steps Performed

Using pandas:

Converted Launch Date to datetime format

Converted numeric columns using pd.to_numeric()

Removed missing and invalid values

Extracted launch year for trend analysis

Created mission labels for dropdown selection

Applied filtering using:

Mission Type

Launch Vehicle

Minimum Success %

Exploration methods used:

.head()

.info()

.describe()

.dropna()

The dataset was fully cleaned before visualization and simulation comparison.

3️⃣ Data Visualization & Analysis 

The application includes all required visualization types with interactive controls.

📊 1. Scatter Plot

Payload vs Fuel Consumption

Shows positive correlation

Color-coded by mission type

Interactive hover tooltips

Insight: Heavier payloads require significantly more fuel.

📊 2. Bar Chart

Average Mission Cost by Launch Vehicle

Insight: Launch vehicle selection significantly affects mission cost.

📊 3. Line Chart

Mission Success Rate Over Time

Insight: Success rates improve across years for certain mission types.

📊 4. Box Plot

Mission Cost Distribution by Mission Type

Insight: Some mission types show large cost variability and outliers.

📊 5. Success Category Comparison

Cost vs Success Category

Insight: Higher mission cost does not guarantee higher success rate.

🎛 Interactive Features

Sidebar filters

Dropdown selection

Slider for minimum success rate

Real-time chart updates

Dark professional dashboard layout

4️⃣ Simulation & Mathematical Modeling 
📐 Mathematical Foundation

The simulation applies Newton’s Second Law:

𝐹
=
𝑚
𝑎
F=ma

Acceleration is calculated as:

𝑎
=
𝑇
−
𝑚
𝑔
𝑚
a=
m
T−mg
	​


Where:

𝑇
T = thrust

𝑚
m = total mass

𝑔
g = gravitational acceleration

⏱ Numerical Time-Step Updates

Velocity update:

𝑣
𝑡
+
1
=
𝑣
𝑡
+
𝑎
Δ
𝑡
v
t+1
	​

=v
t
	​

+aΔt

Altitude update:

ℎ
𝑡
+
1
=
ℎ
𝑡
+
𝑣
Δ
𝑡
h
t+1
	​

=h
t
	​

+vΔt

Fuel burn reduces mass at each iteration:

𝑚
𝑡
𝑜
𝑡
𝑎
𝑙
=
𝑚
𝑏
𝑎
𝑠
𝑒
+
𝑚
𝑝
𝑎
𝑦
𝑙
𝑜
𝑎
𝑑
+
𝑚
𝑓
𝑢
𝑒
𝑙
m
total
	​

=m
base
	​

+m
payload
	​

+m
fuel
	​


This produces a dynamic acceleration model.

🚀 Simulation Modes
🔹 Historical Mission Mode

Uses real mission parameters from dataset.

🔹 Manual Design Mode

User controls:

Payload

Fuel mass

Thrust

Outputs:

Altitude vs Time graph

Velocity vs Time graph

Dynamic mission success logic

Animated trajectory visualization

📌 Simulation Insights

Higher thrust-to-mass ratio increases altitude.

Insufficient thrust leads to early mission failure.

Simulation results align with dataset fuel–payload trends.

5️⃣ GitHub Repository & Streamlit Deployment 
📁 Repository Structure
IDAI104(StudentID)-StudentName/
│
├── app.py
├── requirements.txt
├── space_missions_dataset.csv
└── README.md

⚙️ Requirements

streamlit
pandas
numpy
plotly
matplotlib
seaborn

🌍 Live Streamlit Web App

🔗 Live App Link: https://1000408sriprasathpaiy1sa-rlpv-ansjqlhirp4lee59half7u.streamlit.app/

📊 Final Conclusion

This project demonstrates:

Application of calculus and Newtonian physics

Statistical reasoning and exploratory data analysis

Interactive dashboard design

Simulation modeling using numerical methods

Professional cloud deployment

The integration of mathematical theory, Python programming, and real-world aerospace data fulfills all intended learning outcomes of the course.

Project Summary – Rocket Launch Path Visualization 

This project presents an interactive Streamlit web application designed to analyze and simulate rocket launch missions using real-world mission data and mathematical modeling.

The application integrates Newton’s Second Law to simulate rocket motion dynamically. By calculating acceleration as a function of thrust and gravitational force and updating velocity and altitude over discrete time steps, the simulation models realistic launch behavior. Fuel mass reduction dynamically alters total mass, affecting acceleration over time.

Data preprocessing was performed using pandas, including type conversions, missing value handling, and structured filtering. Exploratory Data Analysis was conducted through five required visualizations: scatter plot, bar chart, line chart, box plot, and success comparison plot.

Key findings include:

Positive correlation between payload weight and fuel consumption.

Higher mission cost does not necessarily result in higher success.

Launch vehicle selection influences mission cost efficiency.

Thrust-to-mass ratio significantly impacts launch altitude.

The final application was deployed successfully on Streamlit Cloud, and the GitHub repository contains organized code, documentation, and deployment details.

This project demonstrates the practical integration of mathematical modeling, statistical reasoning, Python programming, and interactive data visualization in solving real-world aerospace problems.

Credits:

Student Name: Sri Prasath. P

Mentor Name: Syed Ali Beema. S

Course: Mathematics for AI

School Name: Jain Vidyalaya IB World School
