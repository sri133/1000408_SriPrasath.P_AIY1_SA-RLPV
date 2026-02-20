import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Rocket Launch Analysis & AI Prediction", layout="wide")

st.title("🚀 Rocket Launch Data Analysis & AI Prediction App")

DATA_URL = "space_missions_dataset.csv"

# ----------------------------
# DATA LOADING
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)

    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("_", " ")
    df.columns = df.columns.str.lower()

    # Convert possible date column
    if "launch date" in df.columns:
        df["launch date"] = pd.to_datetime(df["launch date"], errors="coerce")

    # Convert numeric columns safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    df = df.drop_duplicates()

    return df


df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.write("Columns detected:", df.columns.tolist())

# ----------------------------
# BASIC STATISTICS
# ----------------------------
st.header("📈 Statistical Summary")
st.write(df.describe(include="all"))

# ----------------------------
# VISUALIZATIONS
# ----------------------------
st.header("📊 Data Visualizations")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if numeric_cols:
    # Histogram
    st.subheader("Distribution of First Numeric Feature")
    fig1 = plt.figure()
    sns.histplot(df[numeric_cols[0]], kde=True)
    st.pyplot(fig1)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig2 = plt.figure()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig2)

# Scatter Plot (Payload vs Fuel if exists)
payload_col = [c for c in df.columns if "payload" in c]
fuel_col = [c for c in df.columns if "fuel" in c]
success_col = [c for c in df.columns if "success" in c]

if payload_col and fuel_col:
    st.subheader("Payload vs Fuel Consumption")
    fig3 = plt.figure()
    sns.scatterplot(
        data=df,
        x=payload_col[0],
        y=fuel_col[0],
        hue=success_col[0] if success_col else None
    )
    st.pyplot(fig3)

# Bar chart (Success count)
if success_col:
    st.subheader("Mission Success Count")
    fig4 = plt.figure()
    df[success_col[0]].value_counts().plot(kind="bar")
    st.pyplot(fig4)

# Pie chart (if success exists)
if success_col:
    st.subheader("Mission Success Distribution")
    fig5 = px.pie(
        df,
        names=success_col[0],
        title="Success vs Failure"
    )
    st.plotly_chart(fig5)

# ----------------------------
# AI MODEL SECTION
# ----------------------------
st.header("🤖 AI Model: Mission Success Prediction")

if success_col:

    target = success_col[0]

    numeric_df = df.select_dtypes(include=np.number)

    if target in numeric_df.columns and len(numeric_df.columns) > 1:

        X = numeric_df.drop(columns=[target])
        y = df[target]

        if y.dtype == "object":
            y = y.astype("category").cat.codes

        X = X.fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("Model Accuracy")
        st.write(f"Accuracy: {accuracy:.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig_cm = plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig_cm)

        # Feature Importance
        st.subheader("Feature Importance")
        importance = pd.Series(
            model.coef_[0],
            index=X.columns
        ).sort_values()

        fig_imp = plt.figure()
        importance.plot(kind="barh")
        st.pyplot(fig_imp)

    else:
        st.warning("Target column not numeric or insufficient features.")

else:
    st.warning("Success column not found in dataset.")

# ----------------------------
# INTERACTIVE PREDICTION
# ----------------------------
st.header("🔮 Try Your Own Prediction")

if success_col and numeric_cols:

    input_data = {}

    for col in numeric_cols:
        if col != success_col[0]:
            input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

    if st.button("Predict Mission Outcome"):

        input_df = pd.DataFrame([input_data])

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("🚀 Predicted: Mission Likely Successful")
        else:
            st.error("❌ Predicted: Mission Likely Failure")
