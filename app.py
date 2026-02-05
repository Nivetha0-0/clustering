import streamlit as st
import numpy as np

# --------------------------------------------------
# Load models AFTER dependencies are ready
# --------------------------------------------------
@st.cache_resource
def load_models():
    import joblib

    kmeans = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("scaler.joblib")

    return kmeans, scaler


kmeans, scaler = load_models()

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("🌍 Country Development Clustering App")

st.write(
    """
    This app uses an **unsupervised K-Means clustering model**
    to group countries based on socio-economic indicators.
    """
)

st.subheader("Enter Country Indicators")

# User inputs
child_mort = st.number_input("Child Mortality (per 1000 births)", min_value=0.0)
exports = st.number_input("Exports (% of GDP)", min_value=0.0)
health = st.number_input("Health Expenditure (% of GDP)", min_value=0.0)
imports = st.number_input("Imports (% of GDP)", min_value=0.0)
income = st.number_input("Income (per capita)", min_value=0.0)
inflation = st.number_input("Inflation (%)", min_value=0.0)
life_expec = st.number_input("Life Expectancy (years)", min_value=0.0)
total_fer = st.number_input("Total Fertility Rate", min_value=0.0)
gdpp = st.number_input("GDP per capita", min_value=0.0)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Cluster"):
    input_data = np.array([[
        child_mort,
        exports,
        health,
        imports,
        income,
        inflation,
        life_expec,
        total_fer,
        gdpp
    ]])

    input_scaled = scaler.transform(input_data)
    cluster = int(kmeans.predict(input_scaled)[0])

    st.success(f"Predicted Cluster: {cluster}")

    cluster_meanings = {
        0: "Moderately developed countries",
        1: "Developing countries",
        2: "Well-developed countries",
        3: "Extreme outlier economy",
        4: "Highly underdeveloped countries"
    }

    st.info(cluster_meanings.get(cluster, "Cluster interpretation not available"))
