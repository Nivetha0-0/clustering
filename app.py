import pickle
import streamlit as st
import numpy as np

# load model
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🌍 Country Development Clustering App")

st.write(
    """
    This app uses an **unsupervised K-Means clustering model**
    to classify countries based on socio-economic indicators.
    """
)

st.subheader("Enter Country Indicators")

# user inputs
child_mort = st.number_input("Child Mortality (per 1000 births)", min_value=0.0)
exports = st.number_input("Exports (% of GDP)", min_value=0.0)
health = st.number_input("Health Expenditure (% of GDP)", min_value=0.0)
imports = st.number_input("Imports (% of GDP)", min_value=0.0)
income = st.number_input("Income (per capita)", min_value=0.0)
inflation = st.number_input("Inflation (%)", min_value=0.0)
life_expec = st.number_input("Life Expectancy (years)", min_value=0.0)
total_fer = st.number_input("Total Fertility Rate", min_value=0.0)
gdpp = st.number_input("GDP per capita", min_value=0.0)

if st.button("Predict Cluster"):
    # arrange input in correct order
    input_data = np.array([[child_mort, exports, health, imports,
                            income, inflation, life_expec,
                            total_fer, gdpp]])

    # scale input
    input_scaled = scaler.transform(input_data)

    # predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Cluster: {cluster}")

    # cluster interpretation
    cluster_meanings = {
        0: "Moderately developed countries",
        1: "Developing countries",
        2: "Well-developed countries",
        3: "Extreme outlier economy",
        4: "Highly underdeveloped countries"
    }

    st.info(cluster_meanings.get(cluster, "Cluster interpretation not available"))
