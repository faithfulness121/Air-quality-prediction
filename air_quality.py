
import streamlit as st
import joblib
import numpy as np
import pickle


# Load the model
with open('rf_best_model.pkl', 'rb') as f:
    model = pickle.load(f)
    

# Title
st.title("🌆 City Type Classifier")
st.write("Enter pollution values to predict the type of city (Urban, Suburban, Rural)")

# Feature inputs
CO = st.number_input("CO (Carbon Monoxide)", min_value=0.0, max_value=50.0, value=1.0)
NO2 = st.number_input("NO2 (Nitrogen Dioxide)", min_value=0.0, max_value=200.0, value=30.0)
SO2 = st.number_input("SO2 (Sulfur Dioxide)", min_value=0.0, max_value=100.0, value=15.0)
O3 = st.number_input("O3 (Ozone)", min_value=0.0, max_value=200.0, value=50.0)
PM25 = st.number_input("PM2.5 (Particulate Matter)", min_value=0.0, max_value=300.0, value=70.0)
PM10 = st.number_input("PM10 (Particulate Matter)", min_value=0.0, max_value=400.0, value=120.0)

# Predict button
if st.button("Predict City Type"):
    input_data = np.array([[CO, NO2, SO2, O3, PM25, PM10]])
    prediction = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)[0]

    # Label mapping (change if your label encoder is different)
    label_mapping = {0: "Rural", 1: "Suburban", 2: "Urban"}
    st.success(f"🌍 Predicted City Type: **{label_mapping[prediction]}**")
    
    # Show probability chart
    st.subheader("Prediction Confidence")
    st.write({label_mapping[i]: f"{p*100:.2f}%" for i, p in enumerate(probas)})
