
  
import streamlit as st
import numpy as np
import pickle
import pandas as pd


# Load your trained model
with open('rf_best_model.pkl', 'rb') as f:
    col = pickle.load(f)

     
 
   
# Mapping predictions to city types
label_mapping = {0: "Industrial", 1: "Residentail"}


# App title and description
st.title("🌆 Pollution-Based City Type Classifier")
st.markdown("Use air quality metrics to predict whether a city is Urban, Suburban, or Rural.")

def MainMethod():
# Inputs 
 CO = st.slider("CO (Carbon Monoxide)", 0.0, 50.0, 1.0)
 NO2 = st.slider("NO2 (Nitrogen Dioxide)", 0.0, 200.0, 30.0)
 SO2 = st.slider("SO2 (Sulfur Dioxide)", 0.0, 100.0, 15.0)
 O3 = st.slider("O3 (Ozone)", 0.0, 200.0, 50.0)
 PM25 = st.slider("PM2.5 (Fine Particulate Matter)", 0.0, 300.0, 70.0)
 PM10 = st.slider("PM10 (Coarse Particulate Matter)", 0.0, 400.0, 120.0)

 pred=""

# Prediction
 if st.button("Predict City Type"):
    #pred=([CO, NO2, SO2, O3, PM25, PM10])

    features = np.array([[CO, NO2, SO2, O3, PM25, PM10]])
    prediction = col.predict(features)[0]
    confidence = col.predict_proba(features)[0]


    st.success(f"🌍 Predicted City Type: **{label_mapping[prediction]}**")

    # Confidence Bar Chart

    confidence_df = pd.DataFrame({
        "City Type": [label_mapping[i] for i in range(len(confidence))],
        "Confidence (%)": [p * 100 for p in confidence]
    })


    st.subheader("Prediction Confidence")
    st.bar_chart(confidence_df.set_index("City Type"))
def main():
     MainMethod()

if __name__=="__main__":
       main()

  