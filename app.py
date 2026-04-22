import streamlit as st
import pickle
import numpy as np
import os

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

st.title("🏠 House Price Prediction App")

st.write("Enter house details:")

# 👉 CHANGE these inputs according to your dataset
area = st.number_input("Area (sq ft)", min_value=0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)
stories = st.number_input("Stories", min_value=0)
parking = st.number_input("Parking", min_value=0)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")
