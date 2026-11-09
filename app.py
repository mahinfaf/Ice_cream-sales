import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Ice Cream Sales Predictor", layout="centered")

st.title("🍦 Ice Cream Sales Predictor")
st.write("Predict daily ice cream sales revenue based on outdoor temperature and day type.")

# ---------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model_pkl.py", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# ---------------------------------------------------------
# User Inputs
# ---------------------------------------------------------
st.header("1. Enter Daily Conditions")

col1, col2 = st.columns(2)

with col1:
    temp_input = st.number_input(
        "Temperature (°C):", 
        min_value=0.0, 
        max_value=50.0, 
        value=25.0, 
        step=0.5
    )

with col2:
    is_weekend = st.selectbox(
        "Is it a Weekend / Holiday?", 
        options=["No (Weekday)", "Yes (Weekend/Holiday)"]
    )

# Convert categorical selectbox to binary feature
weekend_numeric = 1 if is_weekend == "Yes (Weekend/Holiday)" else 0

# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
st.header("2. Revenue Prediction")

if st.button("Predict Daily Sales"):
    # Format inputs into 2D array matching training features: [[temperature, weekend]]
    input_data = np.array([[temp_input, weekend_numeric]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Estimated Daily Sales Revenue: **${prediction:.2f}**")

# Context message
st.markdown("---")
st.caption(
    "Note: This model uses multiple linear regression based on historical temperature and day-of-week data."
)
