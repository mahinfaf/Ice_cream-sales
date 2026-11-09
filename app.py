import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Ice Cream Sales Predictor", layout="centered")

st.title("🍦 Ice Cream Sales Predictor")
st.write("Predict daily ice cream sales revenue based on outdoor temperature and day type.")

# ---------------------------------------------------------
# Step 1: Create Dataset & Train Model Automatically in Code
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    """Generates training data and trains the linear regression model directly in Python."""
    np.random.seed(42)
    n_samples = 150

    # Feature 1: Temperature in Celsius (15°C to 35°C)
    temperature = np.random.uniform(15, 35, n_samples)
    
    # Feature 2: Weekend / Holiday status (0 = Weekday, 1 = Weekend/Holiday)
    weekend = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Target: Sales revenue ($) calculated with temperature and weekend effects + noise
    sales = 50 + (temperature * 12.5) + (weekend * 85.0) + np.random.normal(0, 15, n_samples)

    df = pd.DataFrame({
        'temperature': temperature,
        'weekend': weekend,
        'sales': sales
    })

    # Fit Multiple Linear Regression Model
    X = df[['temperature', 'weekend']]
    y = df['sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Train or retrieve cached model
model = train_model()

# ---------------------------------------------------------
# Step 2: User Inputs
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

# Convert selection to binary numeric indicator (0 or 1)
weekend_numeric = 1 if is_weekend == "Yes (Weekend/Holiday)" else 0

# ---------------------------------------------------------
# Step 3: Run Prediction
# ---------------------------------------------------------
st.header("2. Sales Prediction")

if st.button("Predict Daily Sales"):
    # Format inputs into array matching training shape: [[temperature, weekend]]
    input_data = np.array([[temp_input, weekend_numeric]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Estimated Daily Sales Revenue: **${prediction:.2f}**")

st.markdown("---")
st.caption(
    "Note: The linear regression model is dynamically trained in-memory using temperature and weekend indicators as predictors."
)
