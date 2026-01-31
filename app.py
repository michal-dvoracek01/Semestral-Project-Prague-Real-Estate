import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

# Load the model
@st.cache_resource
def load_model():
    with open('estate_model.pkl', 'rb') as f:
        return pickle.load(f)

data = load_model()
model = data['model']
model_features = data['features']

st.set_page_config(page_title="Prague Estate Estimator", layout="wide")
st.title("🏠 Prague Real Estate Price Predictor")
st.markdown("Enter the property details below to estimate the market price using the **Feasible GLS** model.")

# Create Form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Basic Info")
        area = st.number_input("Usable Area (m²)", min_value=10, max_value=500, value=60)
        floor = st.number_input("Floor Number", min_value=-1, max_value=20, value=2)
        elevator = st.checkbox("Has Elevator")
        terrace = st.number_input("Terrace (m²)", min_value=0, max_value=300, value=0)

    with col2:
        st.subheader("🏢 Building & Layout")
        layout = st.selectbox("Apartment Layout", 
            ["1kk", "2kk", "21", "3kk", "31", "4kk", "41", "5kk", "51", "6_plus", "atypical"])
        
        b_type = st.selectbox("Building Type", 
            ["panel", "brick", "mixed", "skeleton", "wood", "stone", "assembled"])
        
        condition = st.selectbox("Condition", 
            ["new", "very_good", "development"])

    with col3:
        st.subheader("🚇 Location & POI")
        metro_station = st.selectbox("Nearest Metro Station", pd.read_csv("metro_stations.csv")['name'].tolist())
        
        # Simple distance assumptions for the UI
        dist_metro = st.checkbox("Distance to Metro (<500m)")

    submit = st.form_submit_button("💰 Calculate Estimated Price")

# Prediction Logic
if submit:
    # Initialize a dictionary with 0 for all features the model expects
    input_data = {feat: 0.0 for feat in model_features}
    
    # Map basic inputs
    input_data['area_usable'] = float(area)
    input_data['floor_number'] = float(floor)
    input_data['has_elevator'] = 1.0 if elevator else 0.0
    input_data['has_terrace'] = 1.0 if terrace else 0.0
    
    # Map Layout (e.g. layout_2kk)
    layout_key = f"layout_{layout}"
    if layout_key in input_data: input_data[layout_key] = 1.0
        
    # Map Building Type
    type_key = f"building_type_{b_type}"
    if type_key in input_data: input_data[type_key] = 1.0
        
    # Map Condition
    cond_key = f"status_{condition}"
    if cond_key in input_data: input_data[cond_key] = 1.0
        
    # Map Nearest Station
    station_key = f"nearest_station_{metro_station}"
    if station_key in input_data: input_data[station_key] = 1.0

    # POI Logic (Binary flags based on your 500m logic)
    input_data['500m_metro_distance'] = 1.0 if dist_metro else 0.0

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Ensure intercept constant is added
    df_input = sm.add_constant(df_input, has_constant='add')
    
    # Ensure column order matches EXACTLY
    df_input = df_input[['const'] + model_features]

    # Predict
    prediction = model.predict(df_input)[0]

    # Display Result
    st.success(f"### Estimated Price: {prediction:,.0f} CZK")
    st.metric("Price per m²", f"{prediction/area:,.0f} CZK/m²")