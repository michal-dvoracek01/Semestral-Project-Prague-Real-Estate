import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium

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
        terrace = st.number_input("Terrace (m²)", min_value=0, max_value=300, value=0)
        elevator = st.checkbox("Has Elevator")

    with col2:
        st.subheader("🏢 Building & Layout")
        layout = st.selectbox("Apartment Layout", 
            ["1kk", "2kk", "21", "3kk", "31", "4kk", "41", "5kk", "51", "6_plus", "atypical"])
        
        b_type = st.selectbox("Building Type", 
            ["panel", "brick", "mixed", "skeleton", "wood", "stone", "assembled"])
        
        condition = st.selectbox("Condition", 
            ["new", "very_good", "development"])

    with col3:
        st.subheader("🚇 Location")
        metro_station = st.selectbox("Nearest Metro Station", pd.read_csv("metro_stations.csv")['name'].tolist())
        
        # distance checkboxes
        dist_A = st.checkbox("Distance to Metro A (<1000m)")
        dist_B = st.checkbox("Distance to Metro B (<1000m)")
        dist_C = st.checkbox("Distance to Metro C (<1000m)")



    submit = st.form_submit_button("💰 Calculate Estimated Price")

# Prediction Logic
if submit:
    # Initialize a dictionary with 0 for all features the model expects
    input_data = {feat: 0.0 for feat in model_features}
    
    # Map basic inputs
    input_data['area_usable'] = float(area)
    input_data['floor_number'] = float(floor)
    input_data['has_elevator'] = 1.0 if elevator else 0.0
    input_data['has_terrace'] = float(terrace)
    
    # Map Layout (e.g. layout_2kk)
    layout_key = f"layout_{layout}"
    if layout_key in input_data: input_data[layout_key] = 1.0
        
    # Map building type
    type_key = f"building_type_{b_type}"
    if type_key in input_data: input_data[type_key] = 1.0
        
    # Map Condition
    cond_key = f"status_{condition}"
    if cond_key in input_data: input_data[cond_key] = 1.0
        
    # Map Nearest station
    station_key = f"nearest_station_{metro_station}"
    if station_key in input_data: input_data[station_key] = 1.0

    # Metro distance Logic 
    input_data['1000m_A'] = 1.0 if dist_A else 0.0
    input_data['1000m_B'] = 1.0 if dist_B else 0.0
    input_data['1000m_C'] = 1.0 if dist_C else 0.0

    # Convert to df
    df_input = pd.DataFrame([input_data])
    
    # Ensure intercept constant is added
    df_input = sm.add_constant(df_input, has_constant='add')
    
    # Ensure column order matches
    df_input = df_input[['const'] + model_features]

    # Predict
    prediction = model.predict(df_input)[0]
    # Predict with confidence intervals
    prediction_obj = model.get_prediction(df_input)
    summary_frame = prediction_obj.summary_frame(alpha=0.05)  # alpha 0.05 = 95% CI

    prediction = summary_frame['mean'][0]
    lower_ci = summary_frame['obs_ci_lower'][0] # Interval for individual prediction
    upper_ci = summary_frame['obs_ci_upper'][0]

    # Display Result
    # st.success(f"### Estimated Price: {prediction:,.0f} CZK")

    # st.info(f"**Range:** {lower_ci:,.0f} CZK - {upper_ci:,.0f} CZK")

    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric("Estimated Price", f"{prediction:,.0f} CZK")
    col_metric1.metric("Price per m²", f"{prediction/area:,.0f} CZK/m²")
    col_metric2.metric("Range", f"{lower_ci:,.0f} CZK - {upper_ci:,.0f} CZK")
    col_metric2.metric("Range per m²", f"{lower_ci/area:,.0f} CZK - {upper_ci/area:,.0f} CZK")

# Market Insights Section
st.markdown("---")
st.header("📊 Market Insights (By Metro Station)")

@st.cache_data
def load_market_data():
    try:
        df = pd.read_csv("data_estate_processed.csv")
        # Calculate price per m2
        df['price_m2'] = df['price'] / df['area_usable']
        
        # Group, calculate mean, and count listings for context
        stats = df.groupby('nearest_station_name')['price_m2'].agg(['mean', 'count']).reset_index()
        stats.columns = ['Metro Station', 'Avg. Price / m²', 'Total Listings']
        return stats
    except FileNotFoundError:
        return None

stats_df = load_market_data()

if stats_df is not None:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📉 10 Most Affordable Stations")
        cheap_10 = stats_df.sort_values("Avg. Price / m²", ascending=True).head(10)
        # Use st.dataframe for an interactive table with formatting
        st.dataframe(
            cheap_10.style.format({"Avg. Price / m²": "{:,.0f} Kč"}),
            use_container_width=True,
            hide_index=True
        )

    with col_right:
        st.subheader("📈 10 Most Expensive Stations")
        expensive_10 = stats_df.sort_values("Avg. Price / m²", ascending=False).head(10)
        st.dataframe(
            expensive_10.style.format({"Avg. Price / m²": "{:,.0f} Kč"}),
            use_container_width=True,
            hide_index=True
        )

# Interactive Map Section
    st.markdown("---")
    st.header("📍 Interactive Property Explorer")
    st.markdown("Click on a marker to see the price, size, and layout.")

    @st.cache_data
    def load_map_data():
        df = pd.read_csv("data_estate_processed.csv")
        return df.dropna(subset=['latitude', 'longitude'])

    map_df = load_map_data()

    # Map centered around Prague
    m = folium.Map(location=[50.0755, 14.4378], zoom_start=15, tiles="cartodbpositron")

    # Add markers for each estate
    for idx, row in map_df.iterrows():
        layout = "Unknown"
        layout_cols = [col for col in row.index if col.startswith('layout_')]
        for col in layout_cols:
            if row[col] == 1:
                layout = col.replace('layout_', '').upper()
                break
        # Define content of click popup
        popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 150px;">
                <h4>{row['price']/row['area_usable']:,.0f} CZK/m²</h4>
                <b>Layout:</b> {layout}<br>
                <hr>
                <b>Price:</b> {row['price']:,.0f} CZK<br>
                <b>Size:</b> {row['area_usable']} m²<br>
            </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['price']:,.0f} CZK", # Hover text
            icon=folium.Icon(color="blue", icon="home")
        ).add_to(m)

    # Display map
    st_folium(m, width=1400, height=500, returned_objects=[])

else:
    st.warning("Market data file (data_estate_processed.csv) not found.")