# 🏠 Prague Real Estate Price Predictor and Metro Index

A comprehensive end-to-end data science project that scrapes real estate listings from Sreality.cz, integrates geospatial data from the Prague Metro system (via Google Maps API), and creates a **Feasible Generalized Least Squares (FGLS)** regression model to predict apartment prices in Prague.

---

## 📋 Project Overview

This repository contains a full pipeline that:

1. **Scrapes Data**: Fetches hundreds of real estate listings (~2500-3500) from [Sreality.cz](https://www.sreality.cz) (the largest Czech real estate portal).

2. **Geolocates**: Uses the Google Maps API to obtain exact GPS coordinates for all Prague Metro stations.

3. **Feature Engineering**:
   - Calculates the **Haversine distance** from every apartment to every metro station
   - Identifies the **nearest station** and specific metro line (A, B, C)
   - Parses complex JSON metadata (floor, usable area, building material, layout)

4. **Modeling**: Trains a **Feasible GLS** regression model to identify key price drivers while addressing heteroskedasticity in the data.

5. **Interactive Web Application**: Deploys a Streamlit dashboard with:
   - Price prediction tool with confidence intervals
   - Market insights by metro station
   - Interactive map of property listings

---

## 📁 Project Structure

```
SemestralProject/
│
├── app.py                              # Streamlit web application
├── Estate_PRG.ipynb                    # Main analysis notebook (EDA, visualization, modeling)
├── main.py                             # Main script: Scraping logic for Sreality.cz and Google Maps API, preprocessing, modelling
│
├── estate_model.pkl                    # Trained FGLS model (pickled)
├── data_estate_processed.csv           # Final cleaned dataset with features
├── metro_stations.csv                  # GPS coordinates of Prague metro stations
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Conda** or **pip** for package management
- **Google Maps API Key** (for geocoding metro stations)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/michal-dvoracek01/Semestral-Project-Prague-Real-Estate.git
   cd Semestral-Project-Prague-Real-Estate
   ```

2. **Create a virtual environment:**
   ```bash
   conda create -n project_real_estate python=3.11
   conda activate project_real_estate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Set up Google Maps API:**
   - Obtain an API key from [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the **Places API** and **Geocoding API**
   - Store your API key in an environment variable or configuration file:
     ```bash
     export GOOGLE_MAPS_API_KEY='your_api_key_here'
     ```

---

## 🔧 Configuration

### Data Collection

The scraping module (`main.py`) uses:
- **Sreality.cz API**: No authentication required (public listings)
- **Google Maps API**: Requires valid API key with Places/Geocoding enabled

Update the API key in `main.py`:
```python
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
```

### Model Parameters

The FGLS model is pre-trained and stored in `estate_model.pkl`. To retrain:
1. Open `Estate_PRG.ipynb` to see the configuration in notebook
2. Change parameters in `main.py` and run
3. The model will be saved automatically
---

## 🐍 Running the Script

### Main.py

```bash
python main.py
```

---

## 💻 Running the Application

### Streamlit Web App

Launch the interactive price predictor:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- 📊 Input apartment characteristics (area, floor, layout, condition)
- 🚇 Select nearest metro station and distance indicators
- 💰 Get estimated price with 95% confidence interval
- 📈 View market insights by metro station
- 🗺️ Explore interactive map of all listings

### Jupyter Notebooks

For data exploration and model development:

```bash
jupyter notebook Estate_PRG.ipynb
```

---

## 📊 Dataset Description

### Raw Data Source
- **Sreality.cz listings** (as of January 2026)
- **Prague Metro stations** geocoded via Google Maps

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `price` | Numeric | Listing price in CZK |
| `area_usable` | Numeric | Usable apartment area in m² |
| `floor_number` | Numeric | Floor number (-1 for basement) |
| `has_elevator` | Binary | Elevator presence (0/1) |
| `has_terrace` | Numeric | Terrace area in m² |
| `layout_*` | One-hot | Apartment layout (1kk, 2kk, 3+1, etc.) |
| `building_type_*` | One-hot | Building material (panel, brick, etc.) |
| `status_*` | One-hot | Condition (new, very_good, development) |
| `nearest_station_*` | One-hot | Closest metro station |
| `1000m_A/B/C` | Binary | Within 1km of metro line A/B/C |
| `latitude`, `longitude` | Numeric | GPS coordinates |

## 🤖 Model Details

### Algorithm
**Feasible Generalized Least Squares (FGLS)**
- Addresses heteroskedasticity in error terms
- More efficient than OLS when variance is non-constant
- Implemented using `statsmodels`

### Performance Metrics
- **R²**: >0.80 (>80% variance explained)

---

## ⚠️ Disclaimer

**This project is for educational and research purposes only.**

- **Data Accuracy**: Listings are scraped from public sources and may contain errors or outdated information
- **Model Limitations**: Predictions are estimates based on historical data and should not be used as sole basis for financial decisions
- **Legal Notice**: Web scraping should comply with the target website's Terms of Service. Sreality.cz data is used for academic purposes only
- **No Warranty**: The authors provide no warranty regarding the accuracy, completeness, or fitness for any particular purpose

**By using this software, you agree that:**
- You will not use it for commercial purposes without proper authorization
- You will respect rate limits and robots.txt when scraping
- You understand that real estate prices are influenced by many factors not captured in this model

---

## 🔮 Future Improvements

- [ ] Add time-series forecasting (requires periodic scraping with timestamps)
- [ ] Deploy on cloud (AWS/Google Cloud)
- [ ] Add user authentication and saved predictions
- [ ] Create API endpoint for programmatic access

---

## 📧 Contact

**Author**: Michal Dvořáček  
**Institution**: IES FSV UK (Charles University)
**Email**: [76182031@fsv.cuni.com]  
**GitHub**: [@michal-dvoracek01](https://github.com/michal-dvoracek01)

---

**Last Updated**: February 1, 2026  
**Version**: 1.0.0
