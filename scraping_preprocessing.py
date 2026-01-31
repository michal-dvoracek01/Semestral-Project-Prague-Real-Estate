import ast
import asyncio
import re
import httpx
import pandas as pd
import numpy as np
import googlemaps
import time
import requests
from google.cloud import secretmanager
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

def sreality_scrape():
    """Scrapes flat listings from Sreality.cz for Prague based on building conditions.
    """
    BASE_URL = (
        "https://www.sreality.cz/api/v1/estates/search?"
        "category_main_cb=1"
        "&category_type_cb=1"
        "&locality_country_id=112"
        "&locality_region_id=10"
        "&building_condition={condition}"
        "&ownership=1"
        "&limit={limit}"
        "&offset={offset}"
        "&sort=-date"
        "&lang=cs"
    )

    limit = 22
    all_dfs = []

    condition_map = {
        1: "very good",
        2: "good",
        4: "in development",
        6: "new"
    }

    for condition_code, condition_name in condition_map.items():
        print(f"\nDownloading flats: {condition_name}")
        offset = 0
        total_results = float("inf")
        all_results = []

        try:
            while offset < total_results:
                url = BASE_URL.format(
                    condition=condition_code,
                    limit=limit,
                    offset=offset
                )

                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                all_results.extend(results)

                if total_results == float("inf"):
                    total_results = data["pagination"]["total"]

                offset += limit

            df = pd.DataFrame(all_results)
            df["condition"] = condition_name

            print(f"{len(df)} flats downloaded ({condition_name})")
            all_dfs.append(df)
            
        except requests.exceptions.RequestException as e:
            print(f"Error for condition {condition_name}: {e}")

    # Return concatenated DataFrame
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

async def get_estate_detail(client, hash_id, semaphore):
    url = f"https://www.sreality.cz/api/cs/v2/estates/{hash_id}"
    headers = {"User-Agent": "Mozilla/5.0"}

    async with semaphore:
        try:
            await asyncio.sleep(0.02)
            r = await client.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            data_json = r.json()
            items = data_json.get("items", [])
            detail = items if items else None
            return {'hash_id': hash_id, 'detail': detail}
        except Exception as e:
            return {'hash_id': hash_id, 'detail': f"Chyba: {e}"}

async def run_scraping(hash_ids):
    semaphore = asyncio.Semaphore(15)
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)

    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        tasks = [asyncio.create_task(get_estate_detail(client, hid, semaphore)) for hid in hash_ids]

        results = []
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                res = await fut
            except Exception as e:
                res = {'hash_id': None, 'detail': f'Chyba: {e}'}
            results.append(res)

        for t in tasks:
            if not t.done():
                t.cancel()

        return results
    
def safe_eval(x):
        try:
            return ast.literal_eval(x) if pd.notna(x) else None
        except (ValueError, SyntaxError):
            return None
        
# Mapping for column renaming to ensure consistency and snake_case naming
COLUMN_MAPPING = {
    'price_czk': 'price',
    'lat': 'latitude',
    'lon': 'longitude',
    'usable_area': 'area_usable',
    'terrace': 'has_terrace',
    'elevator': 'has_elevator',
    'floor_num': 'floor_number',
    
    # Mapping for layout types (one-hot encoded)
    'category_sub_cb_1+kk': 'layout_1kk',
    'category_sub_cb_2+1': 'layout_21',
    'category_sub_cb_2+kk': 'layout_2kk',
    'category_sub_cb_3+1': 'layout_31',
    'category_sub_cb_3+kk': 'layout_3kk',
    'category_sub_cb_4+1': 'layout_41',
    'category_sub_cb_4+kk': 'layout_4kk',
    'category_sub_cb_5+1': 'layout_51',
    'category_sub_cb_5+kk': 'layout_5kk',
    'category_sub_cb_6 a více': 'layout_6_plus',
    'category_sub_cb_Atypický': 'layout_atypical',
    
    # Mapping for districts
    'district_Praha 1': 'district_praha_1',
    'district_Praha 10': 'district_praha_10',
    'district_Praha 2': 'district_praha_2',
    'district_Praha 3': 'district_praha_3',
    'district_Praha 4': 'district_praha_4',
    'district_Praha 5': 'district_praha_5',
    'district_Praha 6': 'district_praha_6',
    'district_Praha 7': 'district_praha_7',
    'district_Praha 8': 'district_praha_8',
    'district_Praha 9': 'district_praha_9',
    
    # Mapping for building materials
    'building_type_Dřevostavba': 'building_type_wood',
    'building_type_Kamenná': 'building_type_stone',
    'building_type_Modulární': 'building_type_modular',
    'building_type_Montovaná': 'building_type_assembled',
    'building_type_Panelová': 'building_type_panel',
    'building_type_Skeletová': 'building_type_skeleton',
    'building_type_Smíšená': 'building_type_mixed',
    
    # Mapping for property status
    'condition_in development': 'status_development',
    'condition_new': 'status_new',
    'condition_very good': 'status_very_good'
}

def parse_details_list(details_list):
    """
    Parses the 'items' list from Sreality API detail to extract specific property attributes.
    """
    extracted = {}
    if isinstance(details_list, str):
        try:
            details_list = ast.literal_eval(details_list)
        except:
            return extracted
    if not isinstance(details_list, list):
        return extracted
    
    for item in details_list:
        if isinstance(item, dict):
            name = item.get('name')
            value = item.get('value')
            # Extract relevant fields based on Czech names in API response
            if name == 'Stav objektu':
                extracted['building_condition'] = value
            elif name in ['Užitná plocha', 'Užitná ploch']:
                extracted['usable_area'] = value
            elif name == 'Stavba':
                extracted['building_type'] = value
            elif name == 'Terasa':
                extracted['terrace'] = value
            elif name == 'Výtah':
                extracted['elevator'] = value
            elif name == 'Podlaží':
                extracted['floor'] = value
    return extracted

def extract_floor_num(floor_str):
    """
    Converts floor description string (e.g., '4. podlaží', 'přízemí') to an integer.
    """
    if not isinstance(floor_str, str):
        return 0
    floor_str = floor_str.lower()
    if 'přízemí' in floor_str: return 0 # Ground floor
    elif 'suterén' in floor_str: return -1 # Basement
    match = re.search(r'(-?\d+)', floor_str)
    return int(match.group(1)) if match else 0

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the great-circle distance between two points in meters.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371000

# (Note: sreality_scrape and run_scraping functions are assumed to be present as per previous logic)

if __name__ == "__main__":
    # Start scraping basic listing data
    sreality_df = sreality_scrape()
    
    if not sreality_df.empty:
        # Step 1: Detailed scraping (asynchronous) to get specific property parameters
        print(f"Starting detailed scraping for {len(sreality_df)} items...")
        details_list = asyncio.run(run_scraping(sreality_df['hash_id'].tolist()))
        estates_details_df = pd.DataFrame(details_list)

        # Merge basic listings with details
        all_df = pd.merge(sreality_df, estates_details_df, on='hash_id', how='left')

        # Step 2: Extract technical parameters from the 'detail' JSON column
        print("Extracting detailed parameters...")
        details_extracted = all_df['detail'].apply(parse_details_list)
        details_df = pd.DataFrame(details_extracted.tolist())
        all_df = pd.concat([all_df, details_df], axis=1)

        # Step 3: Data Cleaning and Type Conversion
        if 'usable_area' in all_df.columns:
            # Convert area string (e.g. "85 m2") to numeric float
            all_df['usable_area'] = all_df['usable_area'].astype(str).str.extract('(\d+)').astype(float)
        
        if 'floor' in all_df.columns:
            # Parse floor string to integer
            all_df['floor_num'] = all_df['floor'].apply(extract_floor_num)

        for col in ['terrace', 'elevator']:
            if col in all_df.columns:
                # Convert boolean/string indicators to binary 0/1
                all_df[col] = all_df[col].apply(lambda x: 1 if x in [True, 'ano', 1, '1'] else 0)

        # Step 4: Extract Location and GPS data
        locality_df = pd.json_normalize(all_df['locality'])
        all_df['lat'] = locality_df.get('gps_lat')
        all_df['lon'] = locality_df.get('gps_lon')
        all_df['district'] = locality_df.get('district')

        # Step 5: Prepare Categorical Columns for Encoding
        categorical_cols = ['category_sub_cb', 'district', 'building_type', 'condition']
        categorical_cols = [c for c in categorical_cols if c in all_df.columns]

        for col in categorical_cols:
            if all_df[col].dtype == 'object':
                # Extract 'name' if column contains dictionary objects
                all_df[col] = all_df[col].apply(lambda x: x.get('name') if isinstance(x, dict) else x)

        # Step 6: Create POI Flags (500m proximity)
        dist_cols = [col for col in all_df.columns if col.startswith('poi_')]
        for col in dist_cols:
            all_df[col.replace('poi_', '500m_')] = (all_df[col] <= 500).astype(int)

        # Step 7: One-Hot Encoding and Column Renaming
        df_encoded = pd.get_dummies(all_df, columns=categorical_cols, drop_first=True)
        df_encoded = df_encoded.rename(columns=COLUMN_MAPPING)

        # Step 8: Calculate Metro Distances and create 1000m Flags
        try:
            metro = pd.read_csv("metro_stations.csv")
            def get_metro_data(row):
                res = {'dist_A': 999999, 'dist_B': 999999, 'dist_C': 999999}
                for _, s in metro.iterrows():
                    d = haversine(row['longitude'], row['latitude'], s['lng'], s['lat'])
                    k = f"dist_{s['line']}"
                    if d < res[k]: res[k] = d
                return pd.Series([res['dist_A'], res['dist_B'], res['dist_C']])

            m_cols = ['metro_dist_A', 'metro_dist_B', 'metro_dist_C']
            df_encoded[m_cols] = df_encoded.apply(get_metro_data, axis=1)
            
            # Create proximity flags for each metro line
            for line in ['A', 'B', 'C']:
                df_encoded[f'1000m_dist_{line}'] = (df_encoded[f'metro_dist_{line}'] <= 1000).astype(int)
        except:
            print("Metro stations file not found, skipping metro calculations.")

        # Step 9: Final Dataset Filtering
        # Keep only numeric columns and drop rows with missing prices
        df_encoded = df_encoded.select_dtypes(include=[np.number]).dropna(subset=['price'])
        # Filter out extreme outliers or incorrect listings below 1M CZK
        df_encoded = df_encoded[df_encoded['price'] > 1_000_000]
        
        # Save the finalized processed dataset
        df_encoded.to_csv('data_estate_processed.csv', index=False)
        print(f"Dataset saved with {df_encoded.shape[1]} parameters.")

        # Step 10: Modeling using Weighted Least Squares (WLS)
        X = df_encoded.drop(columns=['price', 'latitude', 'longitude'], errors='ignore').fillna(0)
        y = df_encoded['price']
        X_const = sm.add_constant(X)
        
        # Fit initial OLS model to estimate residuals for weighting
        ols_res = sm.OLS(y, X_const).fit()
        
        # Estimate variance to compute weights (addressing heteroskedasticity)
        log_resid_sq = np.log(ols_res.resid**2 + 1e-10)
        var_model = sm.OLS(log_resid_sq, X_const).fit()
        weights = 1.0 / np.exp(var_model.fittedvalues)
        
        # Fit final WLS model
        wls_model = sm.WLS(y, X_const, weights=weights).fit()
        
        print("\n--- WLS Model Summary ---")
        print(wls_model.summary())