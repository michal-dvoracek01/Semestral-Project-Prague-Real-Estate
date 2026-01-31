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
import pickle

COLUMN_MAPPING ={
    # Original : New, snake_case
    'price_czk': 'price',
    'lat': 'latitude',
    'lon': 'longitude',
    'usable_area': 'area_usable',
    'terrace': 'has_terrace',
    'elevator': 'has_elevator',
    'floor_num': 'floor_number',
    
    # One-hot encoded columns (cleaning up the prefixes)
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
    
    'building_type_Dřevostavba': 'building_type_wood',
    'building_type_Kamenná': 'building_type_stone',
    'building_type_Modulární': 'building_type_modular',
    'building_type_Montovaná': 'building_type_assembled',
    'building_type_Panelová': 'building_type_panel',
    'building_type_Skeletová': 'building_type_skeleton',
    'building_type_Smíšená': 'building_type_mixed',
    
    'building_condition_Novostavba': 'condition_new_build',
    'building_condition_Ve výstavbě': 'condition_under_construction',
    'building_condition_Velmi dobrý': 'condition_very_good',
    
    'condition_in development': 'status_development',
    'condition_new': 'status_new',
    'condition_very good': 'status_very_good'
}

def get_secret(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")


# def load_metro_data(api_key, sleep_time=0.1):
#     """
#     Loads Prague metro stations and fetches GPS coordinates using Google Maps API.
    
#     Returns:
#         pandas.DataFrame with columns: line, name, lat, lng
#     """

#     prague_metro = {
#         "A": [
#             "Nemocnice Motol", "Petřiny", "Nádraží Veleslavín", "Bořislavka",
#             "Dejvická", "Hradčanská", "Malostranská", "Staroměstská",
#             "Můstek", "Muzeum", "Náměstí Míru", "Jiřího z Poděbrad",
#             "Flora", "Želivského", "Strašnická", "Skalka", "Depo Hostivař"
#         ],
#         "B": [
#             "Zličín", "Stodůlky", "Luka", "Lužiny", "Hůrka",
#             "Nové Butovice", "Jinonice", "Radlická", "Smíchovské nádraží",
#             "Anděl", "Karlovo náměstí", "Národní třída", "Můstek",
#             "Náměstí Republiky", "Florenc", "Křižíkova", "Invalidovna",
#             "Palmovka", "Českomoravská", "Vysočanská", "Kolbenova",
#             "Hloubětín", "Rajská zahrada", "Černý Most"
#         ],
#         "C": [
#             "Letňany", "Prosek", "Střížkov", "Ládví", "Kobylisy",
#             "Nádraží Holešovice", "Vltavská", "Florenc", "Hlavní nádraží",
#             "Muzeum", "I. P. Pavlova", "Vyšehrad", "Pražského povstání",
#             "Pankrác", "Budějovická", "Kačerov", "Roztyly",
#             "Chodov", "Opatov", "Háje"
#         ]
#     }

#     gmaps = googlemaps.Client(key=api_key)

#     records = []

#     def get_station_coordinates(line, station_name):
#         try:
#             query = f"Metro {line} {station_name}, Prague, Czech Republic"
#             response = gmaps.places(query=query)

#             if response["status"] == "OK" and response["results"]:
#                 loc = response["results"][0]["geometry"]["location"]
#                 return loc["lat"], loc["lng"]
#             else:
#                 return None, None

#         except Exception as e:
#             print(f"Error for {station_name}: {e}")
#             return None, None

#     print("Starting GPS extraction...")

#     for line, stations in prague_metro.items():
#         print(f"--- Line {line} ---")

#         for station in stations:
#             lat, lng = get_station_coordinates(line, station)

#             records.append({
#                 "line": line,
#                 "name": station,
#                 "lat": lat,
#                 "lng": lng
#             })

#             print(f"{station}: {lat}, {lng}")
#             time.sleep(sleep_time)

#     return pd.DataFrame(records)

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

# Function to parse the 'detail' column
def parse_details_list(details_list):
    extracted = {}
    
    # If the details_list is a string, try to convert it to a list
    if isinstance(details_list, str):
        try:
            details_list = ast.literal_eval(details_list)
        except:
            return extracted

    if not isinstance(details_list, list):
        return extracted

    # Iterate through each item in list
    for item in details_list:
        if isinstance(item, dict):
            name = item.get('name')
            value = item.get('value')
            
            # Define the mappings based on 'name'
            if name == 'Stav objektu':
                extracted['building_condition'] = value
            elif name == 'Užitná plocha' or name == 'Užitná ploch':
                extracted['usable_area'] = value
            elif name == 'Stavba':
                extracted['building_type'] = value
            elif name == 'Celková cena':
                extracted['price'] = value
            elif name == 'Terasa':
                extracted['terrace'] = value
            elif name == 'Výtah':
                extracted['elevator'] = value
            elif name == 'Podlaží':
                extracted['floor'] = value

    return extracted

def extract_details(detail):
    if isinstance(detail, str):
        try:
            detail = ast.literal_eval(detail)
        except:
            return {}
    if not isinstance(detail, list):
        return {}   
    details_dict = {}
    for item in detail:
        if isinstance(item, dict):
            name = item.get('name')
            value = item.get('value')
            details_dict[name] = value
    return details_dict

def extract_name(category_cb):
    if isinstance(category_cb, str):
        try:
            category_cb = ast.literal_eval(category_cb)
        except:
            return None
    if isinstance(category_cb, dict):
        return category_cb.get('name')
    return None


if __name__ == "__main__":

    # API_KEY = get_secret(
    # project_id="{insert your key here}",
    # secret_id="API_KEY_Google"
    # )
    # metro_df = load_metro_data(API_KEY)
    # print("\nMetro data preview:")
    # print(metro_df.head())

    sreality_df = sreality_scrape()
    
    if not sreality_df.empty:
        print("\nSreality data preview:")
        print(sreality_df.head())

        print(f"\nStarting detailed scraping for {len(sreality_df)} items...")
        details_list = asyncio.run(run_scraping(sreality_df['hash_id'].tolist()))
        
        estates_details_df = pd.DataFrame(details_list)
        print("\nEstate details preview:")
        print(estates_details_df.head())

        # Merge and ensure unique index
        sreality_df = sreality_df.drop_duplicates(subset='hash_id')
        all_df = pd.merge(sreality_df, estates_details_df, on='hash_id', how='left')
        all_df = all_df.reset_index(drop=True)
        print("\nFinal Merged Data Preview:")
        print(all_df.head())

        # Extract detail column BEFORE dropping it
        if 'detail' in all_df.columns:
            details_extracted = all_df['detail'].apply(parse_details_list)
            details_df = pd.DataFrame(details_extracted.tolist())
            all_df = pd.concat([all_df, details_df], axis=1)

        if 'usable_area' in all_df.columns:
            # Convert usable_area to numeric (extract numbers)
            all_df['usable_area'] = all_df['usable_area'].astype(str).str.extract(r'(\d+)').astype(float)

        # Drop unnecessary columns (metadata, images, logos, etc.)
        cols_to_drop = [
            'Unnamed: 0', 'hash_id', 'advert_images', 'advert_images_all',
            'premise_logo', 'user_id', 'premise_id', 'price_summary',
            'price_currency_cb', 'price_unit_cb', 'price_summary_unit_cb',
            'has_matterport_url', 'has_video', 'advert_name', 'premise'
        ]
        all_df = all_df.drop(columns=cols_to_drop, errors='ignore')

        # Remove rows with missing price
        all_df = all_df.dropna(subset=['price_czk'])
        all_df = all_df.reset_index(drop=True)

        # Extract locality information
        locality_df = pd.json_normalize(all_df['locality'])
        all_df['city'] = locality_df.get('city')
        all_df['district'] = locality_df.get('district')
        all_df['lat'] = locality_df.get('gps_lat')
        all_df['lon'] = locality_df.get('gps_lon')

        # Extract floor number
        def extract_floor_num(floor_str):
            if not isinstance(floor_str, str):
                return 0
            floor_str = floor_str.lower()
            if 'přízemí' in floor_str:
                return 0
            elif 'suterén' in floor_str:
                return -1
            match = re.search(r'(-?\d+)', floor_str)
            if match:
                return int(match.group(1))
            return 0

        if 'floor' in all_df.columns:
            all_df['floor_num'] = all_df['floor'].apply(extract_floor_num)

        # Convert terrace and elevator
        for col in ['terrace', 'elevator']:
            if col in all_df.columns:
                all_df[col] = all_df[col].fillna(0).astype(int)

        # Drop unnecessary columns (detail already extracted and parsed)
        cols_to_remove = ['locality', 'locality_dict', 'detail', 'price', 'price_czk_m2', 'price_summary_czk', 'floor', 'discount_show', 'condition']
        all_df = all_df.drop(columns=cols_to_remove, errors='ignore')

        # Prepare categorical columns for encoding
        categorical_cols = [
            'category_main_cb', 'category_sub_cb', 'category_type_cb',
            'city', 'district', 'building_type', 'building_condition',
            'ownership', 'condition'
        ]
        categorical_cols = [c for c in categorical_cols if c in all_df.columns]

        # Extract 'name' field from dictionary columns before encoding
        for col in categorical_cols:
            if all_df[col].dtype == 'object':
                if all_df[col].apply(lambda x: isinstance(x, dict)).any():
                    all_df[col] = all_df[col].apply(lambda x: x.get('name') if isinstance(x, dict) else x)

        # Create distance-based binary features
        dist_cols = [col for col in all_df.columns if col.startswith('poi_')]
        for col in dist_cols:
            new_col_name = col.replace('poi_', '500m_')
            all_df[new_col_name] = (all_df[col] <= 500).astype(int)

        all_df.drop(columns=dist_cols, inplace=True)

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(all_df, columns=categorical_cols, drop_first=True)

        print(f"Data ready! Shape: {df_encoded.shape}")

        # Filter out very low-price listings and cast booleans
        if 'price_czk' in df_encoded.columns:
            df_encoded = df_encoded[df_encoded['price_czk'] > 1_000_000]
        bool_cols = df_encoded.select_dtypes(include=['bool']).columns
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

        df_encoded = df_encoded.rename(columns=COLUMN_MAPPING)

        # Load metro stations and compute distances
        try:
            metro = pd.read_csv("metro_stations.csv")
        except Exception:
            metro = pd.DataFrame()

        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371000
            return c * r

        def get_metro_distances(row):
            flat_lat = row.get('latitude') if 'latitude' in row else row.get('lat')
            flat_lon = row.get('longitude') if 'longitude' in row else row.get('lon')

            results = {'dist_A': 999999, 'dist_B': 999999, 'dist_C': 999999, 'nearest_name': None, 'global_min_dist': 999999}

            if metro.empty or pd.isna(flat_lat) or pd.isna(flat_lon):
                return pd.Series([results['dist_A'], results['dist_B'], results['dist_C'], results['nearest_name']])

            for _, station in metro.iterrows():
                station_line = station.get('line')
                station_name = station.get('name')
                station_lat = station.get('lat')
                station_lng = station.get('lng')
                try:
                    dist = haversine(flat_lon, flat_lat, station_lng, station_lat)
                except Exception:
                    continue

                col_name = f'dist_{station_line}'
                if col_name in results and dist < results[col_name]:
                    results[col_name] = dist

                if dist < results['global_min_dist']:
                    results['global_min_dist'] = dist
                    results['nearest_name'] = station_name

            return pd.Series([results['dist_A'], results['dist_B'], results['dist_C'], results['nearest_name']])

        if 'latitude' in df_encoded.columns or 'lat' in df_encoded.columns:
            df_encoded[['metro_dist_A', 'metro_dist_B', 'metro_dist_C', 'nearest_station']] = df_encoded.apply(get_metro_distances, axis=1)

            # Create 1000m proximity flags
            dist_cols = [c for c in df_encoded.columns if c.startswith('metro_dist_')]
            for col in dist_cols:
                new_col = col.replace('metro_dist_', '1000m_')
                df_encoded[new_col] = (df_encoded[col] <= 1000).astype(int)

            # One-hot encode nearest station
            if 'nearest_station' in df_encoded.columns:
                df_encoded = pd.get_dummies(df_encoded, columns=['nearest_station'], drop_first=True)

        # Save processed dataset (intermediate)
        df_encoded.to_csv('data_estate_processed.csv', index=False)
        print(f"Saved processed dataset to data_estate_processed.csv (shape {df_encoded.shape})")
print("\nModelling (Feasible GLS/WLS)")

if 'price' in df_encoded.columns:
    # Define all 95 feature parameters
    FEATURES_LIST = [
        'area_usable', 'has_elevator', 'has_terrace', 'floor_number',
        '500m_atm_distance', '500m_bus_public_transport_distance', '500m_kindergarten_distance',
        '500m_medic_distance', '500m_metro_distance', '500m_playground_distance',
        '500m_post_office_distance', '500m_restaurant_distance', '500m_school_distance',
        '500m_shop_distance', '500m_small_shop_distance', '500m_train_distance', '500m_vet_distance',
        'layout_1kk', 'layout_21', 'layout_2kk', 'layout_31', 'layout_3kk', 'layout_41', 'layout_4kk',
        'layout_51', 'layout_5kk', 'layout_6_plus', 'layout_atypical',
        'building_type_wood', 'building_type_stone', 'building_type_modular', 'building_type_assembled',
        'building_type_panel', 'building_type_skeleton', 'building_type_mixed',
        'status_development', 'status_new', 'status_very_good'
    ]
    
    # All 57 Prague metro stations
    stations = [
        'Bořislavka', 'Budějovická', 'Chodov', 'Dejvická', 'Depo Hostivař', 'Flora', 'Florenc',
        'Hlavní nádraží', 'Hloubětín', 'Hradčanská', 'Háje', 'Hůrka', 'I. P. Pavlova', 'Invalidovna',
        'Jinonice', 'Jiřího z Poděbrad', 'Karlovo náměstí', 'Kačerov', 'Kobylisy', 'Kolbenova',
        'Křižíkova', 'Letňany', 'Luka', 'Lužiny', 'Ládví', 'Malostranská', 'Muzeum', 'Můstek',
        'Nemocnice Motol', 'Nové Butovice', 'Nádraží Holešovice', 'Nádraží Veleslavín', 'Náměstí Míru',
        'Náměstí Republiky', 'Národní třída', 'Opatov', 'Palmovka', 'Pankrác', 'Petřiny',
        'Pražského povstání', 'Prosek', 'Radlická', 'Rajská zahrada', 'Roztyly', 'Skalka',
        'Smíchovské nádraží', 'Staroměstská', 'Stodůlky', 'Strašnická', 'Střížkov', 'Vltavská',
        'Vysočanská', 'Vyšehrad', 'Zličín', 'Černý Most', 'Českomoravská', 'Želivského'
    ]
    FEATURES_LIST += [f'nearest_station_{s}' for s in stations]
    
    # Fill missing features with 0
    for col in FEATURES_LIST:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Extract X explicitly (all 95 features)
    X = df_encoded[FEATURES_LIST].astype(float)
    y = df_encoded['price']
    
    print(f"Model will use {X.shape[1]} features (before cleaning)")
    print(f"Initial samples: {len(y)}")
    
    # Remove rows with NaN or Inf in X or y
    valid_idx = ~(X.isna().any(axis=1) | y.isna() | np.isinf(X.values).any(axis=1) | np.isinf(y.values))
    X = X[valid_idx].reset_index(drop=True)
    y = y[valid_idx].reset_index(drop=True)
    
    print(f"Samples after removing NaN/Inf: {len(y)}")
    
    if len(y) < 10:
        print("ERROR: Not enough valid samples for modeling")
        exit(1)
    
    # Add constant for intercept
    X_const = sm.add_constant(X)
    
    # 1. OLS to detect heteroscedasticity
    print("Computing OLS to detect heteroscedasticity...")
    ols_model = sm.OLS(y, X_const).fit()
    
    # Breusch-Pagan test
    residuals = ols_model.resid
    try:
        bp_test = het_breuschpagan(residuals, X_const)
        print(f'Breusch-Pagan p-value: {bp_test[1]:.4f} (if < 0.05, WLS is recommended)')
    except Exception as e:
        print(f"Breusch-Pagan test failed: {e}")
    
    # Remove specific high-VIF variables
    print("\n" + "="*80)
    print("Removing high-VIF variables...")
    print("="*80)
    high_vif_vars = ['500m_playground_distance', '500m_bus_public_transport_distance']
    X_clean = X.drop(columns=high_vif_vars, errors='ignore')
    print(f"Removed {len(high_vif_vars)} variables: {high_vif_vars}")
    print(f"Features after VIF filtering: {X_clean.shape[1]} (reduced from {X.shape[1]})")
    
    # Feasible GLS: estimate variance weights (on cleaned X)
    print("\nComputing Feasible GLS (WLS) model...")
    X_const_clean = sm.add_constant(X_clean)
    
    ols_model_clean = sm.OLS(y, X_const_clean).fit()
    residuals_clean = ols_model_clean.resid
    log_resid_sq = np.log(residuals_clean**2 + 1e-10)
    var_model = sm.OLS(log_resid_sq, X_const_clean).fit()
    weights = 1.0 / np.exp(var_model.fittedvalues)
    
    # 3. Final WLS model
    wls_model = sm.WLS(y, X_const_clean, weights=weights).fit()
    print("\n" + "="*80)
    print("FEASIBLE GLS (WLS) MODEL SUMMARY")
    print("="*80)
    print(wls_model.summary())
    
    # VIF for non-constant columns with variance > 0
    print("\n" + "="*80)
    print("VIF FACTORS (Multicollinearity Check)")
    print("="*80)
    try:
        vif_cols = [c for c in X_clean.columns if X_clean[c].var() > 1e-6]
        vif_data = pd.DataFrame()
        vif_data["Variable"] = vif_cols
        vif_data["VIF"] = [variance_inflation_factor(X_clean[vif_cols].values, i) for i in range(len(vif_cols))]
        print(vif_data.sort_values("VIF", ascending=False).head(60))
    except Exception as e:
        print(f"VIF calculation error: {e}")
    
    # Plot residuals
    # try:
    #     plt.figure(figsize=(12, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.scatter(wls_model.fittedvalues, wls_model.resid, alpha=0.5)
    #     plt.axhline(0, color='red', linestyle='--')
    #     plt.xlabel('Predicted prices')
    #     plt.ylabel('Residuals')
    #     plt.title('WLS Residuals vs Fitted')
        
    #     plt.subplot(1, 2, 2)
    #     plt.hist(wls_model.resid, bins=50, edgecolor='black')
    #     plt.xlabel('Residuals')
    #     plt.ylabel('Frequency')
    #     plt.title('Distribution of Residuals')
    #     plt.tight_layout()
    #     plt.show()
    # except Exception as e:
    #     print(f"Plotting error: {e}")

# Save the model and the feature list
clean_feature_names = [str(col) for col in X_clean.columns.tolist()]

model_data = {
    'model': wls_model,
    'features': clean_feature_names 
}

# Save
with open('estate_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model successfully re-exported with plain Python types.")