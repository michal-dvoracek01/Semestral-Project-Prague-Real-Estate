import pandas as pd
import numpy as np
from scraping_functions import (
    get_name,
    get_first_image_url,
    get_first_plain_image_url,
    get_city,
    get_gps_lon,
    get_gps_lat,
    get_region,
    get_district,
    get_citypart,
    assign_nearest_metro 
)

def preprocess_flats(input_file):
    df = pd.read_csv(f"{input_file}")

    df['category_main_name'] = df['category_main_cb'].apply(get_name)
    df['category_sub_name'] = df['category_sub_cb'].apply(get_name)
    df['category_type_name'] = df['category_type_cb'].apply(get_name)
    df['price_currency_name'] = df['price_currency_cb'].apply(get_name)
    df['price_summary_unit_name'] = df['price_summary_unit_cb'].apply(get_name)
    df['price_unit_name'] = df['price_unit_cb'].apply(get_name)

    df['first_advert_image'] = df['advert_images'].apply(get_first_plain_image_url)
    df['first_advert_image_all'] = df['advert_images_all'].apply(get_first_image_url)

    df['city'] = df['locality'].apply(get_city)
    df['region'] = df['locality'].apply(get_region)
    df['gps_lat'] = df['locality'].apply(get_gps_lat)
    df['gps_lon'] = df['locality'].apply(get_gps_lon)
    df['district'] = df['locality'].apply(get_district)
    df['citypart'] = df['locality'].apply(get_citypart)

    return df

if __name__ == "__main__":
    metro_df = pd.read_csv("metro_stations.csv")
    flats_df = preprocess_flats("sreality_flats.csv")
    print("\nPreprocessed flats data preview:")
    print(flats_df.head())