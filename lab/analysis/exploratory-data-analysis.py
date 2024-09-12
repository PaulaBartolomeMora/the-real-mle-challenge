import os
from pathlib import Path
import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIR_REPO = Path.cwd().parent.parent
logger.info(f"{DIR_REPO}")
DIR_DATA_RAW = Path(DIR_REPO) / "data" / "raw"
DIR_DATA_PROCESSED = Path(DIR_REPO) / "data" / "processed"

COLUMNS = [
    'id', 'neighbourhood_group_cleansed', 'property_type', 'room_type',
    'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'amenities', 'price'
]

# CARGA DE DATOS
def load_data(file_path: Path) -> DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# PREPROCESAMIENTO 'BATHROOMS'
def num_bathroom_from_text(text: str) -> float:
    try:
        if isinstance(text, str):
            bath_num = text.split(" ")[0]
            return float(bath_num)
        return np.NaN
    except ValueError:
        return np.NaN

# PREPROCESAMIENTO
def preprocess_data(df: DataFrame) -> DataFrame:
    df = df.copy()
    
    tqdm.pandas(desc="Processing bathrooms")
    df['bathrooms'] = df['bathrooms_text'].progress_apply(num_bathroom_from_text)
    
    df = df[COLUMNS].copy()
    df.rename(columns={'neighbourhood_group_cleansed': 'neighbourhood'}, inplace=True)
   
    df.dropna(axis=0, inplace=True)
    
    df['price'] = df['price'].str.extract(r"(\d+).")
    df['price'] = df['price'].astype(int)
    
    return df

# VISUALIZACIÓN DE LA DISTRIBUCIÓN DE PRECIOS
def plot_price_distribution(df: DataFrame):
    fontsize_labels = 12
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['price'], bins=range(0, max(df['price']), 10))
    ax.grid(alpha=0.2)
    ax.set_title('Price distribution', fontsize=fontsize_labels)
    ax.set_ylabel('Number of properties', fontsize=fontsize_labels)
    ax.set_xlabel('Price ($)', fontsize=fontsize_labels)
    plt.show()
    
# VISUALIZACIÓN DE LA DISTRIBUCIÓN SEGÚN EL VECINDARIO 
def plot_neighbourhood_distribution(df: DataFrame):
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    neighbourhoods = ['Manhattan', 'Brooklyn', 'Queens', 'Staten Island', 'Bronx']
    
    for i, ax in enumerate(axes):
        values = df[df['neighbourhood'] == neighbourhoods[i]]['price']
        avg = round(values.mean(), 1)
        ax.hist(values, bins=range(0, max(df['price']), 20))
        ax.set_title(f'{neighbourhoods[i]}. Avg price: ${avg}', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

    axes[-1].set_xlabel('Price ($)', fontsize=12)
    plt.tight_layout()
    plt.show()

# CATEGORIZACIÓN DE PRECIOS
def categorize_price(df: DataFrame) -> DataFrame:
    df['category'] = pd.cut(df['price'], bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3]) 
    return df

# PREPROCESAMIENTO COLUMNA 'AMENITIES'
def preprocess_amenities_column(df: DataFrame) -> DataFrame:
    amenities_to_extract = [
        'TV', 'Internet', 'Air conditioning', 'Kitchen', 'Heating',
        'Wifi', 'Elevator', 'Breakfast'
    ]
    
    tqdm.pandas(desc="Processing amenities")
    for amenity in amenities_to_extract:
        df[amenity] = df['amenities'].progress_apply(lambda x: int(amenity in x))
    
    df.drop('amenities', axis=1, inplace=True)
    return df

# ALMACENAMIENTO DATOS PROCESADOS EN CSV
def save_processed_data(df: DataFrame, output_path: Path):
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")

# FUNCIÓN MAIN
def main():
    filepath_data = DIR_DATA_RAW / "listings.csv"
    df_raw = load_data(filepath_data)
    
    # PREPROCESAMIENTO
    df = preprocess_data(df_raw)
    df = categorize_price(df)
    df = preprocess_amenities_column(df)
    
    # PLOT
    plot_price_distribution(df)
    plot_neighbourhood_distribution(df)
    
    # ALMACENAMIENTO
    filepath_processed = DIR_DATA_PROCESSED / "preprocessed_listings.csv"
    save_processed_data(df, filepath_processed)

if __name__ == "__main__":
    main()
