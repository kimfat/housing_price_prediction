import argparse
import os
import datetime
import pandas as pd
import joblib
import logging
from cianparser import CianParser
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# Настройка логирования
logging.basicConfig(
    filename="./logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def preprocess_data(input_csv: str, output_dir: str) -> str:
    """
    Preprocesses the raw data CSV.

    Args:
        input_csv (str): Path to the raw data file.
        output_dir (str): Directory to save the processed data.

    Returns:
        str: Path to the processed CSV file.
    """
    raw_df = pd.read_csv(input_csv)
    df = raw_df[['url', 'floor', 'floors_count', 'rooms_count', 'total_meters', 'author_type', 'price']]
    def get_url_id(url):
        return url[30:-1]
    df['id'] = df['url'].apply(get_url_id)
    df = df.drop('url', axis=1)
    df = df.drop_duplicates()
    df = df[df['price'] > 0]  # Удалим строки с нереальной ценой -1


    # Категориальные переменные
    cat_features = ['author_type']

    for col in cat_features:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    processed_path = os.path.join(output_dir, f"processed_{os.path.basename(input_csv)}")
    df.to_csv(processed_path, index=False)
    logging.info(f"Processed data saved to {processed_path}")
    logging.info(f"Sample data:\n{df.head()}")
    return processed_path

def main():
    """
    Main function to execute preprocess data.
    """
    parser = argparse.ArgumentParser(description="Run ML pipeline.")
    parser.add_argument("--raw_data_source", type=str, required=False, default=os.path.join("data/raw", "raw_2025-05-19_20-58.csv"), help="raw data path")

    args = parser.parse_args()

    data_processed_dir = "data/processed"

    raw_csv = args.raw_data_source
    processed_csv = preprocess_data(raw_csv, data_processed_dir)
    logging.info(f"preproces finished. at {processed_csv}")


if __name__ == "__main__":
    main()