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
    filename="./logs/collect_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def collect_data(output_dir: str) -> str:
    """
    Collects real estate data from cian.ru and saves it as CSV.

    Args:
        output_dir (str): Directory to save the raw data.

    Returns:
        str: Path to the saved raw CSV file.
    """
    parser = CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = os.path.join(output_dir, f"raw_{t}.csv")
    rooms = (1, 2, 3)
    dfs = []
    for room in rooms:
        data = parser.get_flats(
            deal_type="sale",
            rooms=(1, 2, 3),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 56,
                "object_type": "secondary",
                "min_price": 1000000,
                "max_price": 100000000
            }
        )
        dfs.append(pd.DataFrame(data))
    df = pd.concat(dfs)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logging.info(f"Raw data saved to {csv_path}")
    return csv_path

def main():
    """
    Main function to execute collecting data
    """
    data_raw_dir = "data/raw"
    data_processed_dir = "data/processed"
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_processed_dir, exist_ok=True)

    csv_path = collect_data(data_raw_dir)
    logging.info(f"collecting data finished at {csv_path}")

if __name__ == "__main__":
    main()