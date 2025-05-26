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
from xgboost import XGBRegressor


# Настройка логирования
logging.basicConfig(
    filename="./logs/pipeline.log",
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


def train_and_evaluate(csv_path: str, model_name: str, test_size: float) -> None:
    """
    Trains a model and evaluates it using test data sorted by url_id.

    Args:
        csv_path (str): Path to processed data.
        model_name (str): Name for the saved model file.
        test_size (float): Fraction of data to be used for testing.
    """
    df = pd.read_csv(csv_path)
    if 'price' not in df.columns:
        raise ValueError("Missing 'price' column in data.")
    if 'id' not in df.columns:
        raise ValueError("Missing 'url_id' column to sort data.")

    df.sort_values(by='id', inplace=True)  # Use id to sort chronologically
        # Добавляем признаки первого и последнего этажа
    df['first_floor'] = df['floor'] == 1
    df['last_floor'] = df['floor'] == df['floors_count']
    df = df.drop(columns=['id'])

    y = df['price']
    X = df.drop(columns=['price'])

    split_index = int((1 - test_size) * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    logging.info(f"Model: {model_name}")
    logging.info(f"MSE: {mse:.2f}")
    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"R2: {r2:.2f}")
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"Sample predictions: {y_pred[:5]}")

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', model_name)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")


def main():
    """
    Main function to execute full pipeline via CLI.
    """
    parser = argparse.ArgumentParser(description="Run ML pipeline.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to save (e.g., xgbregressor.pkl)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (0.1 - 0.5)")
    parser.add_argument("--ready_raw_data", type=str, default=None, help="Name of already parsed raw data (e.g., raw_2025-05-19_20-58.csv)")

    args = parser.parse_args()

    data_raw_dir = "data/raw"
    data_processed_dir = "data/processed"
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_processed_dir, exist_ok=True)

    logging.info("Pipeline started.")
    if args.ready_raw_data is None:
        raw_csv = collect_data(data_raw_dir)
    else:
        raw_csv = os.path.join(data_raw_dir, args.ready_raw_data)
    processed_csv = preprocess_data(raw_csv, data_processed_dir)
    train_and_evaluate(processed_csv, args.model_name, args.test_size)
    logging.info("Pipeline finished.")


if __name__ == "__main__":
    main()