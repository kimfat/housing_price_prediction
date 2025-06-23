from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def collect_data():
    import subprocess
    subprocess.run(["python", "collect_data.py"])

def preprocess_data():
    import subprocess
    subprocess.run(["python", "preprocess_data.py"])

def train_and_evaluate():
    import subprocess
    subprocess.run(["python", "train_and_evaluate.py"])

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 23),
    'retries': 1,
}

dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='A simple data pipeline',
    schedule_interval='@daily',  # Задайте расписание по необходимости
)

task_collect_data = PythonOperator(
    task_id='collect_data',
    python_callable=collect_data,
    dag=dag,
)

task_preprocess_data = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

task_train_and_evaluate = PythonOperator(
    task_id='train_and_evaluate',
    python_callable=train_and_evaluate,
    dag=dag,
)

task_collect_data >> task_preprocess_data >> task_train_and_evaluate
