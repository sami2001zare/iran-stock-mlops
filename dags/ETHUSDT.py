from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pendulum
import requests
import zipfile
import os
import pandas as pd
from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='binance_ethusdt_trades_to_hdfs',
    default_args=default_args,
    description='Download Binance ETHUSDT trades → unzip → load to HDFS (for Hive table)',
    schedule='0 2 * * *',
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=['binance', 'hdfs', 'hive'],
) as dag:

    def download_zip(**context):
        ds = context['ds']
        year, month, day = ds.split('-')
        url = f"https://data.binance.vision/data/spot/daily/trades/ETHUSDT/ETHUSDT-trades-{year}-{month}-15.zip"
        local_zip = f"/tmp/ETHUSDT-trades-{ds}.zip"

        print(f"Downloading {url}")
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        with open(local_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_zip

    def unzip_file(zip_path: str, **context):
        extract_dir = "/tmp/binance_extract"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("CSV not found")
        csv_path = os.path.join(extract_dir, csv_files[0])
        print(f"Unzipped: {csv_path}")
        return csv_path

    def prepare_csv(csv_path: str, **context):
        """Read, fix timestamp if needed, and prepare for Hive TEXTFILE table"""
        ds = context['ds']
        
        df = pd.read_csv(
            csv_path,
            header=None,
            names=['trade_id', 'price', 'quantity', 'quote_quantity', 
                   'timestamp', 'is_buyer_maker', 'is_best_match']
        )
        
        # Ensure timestamp is integer (no conversion to datetime needed for your table)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('Int64')
        
        # Optional: add human readable timestamp if you want
        # df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='us', errors='coerce')
        
        print(f"Prepared {len(df):,} rows for Hive table")

        final_csv = f"/tmp/ETHUSDT-trades-{ds}.csv"
        df.to_csv(final_csv, index=False, header=False)   # No header for your table
        return final_csv

    # Tasks
    download_task = PythonOperator(
        task_id='download_zip',
        python_callable=download_zip,
    )

    unzip_task = PythonOperator(
        task_id='unzip_file',
        python_callable=unzip_file,
        op_kwargs={'zip_path': "{{ task_instance.xcom_pull(task_ids='download_zip') }}"},
    )

    prepare_task = PythonOperator(
        task_id='prepare_csv',
        python_callable=prepare_csv,
        op_kwargs={'csv_path': "{{ task_instance.xcom_pull(task_ids='unzip_file') }}"},
    )

    upload_task = BashOperator(
        task_id='upload_to_hdfs',
        bash_command="""
            set -e
            ds={{ ds }}
            csv_file="/tmp/ETHUSDT-trades-${ds}.csv"
            
            hdfs_dir="/user/airflow/ethusdt_trades/day=${ds}"
            
            echo "Creating HDFS partition directory..."
            hdfs dfs -mkdir -p ${hdfs_dir}
            
            echo "Uploading CSV to HDFS..."
            hdfs dfs -put -f ${csv_file} ${hdfs_dir}/ETHUSDT-trades-${ds}.csv
            
            echo "✅ Uploaded successfully to ${hdfs_dir}"
        """,
    )

    # Dependencies
    download_task >> unzip_task >> prepare_task >> upload_task