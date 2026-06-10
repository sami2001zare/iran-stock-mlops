from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.providers.standard.operators.empty import EmptyOperator   
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from os import path
import jdatetime
import pandas as pd

EXCEL_FILE_NAME = "daily_trades"
EXCEL_FILE_EXT_XLSX = "xlsx"
EXCEL_FILE_EXT_CSV = "csv"
EXCEL_FILE_PATH = "/tmp"

def preprocess_convert_to_csv(**kwargs):
    execution_date = kwargs['logical_date'].date()
    xlsx_file_path = path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{execution_date.strftime('%Y_%m_%d')}.{EXCEL_FILE_EXT_XLSX}")
    csv_file_path = path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{execution_date.strftime('%Y_%m_%d')}.{EXCEL_FILE_EXT_CSV}")
    df = pd.read_excel(xlsx_file_path, header=0, skiprows=2, engine='openpyxl')
    today = jdatetime.date.today()
    df = df.assign(fa_date=today.strftime("%Y-%m-%d"))
    df = df.assign(en_date=execution_date.isoformat())
    df = df.assign(fa_year=today.year)
    df.to_csv(csv_file_path, index=None, header=True, encoding="utf-8")
    print("%" * 20)
    print(f"{csv_file_path} created successfully!")
    print("%" * 20)


default_args = {
    "owner": "ZSQ",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email": ["saman.24.zare@gmail.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "Stock-Exchange-V2",
    default_args=default_args,
    schedule="0 13 * * 6,0-3",
    catchup=False,
)

task_read_stock_exchange_xlsx_file = BashOperator(
    task_id='Download-Stock-Exchange-Xlsx-File',
    bash_command='curl --retry 10 --output {0} -L -H "User-Agent:Chrome/61.0" --compressed "http://old.tsetmc.com/tsev2/excel/MarketWatchPlus.aspx?d=0"'.format(
        path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds_nodash }}}}.{EXCEL_FILE_EXT_XLSX}")
    ),
    dag=dag,
)

task_waiting_file_xlsx = FileSensor(
    task_id="Waiting-Excel-File",
    fs_conn_id="fs_temp",
    filepath=path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds_nodash }}}}.{EXCEL_FILE_EXT_XLSX}"),
    poke_interval=10,
    dag=dag,
)

task_dummy = EmptyOperator(task_id="Dummy-Operator", dag=dag)

task_preprocess_convert_to_csv = PythonOperator(
    task_id='Preprocess-Convert_To_CSV',
    python_callable=preprocess_convert_to_csv,
    dag=dag,
)


task_read_stock_exchange_xlsx_file >> task_waiting_file_xlsx >> task_preprocess_convert_to_csv >> task_dummy