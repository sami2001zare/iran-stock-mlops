from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
from os import path

EXCEL_FILE_NAME = "daily_trades"
EXCEL_FILE_EXT = "xlsx"
EXCEL_FILE_PATH = "/tmp"

default_args = {
    "owner": "ZSQ",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email": ["saman.24.zare@gmail.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "Stock-Exchange-V0",
    default_args=default_args,
    schedule="0 14 * * 6,0-3",
    catchup=False,
)

read_stock_exchange_xlsx_file = BashOperator(
    task_id='Download-Stock-Exchange-Xlsx-File',
    bash_command='curl --retry 10 --output {0} -L -H "User-Agent:Chrome/61.0" --compressed "https://old.tsetmc.com/tsev2/excel/MarketWatchPlus.aspx?d"'.format(
        path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds_nodash }}}}.{EXCEL_FILE_EXT}")
    ),
    dag=dag,
)