from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.providers.standard.operators.empty import EmptyOperator
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
    "Stock-Exchange-V1",
    default_args=default_args,
    schedule="0 13 * * 6,0-3",
    catchup=False,
)

# Jinja template {{ ds_nodash }} instead of date.today()
task_read_stock_exchange_xlsx_file = BashOperator(
    task_id='Download-Stock-Exchange-Xlsx-File',
    bash_command='curl --retry 10 --output {0} -L -H "User-Agent:Chrome/61.0" --compressed "http://old.tsetmc.com/tsev2/excel/MarketWatchPlus.aspx?d=0"'.format(
        path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds_nodash }}}}.{EXCEL_FILE_EXT}")
    ),
    dag=dag,
)

task_waiting_file_xlsx = FileSensor(
    task_id="Waiting-Excel-File",
    fs_conn_id="fs_temp",
    filepath=path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds_nodash }}}}.{EXCEL_FILE_EXT}"),
    poke_interval=10,
    dag=dag,
)

task_dummy = EmptyOperator(task_id="Dummy-Operator", dag=dag)
task_dummy2 = EmptyOperator(task_id="Dummy-Operator-2", dag=dag)
task_dummy3 = EmptyOperator(task_id="Dummy-Operator-3", dag=dag)

task_read_stock_exchange_xlsx_file >> task_waiting_file_xlsx >> [task_dummy, task_dummy2] >> task_dummy3