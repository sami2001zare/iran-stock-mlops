from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_model_daily_retraining',
    default_args=default_args,
    description='Daily retrain stock prediction model',
    schedule_interval='0 20 * * 1-5',  # 8 PM every weekday after market close
    catchup=False,
)

ingest = BashOperator(
    task_id='ingest_data',
    bash_command='cd /opt/airflow/project && python pipeline/ingest.py',
    dag=dag,
)

preprocess = BashOperator(
    task_id='preprocess',
    bash_command='cd /opt/airflow/project && python pipeline/preprocess.py',
    dag=dag,
)

features = BashOperator(
    task_id='feature_engineering',
    bash_command='cd /opt/airflow/project && python pipeline/features.py',
    dag=dag,
)

train = BashOperator(
    task_id='train_model',
    bash_command='cd /opt/airflow/project && python pipeline/train.py',
    dag=dag,
)

evaluate = BashOperator(
    task_id='evaluate_model',
    bash_command='cd /opt/airflow/project && python pipeline/evaluate.py',
    dag=dag,
)
