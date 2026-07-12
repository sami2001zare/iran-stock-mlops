# v7: MLflow-integrated ETH model pipeline + stock exchange pipeline
#
# Two parallel branches that converge at Done:
#   Branch A (Stock Exchange): Download → Wait → Preprocess-Stock → Upload-HDFS → Delete
#   Branch B (ETH ML):         Preprocess-ETH → Train-ETH → Evaluate-ETH → Register-ETH


import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import eth_price_model_v4 as m

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
from os import path
import jdatetime
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Airflow Variables (with defaults so DAG loads without pre-configuration)
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_FILE_NAME     = Variable.get("EXCEL_FILE_NAME",     default_var="daily_trades")
EXCEL_FILE_EXT_XLSX = Variable.get("EXCEL_FILE_EXT_XLSX", default_var="xlsx")
EXCEL_FILE_EXT_CSV  = Variable.get("EXCEL_FILE_EXT_CSV",  default_var="csv")
EXCEL_FILE_PATH     = Variable.get("EXCEL_FILE_PATH",     default_var="/tmp")
HDFS_CSV_FOLDER     = Variable.get("HDFS_CSV_FOLDER",     default_var="/data/data_lake/stock/fa_year=")

ETH_DATA_PATH       = Variable.get("ETH_DATA_PATH",       default_var="/opt/airflow/model/ETHUSDT-trades-2026-06-13.csv")
ETH_MODEL_DIR       = Variable.get("ETH_MODEL_DIR",       default_var="/opt/airflow/model/eth_model_artifacts")
ETH_PIPELINE_DIR    = Variable.get("ETH_PIPELINE_DIR",    default_var="/tmp/eth_pipeline")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI", default_var="http://mlflow:5000")
MLFLOW_EXPERIMENT   = Variable.get("MLFLOW_EXPERIMENT",   default_var="eth-price-prediction")

default_args = {
    "owner": "ZSQ",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email": ["saman.24.zare@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "Stock-Exchange-V7",
    default_args=default_args,
    schedule="0 13 * * 6,0-3",
    catchup=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# Branch A — Stock Exchange Pipeline (unchanged from v6)
# ═════════════════════════════════════════════════════════════════════════════

def put_csv_to_hdfs(**kwargs):
    print("HDFS upload skipped – provider not installed")


def is_symbol(row):
    try:
        return int("".join(filter(str.isdigit, row["نماد"]))) < 20
    except Exception:
        return True


def preprocess_convert_to_csv(**kwargs):
    ti  = kwargs["ti"]
    ds  = kwargs["ds"]
    xlsx_file_path = path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{ds}.{EXCEL_FILE_EXT_XLSX}")
    csv_file_path  = path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{ds}.{EXCEL_FILE_EXT_CSV}")
    df    = pd.read_excel(xlsx_file_path, header=0, skiprows=2, engine="openpyxl")
    today = jdatetime.date.today()
    df    = df.assign(fa_date=today.strftime("%Y-%m-%d"))
    df    = df.assign(en_date=datetime.fromisoformat(ds).date().isoformat())
    df    = df.assign(fa_year=today.year)
    df    = df[df.apply(is_symbol, axis=1)]
    df.to_csv(csv_file_path, index=None, header=True, encoding="utf-8")
    ti.xcom_push(key="fa_year", value=today.year)


task_download_xlsx = BashOperator(
    task_id="Download-Stock-Exchange-Xlsx-File",
    bash_command='curl --retry 10 --output {0} -L -H "User-Agent:Chrome/61.0" --compressed "http://old.tsetmc.com/tsev2/excel/MarketWatchPlus.aspx?d={{{{ds}}}}"'.format(
        path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds }}}}.{EXCEL_FILE_EXT_XLSX}")
    ),
    dag=dag,
)

task_wait_xlsx = FileSensor(
    task_id="Waiting-Excel-File",
    fs_conn_id="fs_temp",
    filepath=path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds }}}}.{EXCEL_FILE_EXT_XLSX}"),
    poke_interval=10,
    dag=dag,
)

task_preprocess_stock = PythonOperator(
    task_id="Preprocess-Convert-To-CSV",
    python_callable=preprocess_convert_to_csv,
    dag=dag,
)

task_upload_hdfs = PythonOperator(
    task_id="Put-CSV-To-HDFS",
    python_callable=put_csv_to_hdfs,
    op_kwargs={
        "target_path": HDFS_CSV_FOLDER,
        "input_file":  path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds }}}}.{EXCEL_FILE_EXT_CSV}"),
    },
    dag=dag,
)

task_delete_files = BashOperator(
    task_id="Delete-Stock-Exchange-Xlsx-File",
    bash_command="rm {0} {1}".format(
        path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds }}}}.{EXCEL_FILE_EXT_XLSX}"),
        path.join(EXCEL_FILE_PATH, f"{EXCEL_FILE_NAME}_{{{{ ds }}}}.{EXCEL_FILE_EXT_CSV}"),
    ),
    dag=dag,
)


# ═════════════════════════════════════════════════════════════════════════════
# Branch B — ETH Price Prediction ML Pipeline
# eth_price_model_v4 is imported at the top of this file because PYTHONPATH
# includes /opt/airflow/model (set in docker-compose.yaml).
# ═════════════════════════════════════════════════════════════════════════════

def _preprocess_eth(**kwargs):
    """load_data → feature_engineering → quantize_data → train_test_split_temporal"""
    import os, time
    ti = kwargs["ti"]
    os.makedirs(ETH_PIPELINE_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"eth-airflow-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        df = m.load_data(ETH_DATA_PATH, m.COL_NAMES)
        df, feature_cols, target_col, X, y = m.feature_engineering(df, m.WINDOWS, m.LAG_STEPS, m.DROP_COLS)
        X_model, X_q8, qt = m.quantize_data(X)
        X_train, X_test, y_train, y_test = m.train_test_split_temporal(X_model, y)
        run_id = run.info.run_id

    joblib.dump(
        {"X_train": X_train, "X_test": X_test,
         "y_train": y_train, "y_test": y_test,
         "feature_cols": feature_cols, "qt": qt},
        path.join(ETH_PIPELINE_DIR, "preprocess_output.pkl"),
    )

    ti.xcom_push(key="mlflow_run_id",  value=run_id)
    ti.xcom_push(key="pipeline_dir",   value=ETH_PIPELINE_DIR)
    ti.xcom_push(key="n_features_raw", value=len(feature_cols))
    print(f"[Preprocess-ETH] run_id={run_id} | features={len(feature_cols)} | "
          f"train={len(X_train):,} | test={len(X_test):,}")


def _train_eth(**kwargs):
    """train_model → prune_and_retrain; logs per-batch metrics to MLflow."""
    ti           = kwargs["ti"]
    pipeline_dir = ti.xcom_pull(task_ids="Preprocess-ETH-Data", key="pipeline_dir")
    run_id       = ti.xcom_pull(task_ids="Preprocess-ETH-Data", key="mlflow_run_id")

    data             = joblib.load(path.join(pipeline_dir, "preprocess_output.pkl"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_test           = data["X_test"]
    feature_cols     = data["feature_cols"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        model, total_estimators, train_history = m.train_model(
            X_train, y_train, m.BATCH_SIZE, m.N_ESTIMATORS_PER_BATCH, m.MAX_BATCHES,
        )
        (model_pruned, pruned_cols, removed_cols,
         mask_keep, feat_df, X_train_p, X_test_p, threshold) = m.prune_and_retrain(
            model, X_train, X_test, y_train, feature_cols, total_estimators
        )
        mlflow.sklearn.log_model(model_pruned, artifact_path="model_staging")
        feat_path = path.join(pipeline_dir, "feature_importance.csv")
        feat_df.to_csv(feat_path, index=False)
        mlflow.log_artifact(feat_path, artifact_path="artifacts")

    joblib.dump(
        {"model_pruned": model_pruned, "mask_keep": mask_keep,
         "pruned_cols": pruned_cols, "removed_cols": removed_cols,
         "threshold": float(threshold), "total_estimators": int(total_estimators),
         "feat_df": feat_df, "train_history": train_history, "qt": data["qt"]},
        path.join(pipeline_dir, "train_output.pkl"),
    )

    ti.xcom_push(key="n_features_pruned",  value=len(pruned_cols))
    ti.xcom_push(key="n_estimators_final", value=int(total_estimators))
    print(f"[Train-ETH] pruned_features={len(pruned_cols)} | total_estimators={total_estimators}")


def _evaluate_eth(**kwargs):
    """compute_metrics on test set; logs MAE/RMSE/R²/MAPE and predictions CSV to MLflow."""
    ti           = kwargs["ti"]
    pipeline_dir = ti.xcom_pull(task_ids="Preprocess-ETH-Data", key="pipeline_dir")
    run_id       = ti.xcom_pull(task_ids="Preprocess-ETH-Data", key="mlflow_run_id")

    data         = joblib.load(path.join(pipeline_dir, "preprocess_output.pkl"))
    train_output = joblib.load(path.join(pipeline_dir, "train_output.pkl"))

    X_test, y_test = data["X_test"], data["y_test"]
    model_pruned   = train_output["model_pruned"]
    mask_keep      = train_output["mask_keep"]

    X_test_p = X_test[:, mask_keep]
    y_pred   = model_pruned.predict(X_test_p)
    metrics  = m.compute_metrics(y_test, y_pred, "Pruned Model")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_mae":  round(metrics["MAE"],    4),
            "test_rmse": round(metrics["RMSE"],   4),
            "test_r2":   round(metrics["R2"],     6),
            "test_mape": round(metrics["MAPE_%"], 4),
        })
        pred_df = pd.DataFrame({
            "actual_price": y_test, "pred_pruned": y_pred,
            "error": y_test - y_pred, "abs_error": np.abs(y_test - y_pred),
        })
        pred_path = path.join(pipeline_dir, "predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path, artifact_path="results")

    ti.xcom_push(key="test_mae",  value=round(metrics["MAE"],    4))
    ti.xcom_push(key="test_rmse", value=round(metrics["RMSE"],   4))
    ti.xcom_push(key="test_r2",   value=round(metrics["R2"],     6))
    ti.xcom_push(key="test_mape", value=round(metrics["MAPE_%"], 4))
    print(f"[Evaluate-ETH] MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
          f"R²={metrics['R2']:.6f} | MAPE={metrics['MAPE_%']:.4f}%")


def _register_eth(**kwargs):
    """save_artifacts → registers pruned HistGBR as 'ETHPricePredictor' in MLflow Registry."""
    ti           = kwargs["ti"]
    pipeline_dir = ti.xcom_pull(task_ids="Preprocess-ETH-Data", key="pipeline_dir")
    run_id       = ti.xcom_pull(task_ids="Preprocess-ETH-Data", key="mlflow_run_id")

    data         = joblib.load(path.join(pipeline_dir, "preprocess_output.pkl"))
    train_output = joblib.load(path.join(pipeline_dir, "train_output.pkl"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        m.save_artifacts(
            model_pruned     = train_output["model_pruned"],
            qt               = train_output["qt"],
            feature_cols     = data["feature_cols"],
            pruned_cols      = train_output["pruned_cols"],
            mask_keep        = train_output["mask_keep"],
            threshold        = train_output["threshold"],
            total_estimators = train_output["total_estimators"],
            m_pruned         = {
                "MAE":    ti.xcom_pull(task_ids="Evaluate-ETH-Model", key="test_mae"),
                "RMSE":   ti.xcom_pull(task_ids="Evaluate-ETH-Model", key="test_rmse"),
                "R2":     ti.xcom_pull(task_ids="Evaluate-ETH-Model", key="test_r2"),
                "MAPE_%": ti.xcom_pull(task_ids="Evaluate-ETH-Model", key="test_mape"),
                "label":  "Pruned Model",
            },
            feat_df          = train_output["feat_df"],
            model_dir        = ETH_MODEL_DIR,
        )
    print(f"[Register-ETH] Model registered as 'ETHPricePredictor' | run_id={run_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Branch B task definitions
# ─────────────────────────────────────────────────────────────────────────────
task_preprocess_eth = PythonOperator(task_id="Preprocess-ETH-Data",  python_callable=_preprocess_eth, dag=dag)
task_train_eth      = PythonOperator(task_id="Train-ETH-Model",      python_callable=_train_eth,      dag=dag)
task_evaluate_eth   = PythonOperator(task_id="Evaluate-ETH-Model",   python_callable=_evaluate_eth,   dag=dag)
task_register_eth   = PythonOperator(task_id="Register-ETH-Model",   python_callable=_register_eth,   dag=dag)
task_done           = EmptyOperator(task_id="Done", dag=dag)

# ─────────────────────────────────────────────────────────────────────────────
# Task dependencies
# ─────────────────────────────────────────────────────────────────────────────
task_download_xlsx >> task_wait_xlsx >> task_preprocess_stock >> task_upload_hdfs >> task_delete_files >> task_done
task_preprocess_eth >> task_train_eth >> task_evaluate_eth >> task_register_eth >> task_done
