from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta

_current_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _proj_root.endswith("dags"):
    _proj_root = os.path.abspath(os.path.join(_proj_root, ".."))
for _p in [_proj_root, os.path.join(_proj_root, "src"), "/opt/airflow", "/opt/airflow/src"]:
    if _p not in sys.path and os.path.exists(_p):
        sys.path.insert(0, _p)

import pendulum
import polars as pl

try:
    from airflow.sdk.definitions.asset import Asset as Dataset
except ImportError:
    try:
        from airflow.sdk import Asset as Dataset
    except ImportError:
        try:
            from airflow.assets import Asset as Dataset
        except ImportError:
            from airflow.datasets import Dataset

from airflow.decorators import dag, task

from src.data_engine.lakehouse import LakehouseManager
from src.features.feast_definitions import FeastFeatureManager
from src.features.quantitative import QuantitativeFeatureEngine

logger = logging.getLogger("airflow.dag2")

SILVER_DATASET = Dataset("s3://lakehouse/silver/eth_trades_cleaned")
ONLINE_FEATURE_DATASET = Dataset("feast://features/online_ready")

default_args = {
    "owner": "dataops_team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="02_feature_store_materialization",
    default_args=default_args,
    description="Compute quantitative feature matrix → Materialize to Feast Redis & Lakehouse Gold",
    schedule=[SILVER_DATASET],
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["features", "feast", "redis", "polars", "quantitative"],
)
def feature_store_materialization_pipeline():

    @task(task_id="compute_gold_features")
    def compute_gold_features(ds: str = None) -> str:
        lakehouse = LakehouseManager()
        df_silver = lakehouse.query_silver_table(ds=ds)
        logger.info("Loaded %d rows from Silver partition %s", len(df_silver), ds)

        df_features = QuantitativeFeatureEngine.compute_all_features(df_silver)
        gold_path = lakehouse.write_gold_features(df_features, partition_ds=ds)
        return gold_path

    @task(task_id="sync_online_feature_store", outlets=[ONLINE_FEATURE_DATASET])
    def sync_online_feature_store(gold_path: str) -> int:
        df_gold = pl.read_parquet(gold_path).to_pandas()

        feast_mgr = FeastFeatureManager()
        rows_pushed = feast_mgr.materialize_online_features(df_gold, key_col="trade_id")
        logger.info("✅ Successfully synced %d feature vectors into Redis Online Store.", rows_pushed)
        return rows_pushed

    gold_artifact = compute_gold_features()
    sync_online_feature_store(gold_artifact)


dag_instance = feature_store_materialization_pipeline()
