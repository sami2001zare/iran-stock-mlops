from __future__ import annotations

import json
import logging
import os
import shutil
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
from airflow.utils.task_group import TaskGroup

from src.data_engine.extractors import BinanceSpotExtractor, MacroIndicatorsExtractor
from src.data_engine.lakehouse import LakehouseManager
from src.data_engine.validators import DataContractValidator

logger = logging.getLogger("airflow.dag1")

SILVER_DATASET = Dataset("s3://lakehouse/silver/eth_trades_cleaned")

default_args = {
    "owner": "dataops_team",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}


@dag(
    dag_id="01_binance_to_lakehouse",
    default_args=default_args,
    description="Stream daily ETHUSDT spot trades → Validate quality → Promote to Silver Lakehouse via DuckDB",
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ingestion", "lakehouse", "binance", "duckdb", "medallion"],
)
def binance_to_lakehouse_pipeline():

    with TaskGroup("ingestion_stage", tooltip="Download external APIs securely") as ingestion_group:
        @task(task_id="download_spot_zip")
        def download_spot_zip(ds: str = None) -> str:
            """Download raw daily spot trade ZIP safely without hardcoded day bugs."""
            zip_path = BinanceSpotExtractor.download_daily_trades(ds=ds, symbol="ETHUSDT")
            logger.info("Downloaded raw ZIP artifact: %s", zip_path)
            return zip_path

        @task(task_id="fetch_macro_indicators")
        def fetch_macro_indicators(ds: str = None) -> str:
            """Fetch macroeconomic context (FRED Yields, DXY Index, M2 Liquidity)."""
            macro_data = MacroIndicatorsExtractor.fetch_macro_snapshot(ds=ds)
            out_dir = "/tmp/lakehouse/bronze/macro"
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"macro_{ds}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(macro_data, f, indent=2)
            logger.info("Saved macro snapshot artifact: %s", out_file)
            return out_file

        zip_artifact = download_spot_zip()
        macro_artifact = fetch_macro_indicators()

    with TaskGroup("validation_and_chunking_stage", tooltip="Chunk extraction and DQ checks") as validation_group:
        @task(task_id="split_to_parquet_chunks")
        def split_to_parquet_chunks(zip_path: str) -> list[str]:
            """Convert raw uncompressed CSV to chunked snappy Parquet out-of-core."""
            return BinanceSpotExtractor.extract_to_parquet_chunks(zip_path, chunk_rows=250_000)

        @task(task_id="validate_data_quality")
        def validate_data_quality(chunk_paths: list[str]) -> list[str]:
            """Run data quality contracts across chunks (row count, bounds, null limits)."""
            valid_chunks = []
            for path in chunk_paths:
                df_chunk = pl.read_parquet(path).to_pandas()
                is_valid, report = DataContractValidator.validate_dataframe(df_chunk, min_rows=100)
                if not is_valid:
                    logger.error("Chunk %s failed DQ validation! Errors: %s", path, report["errors"])
                    raise ValueError(f"Data Quality failure on chunk {path}: {report['errors']}")
                valid_chunks.append(path)
            logger.info("✅ All %d chunks passed stringent Pydantic/Data Quality contract checks.", len(valid_chunks))
            return valid_chunks

        raw_chunks = split_to_parquet_chunks(zip_artifact)
        verified_chunks = validate_data_quality(raw_chunks)

    @task(task_id="promote_to_silver_lakehouse", outlets=[SILVER_DATASET])
    def promote_to_silver_lakehouse(chunk_paths: list[str], ds: str = None) -> str:
        """Promote validated Bronze chunks to Silver partition using DuckDB SQL engine."""
        lakehouse = LakehouseManager()
        silver_path = lakehouse.promote_bronze_to_silver(chunk_paths, partition_ds=ds)
        
        try:
            from src.data_engine.clickhouse_manager import ClickHouseManager
            ClickHouseManager().sync_silver_parquet_from_minio(partition_ds=ds)
        except Exception as ch_exc:
            logger.warning("ClickHouse sync notice (maybe clickhouse offline during minimal test): %s", ch_exc)
        
        if chunk_paths:
            temp_root = os.path.dirname(chunk_paths[0])
            if os.path.exists(temp_root) and "/tmp/" in temp_root:
                shutil.rmtree(temp_root, ignore_errors=True)
                logger.info("Cleaned up temporary workspace directory: %s", temp_root)

        return silver_path

    ingestion_group >> validation_group
    verified_chunks >> promote_to_silver_lakehouse(chunk_paths=verified_chunks)


dag_instance = binance_to_lakehouse_pipeline()
