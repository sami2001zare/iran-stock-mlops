"""
DuckDB & Polars Lakehouse Storage Manager
=========================================
Manages out-of-core vectorized transformations across Bronze, Silver, and Gold layers
stored in MinIO (or local filesystem) using Apache Iceberg/Parquet partition structures.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


class LakehouseManager:
    """Out-of-core query engine supporting MinIO S3 and local storage endpoints."""

    def __init__(
        self,
        s3_endpoint: str | None = None,
        s3_access_key: str | None = None,
        s3_secret_key: str | None = None,
        s3_bucket: str = "lakehouse",
        use_local_fallback: bool = True,
    ):
        self.s3_endpoint = s3_endpoint or os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        self.s3_access_key = s3_access_key or os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        self.s3_secret_key = s3_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        self.s3_bucket = s3_bucket or os.getenv("S3_BUCKET_LAKEHOUSE", "lakehouse")
        self.local_root = "/tmp/lakehouse"
        self.use_local_fallback = use_local_fallback

        os.makedirs(os.path.join(self.local_root, "bronze"), exist_ok=True)
        os.makedirs(os.path.join(self.local_root, "silver"), exist_ok=True)
        os.makedirs(os.path.join(self.local_root, "gold"), exist_ok=True)

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Create and configure DuckDB connection with S3/HTTPFS secret extensions."""
        conn = duckdb.connect(database=":memory:")
        try:
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            conn.execute(f"""
                CREATE OR REPLACE SECRET minio_secret (
                    TYPE S3,
                    KEY_ID '{self.s3_access_key}',
                    SECRET '{self.s3_secret_key}',
                    ENDPOINT '{self.s3_endpoint.replace("http://", "").replace("https://", "")}',
                    USE_SSL false,
                    URL_STYLE 'path'
                );
            """)
        except Exception as exc:
            logger.debug("S3 extension init notice (or using local file mode): %s", exc)
        return conn

    def promote_bronze_to_silver(self, parquet_chunks: list[str], partition_ds: str) -> str:
        """
        Ingest chunked Bronze Parquet files, deduplicate by trade_id, clean schemas,
        and write to Silver Medallion table partition using vectorized DuckDB C++ engine.
        """
        if not parquet_chunks:
            raise ValueError("No Parquet chunks provided for promotion.")

        conn = self.get_connection()
        chunk_files_sql = ", ".join([f"'{p}'" for p in parquet_chunks])

        silver_local_dir = os.path.join(self.local_root, "silver", "eth_trades_cleaned", f"ds={partition_ds}")
        os.makedirs(silver_local_dir, exist_ok=True)
        out_silver_file = os.path.join(silver_local_dir, "trades.parquet")

        logger.info("Promoting %d Bronze chunks to Silver partition %s via DuckDB...", len(parquet_chunks), partition_ds)
        
        query = f"""
            COPY (
                WITH raw_data AS (
                    SELECT * FROM read_parquet([{chunk_files_sql}])
                ),
                deduped AS (
                    SELECT *,
                           ROW_NUMBER() OVER (PARTITION BY trade_id ORDER BY timestamp DESC) AS row_num
                    FROM raw_data
                    WHERE price > 0 AND quantity > 0
                )
                SELECT
                    trade_id,
                    CAST(price AS DOUBLE) AS price,
                    CAST(quantity AS DOUBLE) AS quantity,
                    CAST(quote_quantity AS DOUBLE) AS quote_quantity,
                    CAST(timestamp AS BIGINT) AS timestamp,
                    CAST(is_buyer_maker AS BOOLEAN) AS is_buyer_maker,
                    CAST(is_best_match AS BOOLEAN) AS is_best_match,
                    '{partition_ds}' AS partition_date
                FROM deduped
                WHERE row_num = 1
                ORDER BY timestamp ASC
            ) TO '{out_silver_file}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
        """
        conn.execute(query)
        conn.close()

        logger.info("✅ Successfully created Silver Parquet table partition: %s", out_silver_file)
        return out_silver_file

    def query_silver_table(self, ds: str | None = None) -> pl.DataFrame:
        """Read Silver Lakehouse partition using high-speed Polars engine."""
        if ds:
            partition_path = os.path.join(self.local_root, "silver", "eth_trades_cleaned", f"ds={ds}", "*.parquet")
        else:
            partition_path = os.path.join(self.local_root, "silver", "eth_trades_cleaned", "**", "*.parquet")

        logger.info("Loading Silver dataset from %s into Polars...", partition_path)
        try:
            return pl.read_parquet(partition_path)
        except Exception:
            # Fallback if specific ds folder check needs exact file path
            exact_file = os.path.join(self.local_root, "silver", "eth_trades_cleaned", f"ds={ds}", "trades.parquet")
            if os.path.exists(exact_file):
                return pl.read_parquet(exact_file)
            raise FileNotFoundError(f"Silver partition not found: {partition_path}")

    def write_gold_features(self, df: pl.DataFrame | pd.DataFrame, partition_ds: str) -> str:
        """Save Gold feature matrix to compressed Lakehouse storage."""
        if isinstance(df, pd.DataFrame):
            df_pl = pl.from_pandas(df)
        else:
            df_pl = df

        gold_dir = os.path.join(self.local_root, "gold", "eth_daily_features", f"ds={partition_ds}")
        os.makedirs(gold_dir, exist_ok=True)
        out_gold_file = os.path.join(gold_dir, "features.parquet")

        df_pl.write_parquet(out_gold_file, compression="snappy")
        logger.info("✅ Successfully written %d rows to Gold feature table: %s", len(df_pl), out_gold_file)
        return out_gold_file
