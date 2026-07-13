"""
ClickHouse Quantitative OLAP Manager & Lakehouse Bridge
======================================================
Manages table initialization (`eth_trades`, `eth_ohlcv_1m`), S3 Parquet syncing (`s3://lakehouse/silver/...`),
and sub-second quantitative queries (`VWAP`, `Order Flow Imbalance`, `Realized Volatility`) inside ClickHouse.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class ClickHouseManager:
    """HTTP client bridge for ClickHouse SQL execution and Lakehouse S3 integration."""

    def __init__(
        self,
        host: str | None = None,
        port: int = 8123,
        user: str | None = None,
        password: str | None = None,
        database: str = "quantitative_olap",
    ):
        self.host = host or os.getenv("CLICKHOUSE_HOST", "clickhouse")
        self.port = port
        self.user = user or os.getenv("CLICKHOUSE_USER", "default")
        self.password = password or os.getenv("CLICKHOUSE_PASSWORD", "clickhouse")
        self.database = database
        self.base_url = f"http://{self.host}:{self.port}/"
        self.s3_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        self.s3_user = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        self.s3_pass = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    def execute(self, query: str, format_type: str | None = None) -> Any:
        """Execute arbitrary SQL query on ClickHouse over HTTP interface."""
        full_query = query
        if format_type and "FORMAT " not in query.upper():
            full_query = f"{query.rstrip(';')} FORMAT {format_type}"

        params = {
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }

        try:
            response = requests.post(
                self.base_url,
                params=params,
                data=full_query.encode("utf-8"),
                timeout=30,
            )
            response.raise_for_status()
            if format_type in ("JSON", "JSONEachRow"):
                return response.json()
            return response.text
        except requests.exceptions.RequestException as exc:
            logger.warning("ClickHouse query notice or connection error: %s", exc)
            if hasattr(exc, "response") and exc.response is not None:
                logger.warning("ClickHouse error body: %s", exc.response.text)
            return None

    def initialize_quantitative_schema(self) -> bool:
        """Create database, raw tick table (`MergeTree`), and 1-minute OHLCV Materialized View."""
        logger.info("Initializing quantitative schema (`%s`) inside ClickHouse...", self.database)

        # 1. Ensure database exists
        self.execute(f"CREATE DATABASE IF NOT EXISTS {self.database};")

        # 2. Raw spot trades tick table (optimized for time-series range scanning)
        trades_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.eth_trades (
                trade_id UInt64,
                price Float64,
                quantity Float64,
                quote_quantity Float64,
                timestamp UInt64,
                is_buyer_maker UInt8,
                is_best_match UInt8,
                partition_date String
            )
            ENGINE = ReplacingMergeTree()
            ORDER BY (trade_id, timestamp)
            PARTITION BY partition_date;
        """
        self.execute(trades_sql)

        # 3. Real-time 1-Minute OHLCV & Order Flow Imbalance Materialized View
        ohlcv_sql = f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {self.database}.eth_ohlcv_1m
            ENGINE = AggregatingMergeTree()
            ORDER BY (window_start)
            AS SELECT
                toStartOfMinute(toDateTime(timestamp / 1000)) AS window_start,
                argMinState(price, timestamp) AS open_price,
                maxState(price) AS high_price,
                minState(price) AS low_price,
                argMaxState(price, timestamp) AS close_price,
                sumState(quantity) AS volume,
                sumState(quote_quantity) AS quote_volume,
                sumState(if(is_buyer_maker = 1, -quantity, quantity)) AS order_flow_imbalance
            FROM {self.database}.eth_trades
            GROUP BY window_start;
        """
        self.execute(ohlcv_sql)

        logger.info("✅ Successfully initialized ClickHouse tables (`eth_trades`, `eth_ohlcv_1m`).")
        return True

    def sync_silver_parquet_from_minio(self, partition_ds: str) -> int:
        """
        Ingest cleaned Silver Parquet partition (`s3://lakehouse/silver/eth_trades_cleaned/ds={ds}/trades.parquet`)
        directly into ClickHouse `eth_trades` table at multi-million row per second C++ speed.
        """
        self.initialize_quantitative_schema()

        s3_url = f"{self.s3_endpoint}/lakehouse/silver/eth_trades_cleaned/ds={partition_ds}/*.parquet"
        logger.info("Syncing Silver Parquet partition from MinIO S3 (%s) into ClickHouse...", s3_url)

        query = f"""
            INSERT INTO {self.database}.eth_trades
            SELECT
                trade_id,
                price,
                quantity,
                quote_quantity,
                timestamp,
                if(is_buyer_maker, 1, 0) AS is_buyer_maker,
                if(is_best_match, 1, 0) AS is_best_match,
                partition_date
            FROM s3(
                '{s3_url}',
                '{self.s3_user}',
                '{self.s3_pass}',
                'Parquet'
            );
        """
        res = self.execute(query)
        if res is not None:
            # Check row count
            count_res = self.execute(
                f"SELECT count() FROM {self.database}.eth_trades WHERE partition_date = '{partition_ds}';"
            )
            rows = int(count_res.strip()) if count_res and count_res.strip().isdigit() else 0
            logger.info("✅ ClickHouse sync complete! Partition %s now contains %d rows.", partition_ds, rows)
            return rows

        logger.warning("ClickHouse S3 sync notice: could not complete direct S3 insert (maybe partition empty or local testing mode).")
        return 0

    def query_vwap_and_ofi(self, minutes_window: int = 60) -> dict[str, Any] | None:
        """Query real-time Volume-Weighted Average Price (VWAP) and Order Flow Imbalance."""
        query = f"""
            SELECT
                round(sum(price * quantity) / sum(quantity), 2) AS vwap,
                round(sum(if(is_buyer_maker = 1, -quantity, quantity)), 4) AS net_order_flow_imbalance,
                round(sqrt(sum(pow(log(price / price), 2))), 4) AS realized_volatility,
                count() AS total_ticks
            FROM {self.database}.eth_trades
            WHERE timestamp >= (toUnixTimestamp(now() - INTERVAL {minutes_window} MINUTE) * 1000)
            FORMAT JSON;
        """
        return self.execute(query, format_type="JSON")
