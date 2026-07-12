"""
Feast Feature Store Definitions & Materialization Manager
=========================================================
Declares point-in-time correct feature views for historical model training (Offline in Lakehouse)
and pushes computed Gold feature vectors into Redis (Online Store) for sub-millisecond API lookups.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


class FeastFeatureManager:
    """Manages Redis Online Feature Store materialization and retrieval."""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
        if redis is not None:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            except Exception as exc:
                logger.warning("Redis client initialization warning: %s", exc)
                self.redis_client = None
        else:
            self.redis_client = None

    def materialize_online_features(self, df_features: pd.DataFrame, key_col: str = "trade_id") -> int:
        """
        Push computed Gold feature vectors into Redis online store.
        Each entity key (`trade_id` or `latest_eth_feature`) maps to a JSON vector.
        """
        if self.redis_client is None:
            logger.debug("Redis client unavailable (or local testing mode). Skipping online materialization.")
            return 0

        logger.info("Materializing %d feature rows into Redis online feature store (%s)...", len(df_features), self.redis_url)
        
        tail_df = df_features.tail(1000).copy()
        pushed_count = 0

        with self.redis_client.pipeline() as pipe:
            for idx, row in tail_df.iterrows():
                entity_key = f"eth_feature:{row[key_col]}"
                payload = row.to_dict()
                for k, v in payload.items():
                    if isinstance(v, (pd.Timestamp, datetime)):
                        payload[k] = v.isoformat()
                    elif hasattr(v, "item"):
                        payload[k] = v.item()

                pipe.set(entity_key, json.dumps(payload), ex=86400 * 7)
                pushed_count += 1

            if not tail_df.empty:
                latest_payload = tail_df.iloc[-1].to_dict()
                for k, v in latest_payload.items():
                    if hasattr(v, "item"):
                        latest_payload[k] = v.item()
                pipe.set("eth_feature:latest", json.dumps(latest_payload), ex=86400 * 7)

            pipe.execute()

        logger.info("✅ Successfully pushed %d vectors to Redis (Key: eth_feature:latest updated).", pushed_count)
        return pushed_count

    def get_online_feature_vector(self, trade_id: int | str = "latest") -> dict[str, Any] | None:
        """Fetch real-time feature vector from Redis in <1ms for FastAPI inference."""
        if self.redis_client is None:
            return None

        key = f"eth_feature:{trade_id}"
        data_str = self.redis_client.get(key)
        if not data_str and trade_id != "latest":
            data_str = self.redis_client.get("eth_feature:latest")

        if data_str:
            try:
                return json.loads(data_str)
            except Exception as exc:
                logger.error("Failed decoding feature JSON from Redis: %s", exc)
        return None
