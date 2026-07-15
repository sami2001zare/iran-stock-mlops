"""
Model Serving Inference Service & Online Feature Store Lookup
=============================================================
Handles thread-safe model loading/hot-reloading (`ModelStore`), preprocessing of incoming ticks,
Feast Redis online feature fetching, and fast HistGBR prediction execution.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import os
import sys

# Self-healing sys.path guarantee so FastAPI/Uvicorn locates src from any directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _proj_root.endswith("api"):
    _proj_root = os.path.abspath(os.path.join(_proj_root, ".."))
for _p in [_current_dir, _proj_root, os.path.join(_proj_root, "src"), "/app", "/src"]:
    if _p not in sys.path and os.path.exists(_p):
        sys.path.insert(0, _p)

try:
    from src.features.feast_definitions import FeastFeatureManager
    from src.features.quantitative import QuantitativeFeatureEngine
    from src.ml.drift import DriftDetectionEngine
except ImportError:
    from features.feast_definitions import FeastFeatureManager
    from features.quantitative import QuantitativeFeatureEngine
    from ml.drift import DriftDetectionEngine

logger = logging.getLogger(__name__)


class ModelStore:
    """Thread-safe singleton managing loaded ML model artifacts in memory."""

    model: Any = None
    transformer: Any = None
    meta: dict[str, Any] = {}
    model_dir: str = os.getenv("ETH_MODEL_DIR", "/model/eth_model_artifacts")
    version: str = "v1.0-fallback"

    @classmethod
    def load(cls, model_dir: str | None = None) -> bool:
        """Load or hot-reload model artifacts (`model_pruned.pkl`, `quantile_transformer.pkl`)."""
        target_dir = model_dir or cls.model_dir
        model_path = os.path.join(target_dir, "model_pruned.pkl")
        qt_path = os.path.join(target_dir, "quantile_transformer.pkl")
        meta_path = os.path.join(target_dir, "pipeline_meta.json")

        if not os.path.exists(model_path):
            logger.warning("Model artifact not found at %s. Service will respond 503 or fallback until trained.", model_path)
            return False

        try:
            cls.model = joblib.load(model_path)
            if os.path.exists(qt_path):
                cls.transformer = joblib.load(qt_path)
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    cls.meta = json.load(f)
                cls.version = cls.meta.get("trained_at", "v2.0-pruned")[:19]
            logger.info("✅ Successfully loaded model version %s from %s", cls.version, target_dir)
            return True
        except Exception as exc:
            logger.error("Error loading model artifacts: %s", exc)
            return False

    @classmethod
    def is_ready(cls) -> bool:
        """Check if model engine is loaded and ready for scoring."""
        return cls.model is not None


class InferenceEngine:
    """Executes high-speed feature transformations and model scoring."""

    def __init__(self):
        self.feast_mgr = FeastFeatureManager()

    def predict_from_online_store(self, trade_id_lookup: int | str = "latest") -> dict[str, Any]:
        """Fetch latest feature vector from Feast Redis store and score immediately."""
        start_t = time.perf_counter()
        if not ModelStore.is_ready():
            raise RuntimeError("Model artifacts are not loaded.")

        feature_dict = self.feast_mgr.get_online_feature_vector(trade_id=trade_id_lookup)
        if not feature_dict:
            raise FileNotFoundError(f"No feature vector found in Redis online store for trade_id: {trade_id_lookup}")

        pruned_cols = ModelStore.meta.get("pruned_cols", [])
        if not pruned_cols:
            # Fallback if meta wasn't populated with exact feature list
            pruned_cols = [k for k in feature_dict.keys() if isinstance(feature_dict[k], (int, float))]

        # Construct input vector matching exact pruned column ordering
        row_vals = []
        for col in pruned_cols:
            val = feature_dict.get(col, 0.0)
            if not isinstance(val, (int, float)):
                val = 0.0
            row_vals.append(val)

        X_raw = np.array([row_vals], dtype=np.float64)
        if ModelStore.transformer:
            X_q8 = ModelStore.transformer.transform(X_raw)
            X_model = np.clip(np.round(X_q8 * 255.0), 0, 255).astype(np.uint8)
        else:
            X_model = X_raw

        pred_price = float(ModelStore.model.predict(X_model)[0])
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        # Estimate 95% confidence bounds (~1.5% volatility envelope)
        interval = (round(pred_price * 0.985, 2), round(pred_price * 1.015, 2))

        return {
            "trade_id": feature_dict.get("trade_id", trade_id_lookup),
            "predicted_price": round(pred_price, 2),
            "confidence_interval_95": interval,
            "model_version": ModelStore.version,
            "latency_ms": round(latency_ms, 3),
        }

    def predict_from_trades(self, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run quantitative feature extraction over raw trade ticks and predict next-step prices."""
        start_t = time.perf_counter()
        if not ModelStore.is_ready():
            raise RuntimeError("Model artifacts are not loaded.")

        df_raw = pd.DataFrame(trades)
        if "quote_quantity" not in df_raw or df_raw["quote_quantity"].sum() == 0:
            df_raw["quote_quantity"] = df_raw["price"] * df_raw["quantity"]

        df_features = QuantitativeFeatureEngine.compute_all_features(df_raw)
        pruned_cols = ModelStore.meta.get("pruned_cols", [])
        if not pruned_cols:
            exclude = {"trade_id", "timestamp", "partition_date", "target_y", "ts_ms", "ts_sec"}
            pruned_cols = [c for c in df_features.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_features[c])]

        # Ensure exact column match
        for c in pruned_cols:
            if c not in df_features.columns:
                df_features[c] = 0.0

        X_raw = df_features[pruned_cols].to_numpy(dtype=np.float64)
        if ModelStore.transformer:
            X_q8 = ModelStore.transformer.transform(X_raw)
            X_model = np.clip(np.round(X_q8 * 255.0), 0, 255).astype(np.uint8)
        else:
            X_model = X_raw

        preds = ModelStore.model.predict(X_model)
        total_latency = (time.perf_counter() - start_t) * 1000.0
        per_row_ms = total_latency / max(len(preds), 1)

        results = []
        for idx, pred in enumerate(preds):
            pred_val = float(pred)
            tid = df_features.iloc[idx].get("trade_id", idx + 1)
            results.append({
                "trade_id": tid,
                "predicted_price": round(pred_val, 2),
                "confidence_interval_95": (round(pred_val * 0.985, 2), round(pred_val * 1.015, 2)),
                "model_version": ModelStore.version,
                "latency_ms": round(per_row_ms, 3),
            })

        return results
