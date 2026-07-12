"""
ETH Price Prediction — FastAPI Service
=======================================
Endpoints:
  GET  /health               liveness probe
  GET  /model/info           metadata about the loaded model
  POST /predict              predict ETH price for one or more trade rows
  GET  /metrics/latest       latest registered model metrics from MLflow
  POST /drift/detect         run PSI-based drift detection on incoming data
"""

from __future__ import annotations

import os
import json
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Config (override via env vars)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR           = os.getenv("ETH_MODEL_DIR",       "/opt/airflow/model/eth_model_artifacts")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MLFLOW_MODEL_NAME",   "ETHPricePredictor")
PSI_THRESHOLD       = float(os.getenv("PSI_THRESHOLD", "0.2"))


# ─────────────────────────────────────────────────────────────────────────────
# Load model artifacts at startup
# ─────────────────────────────────────────────────────────────────────────────
class ModelStore:
    model       = None
    transformer = None
    meta: dict  = {}

    @classmethod
    def load(cls):
        model_path = os.path.join(MODEL_DIR, "model_pruned.pkl")
        qt_path    = os.path.join(MODEL_DIR, "quantile_transformer.pkl")
        meta_path  = os.path.join(MODEL_DIR, "pipeline_meta.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        cls.model       = joblib.load(model_path)
        cls.transformer = joblib.load(qt_path)
        with open(meta_path, encoding="utf-8") as f:
            cls.meta = json.load(f)

    @classmethod
    def is_ready(cls) -> bool:
        return cls.model is not None


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ModelStore.load()
    except Exception as exc:
        print(f"[WARN] Model not loaded on startup: {exc}")
    yield


app = FastAPI(
    title="ETH Price Prediction API",
    description="Serves the pruned HistGBR model trained by eth_price_model_v4",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class TradeRow(BaseModel):
    trade_id:       int
    price:          float
    quantity:       float
    quote_quantity: float
    timestamp:      int
    is_buyer_maker: bool
    is_best_match:  bool


class PredictRequest(BaseModel):
    trades: list[TradeRow]


class PredictResponse(BaseModel):
    predictions:   list[dict[str, Any]]
    model_version: str
    latency_ms:    float


class DriftRequest(BaseModel):
    reference_prices: list[float] = Field(..., description="Training distribution sample")
    current_prices:   list[float] = Field(..., description="Current window of prices")


class DriftResponse(BaseModel):
    psi:       float
    drifted:   bool
    threshold: float
    message:   str


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (mirrors eth_price_model_v4 exactly)
# ─────────────────────────────────────────────────────────────────────────────
WINDOWS   = [5, 10, 20, 50, 100]
LAG_STEPS = [1, 2, 3, 5, 10, 20]
DROP_COLS = {
    "trade_id", "price", "timestamp", "ts_ms", "ts_sec",
    "is_buyer_maker", "is_best_match",
}


def _build_features(trades: list[TradeRow]) -> np.ndarray:
    col_names = ["trade_id", "price", "quantity", "quote_quantity",
                 "timestamp", "is_buyer_maker", "is_best_match"]
    df = pd.DataFrame([t.model_dump() for t in trades], columns=col_names)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["ts_ms"]       = df["timestamp"] / 1000.0
    df["ts_sec"]      = df["ts_ms"] / 1000.0
    df["elapsed_sec"] = df["ts_sec"] - df["ts_sec"].iloc[0]

    epoch_dt = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["hour_of_day"]   = epoch_dt.dt.hour
    df["minute_of_day"] = epoch_dt.dt.hour * 60 + epoch_dt.dt.minute
    df["second_sin"]    = np.sin(2 * np.pi * epoch_dt.dt.second / 60.0)
    df["second_cos"]    = np.cos(2 * np.pi * epoch_dt.dt.second / 60.0)

    df["log_quantity"]       = np.log1p(df["quantity"])
    df["log_quote_quantity"] = np.log1p(df["quote_quantity"])
    df["buyer_maker_int"]    = df["is_buyer_maker"].astype(int)
    df["implied_price"]      = df["quote_quantity"] / (df["quantity"] + 1e-9)
    df["trade_size_bucket"]  = pd.qcut(df["quantity"], q=10, labels=False, duplicates="drop")

    for w in WINDOWS:
        df[f"price_roll_mean_{w}"] = df["price"].rolling(w, min_periods=1).mean()
        df[f"price_roll_std_{w}"]  = df["price"].rolling(w, min_periods=1).std().fillna(0)
        df[f"price_roll_max_{w}"]  = df["price"].rolling(w, min_periods=1).max()
        df[f"price_roll_min_{w}"]  = df["price"].rolling(w, min_periods=1).min()
        df[f"qty_roll_mean_{w}"]   = df["quantity"].rolling(w, min_periods=1).mean()
        df[f"qty_roll_sum_{w}"]    = df["quantity"].rolling(w, min_periods=1).sum()
        df[f"buyer_roll_sum_{w}"]  = df["buyer_maker_int"].rolling(w, min_periods=1).sum()

    for lag in LAG_STEPS:
        df[f"price_lag_{lag}"]  = df["price"].shift(lag)
        df[f"qty_lag_{lag}"]    = df["quantity"].shift(lag)
        df[f"return_lag_{lag}"] = df["price"].pct_change(lag)

    df["price_momentum_5"]   = df["price"] - df["price_roll_mean_5"]
    df["price_momentum_20"]  = df["price"] - df["price_roll_mean_20"]
    df["volatility_ratio"]   = df["price_roll_std_10"] / (df["price_roll_mean_10"] + 1e-9)
    df["bollinger_upper_20"] = df["price_roll_mean_20"] + 2 * df["price_roll_std_20"]
    df["bollinger_lower_20"] = df["price_roll_mean_20"] - 2 * df["price_roll_std_20"]
    df["bollinger_band_pct"] = (df["price"] - df["bollinger_lower_20"]) / (
        df["bollinger_upper_20"] - df["bollinger_lower_20"] + 1e-9
    )
    df["vwap_20"]       = (
        (df["price"] * df["quantity"]).rolling(20, min_periods=1).sum()
        / df["quantity"].rolling(20, min_periods=1).sum()
    )
    df["price_vs_vwap"] = df["price"] - df["vwap_20"]

    df = df.ffill().fillna(0)

    feature_cols = ModelStore.meta.get("feature_cols", [c for c in df.columns if c not in DROP_COLS])
    return df[feature_cols].values.astype(np.float32)


def _apply_pipeline(X_raw: np.ndarray) -> np.ndarray:
    X_qt      = ModelStore.transformer.transform(X_raw).astype(np.float32)
    mask_keep = np.array(ModelStore.meta["mask_keep"])
    return X_qt[:, mask_keep]


# ─────────────────────────────────────────────────────────────────────────────
# PSI drift detection
# ─────────────────────────────────────────────────────────────────────────────
def _psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current,   bins=breakpoints)[0]

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ModelStore.is_ready()}


@app.get("/model/info")
def model_info():
    if not ModelStore.is_ready():
        raise HTTPException(503, "Model not loaded")
    meta = ModelStore.meta
    return {
        "model_name":           MODEL_NAME,
        "n_features_raw":       len(meta.get("feature_cols", [])),
        "n_features_pruned":    len(meta.get("pruned_cols",  [])),
        "n_estimators":         meta.get("n_estimators_final"),
        "importance_threshold": meta.get("importance_threshold"),
        "metrics":              meta.get("metrics_pruned", {}),
        "mlflow_tracking_uri":  MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not ModelStore.is_ready():
        raise HTTPException(503, "Model not loaded — run the training pipeline first")

    t0     = time.perf_counter()
    X_raw  = _build_features(req.trades)
    X_pipe = _apply_pipeline(X_raw)
    preds  = ModelStore.model.predict(X_pipe)
    ms     = (time.perf_counter() - t0) * 1000

    results = [
        {"trade_id": t.trade_id, "actual_price": t.price,
         "predicted_price": round(float(p), 4),
         "error": round(float(t.price - p), 4)}
        for t, p in zip(req.trades, preds)
    ]
    return PredictResponse(
        predictions=results,
        model_version=MODEL_NAME,
        latency_ms=round(ms, 2),
    )


@app.get("/metrics/latest")
def latest_metrics():
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME)
        if not versions:
            raise HTTPException(404, f"No versions for model '{MODEL_NAME}'")
        latest = versions[-1]
        run    = client.get_run(latest.run_id)
        return {
            "model_name": MODEL_NAME,
            "version":    latest.version,
            "run_id":     latest.run_id,
            "metrics":    run.data.metrics,
            "params":     run.data.params,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"MLflow unreachable: {exc}")


@app.post("/drift/detect", response_model=DriftResponse)
def detect_drift(req: DriftRequest):
    ref = np.array(req.reference_prices, dtype=np.float64)
    cur = np.array(req.current_prices,   dtype=np.float64)

    if len(ref) < 10 or len(cur) < 10:
        raise HTTPException(422, "Need at least 10 prices in each window")

    psi_score = _psi(ref, cur)
    drifted   = psi_score > PSI_THRESHOLD
    return DriftResponse(
        psi=round(psi_score, 4),
        drifted=drifted,
        threshold=PSI_THRESHOLD,
        message="Drift detected — consider retraining" if drifted else "No significant drift",
    )
