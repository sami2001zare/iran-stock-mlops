"""
FastAPI Model Serving Engine & Quantitative Inference Service
=============================================================
Serves pruned HistGradientBoostingRegressor models over REST API,
queries Feast Redis Online Feature Store for sub-millisecond scoring,
exposes Prometheus telemetry metrics (`/metrics`), and handles PSI drift alarms.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator

from schemas import (
    DriftDetectRequest,
    DriftDetectResponse,
    PredictRequest,
    PredictResponse,
)
from services.inference import InferenceEngine, ModelStore
from src.ml.drift import DriftDetectionEngine

logger = logging.getLogger("api.main")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Lifecycle startup: attempt loading model artifacts from shared volume."""
    logger.info("Initializing FastAPI inference engine...")
    ModelStore.load()
    yield
    logger.info("Shutting down FastAPI inference engine...")


app = FastAPI(
    title="ETH/USDT Quantitative Lakehouse MLOps API",
    description="Production-grade RESTful model serving engine with Feast Redis lookups & OTel/Prometheus telemetry.",
    version="2.0.0",
    lifespan=lifespan,
)

# Instrument Prometheus metrics exporter (`/metrics`)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, Any]:
    """Liveness probe for Kubernetes / Docker health checks."""
    return {
        "status": "ONLINE",
        "model_loaded": ModelStore.is_ready(),
        "model_version": ModelStore.version,
    }


@app.get("/model/info", status_code=status.HTTP_200_OK)
def get_model_info() -> dict[str, Any]:
    """Return metadata regarding the currently loaded champion model."""
    if not ModelStore.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded yet.")

    return {
        "version": ModelStore.version,
        "active_pruned_features": len(ModelStore.meta.get("pruned_cols", [])),
        "pruned_cols": ModelStore.meta.get("pruned_cols", []),
        "metrics": ModelStore.meta.get("metrics", {}),
    }


@app.post("/model/reload", status_code=status.HTTP_200_OK)
def reload_model_artifacts() -> dict[str, Any]:
    """Webhook triggered by Airflow/MLflow pipeline when a new model is promoted."""
    success = ModelStore.load()
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed reloading model artifacts.")
    return {"status": "SUCCESS", "message": f"Successfully reloaded model version {ModelStore.version}"}


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict_price(payload: PredictRequest) -> PredictResponse:
    """Run low-latency model inference on input trades or from Feast online feature store."""
    if not ModelStore.is_ready():
        # Try hot-loading one more time
        if not ModelStore.load():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts not loaded. Please wait for initial Airflow training DAG run.",
            )

    engine = InferenceEngine()
    try:
        if payload.use_online_feature_store:
            lookup_key = payload.trade_id_lookup or "latest"
            single_res = engine.predict_from_online_store(trade_id_lookup=lookup_key)
            predictions = [single_res]
        else:
            if not payload.trades:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No trades provided for scoring.")
            trade_dicts = [t.model_dump() for t in payload.trades]
            predictions = engine.predict_from_trades(trade_dicts)

        return PredictResponse(
            status="SUCCESS",
            model_name=ModelStore.meta.get("model_name", "ETHPricePredictor"),
            active_features_count=len(ModelStore.meta.get("pruned_cols", [])),
            predictions=predictions,
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as exc:
        logger.error("Inference prediction failure: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {exc}")


@app.post("/drift/detect", response_model=DriftDetectResponse, status_code=status.HTTP_200_OK)
def detect_distribution_drift(payload: DriftDetectRequest) -> DriftDetectResponse:
    """Run statistical Population Stability Index (PSI) & KS test between price vectors."""
    psi_score = DriftDetectionEngine.calculate_psi(payload.reference_prices, payload.current_prices)
    ks_stat, ks_pval = DriftDetectionEngine.calculate_ks_test(payload.reference_prices, payload.current_prices)
    is_drift = psi_score >= payload.psi_threshold or ks_pval < 0.01

    return DriftDetectResponse(
        status="SUCCESS",
        psi_score=round(psi_score, 4),
        ks_statistic=round(ks_stat, 4),
        ks_pvalue=round(ks_pval, 4),
        is_drift_detected=is_drift,
        threshold=payload.psi_threshold,
    )
