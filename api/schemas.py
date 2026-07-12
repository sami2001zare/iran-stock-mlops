"""
FastAPI Request & Response Pydantic v2 Schemas
==============================================
Provides type-safe data contracts for real-time model scoring and drift detection.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class TradeTickInput(BaseModel):
    """Input payload representing a single exchange tick or pre-aggregated feature row."""

    trade_id: int = Field(..., gt=0, description="Exchange trade ID")
    price: float = Field(..., gt=0.0, description="Current execution price")
    quantity: float = Field(..., gt=0.0, description="Execution volume")
    quote_quantity: float = Field(0.0, ge=0.0, description="Total quote value")
    timestamp: int = Field(..., gt=1_000_000_000_000, description="Execution timestamp (ms)")
    is_buyer_maker: bool = Field(False, description="Maker trade flag")
    is_best_match: bool = Field(True, description="Best price match flag")


class PredictRequest(BaseModel):
    """Batch prediction request payload."""

    trades: list[TradeTickInput] | None = Field(None, description="Raw trade ticks to score")
    use_online_feature_store: bool = Field(
        False, description="If True, fetches latest feature vector from Feast Redis store"
    )
    trade_id_lookup: int | str | None = Field(
        "latest", description="If using online store, specify trade_id key or 'latest'"
    )


class PredictionResult(BaseModel):
    """Prediction output schema for an individual record."""

    trade_id: int | str
    predicted_price: float
    confidence_interval_95: tuple[float, float]
    model_version: str
    latency_ms: float


class PredictResponse(BaseModel):
    """Response container returning multiple predictions and server status."""

    status: str = "SUCCESS"
    model_name: str
    active_features_count: int
    predictions: list[PredictionResult]


class DriftDetectRequest(BaseModel):
    """Request payload to test data drift between reference and incoming price distributions."""

    reference_prices: list[float] = Field(..., min_length=10, description="Baseline reference price vector")
    current_prices: list[float] = Field(..., min_length=10, description="Incoming production price vector")
    psi_threshold: float = Field(0.20, gt=0.0, lt=1.0, description="PSI threshold for drift alarm")


class DriftDetectResponse(BaseModel):
    """Response reporting PSI score and drift alarm status."""

    status: str = "SUCCESS"
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    is_drift_detected: bool
    threshold: float
