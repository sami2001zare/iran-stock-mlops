"""
Data Quality Validation Contracts & Schema Checkers
===================================================
Enforces strict schema compliance and domain bound assertions
before Bronze trade data is promoted to Silver Lakehouse tables.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class TradeRowSchema(BaseModel):
    """Pydantic v2 contract definition for an individual spot trade tick."""

    trade_id: int = Field(..., gt=0, description="Unique numeric exchange trade ID")
    price: float = Field(..., gt=0.0, lt=1_000_000.0, description="Trade execution price in USD/USDT")
    quantity: float = Field(..., gt=0.0, lt=100_000.0, description="Base asset traded quantity")
    quote_quantity: float = Field(..., ge=0.0, description="Total quote value (price * quantity)")
    timestamp: int = Field(..., gt=1_500_000_000_000, description="Epoch timestamp in milliseconds")
    is_buyer_maker: bool = Field(..., description="True if buyer was market maker")
    is_best_match: bool = Field(True, description="True if trade was matched at best price")

    @field_validator("quote_quantity")
    @classmethod
    def verify_quote_product(cls, v: float, info: Any) -> float:
        """Assert quote quantity matches price * quantity within rounding tolerance."""
        if "price" in info.data and "quantity" in info.data:
            expected = info.data["price"] * info.data["quantity"]
            if abs(v - expected) / (expected + 1e-9) > 0.05:
                raise ValueError(f"Quote quantity mismatch: got {v}, expected ~{expected}")
        return v


class DataContractValidator:
    """Batch-level data quality validation engine."""

    @classmethod
    def validate_dataframe(
        cls,
        df: pd.DataFrame,
        min_rows: int = 1_000,
        max_null_pct: float = 0.001,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Execute comprehensive data quality checks on chunk or partition DataFrame.
        Returns (is_valid, validation_report).
        """
        report: dict[str, Any] = {
            "total_rows": len(df),
            "checks_passed": True,
            "errors": [],
            "metrics": {},
        }

        # Check 1: Minimum row volume assertion
        if len(df) < min_rows:
            report["checks_passed"] = False
            report["errors"].append(f"Row count {len(df)} is below minimum threshold ({min_rows})")

        # Check 2: Null value percentage
        null_counts = df.isnull().sum()
        max_col_null_pct = (null_counts / max(len(df), 1)).max()
        report["metrics"]["max_null_pct"] = float(max_col_null_pct)
        if max_col_null_pct > max_null_pct:
            report["checks_passed"] = False
            report["errors"].append(f"Null percentage {max_col_null_pct:.4f} exceeds threshold ({max_null_pct})")

        # Check 3: Domain bound anomalies (prices <= 0 or quantities <= 0)
        invalid_prices = int((df["price"] <= 0).sum())
        invalid_quantities = int((df["quantity"] <= 0).sum())
        report["metrics"]["invalid_prices"] = invalid_prices
        report["metrics"]["invalid_quantities"] = invalid_quantities

        if invalid_prices > 0 or invalid_quantities > 0:
            report["checks_passed"] = False
            report["errors"].append(f"Found {invalid_prices} non-positive prices and {invalid_quantities} non-positive quantities.")

        # Check 4: Duplicate trade IDs
        duplicate_trades = int(df["trade_id"].duplicated().sum())
        report["metrics"]["duplicate_trades"] = duplicate_trades
        if duplicate_trades > len(df) * 0.05:
            report["checks_passed"] = False
            report["errors"].append(f"Excessive duplicate trade_ids found: {duplicate_trades}")

        if report["checks_passed"]:
            logger.info("✅ Data quality validation passed on %d rows.", len(df))
        else:
            logger.error("❌ Data quality validation FAILED: %s", report["errors"])

        return report["checks_passed"], report
