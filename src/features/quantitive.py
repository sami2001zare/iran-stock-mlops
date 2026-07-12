"""
Quantitative Feature Engineering & Mathematical Transformations
=============================================================
Calculates high-precision time-series features (Order Flow Imbalance, Realized Volatility,
Multi-horizon Log Returns, Bollinger Bands, and Temporal Harmonics) using Polars/Pandas & NumPy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None

logger = logging.getLogger(__name__)


class QuantitativeFeatureEngine:
    """Computes quantitative trading signals over Silver Lakehouse trade datasets."""

    @classmethod
    def compute_all_features(
        cls,
        df: Any,
        windows: list[int] | None = None,
        lag_steps: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates complete feature vector matrix matching both training and inference.
        Returns Pandas DataFrame formatted cleanly for ML models & Feast materialization.
        """
        if windows is None:
            windows = [5, 10, 20, 50]
        else:
            # Ensure core windows 5, 10, 20 are always present for momentum and Bollinger Bands
            windows = sorted(list(set(windows) | {5, 10, 20}))

        if lag_steps is None:
            lag_steps = [1, 2, 5, 10, 20]

        if pl is not None and isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy()

        # Sort strictly by execution timestamp
        df_pd = df_pd.sort_values("timestamp").reset_index(drop=True)

        # 1. Temporal & Harmonic Features
        df_pd["ts_ms"] = df_pd["timestamp"] / 1000.0
        df_pd["ts_sec"] = df_pd["ts_ms"] / 1000.0
        df_pd["elapsed_sec"] = df_pd["ts_sec"] - df_pd["ts_sec"].iloc[0]

        epoch_dt = pd.to_datetime(df_pd["ts_ms"], unit="ms", utc=True)
        df_pd["hour_of_day"] = epoch_dt.dt.hour
        df_pd["minute_of_day"] = epoch_dt.dt.hour * 60 + epoch_dt.dt.minute
        df_pd["second_sin"] = np.sin(2 * np.pi * epoch_dt.dt.second / 60.0)
        df_pd["second_cos"] = np.cos(2 * np.pi * epoch_dt.dt.second / 60.0)

        # 2. Base Trading Metrics
        df_pd["log_quantity"] = np.log1p(df_pd["quantity"])
        df_pd["log_quote_quantity"] = np.log1p(df_pd["quote_quantity"])
        df_pd["buyer_maker_int"] = df_pd["is_buyer_maker"].astype(int)
        df_pd["implied_price"] = df_pd["quote_quantity"] / (df_pd["quantity"] + 1e-9)
        try:
            df_pd["trade_size_bucket"] = pd.qcut(df_pd["quantity"], q=10, labels=False, duplicates="drop")
        except Exception:
            df_pd["trade_size_bucket"] = 0

        # 3. Rolling Mean, Std, & Order Flow Imbalance (OFI)
        df_pd["tick_direction"] = np.where(df_pd["is_buyer_maker"], -1.0, 1.0)
        df_pd["signed_volume"] = df_pd["tick_direction"] * df_pd["quantity"]

        for w in windows:
            df_pd[f"price_roll_mean_{w}"] = df_pd["price"].rolling(w, min_periods=1).mean()
            df_pd[f"price_roll_std_{w}"] = df_pd["price"].rolling(w, min_periods=1).std().fillna(0.0)
            df_pd[f"price_roll_max_{w}"] = df_pd["price"].rolling(w, min_periods=1).max()
            df_pd[f"price_roll_min_{w}"] = df_pd["price"].rolling(w, min_periods=1).min()
            df_pd[f"qty_roll_mean_{w}"] = df_pd["quantity"].rolling(w, min_periods=1).mean()
            df_pd[f"qty_roll_sum_{w}"] = df_pd["quantity"].rolling(w, min_periods=1).sum()
            df_pd[f"buyer_roll_sum_{w}"] = df_pd["buyer_maker_int"].rolling(w, min_periods=1).sum()
            df_pd[f"ofi_roll_{w}"] = df_pd["signed_volume"].rolling(w, min_periods=1).sum()

        # 4. Multi-Horizon Realized Volatility (RV) & Lags
        df_pd["log_return_tick"] = np.log(df_pd["price"] / df_pd["price"].shift(1).fillna(df_pd["price"]))
        df_pd["sq_return"] = df_pd["log_return_tick"] ** 2

        for w in [10, 50, 100]:
            df_pd[f"realized_volatility_{w}"] = np.sqrt(df_pd["sq_return"].rolling(w, min_periods=1).sum())

        for lag in lag_steps:
            df_pd[f"price_lag_{lag}"] = df_pd["price"].shift(lag).bfill()
            df_pd[f"qty_lag_{lag}"] = df_pd["quantity"].shift(lag).bfill()
            df_pd[f"return_lag_{lag}"] = df_pd["price"].pct_change(lag).fillna(0.0)

        # 5. Derivative Signals & Bollinger Bands
        df_pd["price_momentum_5"] = df_pd["price"] - df_pd["price_roll_mean_5"]
        df_pd["price_momentum_20"] = df_pd["price"] - df_pd["price_roll_mean_20"]
        df_pd["volatility_ratio"] = df_pd["price_roll_std_10"] / (df_pd["price_roll_mean_10"] + 1e-9)
        df_pd["bollinger_upper_20"] = df_pd["price_roll_mean_20"] + 2.0 * df_pd["price_roll_std_20"]
        df_pd["bollinger_lower_20"] = df_pd["price_roll_mean_20"] - 2.0 * df_pd["price_roll_std_20"]
        df_pd["bollinger_band_pct"] = (df_pd["price"] - df_pd["bollinger_lower_20"]) / (
            df_pd["bollinger_upper_20"] - df_pd["bollinger_lower_20"] + 1e-9
        )

        # Fill any minor NaN values from initial window shifts
        df_pd = df_pd.fillna(0.0)
        logger.info("✅ Calculated %d quantitative features across %d trade ticks.", len(df_pd.columns), len(df_pd))
        return df_pd
