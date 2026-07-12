"""
Tests for the ETH model pipeline (eth_price_model_v4).
Uses synthetic data so no real CSV is required.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import eth_price_model_v4 as m


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_df(tmp_path):
    """500-row synthetic trade CSV that mimics ETHUSDT format."""
    n = 500
    rng = np.random.default_rng(42)
    ts_base = 1_718_265_600_000  # 2026-06-13 00:00 UTC in ms
    df = pd.DataFrame({
        "trade_id":       np.arange(n),
        "price":          3500.0 + rng.standard_normal(n).cumsum() * 0.5,
        "quantity":       rng.exponential(0.1, n),
        "quote_quantity":  0.0,
        "timestamp":      ts_base + np.arange(n) * 500,
        "is_buyer_maker": rng.integers(0, 2, n).astype(bool),
        "is_best_match":  True,
    })
    df["quote_quantity"] = df["price"] * df["quantity"]

    csv_path = tmp_path / "test_trades.csv"
    df.to_csv(csv_path, header=False, index=False)
    return str(csv_path), df


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_load_data(synthetic_df, tmp_path):
    csv_path, _ = synthetic_df
    df = m.load_data(csv_path, m.COL_NAMES)
    assert len(df) == 500
    assert "price" in df.columns


def test_feature_engineering(synthetic_df):
    _, raw_df = synthetic_df
    df, feature_cols, target_col, X, y = m.feature_engineering(
        raw_df.copy(), m.WINDOWS, m.LAG_STEPS, m.DROP_COLS
    )
    assert len(feature_cols) > 0
    assert X.shape[1] == len(feature_cols)
    assert X.shape[0] == y.shape[0]
    assert target_col == "price"


def test_quantize_data(synthetic_df):
    _, raw_df = synthetic_df
    _, _, _, X, _ = m.feature_engineering(raw_df.copy(), m.WINDOWS, m.LAG_STEPS, m.DROP_COLS)
    X_model, X_q8, qt = m.quantize_data(X)
    assert X_model.dtype == np.float32
    assert X_model.min() >= 0.0
    assert X_model.max() <= 1.0


def test_train_test_split(synthetic_df):
    _, raw_df = synthetic_df
    _, _, _, X, y = m.feature_engineering(raw_df.copy(), m.WINDOWS, m.LAG_STEPS, m.DROP_COLS)
    X_model, _, _ = m.quantize_data(X)
    X_train, X_test, y_train, y_test = m.train_test_split_temporal(X_model, y, 0.8)
    total = len(X_train) + len(X_test)
    assert total == len(X_model)
    assert abs(len(X_train) / total - 0.8) < 0.01


def test_train_and_prune(synthetic_df):
    _, raw_df = synthetic_df
    _, feature_cols, _, X, y = m.feature_engineering(raw_df.copy(), m.WINDOWS, m.LAG_STEPS, m.DROP_COLS)
    X_model, _, _ = m.quantize_data(X)
    X_train, X_test, y_train, _ = m.train_test_split_temporal(X_model, y)

    # Use 2 batches to keep the test fast
    model, total_est, _ = m.train_model(X_train, y_train, batch_size=100, n_est_per_batch=5, max_batches=2)
    assert total_est > 0

    result = m.prune_and_retrain(model, X_train, X_test, y_train, feature_cols, total_est)
    model_pruned, pruned_cols = result[0], result[1]
    assert len(pruned_cols) > 0
    assert len(pruned_cols) <= len(feature_cols)


def test_compute_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 305.0])
    metrics = m.compute_metrics(y_true, y_pred, "test")
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["MAE"] > 0
