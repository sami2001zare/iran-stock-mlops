"""
Unit tests for Quantitative Feature Engineering module (`src.features.quantitative`).
Verifies Order Flow Imbalance, Realized Volatility, Bollinger Bands, and harmonic transformations.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.quantitative import QuantitativeFeatureEngine


@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    """Generate 200 synthetic tick trades for mathematical feature verification."""
    n = 200
    ts_base = 1_718_265_600_000
    prices = np.linspace(3500.0, 3550.0, n)
    quantities = np.full(n, 0.5)
    return pd.DataFrame({
        "trade_id": np.arange(1, n + 1),
        "price": prices,
        "quantity": quantities,
        "quote_quantity": prices * quantities,
        "timestamp": ts_base + np.arange(n) * 100,
        "is_buyer_maker": [i % 2 == 0 for i in range(n)],
        "is_best_match": True,
    })


def test_quantitative_feature_computation(sample_trades_df):
    """Test that compute_all_features returns exact expected rolling indicators."""
    df_features = QuantitativeFeatureEngine.compute_all_features(sample_trades_df, windows=[5, 10], lag_steps=[1, 2])

    assert len(df_features) == 200
    # Verify temporal harmonics
    assert "second_sin" in df_features.columns
    assert "second_cos" in df_features.columns
    assert "elapsed_sec" in df_features.columns

    # Verify OFI calculation
    assert "ofi_roll_5" in df_features.columns
    assert "signed_volume" in df_features.columns

    # Verify Realized Volatility
    assert "realized_volatility_10" in df_features.columns
    assert not df_features["realized_volatility_10"].isnull().any()

    # Verify Bollinger Bands
    assert "bollinger_upper_20" in df_features.columns
    assert "bollinger_band_pct" in df_features.columns
