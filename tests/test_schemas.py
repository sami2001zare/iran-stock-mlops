import pandas as pd
import pytest
from pydantic import ValidationError
from src.data_engine.validators import DataContractValidator, TradeRowSchema


def test_trade_row_schema_valid():
    """Verify clean tick passes Pydantic v2 validation."""
    row = {
        "trade_id": 101,
        "price": 3500.50,
        "quantity": 1.2,
        "quote_quantity": 4200.60,
        "timestamp": 1_718_265_600_000,
        "is_buyer_maker": True,
        "is_best_match": True,
    }
    validated = TradeRowSchema(**row)
    assert validated.trade_id == 101
    assert validated.price == 3500.50


def test_trade_row_schema_invalid_price():
    """Negative or zero price must raise ValidationError."""
    row = {
        "trade_id": 102,
        "price": -50.0,
        "quantity": 1.0,
        "quote_quantity": -50.0,
        "timestamp": 1_718_265_600_000,
        "is_buyer_maker": False,
    }
    with pytest.raises(ValidationError):
        TradeRowSchema(**row)


def test_data_contract_validator_dataframe():
    """Test DataContractValidator DataFrame checking logic."""
    df_good = pd.DataFrame({
        "trade_id": range(1, 150),
        "price": [3500.0] * 149,
        "quantity": [0.1] * 149,
        "quote_quantity": [350.0] * 149,
        "timestamp": [1_718_265_600_000] * 149,
        "is_buyer_maker": [False] * 149,
        "is_best_match": [True] * 149,
    })
    is_valid, report = DataContractValidator.validate_dataframe(df_good, min_rows=100)
    assert is_valid is True
    assert report["checks_passed"] is True

    # Check failure on insufficient rows
    is_valid_fail, report_fail = DataContractValidator.validate_dataframe(df_good.head(10), min_rows=100)
    assert is_valid_fail is False
    assert "below minimum threshold" in report_fail["errors"][0]
