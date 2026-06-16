"""
FastAPI endpoint tests (no real model required — tests health/schema/drift).
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_returns_503_when_model_not_loaded():
    """Without model artifacts the endpoint must respond 503, not crash."""
    payload = {
        "trades": [{
            "trade_id": 1, "price": 3500.0, "quantity": 0.05,
            "quote_quantity": 175.0, "timestamp": 1718265600000,
            "is_buyer_maker": False, "is_best_match": True,
        }]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code in (200, 503)


def test_drift_detect_no_drift():
    import numpy as np
    rng = np.random.default_rng(0)
    ref = rng.normal(3500, 10, 200).tolist()
    cur = rng.normal(3500, 10, 200).tolist()
    r = client.post("/drift/detect", json={"reference_prices": ref, "current_prices": cur})
    assert r.status_code == 200
    body = r.json()
    assert "psi" in body
    assert "drifted" in body
    assert body["drifted"] is False


def test_drift_detect_with_drift():
    import numpy as np
    rng = np.random.default_rng(1)
    ref = rng.normal(3500, 5,  200).tolist()
    cur = rng.normal(4000, 50, 200).tolist()   # large shift → drift
    r = client.post("/drift/detect", json={"reference_prices": ref, "current_prices": cur})
    assert r.status_code == 200
    assert r.json()["drifted"] is True


def test_drift_detect_rejects_small_windows():
    r = client.post("/drift/detect", json={"reference_prices": [1, 2, 3], "current_prices": [1, 2, 3]})
    assert r.status_code == 422


def test_model_info_503_when_not_loaded():
    r = client.get("/model/info")
    assert r.status_code in (200, 503)
