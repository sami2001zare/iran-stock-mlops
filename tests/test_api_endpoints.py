import os
import sys
from unittest.mock import MagicMock
import pytest

fastapi_mod = sys.modules.get("fastapi")
if isinstance(fastapi_mod, MagicMock) or "fastapi" not in sys.modules:
    pytest.skip("Real FastAPI package not installed locally; skipping API endpoint tests (runs inside CI/Docker container)", allow_module_level=True)

from fastapi.testclient import TestClient

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(PROJ_ROOT, "api") not in sys.path:
    sys.path.insert(0, os.path.join(PROJ_ROOT, "api"))

from main import app
from services.inference import ModelStore

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ONLINE"
    assert "model_loaded" in data


def test_predict_endpoint_fallback_when_model_missing():
    ModelStore.model = None
    payload = {
        "trades": [{
            "trade_id": 1, "price": 3500.0, "quantity": 0.05,
            "quote_quantity": 175.0, "timestamp": 1718265600000,
            "is_buyer_maker": False, "is_best_match": True,
        }]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code in (200, 503)


def test_drift_detect_endpoint():
    payload = {
        "reference_prices": [3500.0 + i * 0.1 for i in range(100)],
        "current_prices": [3500.0 + i * 0.1 for i in range(100)],
        "psi_threshold": 0.20,
    }
    r = client.post("/drift/detect", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "SUCCESS"
    assert data["is_drift_detected"] is False


def test_prometheus_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text or "python_gc_" in r.text
