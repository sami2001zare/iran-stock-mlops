"""
Shared fixtures for all test modules.
Patches MLflow globally so unit tests never need a live tracking server.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_mlflow():
    """Patch every MLflow call so tests run without a tracking server."""
    fake_run = MagicMock()
    fake_run.info.run_id = "test-run-id-0000"

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=fake_run),
        patch("mlflow.end_run"),
        patch("mlflow.log_param"),
        patch("mlflow.log_params"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_artifact"),
        patch("mlflow.sklearn.log_model"),
        patch("mlflow.tracking.MlflowClient"),
    ):
        yield
