import os
import sys
from unittest.mock import MagicMock, patch
import pytest

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [PROJ_ROOT, os.path.join(PROJ_ROOT, "src"), os.path.join(PROJ_ROOT, "api")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import mlflow
except ImportError:
    mock_mlflow_mod = MagicMock()
    sys.modules["mlflow"] = mock_mlflow_mod
    sys.modules["mlflow.tracking"] = MagicMock()
    sys.modules["mlflow.sklearn"] = MagicMock()
    sys.modules["mlflow.models"] = MagicMock()
    sys.modules["mlflow.models.signature"] = MagicMock()
    import mlflow

try:
    import fastapi
except ImportError:
    mock_fastapi = MagicMock()
    sys.modules["fastapi"] = mock_fastapi
    sys.modules["fastapi.testclient"] = MagicMock()
    sys.modules["prometheus_fastapi_instrumentator"] = MagicMock()

try:
    import airflow
except ImportError:
    mock_airflow = MagicMock()
    sys.modules["airflow"] = mock_airflow
    sys.modules["airflow.datasets"] = MagicMock()
    sys.modules["airflow.decorators"] = MagicMock()
    sys.modules["airflow.models"] = MagicMock()
    sys.modules["airflow.operators.empty"] = MagicMock()
    sys.modules["airflow.utils.task_group"] = MagicMock()


@pytest.fixture(autouse=True)
def mock_mlflow():
    """Patch every MLflow call so tests run without a live tracking server."""
    fake_run = MagicMock()
    fake_run.info.run_id = "test-run-id-0000"

    with (
        patch("mlflow.set_tracking_uri", MagicMock()),
        patch("mlflow.set_experiment", MagicMock()),
        patch("mlflow.start_run", MagicMock(return_value=fake_run)),
        patch("mlflow.end_run", MagicMock()),
        patch("mlflow.log_param", MagicMock()),
        patch("mlflow.log_params", MagicMock()),
        patch("mlflow.log_metric", MagicMock()),
        patch("mlflow.log_metrics", MagicMock()),
        patch("mlflow.log_artifact", MagicMock()),
        patch("mlflow.log_artifacts", MagicMock()),
        patch("mlflow.sklearn.log_model", MagicMock()),
        patch("mlflow.tracking.MlflowClient", MagicMock()),
    ):
        yield
