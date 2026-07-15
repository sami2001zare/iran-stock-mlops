import os
import sys
from unittest.mock import MagicMock
import pytest

airflow_mod = sys.modules.get("airflow")
if isinstance(airflow_mod, MagicMock) or "airflow" not in sys.modules:
    pytest.skip("Real Airflow package not installed locally; skipping DAG integrity tests (runs inside CI/Docker container)", allow_module_level=True)

from airflow.models import DagBag


@pytest.fixture(scope="session")
def dagbag():
    dag_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dags"))
    return DagBag(dag_folder=dag_folder, include_examples=False)


def test_dagbag_import_errors(dagbag):
    assert len(dagbag.import_errors) == 0, f"DAG import errors found: {dagbag.import_errors}"


def test_expected_dags_present(dagbag):
    expected_dag_ids = {
        "01_binance_to_lakehouse",
        "02_feature_store_materialization",
        "03_automated_retraining_and_drift",
    }
    loaded_dag_ids = set(dagbag.dag_ids)
    assert expected_dag_ids.issubset(loaded_dag_ids), (
        f"Missing expected DAGs! Expected {expected_dag_ids}, got {loaded_dag_ids}"
    )


@pytest.mark.parametrize("dag_id", [
    "01_binance_to_lakehouse",
    "02_feature_store_materialization",
    "03_automated_retraining_and_drift",
])
def test_dag_structure_and_cycles(dagbag, dag_id):
    dag = dagbag.get_dag(dag_id)
    assert dag is not None, f"DAG {dag_id} not found in DagBag"
    assert len(dag.tasks) > 0, f"DAG {dag_id} has no tasks defined"
    dag.test_cycle()
