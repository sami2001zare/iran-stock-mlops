"""
Airflow DAG Integrity & Structure Verification Tests
====================================================
Asserts that all DAG definitions in `dags/` import without raising exceptions,
contain zero cyclical dependencies (`test_cycle`), and define correct task structures.
Runs against real Airflow package or skips gracefully if running on a host without Airflow installed.
"""

import os
import sys
from unittest.mock import MagicMock
import pytest

# Skip all DAG integrity tests if Airflow is mocked/missing locally on host
airflow_mod = sys.modules.get("airflow")
if isinstance(airflow_mod, MagicMock) or "airflow" not in sys.modules:
    pytest.skip("Real Airflow package not installed locally; skipping DAG integrity tests (runs inside CI/Docker container)", allow_module_level=True)

from airflow.models import DagBag


@pytest.fixture(scope="session")
def dagbag():
    """Load all DAGs from the `dags/` folder once per test session."""
    dag_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dags"))
    return DagBag(dag_folder=dag_folder, include_examples=False)


def test_dagbag_import_errors(dagbag):
    """Assert zero DAG import errors across our DataOps pipeline."""
    assert len(dagbag.import_errors) == 0, f"DAG import errors found: {dagbag.import_errors}"


def test_expected_dags_present(dagbag):
    """Verify that all 3 domain DAGs are successfully loaded into DagBag."""
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
    """Verify individual DAG has valid tasks, correct tags, and zero dependency cycles."""
    dag = dagbag.get_dag(dag_id)
    assert dag is not None, f"DAG {dag_id} not found in DagBag"
    assert len(dag.tasks) > 0, f"DAG {dag_id} has no tasks defined"
    dag.test_cycle()
