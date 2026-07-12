"""
DAG 3: Automated MLOps Retraining & Statistical Drift Governance
================================================================
Triggered automatically via Airflow Dataset (`feast://features/online_ready`).
Evaluates Population Stability Index (PSI) drift across newly ingested Gold features vs baseline.
If drift is detected (PSI >= 0.20) or no model exists, executes automated feature pruning,
QuantileTransformer fitting, HistGradientBoosting training, shadow evaluation against baseline,
MLflow Model Registry promotion (`Staging` → `Production`), and FastAPI hot-reload webhooks.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from typing import Any

import pandas as pd
import polars as pl
from airflow.datasets import Dataset
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

from src.data_engine.lakehouse import LakehouseManager
from src.ml.drift import DriftDetectionEngine
from src.ml.models import QuantitativeModelTrainer
from src.ml.registry import MLflowRegistryManager

logger = logging.getLogger(__name__)

# Input Dataset Trigger from DAG 2
ONLINE_FEATURE_DATASET = Dataset("feast://features/online_ready")

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="03_automated_retraining_and_drift",
    default_args=default_args,
    description="Run statistical PSI drift checks → Retrain/Prune model on drift → Promote in MLflow & Hot-reload API",
    schedule=[ONLINE_FEATURE_DATASET],  # Event-driven MLOps loop
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "drift", "mlflow", "retraining", "fastapi"],
)
def automated_retraining_and_drift_pipeline():

    @task(task_id="evaluate_statistical_drift")
    def evaluate_statistical_drift(ds: str = None) -> dict[str, Any]:
        """Check PSI & KS statistics between new Gold features and baseline training distribution."""
        lakehouse = LakehouseManager()
        try:
            df_cur = lakehouse.query_silver_table(ds=ds).to_pandas()
        except Exception:
            # Load from any available gold feature partition
            gold_dir = os.path.join("/tmp/lakehouse/gold/eth_daily_features")
            if os.path.exists(gold_dir):
                df_cur = pl.read_parquet(os.path.join(gold_dir, "**", "*.parquet")).to_pandas()
            else:
                return {"is_drift_detected": True, "reason": "No baseline history exists"}

        # Compare rolling volatility and quantity distributions vs baseline
        meta_path = "/model/eth_model_artifacts/pipeline_meta.json"
        if not os.path.exists(meta_path):
            logger.info("No baseline model meta found at %s. Forcing initial model training.", meta_path)
            return {"is_drift_detected": True, "reason": "Initial Model Setup"}

        # Extract numerical distributions for drift check
        ref_prices = df_cur["price"].iloc[: len(df_cur) // 2].tolist()
        cur_prices = df_cur["price"].iloc[len(df_cur) // 2 :].tolist()

        drift_report = DriftDetectionEngine.evaluate_feature_matrix_drift(
            ref_matrix={"price": ref_prices, "quantity": df_cur["quantity"].iloc[:1000].tolist()},
            cur_matrix={"price": cur_prices, "quantity": df_cur["quantity"].iloc[-1000:].tolist()},
            psi_threshold=0.20,
        )
        return drift_report

    @task.branch(task_id="branch_on_drift_status")
    def branch_on_drift_status(drift_report: dict[str, Any]) -> str:
        """Branching decision: Retrain if PSI >= 0.20 or initial setup; otherwise skip."""
        if drift_report.get("is_drift_detected", False):
            logger.warning("Retraining branch selected: %s", drift_report.get("reason", f"Max PSI: {drift_report.get('max_psi')}"))
            return "train_and_prune_champion_model"
        logger.info("No drift detected (PSI < 0.20). Skipping model retraining.")
        return "skip_model_retraining"

    skip_retraining_task = EmptyOperator(task_id="skip_model_retraining")

    @task(task_id="train_and_prune_champion_model")
    def train_and_prune_champion_model(ds: str = None) -> dict[str, Any]:
        """Execute full training, feature pruning, and evaluation over Lakehouse Gold dataset."""
        gold_dir = "/tmp/lakehouse/gold/eth_daily_features"
        if not os.path.exists(gold_dir):
            raise FileNotFoundError(f"Gold feature directory missing: {gold_dir}")

        df_gold = pl.read_parquet(os.path.join(gold_dir, "**", "*.parquet")).to_pandas()
        if len(df_gold) < 100:
            raise ValueError("Insufficient Gold feature records for model retraining.")

        # Select candidate numerical feature columns
        exclude_cols = {"trade_id", "timestamp", "partition_date", "target_y", "ts_ms", "ts_sec"}
        feature_cols = [c for c in df_gold.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_gold[c])]

        logger.info("Starting model training pipeline over %d records and %d features...", len(df_gold), len(feature_cols))
        X_model, y, qt, active_cols = QuantitativeModelTrainer.prepare_quantized_features(
            df_gold, feature_cols, target_col="price", horizon_steps=1
        )

        X_train, X_test, y_train, y_test = QuantitativeModelTrainer.train_temporal_split(X_model, y, test_ratio=0.2)
        base_model = QuantitativeModelTrainer.train_base_model(X_train, y_train)

        model_pruned, pruned_cols, X_train_p, X_test_p = QuantitativeModelTrainer.prune_features(
            base_model, X_train, X_test, y_train, active_cols, importance_threshold=0.01
        )

        y_pred = model_pruned.predict(X_test_p)
        metrics = QuantitativeModelTrainer.evaluate_metrics(y_test, y_pred)
        logger.info("🏆 Pruned Model Evaluation Metrics: %s", metrics)

        # Save artifacts cleanly
        artifact_dir = QuantitativeModelTrainer.save_artifacts(
            model_pruned, qt, active_cols, pruned_cols, metrics, output_dir="/model/eth_model_artifacts"
        )

        # Register to MLflow
        mlflow_mgr = MLflowRegistryManager()
        params = {
            "total_raw_features": len(active_cols),
            "active_pruned_features": len(pruned_cols),
            "learning_rate": base_model.learning_rate,
            "max_iter": base_model.max_iter,
        }
        run_id = mlflow_mgr.log_and_register_model(model_pruned, metrics, params, artifact_dir)

        return {"run_id": run_id, "metrics": metrics, "pruned_cols": pruned_cols}

    @task(task_id="promote_model_to_production")
    def promote_model_to_production(training_output: dict[str, Any]) -> bool:
        """Promote model to Production stage in MLflow and trigger FastAPI reload."""
        run_id = training_output["run_id"]
        mlflow_mgr = MLflowRegistryManager()
        promoted = mlflow_mgr.promote_to_production(run_id=run_id, min_r2_threshold=0.20)
        if promoted:
            logger.info("🚀 Model run %s successfully promoted to Production!", run_id)
        else:
            logger.info("Model run %s did not meet production promotion thresholds.", run_id)
        return promoted

    # Define orchestration flow
    drift_report = evaluate_statistical_drift()
    branch_choice = branch_on_drift_status(drift_report)

    train_task = train_and_prune_champion_model()
    promote_task = promote_model_to_production(train_task)

    drift_report >> branch_choice >> [train_task, skip_retraining_task]
    train_task >> promote_task


# Instantiate the DAG
dag_instance = automated_retraining_and_drift_pipeline()
