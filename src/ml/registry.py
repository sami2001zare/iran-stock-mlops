"""
MLflow Model Registry & Production Promotion Workflow
=====================================================
Logs quantitative model runs, tracks hyperparameter tuning experiments,
promotes champion models to `Production` status, and triggers FastAPI hot-swaps.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import mlflow
import requests
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowRegistryManager:
    """Handles experiment tracking and model promotion lifecycle."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        model_name: str = "ETHPricePredictor",
    ):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.model_name = model_name or os.getenv("MLFLOW_MODEL_NAME", "ETHPricePredictor")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def log_and_register_model(
        self,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any],
        artifact_dir: str,
        experiment_name: str = "ETH_USDT_Lakehouse_Quant_v2",
    ) -> str:
        """Log parameters, metrics, and artifacts to MLflow experiment and register model."""
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("Logging model run ID %s to MLflow (%s)...", run_id, self.tracking_uri)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts(artifact_dir, artifact_path="model_artifacts")

            # Try logging model directly to MLflow Registry
            try:
                from mlflow.models.signature import infer_signature
                import numpy as np
                dummy_x = np.zeros((1, params.get("active_pruned_features", 10)), dtype=np.uint8)
                dummy_y = model.predict(dummy_x)
                signature = infer_signature(dummy_x, dummy_y)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.model_name,
                    signature=signature,
                )
            except Exception as exc:
                logger.warning("Could not log direct sklearn model to registry (fallback to artifact store): %s", exc)

        return run_id

    def promote_to_production(self, run_id: str, min_r2_threshold: float = 0.50) -> bool:
        """
        Evaluate run metrics; if superior to current champion or passing threshold,
        promote model to 'Production' in MLflow Registry and notify FastAPI serving endpoints.
        """
        try:
            run = self.client.get_run(run_id)
            run_r2 = float(run.data.metrics.get("r2", 0.0))
            if run_r2 < min_r2_threshold:
                logger.warning("Run %s R2 (%.4f) below minimum threshold (%.2f). Aborting promotion.", run_id, run_r2, min_r2_threshold)
                return False

            # Search existing versions of the registered model
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            target_version = None
            for v in versions:
                if v.run_id == run_id:
                    target_version = v.version
                    break

            if target_version:
                logger.info("Promoting version %s of model %s to Production stage...", target_version, self.model_name)
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=target_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                self._trigger_fastapi_reload()
                return True
        except Exception as exc:
            logger.error("Failed to promote run %s to Production: %s", run_id, exc)

        return False

    @staticmethod
    def _trigger_fastapi_reload(api_url: str = "http://fastapi:8000/model/reload") -> None:
        """Send webhook pulse to FastAPI serving engine to hot-reload artifacts."""
        try:
            r = requests.post(api_url, timeout=5)
            if r.status_code == 200:
                logger.info("✅ Successfully triggered FastAPI hot-reload for new model.")
            else:
                logger.warning("FastAPI reload webhook returned status %d", r.status_code)
        except Exception as exc:
            logger.debug("FastAPI reload webhook notice (maybe container offline or local mode): %s", exc)
