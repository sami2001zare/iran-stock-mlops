from __future__ import annotations

import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer

logger = logging.getLogger(__name__)


class QuantitativeModelTrainer:
    @classmethod
    def prepare_quantized_features(
        cls,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "price",
        horizon_steps: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, QuantileTransformer, list[str]]:
        clean_df = df.dropna(subset=feature_cols + [target_col]).copy()
        if len(clean_df) < 50:
            raise ValueError("Insufficient rows for model training after dropna.")

        if horizon_steps == 1:
            clean_df["target_y"] = clean_df[target_col].shift(-1)
        else:
            clean_df["target_y"] = np.log(clean_df[target_col].shift(-horizon_steps) / clean_df[target_col])

        clean_df = clean_df.dropna(subset=["target_y"]).reset_index(drop=True)

        X_raw = clean_df[feature_cols].to_numpy(dtype=np.float64)
        y = clean_df["target_y"].to_numpy(dtype=np.float64)

        qt = QuantileTransformer(n_quantiles=min(1000, len(X_raw)), output_distribution="uniform", random_state=42)
        X_q8 = qt.fit_transform(X_raw)

        X_model = np.clip(np.round(X_q8 * 255.0), 0, 255).astype(np.uint8)

        return X_model, y, qt, feature_cols

    @classmethod
    def train_temporal_split(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_idx = int(len(X) * (1.0 - test_ratio))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    @classmethod
    def train_base_model(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        max_iter: int = 150,
        learning_rate: float = 0.08,
    ) -> HistGradientBoostingRegressor:
        logger.info("Training HistGBR model over %d rows with %d features...", X_train.shape[0], X_train.shape[1])
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.5,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    @classmethod
    def prune_features(
        cls,
        model: HistGradientBoostingRegressor,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        feature_cols: list[str],
        importance_threshold: float = 0.01,
    ) -> tuple[HistGradientBoostingRegressor, list[str], np.ndarray, np.ndarray]:
        from sklearn.inspection import permutation_importance

        logger.info("Evaluating feature importances for pruning...")
        result = permutation_importance(model, X_train[:1000], y_train[:1000], n_repeats=3, random_state=42)
        importances = result.importances_mean

        mask_keep = importances >= (np.max(importances) * importance_threshold)
        if not np.any(mask_keep):
            top_indices = np.argsort(importances)[-15:]
            mask_keep = np.zeros_like(importances, dtype=bool)
            mask_keep[top_indices] = True

        pruned_cols = [col for idx, col in enumerate(feature_cols) if mask_keep[idx]]
        logger.info("Pruning reduced features from %d -> %d active features.", len(feature_cols), len(pruned_cols))

        X_train_p = X_train[:, mask_keep]
        X_test_p = X_test[:, mask_keep]

        model_pruned = HistGradientBoostingRegressor(
            max_iter=model.max_iter,
            learning_rate=model.learning_rate,
            max_leaf_nodes=31,
            random_state=42,
        )
        model_pruned.fit(X_train_p, y_train)
        return model_pruned, pruned_cols, X_train_p, X_test_p

    @classmethod
    def evaluate_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate quantitative validation metrics ($R^2$, RMSE, MAE, MAPE)."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        mape_arr = np.abs((y_true - y_pred) / (y_true + 1e-9))
        mape = float(np.mean(mape_arr) * 100.0)

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "mape_pct": round(mape, 4),
        }

    @classmethod
    def save_artifacts(
        cls,
        model_pruned: Any,
        qt: QuantileTransformer,
        feature_cols: list[str],
        pruned_cols: list[str],
        metrics: dict[str, float],
        output_dir: str = "/model/eth_model_artifacts",
    ) -> str:
        if output_dir == "/model/eth_model_artifacts" and not os.path.exists("/model") and os.path.exists("/opt/airflow/model"):
            output_dir = "/opt/airflow/model/eth_model_artifacts"
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            if output_dir.startswith("/model") and os.path.exists("/opt/airflow/model"):
                output_dir = "/opt/airflow/model/eth_model_artifacts"
                os.makedirs(output_dir, exist_ok=True)
            else:
                raise

        model_path = os.path.join(output_dir, "model_pruned.pkl")
        qt_path = os.path.join(output_dir, "quantile_transformer.pkl")
        meta_path = os.path.join(output_dir, "pipeline_meta.json")

        joblib.dump(model_pruned, model_path)
        joblib.dump(qt, qt_path)

        meta = {
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "total_raw_features": len(feature_cols),
            "active_pruned_features": len(pruned_cols),
            "feature_cols": feature_cols,
            "pruned_cols": pruned_cols,
            "metrics": metrics,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved model artifacts (`model_pruned.pkl`, `pipeline_meta.json`) to %s", output_dir)
        return output_dir
