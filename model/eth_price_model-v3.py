"""
ETHUSDT Price Prediction Pipeline
==================================
Features:
    Feature Engineering from raw trading data
    Quantization (int8) for memory reduction
    Batching for efficient processing
    Gradient Boosting Regression (GBR)
    Pruning with feature importance + threshold
    Train/Test Split: 80/20
"""

import pandas as pd
import numpy as np
import warnings
import time
import os
import json
import joblib

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "ETHUSDT-trades-2026-06-13.csv"
MODEL_DIR  = "eth_model_artifacts"
OUT_JSON   = "eth_model_results.json"
COL_NAMES  = [
    "trade_id", "price", "quantity", "quote_quantity",
    "timestamp", "is_buyer_maker", "is_best_match"
]
WINDOWS    = [5, 10, 20, 50, 100]
LAG_STEPS  = [1, 2, 3, 5, 10, 20]
DROP_COLS  = {
    "trade_id", "price", "timestamp", "ts_ms", "ts_sec",
    "is_buyer_maker", "is_best_match"
}

BATCH_SIZE              = 20_000
N_ESTIMATORS_PER_BATCH  = 25
MAX_BATCHES             = 50


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_path: str, col_names: list) -> pd.DataFrame:
    """Load raw trade data from CSV file."""
    print("\n[1/7] Loading Data...")
    t0 = time.time()

    df = pd.read_csv(data_path, header=None, names=col_names)

    print(f"      {len(df):,} rows loaded  ({time.time()-t0:.2f}s)")
    print(f"      price range: {df['price'].min():.2f} – {df['price'].max():.2f} USDT")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame, windows: list, lag_steps: list,
                         drop_cols: set) -> tuple[pd.DataFrame, list, str, np.ndarray, np.ndarray]:
    """Create temporal, trading, rolling, lag, and derivative features."""
    print("\n[2/7] Feature Engineering...")
    t0 = time.time()

    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Temporal features ---
    df["ts_ms"]     = df["timestamp"] / 1000.0
    df["ts_sec"]    = df["ts_ms"] / 1000.0
    df["elapsed_sec"] = df["ts_sec"] - df["ts_sec"].iloc[0]

    epoch_dt = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["hour_of_day"]   = epoch_dt.dt.hour
    df["minute_of_day"] = epoch_dt.dt.hour * 60 + epoch_dt.dt.minute
    df["second_sin"]    = np.sin(2 * np.pi * epoch_dt.dt.second / 60.0)
    df["second_cos"]    = np.cos(2 * np.pi * epoch_dt.dt.second / 60.0)

    # --- Trading features ---
    df["log_quantity"]       = np.log1p(df["quantity"])
    df["log_quote_quantity"] = np.log1p(df["quote_quantity"])
    df["buyer_maker_int"]    = df["is_buyer_maker"].astype(int)
    df["implied_price"]      = df["quote_quantity"] / (df["quantity"] + 1e-9)
    df["trade_size_bucket"]  = pd.qcut(df["quantity"], q=10, labels=False, duplicates="drop")

    # --- Rolling features ---
    for w in windows:
        df[f"price_roll_mean_{w}"] = df["price"].rolling(w, min_periods=1).mean()
        df[f"price_roll_std_{w}"]  = df["price"].rolling(w, min_periods=1).std().fillna(0)
        df[f"price_roll_max_{w}"]  = df["price"].rolling(w, min_periods=1).max()
        df[f"price_roll_min_{w}"]  = df["price"].rolling(w, min_periods=1).min()
        df[f"qty_roll_mean_{w}"]   = df["quantity"].rolling(w, min_periods=1).mean()
        df[f"qty_roll_sum_{w}"]    = df["quantity"].rolling(w, min_periods=1).sum()
        df[f"buyer_roll_sum_{w}"]  = df["buyer_maker_int"].rolling(w, min_periods=1).sum()

    # --- Lag features ---
    for lag in lag_steps:
        df[f"price_lag_{lag}"]  = df["price"].shift(lag)
        df[f"qty_lag_{lag}"]    = df["quantity"].shift(lag)
        df[f"return_lag_{lag}"] = df["price"].pct_change(lag)

    # --- Derivative features ---
    df["price_momentum_5"]   = df["price"] - df["price_roll_mean_5"]
    df["price_momentum_20"]  = df["price"] - df["price_roll_mean_20"]
    df["volatility_ratio"]   = df["price_roll_std_10"] / (df["price_roll_mean_10"] + 1e-9)
    df["bollinger_upper_20"] = df["price_roll_mean_20"] + 2 * df["price_roll_std_20"]
    df["bollinger_lower_20"] = df["price_roll_mean_20"] - 2 * df["price_roll_std_20"]
    df["bollinger_band_pct"] = (df["price"] - df["bollinger_lower_20"]) / (
        df["bollinger_upper_20"] - df["bollinger_lower_20"] + 1e-9
    )

    df["vwap_20"] = (
        (df["price"] * df["quantity"]).rolling(20, min_periods=1).sum() /
        df["quantity"].rolling(20, min_periods=1).sum()
    )
    df["price_vs_vwap"] = df["price"] - df["vwap_20"]

    # --- Drop NaNs ---
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    target_col   = "price"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    print(f"      {len(feature_cols)} features created ({time.time()-t0:.2f}s)")
    print(f"      {len(df):,} rows remained after dropping NaNs")
    return df, feature_cols, target_col, X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. Quantization (INT8)
# ─────────────────────────────────────────────────────────────────────────────
def quantize_data(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, QuantileTransformer]:
    """Apply QuantileTransformer and map to int8 range."""
    print("\n[3/7] Quantizing data (float32 → int8 mapping)...")
    t0 = time.time()

    qt = QuantileTransformer(n_quantiles=256, output_distribution="uniform",
                             random_state=42)
    X_qt = qt.fit_transform(X)
    X_q8 = (X_qt * 255).astype(np.int16)

    mem_f32 = X.nbytes / 1e6
    mem_q8  = X_q8.nbytes / 1e6
    print(f"      float32: {mem_f32:.1f} MB  →  int8: {mem_q8:.1f} MB"
          f"  (Reduced by {100*(1-mem_q8/mem_f32):.0f}%) ({time.time()-t0:.2f}s)")

    X_model = X_qt.astype(np.float32)
    return X_model, X_q8, qt


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
def train_test_split_temporal(X_model: np.ndarray, y: np.ndarray,
                               split_ratio: float = 0.80) -> tuple:
    """Temporal (non-shuffled) train/test split."""
    print("\n[4/7] Temporal train/test split (80% train / 20% test)...")

    split_idx = int(len(X_model) * split_ratio)
    X_train, X_test = X_model[:split_idx], X_model[split_idx:]
    y_train, y_test = y[:split_idx],       y[split_idx:]

    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 5. Batching + Incremental Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train: np.ndarray, y_train: np.ndarray,
                batch_size: int, n_est_per_batch: int,
                max_batches: int) -> tuple[GradientBoostingRegressor, int, list]:
    """Train GradientBoostingRegressor with warm_start batching."""
    print("\n[5/7] Training the model with Batching (GradientBoostingRegressor)...")

    model = GradientBoostingRegressor(
        n_estimators=n_est_per_batch,
        learning_rate=0.08,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        max_features="sqrt",
        warm_start=True,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=5,
        tol=1e-4,
    )

    total_estimators = 0
    train_history    = []

    for batch_num in range(max_batches):
        batch_start = (batch_num * batch_size) % len(X_train)
        batch_end   = min(batch_start + batch_size, len(X_train))

        Xb = X_train[batch_start:batch_end]
        yb = y_train[batch_start:batch_end]

        t_b = time.time()
        target_estimators = total_estimators + n_est_per_batch
        model.set_params(n_estimators=target_estimators)
        model.fit(Xb, yb)
        total_estimators = model.n_estimators_

        y_pred_b = model.predict(Xb)
        mae_b    = mean_absolute_error(yb, y_pred_b)
        r2_b     = r2_score(yb, y_pred_b)

        train_history.append({
            "batch":        batch_num + 1,
            "n_estimators": total_estimators,
            "batch_size":   len(Xb),
            "mae":          round(float(mae_b), 4),
            "r2":           round(float(r2_b),  4),
            "elapsed":      round(time.time() - t_b, 2)
        })

        print(f"      Batch {batch_num+1}/{max_batches} | "
              f"Trees: {total_estimators} | "
              f"MAE: {mae_b:.4f} | R²: {r2_b:.4f} | "
              f"{time.time()-t_b:.1f}s")

    print(f"     Training finished — total trees: {total_estimators}")
    return model, total_estimators, train_history


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature Pruning + Retraining
# ─────────────────────────────────────────────────────────────────────────────
def prune_and_retrain(model: GradientBoostingRegressor,
                      X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray,
                      feature_cols: list,
                      total_estimators: int) -> tuple:
    """Prune low-importance features and retrain with HistGradientBoostingRegressor."""
    print("\n[6/7] Pruning features with Feature Importance...")
    t0 = time.time()

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    threshold    = importances.mean()
    mask_keep    = importances >= threshold
    pruned_cols  = [feature_cols[i] for i in range(len(feature_cols)) if mask_keep[i]]
    removed_cols = [feature_cols[i] for i in range(len(feature_cols)) if not mask_keep[i]]

    print(f"      Importance threshold: {threshold:.6f}")
    print(f"      Kept features: {len(pruned_cols)} from {len(feature_cols)}")
    print(f"      Removed features: {len(removed_cols)}")

    X_train_p = X_train[:, mask_keep]
    X_test_p  = X_test[:,  mask_keep]

    model_pruned = HistGradientBoostingRegressor(
        max_iter=total_estimators,
        learning_rate=0.08,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
    )
    model_pruned.fit(X_train_p, y_train)

    print(f"      The pruned model has been trained ({time.time()-t0:.2f}s)")
    return model_pruned, pruned_cols, removed_cols, mask_keep, feat_df, X_train_p, X_test_p, threshold


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Calculate MAE, RMSE, R², and MAPE."""
    return {
        "label":   label,
        "MAE":     mean_absolute_error(y_true, y_pred),
        "RMSE":    np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":      r2_score(y_true, y_pred),
        "MAPE_%":  mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def evaluate_models(model: GradientBoostingRegressor,
                    model_pruned: HistGradientBoostingRegressor,
                    X_test: np.ndarray, X_test_p: np.ndarray,
                    y_test: np.ndarray, feat_df: pd.DataFrame) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Evaluate both models and print results."""
    print("\n[7/7] Final evaluation on the Test set...")

    y_pred_full   = model.predict(X_test)
    y_pred_pruned = model_pruned.predict(X_test_p)

    m_full   = compute_metrics(y_test, y_pred_full,   "Full Model")
    m_pruned = compute_metrics(y_test, y_pred_pruned, "Pruned Model")

    print("\n" + "=" * 65)
    print("  Final evaluation results")
    print("=" * 65)

    for m in [m_full, m_pruned]:
        print(f"\n  [{m['label']}]")
        print(f"    MAE   (Mean Absolute Error)         : {m['MAE']:.4f} USDT")
        print(f"    RMSE  (Root Mean Square Error)      : {m['RMSE']:.4f} USDT")
        print(f"    R²    (Coefficient of Determination): {m['R2']:.6f}")
        print(f"    MAPE  (Mean Absolute Percentage Error): {m['MAPE_%']:.4f}%")

    print("\n  Sample predictions (first 10 rows of Test):")
    print(f"  {'Actual':>12} {'Main Model':>12} {'Pruned Model':>12} {'Pruned Error':>12}")
    print("  " + "-" * 54)
    for i in range(min(10, len(y_test))):
        err = y_test[i] - y_pred_pruned[i]
        print(f"  {y_test[i]:>12.4f}  {y_pred_full[i]:>12.4f}  "
              f"{y_pred_pruned[i]:>12.4f}  {err:>+12.4f}")

    print("\n  Top 5 important features (Main Model):")
    print(f"  {'Feature':<30}  {'Importance':>10}")
    print("  " + "-" * 44)
    for _, row in feat_df.head(5).iterrows():
        bar = "█" * int(row["importance"] * 500)
        print(f"  {row['feature']:<30}  {row['importance']:>10.6f}  {bar}")

    return m_full, m_pruned, y_pred_full, y_pred_pruned


# ─────────────────────────────────────────────────────────────────────────────
# 8. Save Model Artifacts
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(model_pruned: HistGradientBoostingRegressor,
                   qt: QuantileTransformer,
                   feature_cols: list, pruned_cols: list,
                   mask_keep: np.ndarray, threshold: float,
                   total_estimators: int, m_pruned: dict,
                   model_dir: str) -> None:
    """Save pruned model, quantile transformer, and pipeline metadata."""
    print("\n[Saving] Saving models and pipeline artifacts...")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model_pruned, os.path.join(model_dir, "model_pruned.pkl"))
    joblib.dump(qt,           os.path.join(model_dir, "quantile_transformer.pkl"))

    pipeline_meta = {
        "feature_cols":         feature_cols,
        "pruned_cols":          pruned_cols,
        "mask_keep":            mask_keep.tolist(),
        "windows":              WINDOWS,
        "lag_steps":            LAG_STEPS,
        "drop_cols":            list(DROP_COLS),
        "n_estimators_final":   int(total_estimators),
        "importance_threshold": float(threshold),
        "metrics_pruned": {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in m_pruned.items()
        },
    }
    with open(os.path.join(model_dir, "pipeline_meta.json"), "w", encoding="utf-8") as f:
        json.dump(pipeline_meta, f, ensure_ascii=False, indent=2)

    print(f"      Model saved      : {model_dir}/model_pruned.pkl")
    print(f"      Transformer saved: {model_dir}/quantile_transformer.pkl")
    print(f"      Metadata saved   : {model_dir}/pipeline_meta.json")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Save Results & Predictions
# ─────────────────────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame,
                 feature_cols: list, pruned_cols: list,
                 removed_cols: list, total_estimators: int,
                 X_train: np.ndarray, X_test: np.ndarray,
                 train_history: list,
                 m_full: dict, m_pruned: dict,
                 feat_df: pd.DataFrame,
                 y_test: np.ndarray,
                 y_pred_full: np.ndarray, y_pred_pruned: np.ndarray,
                 out_json: str) -> None:
    """Save JSON results summary and predictions CSV."""
    results = {
        "dataset_rows":             int(len(df)),
        "features_before_pruning":  len(feature_cols),
        "features_after_pruning":   len(pruned_cols),
        "pruned_out_features":      removed_cols,
        "n_estimators_final":       int(total_estimators),
        "train_size":               int(len(X_train)),
        "test_size":                int(len(X_test)),
        "train_batches":            train_history,
        "metrics_full":   {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in m_full.items()},
        "metrics_pruned": {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in m_pruned.items()},
        "top_features": feat_df.head(20)[["feature", "importance"]]
                               .assign(importance=lambda d: d["importance"].round(6))
                               .to_dict(orient="records"),
    }

    os.makedirs("outputs", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({
        "actual_price":     y_test,
        "pred_full":        y_pred_full,
        "pred_pruned":      y_pred_pruned,
        "error_pruned":     y_test - y_pred_pruned,
        "abs_error_pruned": np.abs(y_test - y_pred_pruned),
    })
    pred_df.to_csv("eth_predictions.csv", index=False)

    print(f"\n  JSON results saved: {out_json}")
    print(f"  Predictions saved: {pred_df.shape[0]} rows")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  ETHUSDT Price Prediction — Gradient Boosting Regression")
    print("=" * 65)

    # 1. Load data
    df = load_data(DATA_PATH, COL_NAMES)

    # 2. Feature engineering
    df, feature_cols, target_col, X, y = feature_engineering(
        df, WINDOWS, LAG_STEPS, DROP_COLS
    )

    # 3. Quantization
    X_model, X_q8, qt = quantize_data(X)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_temporal(X_model, y)

    # 5. Train model
    model, total_estimators, train_history = train_model(
        X_train, y_train, BATCH_SIZE, N_ESTIMATORS_PER_BATCH, MAX_BATCHES
    )

    # 6. Prune and retrain
    (model_pruned, pruned_cols, removed_cols,
     mask_keep, feat_df, X_train_p, X_test_p,
     threshold) = prune_and_retrain(
        model, X_train, X_test, y_train, feature_cols, total_estimators
    )

    # 7. Evaluate
    m_full, m_pruned, y_pred_full, y_pred_pruned = evaluate_models(
        model, model_pruned, X_test, X_test_p, y_test, feat_df
    )

    # 8. Save model artifacts
    save_artifacts(
        model_pruned, qt, feature_cols, pruned_cols,
        mask_keep, threshold, total_estimators, m_pruned, MODEL_DIR
    )

    # 9. Save results and predictions
    save_results(
        df, feature_cols, pruned_cols, removed_cols,
        total_estimators, X_train, X_test, train_history,
        m_full, m_pruned, feat_df, y_test,
        y_pred_full, y_pred_pruned, OUT_JSON
    )

    print("\n" + "=" * 65)
    print("  Pipeline completed!")
    print("=" * 65)


if __name__ == "__main__":
    main()
