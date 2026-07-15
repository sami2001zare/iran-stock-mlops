
import pandas as pd
import numpy as np
import warnings
import time
import os
import json
import joblib

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)

DATA_PATH  = "ETHUSDT-trades-2026-06-13.csv"
MODEL_DIR  = "eth_model_artifacts"
OUT_JSON   = "eth_model_results.json"
COL_NAMES  = [
    "trade_id", "price", "quantity", "quote_quantity",
    "timestamp", "is_buyer_maker", "is_best_match",
]
WINDOWS   = [5, 10, 20, 50, 100]
LAG_STEPS = [1, 2, 3, 5, 10, 20]
DROP_COLS = {
    "trade_id", "price", "timestamp", "ts_ms", "ts_sec",
    "is_buyer_maker", "is_best_match",
}

BATCH_SIZE             = 20_000
N_ESTIMATORS_PER_BATCH = 25
MAX_BATCHES            = 50
LEARNING_RATE          = 0.08
MAX_DEPTH              = 5
MIN_SAMPLES_SPLIT      = 20
MIN_SAMPLES_LEAF       = 10
SUBSAMPLE              = 0.8
N_QUANTILES            = 256
TRAIN_SPLIT            = 0.80

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT   = "eth-price-prediction"


def load_data(data_path: str, col_names: list) -> pd.DataFrame:
    print("\n[1/7] Loading Data...")
    t0 = time.time()
    df = pd.read_csv(data_path, header=None, names=col_names)
    print(f"      {len(df):,} rows loaded  ({time.time()-t0:.2f}s)")
    print(f"      price range: {df['price'].min():.2f} – {df['price'].max():.2f} USDT")

    mlflow.log_params({
        "data_path":  data_path,
        "total_rows": int(len(df)),
        "price_min":  round(float(df["price"].min()), 4),
        "price_max":  round(float(df["price"].max()), 4),
    })
    return df


def feature_engineering(df: pd.DataFrame, windows: list, lag_steps: list,
                         drop_cols: set) -> tuple:
    print("\n[2/7] Feature Engineering...")
    t0 = time.time()

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["ts_ms"]       = df["timestamp"] / 1000.0
    df["ts_sec"]      = df["ts_ms"] / 1000.0
    df["elapsed_sec"] = df["ts_sec"] - df["ts_sec"].iloc[0]

    epoch_dt = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["hour_of_day"]   = epoch_dt.dt.hour
    df["minute_of_day"] = epoch_dt.dt.hour * 60 + epoch_dt.dt.minute
    df["second_sin"]    = np.sin(2 * np.pi * epoch_dt.dt.second / 60.0)
    df["second_cos"]    = np.cos(2 * np.pi * epoch_dt.dt.second / 60.0)

    df["log_quantity"]       = np.log1p(df["quantity"])
    df["log_quote_quantity"] = np.log1p(df["quote_quantity"])
    df["buyer_maker_int"]    = df["is_buyer_maker"].astype(int)
    df["implied_price"]      = df["quote_quantity"] / (df["quantity"] + 1e-9)
    df["trade_size_bucket"]  = pd.qcut(df["quantity"], q=10, labels=False, duplicates="drop")

    for w in windows:
        df[f"price_roll_mean_{w}"] = df["price"].rolling(w, min_periods=1).mean()
        df[f"price_roll_std_{w}"]  = df["price"].rolling(w, min_periods=1).std().fillna(0)
        df[f"price_roll_max_{w}"]  = df["price"].rolling(w, min_periods=1).max()
        df[f"price_roll_min_{w}"]  = df["price"].rolling(w, min_periods=1).min()
        df[f"qty_roll_mean_{w}"]   = df["quantity"].rolling(w, min_periods=1).mean()
        df[f"qty_roll_sum_{w}"]    = df["quantity"].rolling(w, min_periods=1).sum()
        df[f"buyer_roll_sum_{w}"]  = df["buyer_maker_int"].rolling(w, min_periods=1).sum()

    for lag in lag_steps:
        df[f"price_lag_{lag}"]  = df["price"].shift(lag)
        df[f"qty_lag_{lag}"]    = df["quantity"].shift(lag)
        df[f"return_lag_{lag}"] = df["price"].pct_change(lag)

    df["price_momentum_5"]   = df["price"] - df["price_roll_mean_5"]
    df["price_momentum_20"]  = df["price"] - df["price_roll_mean_20"]
    df["volatility_ratio"]   = df["price_roll_std_10"] / (df["price_roll_mean_10"] + 1e-9)
    df["bollinger_upper_20"] = df["price_roll_mean_20"] + 2 * df["price_roll_std_20"]
    df["bollinger_lower_20"] = df["price_roll_mean_20"] - 2 * df["price_roll_std_20"]
    df["bollinger_band_pct"] = (df["price"] - df["bollinger_lower_20"]) / (
        df["bollinger_upper_20"] - df["bollinger_lower_20"] + 1e-9
    )
    df["vwap_20"]       = (
        (df["price"] * df["quantity"]).rolling(20, min_periods=1).sum()
        / df["quantity"].rolling(20, min_periods=1).sum()
    )
    df["price_vs_vwap"] = df["price"] - df["vwap_20"]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["price"].values.astype(np.float32)

    print(f"      {len(feature_cols)} features created ({time.time()-t0:.2f}s)")
    print(f"      {len(df):,} rows remained after dropping NaNs")

    mlflow.log_params({
        "n_features_raw": len(feature_cols),
        "rows_after_dropna": int(len(df)),
        "windows":    str(windows),
        "lag_steps":  str(lag_steps),
    })
    return df, feature_cols, "price", X, y


def quantize_data(X: np.ndarray) -> tuple:
    print("\n[3/7] Quantizing data (float32 → int8 mapping)...")
    t0 = time.time()

    qt = QuantileTransformer(
        n_quantiles=N_QUANTILES, output_distribution="uniform", random_state=42
    )
    X_qt  = qt.fit_transform(X)
    X_q8  = (X_qt * 255).astype(np.int16)

    mem_f32 = X.nbytes / 1e6
    mem_q8  = X_q8.nbytes / 1e6
    reduction = 100 * (1 - mem_q8 / mem_f32)
    print(f"      float32: {mem_f32:.1f} MB  →  int8: {mem_q8:.1f} MB"
          f"  (Reduced by {reduction:.0f}%) ({time.time()-t0:.2f}s)")

    mlflow.log_params({
        "n_quantiles":      N_QUANTILES,
        "mem_reduction_pct": round(reduction, 1),
    })

    X_model = X_qt.astype(np.float32)
    return X_model, X_q8, qt


def train_test_split_temporal(X_model: np.ndarray, y: np.ndarray,
                               split_ratio: float = TRAIN_SPLIT) -> tuple:
    print(f"\n[4/7] Temporal train/test split ({int(split_ratio*100)}% train)...")

    split_idx = int(len(X_model) * split_ratio)
    X_train, X_test = X_model[:split_idx], X_model[split_idx:]
    y_train, y_test = y[:split_idx],       y[split_idx:]

    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    mlflow.log_params({
        "train_split":  split_ratio,
        "train_size":   int(len(X_train)),
        "test_size":    int(len(X_test)),
    })
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                batch_size: int, n_est_per_batch: int,
                max_batches: int) -> tuple:
    print("\n[5/7] Training GradientBoostingRegressor (batched)...")

    mlflow.log_params({
        "batch_size":             batch_size,
        "n_estimators_per_batch": n_est_per_batch,
        "max_batches":            max_batches,
        "learning_rate":          LEARNING_RATE,
        "max_depth":              MAX_DEPTH,
        "min_samples_split":      MIN_SAMPLES_SPLIT,
        "min_samples_leaf":       MIN_SAMPLES_LEAF,
        "subsample":              SUBSAMPLE,
        "max_features":           "sqrt",
    })

    model = GradientBoostingRegressor(
        n_estimators=n_est_per_batch,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        subsample=SUBSAMPLE,
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

        mlflow.log_metrics(
            {"train_batch_mae": round(mae_b, 4), "train_batch_r2": round(r2_b, 4)},
            step=batch_num + 1,
        )

        train_history.append({
            "batch":        batch_num + 1,
            "n_estimators": total_estimators,
            "batch_size":   len(Xb),
            "mae":          round(float(mae_b), 4),
            "r2":           round(float(r2_b),  4),
            "elapsed":      round(time.time() - t_b, 2),
        })

        print(f"      Batch {batch_num+1}/{max_batches} | "
              f"Trees: {total_estimators} | MAE: {mae_b:.4f} | R²: {r2_b:.4f} | "
              f"{time.time()-t_b:.1f}s")

    mlflow.log_metric("n_estimators_final", total_estimators)
    print(f"     Training finished — total trees: {total_estimators}")
    return model, total_estimators, train_history


def prune_and_retrain(model: GradientBoostingRegressor,
                      X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray,
                      feature_cols: list,
                      total_estimators: int) -> tuple:
    print("\n[6/7] Pruning features with Feature Importance...")
    t0 = time.time()

    importances  = model.feature_importances_
    feat_df      = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_df      = feat_df.sort_values("importance", ascending=False).reset_index(drop=True)

    threshold    = importances.mean()
    mask_keep    = importances >= threshold
    pruned_cols  = [feature_cols[i] for i in range(len(feature_cols)) if mask_keep[i]]
    removed_cols = [feature_cols[i] for i in range(len(feature_cols)) if not mask_keep[i]]

    print(f"      Importance threshold: {threshold:.6f}")
    print(f"      Kept features: {len(pruned_cols)} from {len(feature_cols)}")

    mlflow.log_params({
        "importance_threshold":   round(float(threshold), 6),
        "n_features_after_prune": len(pruned_cols),
        "n_features_removed":     len(removed_cols),
    })

    X_train_p = X_train[:, mask_keep]
    X_test_p  = X_test[:,  mask_keep]

    model_pruned = HistGradientBoostingRegressor(
        max_iter=total_estimators,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=42,
    )
    model_pruned.fit(X_train_p, y_train)

    print(f"      Pruned model trained ({time.time()-t0:.2f}s)")
    return model_pruned, pruned_cols, removed_cols, mask_keep, feat_df, X_train_p, X_test_p, threshold


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    return {
        "label":  label,
        "MAE":    mean_absolute_error(y_true, y_pred),
        "RMSE":   np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":     r2_score(y_true, y_pred),
        "MAPE_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def evaluate_models(model, model_pruned,
                    X_test, X_test_p, y_test,
                    feat_df) -> tuple:
    print("\n[7/7] Final evaluation on the Test set...")

    y_pred_full   = model.predict(X_test)
    y_pred_pruned = model_pruned.predict(X_test_p)

    m_full   = compute_metrics(y_test, y_pred_full,   "Full Model")
    m_pruned = compute_metrics(y_test, y_pred_pruned, "Pruned Model")

    mlflow.log_metrics({
        "test_mae_full":    round(m_full["MAE"],    4),
        "test_rmse_full":   round(m_full["RMSE"],   4),
        "test_r2_full":     round(m_full["R2"],     6),
        "test_mape_full":   round(m_full["MAPE_%"], 4),
        "test_mae_pruned":  round(m_pruned["MAE"],    4),
        "test_rmse_pruned": round(m_pruned["RMSE"],   4),
        "test_r2_pruned":   round(m_pruned["R2"],     6),
        "test_mape_pruned": round(m_pruned["MAPE_%"], 4),
    })

    print("\n" + "=" * 65)
    print("  Final evaluation results")
    print("=" * 65)
    for m in [m_full, m_pruned]:
        print(f"\n  [{m['label']}]")
        print(f"    MAE   : {m['MAE']:.4f} USDT")
        print(f"    RMSE  : {m['RMSE']:.4f} USDT")
        print(f"    R²    : {m['R2']:.6f}")
        print(f"    MAPE  : {m['MAPE_%']:.4f}%")

    print("\n  Sample predictions (first 10 rows of Test):")
    print(f"  {'Actual':>12} {'Full':>12} {'Pruned':>12} {'Error':>12}")
    print("  " + "-" * 54)
    for i in range(min(10, len(y_test))):
        err = y_test[i] - y_pred_pruned[i]
        print(f"  {y_test[i]:>12.4f}  {y_pred_full[i]:>12.4f}  "
              f"{y_pred_pruned[i]:>12.4f}  {err:>+12.4f}")

    print("\n  Top 5 important features:")
    print(f"  {'Feature':<30}  {'Importance':>10}")
    print("  " + "-" * 44)
    for _, row in feat_df.head(5).iterrows():
        bar = "█" * int(row["importance"] * 500)
        print(f"  {row['feature']:<30}  {row['importance']:>10.6f}  {bar}")

    return m_full, m_pruned, y_pred_full, y_pred_pruned


def save_artifacts(model_pruned, qt,
                   feature_cols, pruned_cols, mask_keep,
                   threshold, total_estimators, m_pruned,
                   feat_df, model_dir) -> None:
    print("\n[Saving] Saving models and pipeline artifacts...")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model_pruned.pkl")
    qt_path    = os.path.join(model_dir, "quantile_transformer.pkl")
    meta_path  = os.path.join(model_dir, "pipeline_meta.json")
    feat_path  = os.path.join(model_dir, "feature_importance.csv")

    joblib.dump(model_pruned, model_path)
    joblib.dump(qt,           qt_path)

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
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_meta, f, ensure_ascii=False, indent=2)

    feat_df.to_csv(feat_path, index=False)


    mlflow.sklearn.log_model(
        sk_model=model_pruned,
        artifact_path="model",
        registered_model_name="ETHPricePredictor",
        input_example=None,
    )


    mlflow.log_artifact(qt_path,   artifact_path="artifacts")
    mlflow.log_artifact(meta_path, artifact_path="artifacts")
    mlflow.log_artifact(feat_path, artifact_path="artifacts")

    print(f"      Model saved      : {model_path}")
    print(f"      Transformer saved: {qt_path}")
    print(f"      Metadata saved   : {meta_path}")
    print(f"      Feature CSV saved: {feat_path}")
    print("      All artifacts logged to MLflow")


def save_results(df, feature_cols, pruned_cols, removed_cols,
                 total_estimators, X_train, X_test,
                 train_history, m_full, m_pruned, feat_df,
                 y_test, y_pred_full, y_pred_pruned, out_json) -> None:
    results = {
        "dataset_rows":            int(len(df)),
        "features_before_pruning": len(feature_cols),
        "features_after_pruning":  len(pruned_cols),
        "pruned_out_features":     removed_cols,
        "n_estimators_final":      int(total_estimators),
        "train_size":              int(len(X_train)),
        "test_size":               int(len(X_test)),
        "train_batches":           train_history,
        "metrics_full":   {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in m_full.items()},
        "metrics_pruned": {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in m_pruned.items()},
        "top_features": feat_df.head(20)[["feature", "importance"]]
                               .assign(importance=lambda d: d["importance"].round(6))
                               .to_dict(orient="records"),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({
        "actual_price":     y_test,
        "pred_full":        y_pred_full,
        "pred_pruned":      y_pred_pruned,
        "error_pruned":     y_test - y_pred_pruned,
        "abs_error_pruned": np.abs(y_test - y_pred_pruned),
    })
    pred_csv = "eth_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    mlflow.log_artifact(out_json,  artifact_path="results")
    mlflow.log_artifact(pred_csv,  artifact_path="results")

    print(f"\n  JSON results saved : {out_json}")
    print(f"  Predictions saved  : {pred_df.shape[0]} rows → {pred_csv}")


def main():
    print("=" * 65)
    print("  ETHUSDT Price Prediction — v4 (MLflow)")
    print("=" * 65)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"eth-gbr-{time.strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n  MLflow Run: {run.info.run_id}")
        print(f"  Experiment: {MLFLOW_EXPERIMENT}")
        print(f"  Tracking URI: {MLFLOW_TRACKING_URI}\n")

        df = load_data(DATA_PATH, COL_NAMES)

        df, feature_cols, target_col, X, y = feature_engineering(
            df, WINDOWS, LAG_STEPS, DROP_COLS
        )

        X_model, X_q8, qt = quantize_data(X)

        X_train, X_test, y_train, y_test = train_test_split_temporal(X_model, y)

        model, total_estimators, train_history = train_model(
            X_train, y_train, BATCH_SIZE, N_ESTIMATORS_PER_BATCH, MAX_BATCHES
        )

        (model_pruned, pruned_cols, removed_cols,
         mask_keep, feat_df, X_train_p, X_test_p,
         threshold) = prune_and_retrain(
            model, X_train, X_test, y_train, feature_cols, total_estimators
        )

        m_full, m_pruned, y_pred_full, y_pred_pruned = evaluate_models(
            model, model_pruned, X_test, X_test_p, y_test, feat_df
        )

        save_artifacts(
            model_pruned, qt, feature_cols, pruned_cols,
            mask_keep, threshold, total_estimators, m_pruned,
            feat_df, MODEL_DIR,
        )

        save_results(
            df, feature_cols, pruned_cols, removed_cols,
            total_estimators, X_train, X_test, train_history,
            m_full, m_pruned, feat_df, y_test,
            y_pred_full, y_pred_pruned, OUT_JSON,
        )

        print("\n" + "=" * 65)
        print("  Pipeline completed!")
        print(f"  MLflow Run ID : {run.info.run_id}")
        print(f"  View results  : {MLFLOW_TRACKING_URI}/#/experiments")
        print("=" * 65)


if __name__ == "__main__":
    main()
