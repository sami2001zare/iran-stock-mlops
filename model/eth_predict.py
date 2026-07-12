"""
ETHUSDT Price Predictor — Inference Script
==========================================
Usage:
    python eth_predict.py --input new_trades.csv
    python eth_predict.py --input new_trades.csv --output my_predictions.csv
    python eth_predict.py --input new_trades.csv --model-dir path/to/eth_model_artifacts

This script:
    1. Loads the saved model + transformer + metadata from eth_model_artifacts/
    2. Applies the exact same feature engineering as training
    3. Runs prediction on the new CSV data
    4. Prints results and saves them to a CSV file
"""

import argparse
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering  (must match eth_price_model.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, windows: list, lag_steps: list) -> pd.DataFrame:
    """Apply the same feature engineering pipeline used during training."""

    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Temporal features ---
    df["ts_ms"]      = df["timestamp"] / 1000.0
    df["ts_sec"]     = df["ts_ms"] / 1000.0
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
    df["trade_size_bucket"]  = pd.qcut(
        df["quantity"], q=10, labels=False, duplicates="drop"
    )

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
        (df["price"] * df["quantity"]).rolling(20, min_periods=1).sum()
        / df["quantity"].rolling(20, min_periods=1).sum()
    )
    df["price_vs_vwap"] = df["price"] - df["vwap_20"]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ETHUSDT price prediction using the pre-trained GBR model."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the new CSV file (same format as training data, no header).",
    )
    parser.add_argument(
        "--output", "-o",
        default="eth_new_predictions.csv",
        help="Path for the output predictions CSV (default: eth_new_predictions.csv).",
    )
    parser.add_argument(
        "--model-dir", "-m",
        default="eth_model_artifacts",
        help="Directory that contains model_pruned.pkl, quantile_transformer.pkl, "
             "and pipeline_meta.json (default: eth_model_artifacts).",
    )
    parser.add_argument(
        "--show-rows", "-n",
        type=int,
        default=10,
        help="Number of sample prediction rows to print (default: 10).",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  ETHUSDT Price Predictor — Inference")
    print("=" * 65)

    # ── 1. Load artifacts ────────────────────────────────────────────────────
    print(f"\n[1/4] Loading model artifacts from '{args.model_dir}'...")

    meta_path  = os.path.join(args.model_dir, "pipeline_meta.json")
    model_path = os.path.join(args.model_dir, "model_pruned.pkl")
    qt_path    = os.path.join(args.model_dir, "quantile_transformer.pkl")

    for p in [meta_path, model_path, qt_path]:
        if not os.path.exists(p):
            print(f"  [ERROR] File not found: {p}")
            print("  Make sure you run eth_price_model.py first to generate the artifacts.")
            sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = joblib.load(model_path)
    qt    = joblib.load(qt_path)

    feature_cols = meta["feature_cols"]   # all 73 features (before pruning)
    mask_keep    = np.array(meta["mask_keep"], dtype=bool)  # pruning mask
    windows      = meta["windows"]
    lag_steps    = meta["lag_steps"]

    print(f"      Model      : {model.__class__.__name__}")
    print(f"      Features   : {int(mask_keep.sum())} (pruned from {len(feature_cols)})")
    print(f"      Trained MAE: {meta['metrics_pruned']['MAE']:.4f} USDT")

    # ── 2. Load new data ─────────────────────────────────────────────────────
    print(f"\n[2/4] Loading new data from '{args.input}'...")
    t0 = time.time()

    COL_NAMES = [
        "trade_id", "price", "quantity", "quote_quantity",
        "timestamp", "is_buyer_maker", "is_best_match",
    ]

    # Accept files with or without a header row
    sample = pd.read_csv(args.input, nrows=1, header=None)
    has_header = str(sample.iloc[0, 0]).lower() == "trade_id"

    df_raw = pd.read_csv(
        args.input,
        header=0 if has_header else None,
        names=None if has_header else COL_NAMES,
    )
    if has_header:
        df_raw.columns = COL_NAMES

    print(f"      {len(df_raw):,} rows loaded  ({time.time()-t0:.2f}s)")
    print(f"      price range : {df_raw['price'].min():.2f} – {df_raw['price'].max():.2f} USDT")

    # ── 3. Feature engineering + transform ───────────────────────────────────
    print("\n[3/4] Building features and transforming...")
    t0 = time.time()

    df_feat = build_features(df_raw.copy(), windows, lag_steps)

    # Select only the columns used during training (in the same order)
    DROP_COLS = set(meta["drop_cols"])
    X_raw = df_feat[feature_cols].values.astype(np.float32)

    # Apply the saved QuantileTransformer
    X_qt = qt.transform(X_raw)

    # Apply the pruning mask
    X_pruned = X_qt[:, mask_keep]

    print(f"      {len(df_feat):,} rows after feature engineering  ({time.time()-t0:.2f}s)")

    # ── 4. Predict ───────────────────────────────────────────────────────────
    print("\n[4/4] Running predictions...")
    t0 = time.time()

    y_pred = model.predict(X_pruned).astype(np.float32)

    elapsed = time.time() - t0
    print(f"      {len(y_pred):,} predictions completed in {elapsed:.3f}s  "
          f"({len(y_pred)/elapsed:,.0f} rows/s)")

    # ── Display sample ───────────────────────────────────────────────────────
    has_actual = True
    try:
        actual_prices = df_feat["price"].values.astype(np.float32)
    except KeyError:
        has_actual = False

    print("\n" + "=" * 65)
    print("  Prediction Results")
    print("=" * 65)

    n_show = min(args.show_rows, len(y_pred))

    if has_actual:
        from sklearn.metrics import (
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
            r2_score,
        )
        mae  = mean_absolute_error(actual_prices, y_pred)
        rmse = np.sqrt(mean_squared_error(actual_prices, y_pred))
        r2   = r2_score(actual_prices, y_pred)
        mape = mean_absolute_percentage_error(actual_prices, y_pred) * 100

        print(f"\n  MAE  (Mean Absolute Error)       : {mae:.4f} USDT")
        print(f"  RMSE (Root Mean Squared Error)   : {rmse:.4f} USDT")
        print(f"  R²   (Coefficient of Determination): {r2:.6f}")
        print(f"  MAPE (Mean Abs Percentage Error) : {mape:.4f}%")

        print(f"\n  Sample predictions (first {n_show} rows):")
        print(f"  {'Row':>5}  {'Actual':>12}  {'Predicted':>12}  {'Error':>10}  {'Error%':>8}")
        print("  " + "-" * 55)
        for i in range(n_show):
            err  = actual_prices[i] - y_pred[i]
            errp = err / actual_prices[i] * 100
            print(f"  {i+1:>5}  {actual_prices[i]:>12.4f}  {y_pred[i]:>12.4f}  "
                  f"{err:>+10.4f}  {errp:>+7.3f}%")
    else:
        print(f"\n  Sample predictions (first {n_show} rows):")
        print(f"  {'Row':>5}  {'Predicted Price (USDT)':>24}")
        print("  " + "-" * 33)
        for i in range(n_show):
            print(f"  {i+1:>5}  {y_pred[i]:>24.4f}")

    # ── Save output ──────────────────────────────────────────────────────────
    out_df = df_feat[["trade_id", "timestamp", "price"]].copy() if "trade_id" in df_feat.columns else df_feat[["timestamp", "price"]].copy()
    out_df["predicted_price"] = y_pred

    if has_actual:
        out_df["error"]     = out_df["price"] - out_df["predicted_price"]
        out_df["abs_error"] = out_df["error"].abs()
        out_df["error_pct"] = (out_df["error"] / out_df["price"] * 100).round(4)

    out_df.to_csv(args.output, index=False)

    print(f"\n  Predictions saved to : {args.output}")
    print(f"  Total rows predicted  : {len(y_pred):,}")
    print("\n" + "=" * 65)
    print("  Done!")
    print("=" * 65)


if __name__ == "__main__":
    main()
