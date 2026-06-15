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
# Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  ETHUSDT Price Prediction — Gradient Boosting Regression")
print("=" * 65)

DATA_PATH = "ETHUSDT-trades-2026-06-13.csv"
COL_NAMES = [
    "trade_id", "price", "quantity", "quote_quantity",
    "timestamp", "is_buyer_maker", "is_best_match"
]

print("\n[1/7] Loading Data...")
t0 = time.time()
df = pd.read_csv(DATA_PATH, header=None, names=COL_NAMES)
print(f"      {len(df):,} rows loaded  ({time.time()-t0:.2f}s)")
print(f"      price range: {df['price'].min():.2f} – {df['price'].max():.2f} USDT")

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Feature Engineering...")
t0 = time.time()

# Sorting by time
df = df.sort_values("timestamp").reset_index(drop=True)

# --- Temporal features ---
# Convert timestamp to milliseconds (original is in microseconds)

df["ts_ms"] = df["timestamp"] / 1000.0
df["ts_sec"] = df["ts_ms"] / 1000.0

# Relative time from the start of the period
df["elapsed_sec"] = df["ts_sec"] - df["ts_sec"].iloc[0]

# Hour and minute from epoch time (UTC)
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

# --- Rolling features (different windows) ---
WINDOWS = [5, 10, 20, 50, 100]
for w in WINDOWS:
    df[f"price_roll_mean_{w}"]  = df["price"].rolling(w, min_periods=1).mean()
    df[f"price_roll_std_{w}"]   = df["price"].rolling(w, min_periods=1).std().fillna(0)
    df[f"price_roll_max_{w}"]   = df["price"].rolling(w, min_periods=1).max()
    df[f"price_roll_min_{w}"]   = df["price"].rolling(w, min_periods=1).min()
    df[f"qty_roll_mean_{w}"]    = df["quantity"].rolling(w, min_periods=1).mean()
    df[f"qty_roll_sum_{w}"]     = df["quantity"].rolling(w, min_periods=1).sum()
    df[f"buyer_roll_sum_{w}"]   = df["buyer_maker_int"].rolling(w, min_periods=1).sum()

# --- Lag features (previous prices) ---
LAG_STEPS = [1, 2, 3, 5, 10, 20]
for lag in LAG_STEPS:
    df[f"price_lag_{lag}"]    = df["price"].shift(lag)
    df[f"qty_lag_{lag}"]      = df["quantity"].shift(lag)
    df[f"return_lag_{lag}"]   = df["price"].pct_change(lag)  # relative return

# --- Derivative features ---
df["price_momentum_5"]   = df["price"] - df["price_roll_mean_5"]
df["price_momentum_20"]  = df["price"] - df["price_roll_mean_20"]
df["volatility_ratio"]   = df["price_roll_std_10"] / (df["price_roll_mean_10"] + 1e-9)
df["bollinger_upper_20"] = df["price_roll_mean_20"] + 2 * df["price_roll_std_20"]
df["bollinger_lower_20"] = df["price_roll_mean_20"] - 2 * df["price_roll_std_20"]
df["bollinger_band_pct"] = (df["price"] - df["bollinger_lower_20"]) / (
    df["bollinger_upper_20"] - df["bollinger_lower_20"] + 1e-9
)

# Simple VWAP (volume-weighted average price) for a window of 20
df["vwap_20"] = (
    (df["price"] * df["quantity"]).rolling(20, min_periods=1).sum() /
    df["quantity"].rolling(20, min_periods=1).sum()
)
df["price_vs_vwap"] = df["price"] - df["vwap_20"]

# --- Removing NaNs caused by lag/rolling ---
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Feature columns (excluding target and raw columns)
DROP_COLS = {"trade_id", "price", "timestamp", "ts_ms", "ts_sec",
             "is_buyer_maker", "is_best_match"}
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
TARGET_COL   = "price"

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)

print(f"      {len(FEATURE_COLS)} features created ({time.time()-t0:.2f}s)")
print(f"      {len(df):,} rows remained after dropping NaNs")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Quantization (INT8)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Quantizing data (float32 → int8 mapping)...")
t0 = time.time()

# QuantileTransformer: transforms the data into a uniform distribution [0,1]
qt = QuantileTransformer(n_quantiles=256, output_distribution="uniform",
                         random_state=42)
X_qt = qt.fit_transform(X)        # shape: [n, features] in the range [0,1]

# mapping to int8 (0-255 → store as int16 for computational safety)
X_q8 = (X_qt * 255).astype(np.int16)

mem_f32 = X.nbytes / 1e6
mem_q8  = X_q8.nbytes / 1e6
print(f"      float32: {mem_f32:.1f} MB  →  int8: {mem_q8:.1f} MB"
      f"  (Reduced by {100*(1-mem_q8/mem_f32):.0f}%) ({time.time()-t0:.2f}s)")

# We use X_qt for the model (float is better than int for GBR)
X_model = X_qt.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train / Test Split (80 / 20) — temporal (without shuffling)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Temporal train/test split (80% train / 20% test)...")

split_idx = int(len(X_model) * 0.80)
X_train, X_test = X_model[:split_idx], X_model[split_idx:]
y_train, y_test = y[:split_idx],       y[split_idx:]

print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Batching + Incremental training (warm_start)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Training the model with Batching (GradientBoostingRegressor)...")

BATCH_SIZE       = 20_000      # How many rows per batch
N_ESTIMATORS_PER_BATCH = 25    # Trees added per batch
MAX_BATCHES      = 10           # Maximum number of batches

model = GradientBoostingRegressor(
    n_estimators=N_ESTIMATORS_PER_BATCH,
    learning_rate=0.08,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    max_features="sqrt",
    warm_start=True,           # Allow adding new trees
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=5,
    tol=1e-4,
)

total_estimators = 0
train_history    = []

for batch_num in range(MAX_BATCHES):
    batch_start = (batch_num * BATCH_SIZE) % len(X_train)
    batch_end   = min(batch_start + BATCH_SIZE, len(X_train))
    
    Xb = X_train[batch_start:batch_end]
    yb = y_train[batch_start:batch_end]
    
    t_b = time.time()
    
    target_estimators = total_estimators + N_ESTIMATORS_PER_BATCH
    model.set_params(n_estimators=target_estimators)
    model.fit(Xb, yb)
    total_estimators = model.n_estimators_
    
    # Quick evaluation on batch
    y_pred_b = model.predict(Xb)
    mae_b    = mean_absolute_error(yb, y_pred_b)
    r2_b     = r2_score(yb, y_pred_b)
    
    train_history.append({
        "batch": batch_num + 1,
        "n_estimators": total_estimators,
        "batch_size": len(Xb),
        "mae": round(float(mae_b), 4),
        "r2":  round(float(r2_b),  4),
        "elapsed": round(time.time() - t_b, 2)
    })
    
    print(f"      Batch {batch_num+1}/{MAX_BATCHES} | "
          f"Trees: {total_estimators} | "
          f"MAE: {mae_b:.4f} | R²: {r2_b:.4f} | "
          f"{time.time()-t_b:.1f}s")

print(f"     Training finished — total trees: {total_estimators}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Pruning with Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Pruning features with Feature Importance...")
t0 = time.time()

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

# Threshold: Features that have importance greater than the mean
threshold     = importances.mean()
mask_keep     = importances >= threshold
PRUNED_COLS   = [FEATURE_COLS[i] for i in range(len(FEATURE_COLS)) if mask_keep[i]]
REMOVED_COLS  = [FEATURE_COLS[i] for i in range(len(FEATURE_COLS)) if not mask_keep[i]]

print(f"      Importance threshold: {threshold:.6f}")
print(f"      Kept features: {len(PRUNED_COLS)} from {len(FEATURE_COLS)}")
print(f"      Removed features: {len(REMOVED_COLS)}")

# Retraining with pruned features
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

# ─────────────────────────────────────────────────────────────────────────────
# 7. Final evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Final evaluation on the Test set...")

# Main model (all features)
y_pred_full   = model.predict(X_test)
# Pruned model
y_pred_pruned = model_pruned.predict(X_test_p)

def metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}

m_full   = metrics(y_test, y_pred_full,   "Full Model")
m_pruned = metrics(y_test, y_pred_pruned, "Pruned Model")

# ─────────────────────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Final evaluation results")
print("=" * 65)

for m in [m_full, m_pruned]:
    print(f"\n  [{m['label']}]")
    print(f"    MAE   (Mean Absolute Error)  : {m['MAE']:.4f} USDT")
    print(f"    RMSE  (Root Mean Square Error): {m['RMSE']:.4f} USDT")
    print(f"    R²    (Coefficient of Determination)        : {m['R2']:.6f}")
    print(f"    MAPE  Mean Absolute Percentage Error)    : {m['MAPE_%']:.4f}%")

# Comparison of predicted vs actual for the first 10 rows of the test set
print("\n  Sample predictions (first 10 rows of Test):")
print(f"  {'Actual':>12} {'Main Model':>12} {'Pruned Model':>12} {'Pruned Error':>12}")
print("  " + "-" * 54)
for i in range(min(10, len(y_test))):
    err = y_test[i] - y_pred_pruned[i]
    print(f"  {y_test[i]:>12.4f}  {y_pred_full[i]:>12.4f}  "
          f"{y_pred_pruned[i]:>12.4f}  {err:>+12.4f}")

# Top important features
print("\n  Top 15 important features (Main Model):")
print(f"  {'Feature':<30}  {'Importance':>10}")
print("  " + "-" * 44)
for _, row in feat_df.head(15).iterrows():
    bar = "█" * int(row["importance"] * 500)
    print(f"  {row['feature']:<30}  {row['importance']:>10.6f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "dataset_rows": int(len(df)),
    "features_before_pruning": len(FEATURE_COLS),
    "features_after_pruning":  len(PRUNED_COLS),
    "pruned_out_features":     REMOVED_COLS,
    "n_estimators_final":      int(total_estimators),
    "train_size":              int(len(X_train)),
    "test_size":               int(len(X_test)),
    "train_batches":           train_history,
    "metrics_full":   {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in m_full.items()},
    "metrics_pruned": {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in m_pruned.items()},
    "top_features": feat_df.head(20)[["feature","importance"]]
                           .assign(importance=lambda d: d["importance"].round(6))
                           .to_dict(orient="records"),
}

out_json = "eth_model_results.json"
os.makedirs("outputs", exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# Save predictions
# ─────────────────────────────────────────────────────────────────────────────
pred_df = pd.DataFrame({
    "actual_price":       y_test,
    "pred_full":          y_pred_full,
    "pred_pruned":        y_pred_pruned,
    "error_pruned":       y_test - y_pred_pruned,
    "abs_error_pruned":   np.abs(y_test - y_pred_pruned),
})
pred_df.to_csv("eth_predictions.csv", index=False)

print(f"\n  JSON results saved: {out_json}")
print(f"  Predictions saved: {pred_df.shape[0]} rows")
print("\n" + "=" * 65)
print("  Pipeline completed!")
print("=" * 65)
