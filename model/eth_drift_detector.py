"""
ETHUSDT Drift Detector
======================
This file works independently of eth_price_model-v3.py and detects drift (deviation) in new data relative to the training data.

Types of drift checked:
    1. Data Drift      — Change in distribution of input features (KS-Test + PSI)
    2. Target Drift    — Change in distribution of price
    3. Prediction Drift— Change in distribution of model predictions
    4. Performance Drift — Model performance drop (MAE / R²)

Usage:
    python eth_drift_detector.py --reference train_data.csv --current new_data.csv
    python eth_drift_detector.py --reference train_data.csv --current new_data.csv --model-dir eth_model_artifacts
    python eth_drift_detector.py --reference train_data.csv --current new_data.csv --output drift_report.json
"""

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
COL_NAMES = [
    "trade_id", "price", "quantity", "quote_quantity",
    "timestamp", "is_buyer_maker", "is_best_match"
]
WINDOWS   = [5, 10, 20, 50, 100]
LAG_STEPS = [1, 2, 3, 5, 10, 20]
DROP_COLS = {
    "trade_id", "price", "timestamp", "ts_ms", "ts_sec",
    "is_buyer_maker", "is_best_match"
}

# Drift detection thresholds
KS_P_VALUE_THRESHOLD  = 0.05   # If p-value < 0.05 → drift is detected
PSI_WARNING_THRESHOLD = 0.10   # PSI > 0.10 → Warning
PSI_ALERT_THRESHOLD   = 0.20   # PSI > 0.20 → Serious drift
MAE_DEGRADATION_PCT   = 20.0   # If new MAE > MAE_train * (1 + 20%) → performance drift
R2_DEGRADATION_ABS    = 0.05   # If new R² < R²_train - 0.05 → performance drift


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureDriftResult:
    feature: str
    ks_statistic: float
    ks_p_value: float
    psi: float
    drift_detected: bool
    severity: str          # "none" | "warning" | "alert"
    ref_mean: float
    cur_mean: float
    ref_std: float
    cur_std: float
    mean_shift_pct: float  # Percentage change in mean


@dataclass
class DriftReport:
    timestamp: str
    reference_rows: int
    current_rows: int

    # Data Drift
    data_drift_detected: bool = False
    drifted_features_count: int = 0
    total_features_checked: int = 0
    data_drift_ratio: float = 0.0
    feature_results: list = field(default_factory=list)

    # Target Drift
    target_drift_detected: bool = False
    target_ks_statistic: float = 0.0
    target_ks_p_value: float = 0.0
    target_psi: float = 0.0
    target_ref_mean: float = 0.0
    target_cur_mean: float = 0.0
    target_mean_shift_pct: float = 0.0

    # Prediction Drift
    prediction_drift_detected: bool = False
    pred_ks_statistic: float = 0.0
    pred_ks_p_value: float = 0.0
    pred_psi: float = 0.0

    # Performance Drift
    performance_drift_detected: bool = False
    ref_mae: float = 0.0
    cur_mae: float = 0.0
    mae_change_pct: float = 0.0
    ref_r2: float = 0.0
    cur_r2: float = 0.0
    r2_change_abs: float = 0.0

    # Overall
    overall_drift_detected: bool = False
    overall_severity: str = "none"   # "none" | "warning" | "alert" | "critical"
    summary: str = ""



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Engineering  (Exactly like eth_price_model-v3.py)
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame, windows: list,
                         lag_steps: list, drop_cols: set) -> tuple:
    """Same feature engineering as training pipeline."""
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["ts_ms"]      = df["timestamp"] / 1000.0
    df["ts_sec"]     = df["ts_ms"] / 1000.0
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
    df["vwap_20"] = (
        (df["price"] * df["quantity"]).rolling(20, min_periods=1).sum() /
        df["quantity"].rolling(20, min_periods=1).sum()
    )
    df["price_vs_vwap"] = df["price"] - df["vwap_20"]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["price"].values.astype(np.float32)
    return df, feature_cols, X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Preparation with Saved Quantile Transformer
# ─────────────────────────────────────────────────────────────────────────────
def prepare_for_model(X: np.ndarray, feature_cols: list,
                       meta: dict, qt) -> np.ndarray:
    """
    Align the column order with training data,
    apply the QuantileTransformer, and apply the pruning mask.
    """
    train_feature_cols = meta["feature_cols"]
    mask_keep          = np.array(meta["mask_keep"], dtype=bool)

    # Creating a DataFrame with the correct columns
    df_x = pd.DataFrame(X, columns=feature_cols)

    # Fill columns that were not in training with 0
    for col in train_feature_cols:
        if col not in df_x.columns:
            df_x[col] = 0.0

    X_ordered = df_x[train_feature_cols].values.astype(np.float32)
    X_qt      = qt.transform(X_ordered)
    X_pruned  = X_qt[:, mask_keep]
    return X_pruned


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute PSI (Population Stability Index)
# ─────────────────────────────────────────────────────────────────────────────
def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> float:
    """
    PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
    PSI < 0.10 → stable
    PSI 0.10–0.20 → warning
    PSI > 0.20 → serious drift, Your model likely needs to be retrained.
    """
    # Determining bin boundaries based on reference
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)     # Removing duplicates

    if len(breakpoints) < 3:
        return 0.0   # There is not enough data for the bins

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current,   bins=breakpoints)

    # Converting frequencies to percentages (normalization)
    eps = 1e-8
    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)

    # Zero Handling
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Checking Data Drift for each feature
# ─────────────────────────────────────────────────────────────────────────────
def detect_data_drift(X_ref: np.ndarray, X_cur: np.ndarray,
                       feature_cols: list,
                       ks_threshold: float = KS_P_VALUE_THRESHOLD,
                       psi_warning: float  = PSI_WARNING_THRESHOLD,
                       psi_alert: float    = PSI_ALERT_THRESHOLD) -> list:
    """
    feature_cols: a list of feature names (columns).

    Checks for data drift for each feature using KS-Test and PSI.
    Returns: list of FeatureDriftResult
    """
    results = []
    n_features = X_ref.shape[1]

    for i, feat_name in enumerate(feature_cols):
        ref_col = X_ref[:, i]
        cur_col = X_cur[:, i]

        # KS-Test
        ks_stat, ks_p = stats.ks_2samp(ref_col, cur_col)

        # PSI
        psi = compute_psi(ref_col, cur_col)

        # Determining severity
        drift_by_ks  = ks_p < ks_threshold
        drift_by_psi = psi  > psi_warning

        if psi > psi_alert:
            severity = "alert"
        elif psi > psi_warning or drift_by_ks:
            severity = "warning"
        else:
            severity = "none"

        drift_detected = drift_by_ks or drift_by_psi

        # Descriptive statistics
        ref_mean = float(np.mean(ref_col))
        cur_mean = float(np.mean(cur_col))
        ref_std  = float(np.std(ref_col))
        cur_std  = float(np.std(cur_col))
        # The formula indicates by what percentage the average of the new data has increased or decreased compared to the past.
        mean_shift_pct = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100

        results.append(FeatureDriftResult(
            feature        = feat_name,
            ks_statistic   = round(ks_stat, 6),
            ks_p_value     = round(ks_p, 6),
            psi            = round(psi, 6),
            drift_detected = drift_detected,
            severity       = severity,
            ref_mean       = round(ref_mean, 6),
            cur_mean       = round(cur_mean, 6),
            ref_std        = round(ref_std, 6),
            cur_std        = round(cur_std, 6),
            mean_shift_pct = round(mean_shift_pct, 4),
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Checking Target Drift (Price Distribution)
# ─────────────────────────────────────────────────────────────────────────────
def detect_target_drift(y_ref: np.ndarray, y_cur: np.ndarray) -> dict:
    """KS-Test and PSI on the actual price distribution."""
    ks_stat, ks_p = stats.ks_2samp(y_ref, y_cur)
    psi           = compute_psi(y_ref, y_cur)

    # The examination of the average shift.
    ref_mean = float(np.mean(y_ref))
    cur_mean = float(np.mean(y_cur))
    mean_shift_pct = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100

    drift_detected = (ks_p < KS_P_VALUE_THRESHOLD) or (psi > PSI_WARNING_THRESHOLD)

    return {
        "drift_detected":    drift_detected,
        "ks_statistic":      round(ks_stat, 6),
        "ks_p_value":        round(ks_p, 6),
        "psi":               round(psi, 6),
        "ref_mean":          round(ref_mean, 4),
        "cur_mean":          round(cur_mean, 4),
        "mean_shift_pct":    round(mean_shift_pct, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Checking Prediction Drift (Model Output Distribution)
# ─────────────────────────────────────────────────────────────────────────────
def detect_prediction_drift(y_pred_ref: np.ndarray,
                             y_pred_cur: np.ndarray) -> dict:
    """KS-Test and PSI on the model predictions."""
    ks_stat, ks_p = stats.ks_2samp(y_pred_ref, y_pred_cur)
    psi           = compute_psi(y_pred_ref, y_pred_cur)

    drift_detected = (ks_p < KS_P_VALUE_THRESHOLD) or (psi > PSI_WARNING_THRESHOLD)

    return {
        "drift_detected": drift_detected,
        "ks_statistic":   round(ks_stat, 6),
        "ks_p_value":     round(ks_p, 6),
        "psi":            round(psi, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Checking Performance Drift (Model Performance Degradation)
# ─────────────────────────────────────────────────────────────────────────────
def detect_performance_drift(y_ref: np.ndarray,   y_pred_ref: np.ndarray,
                              y_cur: np.ndarray,   y_pred_cur: np.ndarray,
                              ref_mae_threshold: float,
                              ref_r2_threshold: float) -> dict:
    """
    Compare MAE and R² on the reference (training) and current data.
    If the performance on the current data degrades significantly → performance drift.
    """
    ref_mae = float(mean_absolute_error(y_ref, y_pred_ref))
    cur_mae = float(mean_absolute_error(y_cur, y_pred_cur))
    ref_r2  = float(r2_score(y_ref, y_pred_ref))
    cur_r2  = float(r2_score(y_cur, y_pred_cur))

    # Calculation of the rate of performance change.
    mae_change_pct = (cur_mae - ref_mae) / (ref_mae + 1e-9) * 100
    r2_change_abs  = cur_r2 - ref_r2

    # Performance drift detection logic (flagging).
    mae_drift = mae_change_pct > MAE_DEGRADATION_PCT
    r2_drift  = r2_change_abs  < -R2_DEGRADATION_ABS

    drift_detected = mae_drift or r2_drift

    return {
        "drift_detected": drift_detected,
        "ref_mae":         round(ref_mae, 6),
        "cur_mae":         round(cur_mae, 6),
        "mae_change_pct":  round(mae_change_pct, 4),
        "mae_drift":       mae_drift,
        "ref_r2":          round(ref_r2, 6),
        "cur_r2":          round(cur_r2, 6),
        "r2_change_abs":   round(r2_change_abs, 6),
        "r2_drift":        r2_drift,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Determination of overall severity.
# ─────────────────────────────────────────────────────────────────────────────
def compute_overall_severity(report: DriftReport) -> str:
    """
    Critical → More than 50% of features have drift, or there is performance drift.
    Alert → More than 25% of features have drift, or there is target/prediction drift.
    Warning → A small number of features have drift.
    None → No drift detected.
    """
    if report.performance_drift_detected:
        return "critical"
    if report.data_drift_ratio > 0.50:
        return "critical"
    if report.data_drift_ratio > 0.25 or report.target_drift_detected:
        return "alert"
    if report.data_drift_detected or report.prediction_drift_detected:
        return "warning"
    return "none"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Display summary of results in the terminal.
# ─────────────────────────────────────────────────────────────────────────────
def print_report(report: DriftReport) -> None:
    sev_icon = {
        "none":     "✅",
        "warning":  "⚠️ ",
        "alert":    "🔶",
        "critical": "🚨",
    }

    print("\n" + "=" * 65)
    print("  DRIFT DETECTION REPORT")
    print("=" * 65)
    print(f"  Timestamp   : {report.timestamp}")
    print(f"  Reference   : {report.reference_rows:,} rows")
    print(f"  Current     : {report.current_rows:,} rows")

    # ── Data Drift ────────────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  [1] Data Drift (Feature Distribution)")
    print(f"  {'─'*60}")
    icon = sev_icon["alert" if report.data_drift_detected else "none"]
    print(f"  Result : {icon} {'DETECTED' if report.data_drift_detected else 'NOT DETECTED'}")
    print(f"  Drifted features : {report.drifted_features_count} / "
          f"{report.total_features_checked}  "
          f"({report.data_drift_ratio*100:.1f}%)")

    # Top drifted features
    drifted = [r for r in report.feature_results if r["drift_detected"]]
    drifted_sorted = sorted(drifted, key=lambda x: x["psi"], reverse=True)
    if drifted_sorted:
        print(f"\n  Top drifted features (by PSI):")
        print(f"  {'Feature':<30} {'PSI':>8} {'KS p-val':>10} {'Mean Shift':>12} {'Severity':>10}")
        print("  " + "─" * 74)
        for r in drifted_sorted[:10]:
            print(f"  {r['feature']:<30} {r['psi']:>8.4f} {r['ks_p_value']:>10.4f} "
                  f"{r['mean_shift_pct']:>10.2f}%  {r['severity']:>10}")

    # ── Target Drift ──────────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  [2] Target Drift (Price Distribution)")
    print(f"  {'─'*60}")
    icon = sev_icon["alert" if report.target_drift_detected else "none"]
    print(f"  Result         : {icon} {'DETECTED' if report.target_drift_detected else 'NOT DETECTED'}")
    print(f"  Ref mean price : {report.target_ref_mean:.4f} USDT")
    print(f"  Cur mean price : {report.target_cur_mean:.4f} USDT")
    print(f"  Mean shift     : {report.target_mean_shift_pct:.4f}%")
    print(f"  KS statistic   : {report.target_ks_statistic:.6f}  "
          f"(p={report.target_ks_p_value:.6f})")
    print(f"  PSI            : {report.target_psi:.6f}")

    # ── Prediction Drift ──────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  [3] Prediction Drift (Model Output Distribution)")
    print(f"  {'─'*60}")
    icon = sev_icon["alert" if report.prediction_drift_detected else "none"]
    print(f"  Result       : {icon} {'DETECTED' if report.prediction_drift_detected else 'NOT DETECTED'}")
    print(f"  KS statistic : {report.pred_ks_statistic:.6f}  "
          f"(p={report.pred_ks_p_value:.6f})")
    print(f"  PSI          : {report.pred_psi:.6f}")

    # ── Performance Drift ─────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  [4] Performance Drift (Model Accuracy)")
    print(f"  {'─'*60}")
    icon = sev_icon["critical" if report.performance_drift_detected else "none"]
    print(f"  Result      : {icon} {'DETECTED' if report.performance_drift_detected else 'NOT DETECTED'}")
    print(f"  MAE ref     : {report.ref_mae:.6f} USDT")
    print(f"  MAE current : {report.cur_mae:.6f} USDT  "
          f"(change: {report.mae_change_pct:+.2f}%)")
    print(f"  R²  ref     : {report.ref_r2:.6f}")
    print(f"  R²  current : {report.cur_r2:.6f}  "
          f"(change: {report.r2_change_abs:+.6f})")

    # ── Overall ───────────────────────────────────────────────────────────────
    print(f"\n  {'═'*60}")
    overall_icon = sev_icon.get(report.overall_severity, "❓")
    print(f"  OVERALL SEVERITY : {overall_icon}  {report.overall_severity.upper()}")
    print(f"  {report.summary}")
    print(f"  {'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Save the report.
# ─────────────────────────────────────────────────────────────────────────────
def save_report(report: DriftReport, output_path: str) -> None:
    report_dict = asdict(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"  Report saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ETHUSDT Drift Detector — detects data, target, prediction, and performance drift."
    )
    parser.add_argument(
        "--reference", "-r", required=True,
        help="CSV file used as reference (e.g., training data, no header)."
    )
    parser.add_argument(
        "--current", "-c", required=True,
        help="CSV file with new/current data to check for drift (no header)."
    )
    parser.add_argument(
        "--model-dir", "-m", default="eth_model_artifacts",
        help="Directory with model_pruned.pkl, quantile_transformer.pkl, pipeline_meta.json."
    )
    parser.add_argument(
        "--output", "-o", default="eth_drift_report.json",
        help="Path to save the drift report JSON (default: eth_drift_report.json)."
    )
    parser.add_argument(
        "--top-features", "-n", type=int, default=10,
        help="Number of most important features to check for drift (default: 10, 0 = all)."
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  ETHUSDT Drift Detector")
    print("=" * 65)

    # ── Upload model artifacts ────────────────────────────────────────────────
    print(f"\n[1/6] Loading model artifacts from '{args.model_dir}'...")
    for fname in ["model_pruned.pkl", "quantile_transformer.pkl", "pipeline_meta.json"]:
        path = os.path.join(args.model_dir, fname)
        if not os.path.exists(path):
            print(f"  [ERROR] Not found: {path}")
            print("  Run eth_price_model-v3.py first to generate artifacts.")
            sys.exit(1)

    model = joblib.load(os.path.join(args.model_dir, "model_pruned.pkl"))
    qt    = joblib.load(os.path.join(args.model_dir, "quantile_transformer.pkl"))
    with open(os.path.join(args.model_dir, "pipeline_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    ref_mae_threshold = meta["metrics_pruned"]["MAE"]
    ref_r2_threshold  = meta["metrics_pruned"]["R2"]
    print(f"  Model loaded — Training MAE: {ref_mae_threshold:.4f} | R²: {ref_r2_threshold:.4f}")

    # ── Upload reference data ───────────────────────────────────────────
    print(f"\n[2/6] Loading reference data from '{args.reference}'...")
    t0 = time.time()
    df_ref_raw = pd.read_csv(args.reference, header=None, names=COL_NAMES)
    print(f"  {len(df_ref_raw):,} rows  ({time.time()-t0:.2f}s)")

    # ── Upload current data ─────────────────────────────────────────────
    print(f"\n[3/6] Loading current data from '{args.current}'...")
    t0 = time.time()
    df_cur_raw = pd.read_csv(args.current, header=None, names=COL_NAMES)
    print(f"  {len(df_cur_raw):,} rows  ({time.time()-t0:.2f}s)")

    # ── Feature engineering for both datasets ─────────────────────────────────────────────
    print("\n[4/6] Feature Engineering on both datasets...")
    t0 = time.time()

    df_ref, feat_cols_ref, X_ref_raw, y_ref = feature_engineering(
        df_ref_raw.copy(), WINDOWS, LAG_STEPS, DROP_COLS
    )
    df_cur, feat_cols_cur, X_cur_raw, y_cur = feature_engineering(
        df_cur_raw.copy(), WINDOWS, LAG_STEPS, DROP_COLS
    )
    print(f"  Reference: {len(df_ref):,} rows | Current: {len(df_cur):,} rows  "
          f"({time.time()-t0:.2f}s)")

    # ── Preparation for the model (column order, quantization, pruning)
    X_ref_model = prepare_for_model(X_ref_raw, feat_cols_ref, meta, qt)
    X_cur_model = prepare_for_model(X_cur_raw, feat_cols_cur, meta, qt)

    # ── Predictions
    y_pred_ref = model.predict(X_ref_model).astype(np.float32)
    y_pred_cur = model.predict(X_cur_model).astype(np.float32)

    # ── Running Drift Detection ─────────────────────────────────────────────────
    print("\n[5/6] Running drift detection...")

    # Selection of features for data drift investigation
    pruned_cols = meta["pruned_cols"]
    top_n       = args.top_features
    check_cols  = pruned_cols if top_n == 0 else pruned_cols[:top_n]

    # Index of columns in X_ref_model / X_cur_model.
    check_indices = list(range(len(check_cols)))

    X_ref_check = X_ref_model[:, check_indices]
    X_cur_check = X_cur_model[:, check_indices]

    # 1. Data Drift
    feature_results = detect_data_drift(
        X_ref_check, X_cur_check, check_cols
    )
    drifted = [r for r in feature_results if r.drift_detected]

    # 2. Target Drift
    target_result = detect_target_drift(y_ref, y_cur)

    # 3. Prediction Drift
    pred_result = detect_prediction_drift(y_pred_ref, y_pred_cur)

    # 4. Performance Drift
    perf_result = detect_performance_drift(
        y_ref, y_pred_ref, y_cur, y_pred_cur,
        ref_mae_threshold, ref_r2_threshold
    )

    # ── Generate overall report ──────────────────────────────────────────────────────
    print("\n[6/6] Compiling drift report...")
    from datetime import datetime, timezone

    report = DriftReport(
        timestamp       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        reference_rows  = len(df_ref),
        current_rows    = len(df_cur),

        # Data Drift
        data_drift_detected     = len(drifted) > 0,
        drifted_features_count  = len(drifted),
        total_features_checked  = len(feature_results),
        data_drift_ratio        = len(drifted) / max(len(feature_results), 1),
        feature_results         = [asdict(r) for r in feature_results],

        # Target Drift
        target_drift_detected   = target_result["drift_detected"],
        target_ks_statistic     = target_result["ks_statistic"],
        target_ks_p_value       = target_result["ks_p_value"],
        target_psi              = target_result["psi"],
        target_ref_mean         = target_result["ref_mean"],
        target_cur_mean         = target_result["cur_mean"],
        target_mean_shift_pct   = target_result["mean_shift_pct"],

        # Prediction Drift
        prediction_drift_detected = pred_result["drift_detected"],
        pred_ks_statistic         = pred_result["ks_statistic"],
        pred_ks_p_value           = pred_result["ks_p_value"],
        pred_psi                  = pred_result["psi"],

        # Performance Drift
        performance_drift_detected = perf_result["drift_detected"],
        ref_mae                    = perf_result["ref_mae"],
        cur_mae                    = perf_result["cur_mae"],
        mae_change_pct             = perf_result["mae_change_pct"],
        ref_r2                     = perf_result["ref_r2"],
        cur_r2                     = perf_result["cur_r2"],
        r2_change_abs              = perf_result["r2_change_abs"],
    )

    # Overall severity and summary
    report.overall_severity     = compute_overall_severity(report)
    report.overall_drift_detected = report.overall_severity != "none"

    severity_msgs = {
        "none":     "No significant drift detected. Model is stable.",
        "warning":  "Minor drift detected. Monitor the model closely.",
        "alert":    "Significant drift detected. Consider retraining soon.",
        "critical": "Critical drift detected! Immediate retraining recommended.",
    }
    report.summary = severity_msgs.get(report.overall_severity, "")

    # Display and save.
    print_report(report)
    save_report(report, args.output)


if __name__ == "__main__":
    main()
