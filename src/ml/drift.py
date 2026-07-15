from __future__ import annotations
import logging
from typing import Any
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetectionEngine:

    @classmethod
    def calculate_psi(
        cls,
        reference: np.ndarray | list[float],
        current: np.ndarray | list[float],
        buckets: int = 10,
    ) -> float:
        ref_arr = np.asarray(reference, dtype=float)
        cur_arr = np.asarray(current, dtype=float)

        if len(ref_arr) == 0 or len(cur_arr) == 0:
            return 0.0

        breakpoints = np.percentile(ref_arr, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts, _ = np.histogram(ref_arr, bins=breakpoints)
        cur_counts, _ = np.histogram(cur_arr, bins=breakpoints)

        ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(ref_arr))
        cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(cur_arr))

        psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return abs(psi_value)

    @classmethod
    def calculate_ks_test(
        cls,
        reference: np.ndarray | list[float],
        current: np.ndarray | list[float],
    ) -> tuple[float, float]:
        ref_arr = np.asarray(reference, dtype=float)
        cur_arr = np.asarray(current, dtype=float)
        if len(ref_arr) < 2 or len(cur_arr) < 2:
            return 0.0, 1.0

        res = stats.ks_2samp(ref_arr, cur_arr)
        return float(res.statistic), float(res.pvalue)

    @classmethod
    def evaluate_feature_matrix_drift(
        cls,
        ref_matrix: dict[str, list[float]],
        cur_matrix: dict[str, list[float]],
        psi_threshold: float = 0.20,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "is_drift_detected": False,
            "max_psi": 0.0,
            "drifted_features": [],
            "feature_scores": {},
        }

        for feat_name, ref_vals in ref_matrix.items():
            if feat_name not in cur_matrix or not cur_matrix[feat_name]:
                continue

            psi_score = cls.calculate_psi(ref_vals, cur_matrix[feat_name])
            ks_stat, ks_pval = cls.calculate_ks_test(ref_vals, cur_matrix[feat_name])

            report["feature_scores"][feat_name] = {
                "psi": round(psi_score, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pval": round(ks_pval, 4),
            }

            if psi_score > report["max_psi"]:
                report["max_psi"] = round(psi_score, 4)

            if psi_score >= psi_threshold or ks_pval < 0.01:
                report["drifted_features"].append(feat_name)

        if len(report["drifted_features"]) > 0 and report["max_psi"] >= psi_threshold:
            report["is_drift_detected"] = True
            logger.warning("Data Drift Detected! Max PSI: %.4f on features: %s", report["max_psi"], report["drifted_features"])
        else:
            logger.info("Data distribution stable. Max PSI: %.4f (Threshold: %.2f)", report["max_psi"], psi_threshold)

        return report
