"""
Unit tests for PSI-based drift detection logic.
Mirrors the _psi() function in api/main.py so it can run without FastAPI.
"""

import numpy as np
import pytest


def psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] = np.inf
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current,   bins=breakpoints)[0]
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


PSI_THRESHOLD = 0.2


def test_psi_identical_distributions():
    rng = np.random.default_rng(0)
    data = rng.normal(3500, 10, 500)
    score = psi(data, data.copy())
    assert score < PSI_THRESHOLD, "Identical distributions must not trigger drift"


def test_psi_no_drift_similar():
    rng = np.random.default_rng(1)
    ref = rng.normal(3500, 10, 500)
    cur = rng.normal(3502, 10, 500)   # tiny shift — well within threshold
    assert psi(ref, cur) < PSI_THRESHOLD


def test_psi_drift_large_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(3500, 5,  500)
    cur = rng.normal(4000, 50, 500)   # large mean + variance shift
    assert psi(ref, cur) > PSI_THRESHOLD, "Large distribution shift must be detected"


def test_psi_drift_heavy_tail():
    rng = np.random.default_rng(3)
    ref = rng.normal(3500, 10, 500)
    cur = np.concatenate([rng.normal(3500, 10, 400), rng.normal(5000, 50, 100)])
    assert psi(ref, cur) > PSI_THRESHOLD, "Heavy-tail outlier injection must be detected"


def test_psi_is_non_negative():
    rng = np.random.default_rng(4)
    ref = rng.normal(3500, 15, 300)
    cur = rng.normal(3600, 20, 300)
    assert psi(ref, cur) >= 0


@pytest.mark.parametrize("buckets", [5, 10, 20])
def test_psi_bucket_sensitivity(buckets):
    rng = np.random.default_rng(5)
    ref = rng.normal(3500, 10, 500)
    cur = rng.normal(4000, 50, 500)
    # Drift should be detectable regardless of bucket count
    assert psi(ref, cur, buckets=buckets) > PSI_THRESHOLD
