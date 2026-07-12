"""
Unit tests for Statistical Drift Detection (`src.ml.drift.DriftDetectionEngine`).
Verifies exact PSI boundary scores across identical, minor shift, and major shift distributions.
"""

import numpy as np
import pytest
from src.ml.drift import DriftDetectionEngine


def test_psi_identical_distribution():
    """Identical distributions must yield near-zero PSI (< 0.01)."""
    rng = np.random.default_rng(42)
    data = rng.normal(3500, 10, 500)
    psi_val = DriftDetectionEngine.calculate_psi(data, data.copy())
    assert psi_val < 0.01, f"Identical sample PSI {psi_val} must be < 0.01"


def test_psi_moderate_shift():
    """Slight shift well within normal threshold (< 0.20)."""
    rng = np.random.default_rng(43)
    ref = rng.normal(3500, 10, 500)
    cur = rng.normal(3502, 10, 500)
    psi_val = DriftDetectionEngine.calculate_psi(ref, cur)
    assert psi_val < 0.20, f"Moderate shift PSI {psi_val} should be < 0.20"


def test_psi_major_drift():
    """Severe distribution shift (> 0.20 PSI) triggers alarm."""
    rng = np.random.default_rng(44)
    ref = rng.normal(3500, 10, 500)
    cur = rng.normal(3600, 25, 500)
    psi_val = DriftDetectionEngine.calculate_psi(ref, cur)
    assert psi_val >= 0.20, f"Major shift PSI {psi_val} should be >= 0.20"


def test_ks_test():
    """Check Kolmogorov-Smirnov statistical test output."""
    rng = np.random.default_rng(45)
    ref = rng.normal(3500, 10, 300)
    cur = rng.normal(3500, 10, 300)
    ks_stat, ks_pval = DriftDetectionEngine.calculate_ks_test(ref, cur)
    assert ks_pval > 0.01, f"KS p-value {ks_pval} should be > 0.01 for identical distributions"
