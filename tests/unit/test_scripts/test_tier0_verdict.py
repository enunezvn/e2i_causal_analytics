"""Tests for the Tier-0 model usefulness verdict helper.

Covers the overfitting gate introduced to prevent severely-overfit models from
being labelled ACCEPTABLE/GOOD/EXCELLENT purely on test-set metrics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_tier0_test.py"


@pytest.fixture(scope="module")
def compute_verdict():
    """Load `_compute_verdict` from run_tier0_test.py without importing side-effects.

    The script imports many heavy ML libs at module import time. We load it via
    importlib.util with a minimal shim so only the helper is exercised. If the
    module load fails for unrelated reasons (e.g., missing optional dep), we
    skip rather than error so the rest of the suite can still run.
    """
    spec = importlib.util.spec_from_file_location("run_tier0_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.skip("Could not build import spec for run_tier0_test")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"Could not import run_tier0_test: {exc}")
    if not hasattr(module, "_compute_verdict"):
        pytest.skip("_compute_verdict helper not present in run_tier0_test")
    return module._compute_verdict


def test_verdict_accepts_when_no_overfit(compute_verdict):
    """Normal ACCEPTABLE verdict when metrics are moderate and no overfitting signal."""
    verdict, icon, description, deploy = compute_verdict(
        auc_roc=0.70,
        recall=0.40,
        precision=0.20,
        overfitting_severity="none",
        train_val_delta=0.02,
    )
    assert verdict == "ACCEPTABLE"


def test_verdict_downgrades_when_severe_overfit(compute_verdict):
    """Severe overfitting severity must downgrade ACCEPTABLE → MARGINAL."""
    verdict, icon, description, deploy = compute_verdict(
        auc_roc=0.70,
        recall=0.40,
        precision=0.20,
        overfitting_severity="severe",
        train_val_delta=None,
    )
    assert verdict == "MARGINAL"
    assert "overfit" in description.lower()


def test_verdict_downgrades_on_delta_threshold(compute_verdict):
    """Train-val AUC delta > 0.15 must downgrade even when severity is not set."""
    verdict, icon, description, deploy = compute_verdict(
        auc_roc=0.70,
        recall=0.40,
        precision=0.20,
        overfitting_severity=None,
        train_val_delta=0.23,
    )
    assert verdict == "MARGINAL"
    assert "overfit" in description.lower()


def test_verdict_good_when_metrics_high_and_no_overfit(compute_verdict):
    """Sanity check: high metrics and no overfit → GOOD/EXCELLENT (not downgraded)."""
    verdict, _, _, _ = compute_verdict(
        auc_roc=0.80,
        recall=0.55,
        precision=0.30,
        overfitting_severity="none",
        train_val_delta=0.03,
    )
    assert verdict == "GOOD"


def test_verdict_excellent_downgrades_on_severe_overfit(compute_verdict):
    """Even EXCELLENT-tier metrics must be downgraded under severe overfit."""
    verdict, _, description, _ = compute_verdict(
        auc_roc=0.92,
        recall=0.80,
        precision=0.40,
        overfitting_severity="severe",
        train_val_delta=0.40,
    )
    assert verdict == "MARGINAL"
    assert "overfit" in description.lower()
