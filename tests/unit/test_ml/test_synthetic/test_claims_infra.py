"""P3 — infra: fail-fast stats deps + split-validator leakage checks (Shard 10).

(a) ``validation.py`` must import scipy + statsmodels with NO silent fallback —
    a missing stats dependency is a hard ImportError, not a degraded numpy path.
(b) The claims split-leakage check must RAISE on a temporal-leakage fixture
    (a feature event dated on/after the index) and on a covariate-imbalance
    fixture, rather than passing silently.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.claims.validation import (
    TemporalLeakageError,
    assert_no_temporal_leakage,
    assert_split_covariate_balance,
)


def test_validation_module_imports_scipy_and_statsmodels_no_fallback():
    mod = importlib.import_module("src.ml.synthetic.claims.validation")
    src = importlib.util.find_spec("src.ml.synthetic.claims.validation").origin
    text = open(src).read()
    # Hard imports at module top — no try/except ImportError silent fallback.
    assert "import scipy" in text
    assert "import statsmodels" in text
    assert "except ImportError" not in text
    # Both modules are actually importable in this env (deps installed).
    assert mod is not None


def test_temporal_leakage_raises_on_feature_dated_at_or_after_index():
    # A claim event dated ON the index date is post-index leakage.
    idx = pd.Timestamp("2025-06-01")
    feats = pd.DataFrame(
        {
            "patid": [1, 1, 2],
            "event_date": [idx - pd.Timedelta(days=10), idx, idx - pd.Timedelta(days=5)],
        }
    )
    index_by_patid = {1: idx, 2: idx}
    with pytest.raises(TemporalLeakageError):
        assert_no_temporal_leakage(feats, index_by_patid, date_col="event_date")


def test_temporal_leakage_passes_when_all_features_strictly_pre_index():
    idx = pd.Timestamp("2025-06-01")
    feats = pd.DataFrame(
        {
            "patid": [1, 2],
            "event_date": [idx - pd.Timedelta(days=10), idx - pd.Timedelta(days=1)],
        }
    )
    # Must NOT raise (returns None on success).
    assert assert_no_temporal_leakage(feats, {1: idx, 2: idx}, date_col="event_date") is None


def test_covariate_imbalance_raises_on_skewed_split():
    rng = np.random.default_rng(0)
    n = 400
    # train has high severity, test has low severity -> standardized mean diff huge.
    df = pd.DataFrame(
        {
            "severity": np.concatenate([rng.normal(2.0, 0.5, n), rng.normal(-2.0, 0.5, n)]),
            "data_split": ["train"] * n + ["test"] * n,
        }
    )
    with pytest.raises(ValueError):
        assert_split_covariate_balance(df, covariates=["severity"], max_smd=0.25)


def test_covariate_balance_passes_on_random_split():
    rng = np.random.default_rng(1)
    n = 400
    sev = rng.normal(0, 1, 2 * n)
    split = rng.permutation(["train"] * n + ["test"] * n)
    df = pd.DataFrame({"severity": sev, "data_split": split})
    # A random split is balanced -> must NOT raise (returns None on success).
    assert assert_split_covariate_balance(df, covariates=["severity"], max_smd=0.25) is None
