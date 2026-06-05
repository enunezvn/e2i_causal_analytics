"""End-to-end RC1/RC2 recovery proof on a synthetic Appendix-A-style cohort:
dense demographics + cardinality-2 sparse clinical flags + all-constant +
too-sparse labs + one genuine post-index leak. After the fixes, detect_leakage
flags ONLY the genuine leak (not the sparse clinical flags), and the runner's
discovery helper retains the sparse flags."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_tier0_test import _discover_model_feature_cols
from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import detect_leakage


def _appendix_a_cohort(n: int = 1294, n_pos: int = 37, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=n_pos, replace=False)] = 1
    neg = np.where(y == 0)[0]
    cols = {"target": y}
    # 6 dense demographics (kept)
    cols["age_at_index"] = rng.integers(30, 85, n).astype(float)
    cols["payer_category"] = rng.integers(0, 4, n).astype(float)
    cols["plan_type"] = rng.integers(0, 3, n).astype(float)
    # 12 cardinality-2 sparse clinical flags (legit pre-index, ~3-5% density,
    # mostly/entirely zero in the tiny positive class) -> the RC1 false positives
    sparse_cols = []
    for i in range(12):
        f = np.zeros(n, dtype=float)
        f[rng.choice(neg, size=rng.integers(20, 60), replace=False)] = 1.0
        name = f"dx_flag_{i}"
        cols[name] = f
        sparse_cols.append(name)
    # all-constant + too-sparse (correctly dropped by discovery, not by leakage)
    cols["office_visits_fill_count"] = np.zeros(n, dtype=float)
    cols["lab_rare"] = np.where(np.arange(n) < int(0.02 * n), 1.0, np.nan)
    # one genuine post-index leak (== target)
    cols["initiated_biologic_180d"] = y.astype(float)
    return pd.DataFrame(cols), sparse_cols


@pytest.mark.asyncio
async def test_rare_event_cohort_recovers_sparse_and_drops_only_the_leak():
    df, sparse_cols = _appendix_a_cohort()
    feature_names = [c for c in df.columns if c != "target"]
    state = {
        "experiment_id": "exp_tier0_recovery",
        "train_df": df,
        "scope_spec": {
            "required_features": feature_names,
            "prediction_target": "target",
            "feature_manifest_source": None,
        },
        "skip_leakage_check": False,
    }
    result = await detect_leakage(state)
    leaked = set(result.get("leaked_features", []))

    # The genuine leak is still caught (logical_dependency / single_feature_auc).
    assert "initiated_biologic_180d" in leaked, f"genuine leak missed: {leaked}"
    # NONE of the cardinality-2 sparse clinical flags are flagged anymore.
    assert not (set(sparse_cols) & leaked), (
        f"RC1 regression: sparse clinical flags wrongly flagged: {set(sparse_cols) & leaked}"
    )

    # The runner's discovery retains every sparse flag and the demographics,
    # subtracting only the genuine leaks; constants/too-sparse drop out.
    cols = _discover_model_feature_cols(df, exclude={"target"} | leaked)
    for s in sparse_cols:
        assert s in cols, f"RC2 regression: sparse flag {s} not retained by discovery"
    assert "initiated_biologic_180d" not in cols
    assert "office_visits_fill_count" not in cols  # all-constant
    assert "lab_rare" not in cols  # too-sparse
    # Recovery direction: many more features retained than the demographics alone.
    assert len(cols) >= 3 + len(sparse_cols)
