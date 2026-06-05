"""RC2: the runner must train on the retained feature set (well-formed columns
minus genuine leaks), via shared discovery helpers — not on the curated
recommended_feature_set survivor list."""

import pandas as pd

from scripts.run_tier0_test import _discover_model_feature_cols, _runner_exclude_set


def _cohort(n: int = 1000):
    return pd.DataFrame(
        {
            "has_asthma": [1.0] * 40 + [0.0] * (n - 40),  # card-2 sparse legit -> KEEP
            "age_at_index": list(range(n)),  # dense numeric -> KEEP
            "payer_category": (["A", "B", "C"] * (n // 3 + 1))[:n],  # low-card categorical -> KEEP
            "genuine_leak": [float(i % 2) for i in range(n)],  # passed in `exclude` -> DROP
            "all_constant": [1.0] * n,  # nunique==1 -> DROP
            "too_sparse": [1.0] * 100 + [None] * (n - 100),  # notna 0.1 < 0.5 -> DROP
            "patient_id": [f"p{i}" for i in range(n)],  # metadata in denylist -> DROP
        }
    )


class TestDiscoverModelFeatureCols_RC2:
    def test_keeps_sparse_and_dense_drops_leaks_and_junk(self):
        df = _cohort()
        cols = _discover_model_feature_cols(df, exclude={"genuine_leak", "patient_id"})
        assert "has_asthma" in cols, "card-2 sparse legitimate predictor must be retained"
        assert "age_at_index" in cols
        assert "payer_category" in cols, "low-cardinality categorical must be retained"
        assert "genuine_leak" not in cols
        assert "all_constant" not in cols
        assert "too_sparse" not in cols
        assert "patient_id" not in cols

    def test_runner_exclude_set_has_the_metadata_denylist(self):
        excl = _runner_exclude_set({"scope_spec": {"excluded_features": ["custom_pii"]}})
        for name in ("patient_journey_id", "journey_status", "treatment_initiated", "custom_pii"):
            assert name in excl, f"{name} must be in the runner exclude set"


import inspect

from scripts import run_tier0_test


class TestRunnerTrainsOnRetainedSet_RC2:
    def test_narrow_subset_selection_is_gone(self):
        src = inspect.getsource(run_tier0_test)
        assert "feature_cols = [f for f in remediated if f in eligible_df.columns]" not in src, (
            "RC2: Site-1 still selects X from the curated remediated sub-list"
        )
        assert "feature_cols = [f for f in _rem_features if f in eligible_df.columns]" not in src, (
            "RC2: Step-5a still selects X from the curated _rem_features sub-list"
        )

    def test_discovery_helper_drives_the_matrix(self):
        src = inspect.getsource(run_tier0_test)
        # The two override sites must route through the helper (definition = 3rd occurrence).
        assert src.count("_discover_model_feature_cols(") >= 3
