"""RC3a: _apply_leakage_remediation must not re-drop declared-safe (manifest
pre-index) features in its per-feature structural re-check. The manifest
contract is authoritative — mirrors adaptive_validity_check's first-pass
immunity and the LLM-drop-list strip in review_and_remediate_leakage."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents.ml_foundation.data_preparer.nodes.leakage_remediation import (
    _apply_leakage_remediation,
)


def _two_safe_sparse(n: int = 400, n_pos: int = 12, seed: int = 0):
    """Two legitimate cardinality-2 sparse pre-index flags (~4-5% density),
    independent of the target (AUC ~ 0.5 so backward elimination never fires),
    each all-zero in the tiny positive class -> the per-feature zero_variance
    re-check rejects them today (when R1's guard is absent on this branch)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=n_pos, replace=False)] = 1
    neg = np.where(y == 0)[0]

    def flag(k: int) -> np.ndarray:
        f = np.zeros(n, dtype=float)
        f[rng.choice(neg, size=k, replace=False)] = 1.0
        return f

    return pd.DataFrame({"safe_sparse_a": flag(16), "safe_sparse_b": flag(20), "target": y})


def _state_and_analysis():
    df = _two_safe_sparse()
    state = {
        "train_df": df,
        "validation_df": None,
        "test_df": None,
        "holdout_df": None,
        "scope_spec": {
            "prediction_target": "target",
            # Truthy manifest source so the immunity branch runs; the actual
            # contract lookup is patched below so the test does not depend on
            # the real manifest contents.
            "feature_manifest_source": "optum",
        },
        "leaked_features": [],
    }
    analysis = {
        "features_to_drop": [],
        "recommended_feature_set": ["safe_sparse_a", "safe_sparse_b"],
    }
    return state, analysis


class TestDeclaredSafeImmuneInRecheck_RC3a:
    def test_declared_safe_features_survive_the_structural_recheck(self):
        state, analysis = _state_and_analysis()
        # Patch at the origin module: the production code lazily imports this
        # name inside `if manifest_source:`, so it is resolved from
        # adaptive_validity_check's module dict at call time.
        with patch(
            "src.agents.ml_foundation.data_preparer.nodes."
            "adaptive_validity_check._declared_safe_immune_features",
            return_value={"safe_sparse_a", "safe_sparse_b"},
        ):
            result = _apply_leakage_remediation(state, analysis)
        assert result.get("success") is True, (
            "RC3a: declared-safe sparse features were re-dropped by the re-check, "
            f"leaving < 2 verified -> remediation failed: {result.get('reason')}"
        )
        final = set(result.get("final_features", []))
        assert {"safe_sparse_a", "safe_sparse_b"} <= final, (
            f"RC3a: declared-safe features missing from final_features: {final}"
        )

    def test_non_immune_sparse_features_are_still_rejected(self):
        """Control: with NO immunity (empty set), the same sparse features are
        rejected by the un-guarded re-check -> remediation reports failure."""
        state, analysis = _state_and_analysis()
        with patch(
            "src.agents.ml_foundation.data_preparer.nodes."
            "adaptive_validity_check._declared_safe_immune_features",
            return_value=set(),
        ):
            result = _apply_leakage_remediation(state, analysis)
        assert result.get("success") is False
