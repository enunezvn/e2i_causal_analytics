"""RC3a: _apply_leakage_remediation must not re-drop declared-safe (manifest
pre-index) features in its per-feature structural re-check. The manifest
contract is authoritative — mirrors adaptive_validity_check's first-pass
immunity and the LLM-drop-list strip in review_and_remediate_leakage.

Test-design note (cross-shard): the RC1 rare-event guard (check_zero_variance_
within_class skips cardinality<=2 features on rare-event cohorts) is now in main,
so a cardinality-2 sparse fixture would survive the re-check via R1 regardless of
R2's immune-skip — it would not isolate R2's behavior. To isolate the immune-skip
independent of R1, this uses a CARDINALITY>2 separator (R1's `n_unique<=2` guard
does not apply) on a BALANCED cohort (R4's pos_rate<5% demotion does not apply):
the separator is rejected per-feature by `single_feature_auc` (HIGH at AUC>0.80)
when NOT immune, and survives ONLY when the immune-skip exempts it. Its AUC stays
below the 0.95 combined-feature backward-elimination threshold so that guard does
not interfere.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents.ml_foundation.data_preparer.nodes.leakage_remediation import (
    _apply_leakage_remediation,
)


def _separator_and_noise(n: int = 240, seed: int = 0):
    """A cardinality>2 separating feature that `single_feature_auc` rejects
    (effective AUC in the ~0.80-0.90 HIGH band, below the 0.95 backward-
    elimination threshold), plus a pure-noise feature that always survives.
    Balanced ~50% target so no rare-event guard/demotion (R1/R4) applies."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # Continuous (cardinality == n >> 2) so R1's n_unique<=2 guard never skips it.
    # signal=1.1 tuned empirically: single-feature effective AUC ~0.86 (fires
    # single_feature_auc HIGH, >0.80) while the combined [sep, noise] CV-AUC stays
    # ~0.85 (well below the 0.95 backward-elimination threshold) — healthy margins
    # on both sides so the test is stable across environments.
    safe_sep = y * 1.1 + rng.normal(0.0, 1.0, n)
    noise = rng.standard_normal(n)
    return pd.DataFrame({"safe_sep": safe_sep, "noise": noise, "target": y})


def _state_and_analysis():
    df = _separator_and_noise()
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
        "recommended_feature_set": ["safe_sep", "noise"],
    }
    return state, analysis


class TestDeclaredSafeImmuneInRecheck_RC3a:
    def test_declared_safe_feature_survives_the_structural_recheck(self):
        """Load-bearing: a declared-safe feature that the structural re-check
        WOULD reject (single_feature_auc HIGH) survives ONLY because the
        immune-skip exempts it. Isolates R2's behavior independent of R1."""
        state, analysis = _state_and_analysis()
        # Patch at the origin module: the production code lazily imports this
        # name inside `if manifest_source:`, so it is resolved from
        # adaptive_validity_check's module dict at call time.
        with patch(
            "src.agents.ml_foundation.data_preparer.nodes."
            "adaptive_validity_check._declared_safe_immune_features",
            return_value={"safe_sep"},
        ):
            result = _apply_leakage_remediation(state, analysis)
        assert result.get("success") is True, (
            "RC3a: a declared-safe feature was re-dropped by the structural "
            f"re-check despite immunity: {result.get('reason')}"
        )
        assert "safe_sep" in set(result.get("final_features", [])), (
            f"RC3a: declared-safe feature missing from final_features: "
            f"{result.get('final_features')}"
        )

    def test_non_immune_separator_is_still_rejected(self):
        """Control: with NO immunity (empty set), the separating feature is
        rejected by the per-feature structural re-check (single_feature_auc),
        leaving only the noise feature (< 2 verified) -> remediation fails.
        Confirms the immune-skip — not some other path — is what saves it."""
        state, analysis = _state_and_analysis()
        with patch(
            "src.agents.ml_foundation.data_preparer.nodes."
            "adaptive_validity_check._declared_safe_immune_features",
            return_value=set(),
        ):
            result = _apply_leakage_remediation(state, analysis)
        assert result.get("success") is False, (
            "control: the non-immune separator should be rejected by the "
            f"structural re-check, but remediation succeeded: {result.get('final_features')}"
        )
