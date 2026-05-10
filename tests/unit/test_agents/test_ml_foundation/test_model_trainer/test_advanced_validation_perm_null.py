"""Plan v3 §3 Tier 1B step 1: permutation-null surface contract.

Pins the new `permutation_null_p95`, `permutation_null_p99`,
`permutation_n_permutations`, and `permutation_n_effective` keys returned by
`compute_permutation_test`, plus the
`_promote_permutation_summary_to_validation_metrics` evaluator helper.

These keys are the prerequisite surface for HBLP variance-inflation gating
(plan §3 Tier 1B step 2), the permutation-anchored AUC floor (plan §4 T2.2),
and the deployer's signal-genuineness input contract (plan §4 T2.6a).
"""

from __future__ import annotations

import numpy as np

from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
    DEFAULT_PERMUTATION_COUNT,
    compute_permutation_test,
)
from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _PERMUTATION_PROMOTED_KEYS,
    _promote_permutation_summary_to_validation_metrics,
)

# --------------------------------------------------------------------------- #
# Module-level constants                                                      #
# --------------------------------------------------------------------------- #


def test_default_permutation_count_is_200() -> None:
    """Plan v3 §3 Tier 1B step 1 default: 200 permutations."""
    assert DEFAULT_PERMUTATION_COUNT == 200


def test_promoted_keys_include_p95_p99_n_perm() -> None:
    """The promotion-helper key list must include the plan-required keys."""
    assert "permutation_null_p95" in _PERMUTATION_PROMOTED_KEYS
    assert "permutation_null_p99" in _PERMUTATION_PROMOTED_KEYS
    assert "permutation_n_permutations" in _PERMUTATION_PROMOTED_KEYS


# --------------------------------------------------------------------------- #
# compute_permutation_test return shape                                       #
# --------------------------------------------------------------------------- #


def _toy_signal(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Toy binary classification with separable AUC ~0.85."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.5).astype(int)
    proba = np.where(y == 1, rng.uniform(0.55, 0.95, size=n), rng.uniform(0.05, 0.45, size=n))
    return y, proba


def test_returns_p95_and_p99_keys_on_normal_run() -> None:
    """When the perm test runs, p95/p99 percentiles of the null are populated."""
    y, proba = _toy_signal()
    result = compute_permutation_test(y, proba, n_permutations=50)
    assert "permutation_null_p95" in result
    assert "permutation_null_p99" in result
    assert result["permutation_null_p95"] is not None
    assert result["permutation_null_p99"] is not None
    # p95 must be no greater than p99 by construction.
    assert result["permutation_null_p95"] <= result["permutation_null_p99"]
    # Both must lie in [0, 1] (AUC bounds).
    assert 0.0 <= result["permutation_null_p95"] <= 1.0
    assert 0.0 <= result["permutation_null_p99"] <= 1.0


def test_returns_n_permutations_keys() -> None:
    """Both the new prefixed key AND the legacy alias must be returned."""
    y, proba = _toy_signal()
    result = compute_permutation_test(y, proba, n_permutations=37)
    assert result["permutation_n_permutations"] == 37
    assert result["n_permutations"] == 37  # legacy alias preserved


def test_n_effective_tracks_completed_shuffles() -> None:
    """`permutation_n_effective` reports the count after dropping shuffles
    that raised in roc_auc_score (single-class shuffle on tiny y)."""
    y, proba = _toy_signal(n=400)
    result = compute_permutation_test(y, proba, n_permutations=20)
    assert result["permutation_n_effective"] >= 1
    assert result["permutation_n_effective"] <= 20


def test_default_n_permutations_is_200_when_omitted() -> None:
    """Calling without `n_permutations` uses DEFAULT_PERMUTATION_COUNT (200)."""
    y, proba = _toy_signal(n=300)
    result = compute_permutation_test(y, proba)
    assert result["permutation_n_permutations"] == DEFAULT_PERMUTATION_COUNT
    assert result["permutation_n_permutations"] == 200


# --------------------------------------------------------------------------- #
# Degenerate-input handling                                                   #
# --------------------------------------------------------------------------- #


def test_none_proba_returns_p95_p99_keys_as_none() -> None:
    """Plan §3 Tier 1B step 1 contract: even on degenerate input, the promoted
    keys must be present (with None) so downstream consumers don't have to
    treat 'key absent' and 'p95=None' as different cases."""
    y = np.array([0, 1, 0, 1])
    result = compute_permutation_test(y, None, n_permutations=10)
    assert "permutation_null_p95" in result
    assert "permutation_null_p99" in result
    assert result["permutation_null_p95"] is None
    assert result["permutation_null_p99"] is None
    assert result["permutation_n_permutations"] == 10
    assert result["signal_genuine"] is None


def test_single_class_y_returns_p95_p99_keys_as_none() -> None:
    """If actual_auc cannot be computed (single-class y), the keys are still
    present but None."""
    y = np.zeros(50, dtype=int)
    proba = np.linspace(0.1, 0.9, 50)
    result = compute_permutation_test(y, proba, n_permutations=10)
    assert "permutation_null_p95" in result
    assert "permutation_null_p99" in result
    assert result["permutation_null_p95"] is None
    assert result["permutation_null_p99"] is None


def test_finite_actual_auc_preserved_when_all_shuffles_nan_filtered() -> None:
    """Codex MEDIUM-1 regression: when actual_auc is finite but every
    shuffled AUC NaN-filters out (extreme corner case where every
    shuffle produces a single-class y), the degenerate-return block must
    still expose the finite actual_auc — discarding it silently hides a
    real measurement from downstream readers."""
    # Construct a y/proba where actual_auc is finite but every shuffle
    # (which has the same class counts as actual y) cannot be NaN; so we
    # instead patch np.random's permutation to return a single-class y.
    y = np.array([0] * 5 + [1] * 5)
    proba = np.linspace(0.05, 0.95, 10)

    # Monkey-patch the random permutation to always return a single-class
    # array, forcing roc_auc_score → NaN under sklearn 1.4+.
    import numpy as _np

    real_default_rng = _np.random.default_rng

    class _AlwaysSingleClassRng:
        def __init__(self, _seed):  # noqa: D401
            pass

        def permutation(self, _y):
            # Always return all-zeros, which yields single-class y_shuffled
            # and triggers the NaN-from-roc_auc_score branch.
            return _np.zeros_like(_y)

    _np.random.default_rng = lambda _seed: _AlwaysSingleClassRng(_seed)
    try:
        result = compute_permutation_test(y, proba, n_permutations=5)
    finally:
        _np.random.default_rng = real_default_rng

    # Degenerate output: pvalue/percentiles None, n_effective=0
    assert result["permutation_pvalue"] is None
    assert result["permutation_null_p95"] is None
    assert result["permutation_null_p99"] is None
    assert result["permutation_n_effective"] == 0
    # But the actual_auc — observed BEFORE the shuffle loop — is preserved.
    assert result["actual_auc"] is not None
    assert _np.isfinite(result["actual_auc"])


def test_p95_p99_match_numpy_percentile_on_known_distribution() -> None:
    """Math correctness: percentiles match np.percentile on the exact same
    shuffled-AUC sequence (deterministic seed 42 in the function)."""
    y, proba = _toy_signal(n=300, seed=1)
    n_perm = 100
    result = compute_permutation_test(y, proba, n_permutations=n_perm)

    # Re-derive the shuffled AUC sequence with the same seed.
    from sklearn.metrics import roc_auc_score

    y_proba_pos = proba
    rng = np.random.default_rng(42)
    expected_aucs = []
    for _ in range(n_perm):
        y_shuffled = rng.permutation(y)
        try:
            expected_aucs.append(float(roc_auc_score(y_shuffled, y_proba_pos)))
        except ValueError:
            continue
    expected_p95 = float(np.percentile(expected_aucs, 95))
    expected_p99 = float(np.percentile(expected_aucs, 99))
    assert result["permutation_null_p95"] == expected_p95
    assert result["permutation_null_p99"] == expected_p99


# --------------------------------------------------------------------------- #
# _promote_permutation_summary_to_validation_metrics                          #
# --------------------------------------------------------------------------- #


def _full_perm_result() -> dict:
    """A permutation_result dict with all promoted keys populated."""
    return {
        "permutation_pvalue": 0.02,
        "permutation_auc_mean": 0.503,
        "permutation_auc_std": 0.041,
        "permutation_null_p95": 0.572,
        "permutation_null_p99": 0.612,
        "permutation_n_permutations": 200,
        "permutation_n_effective": 198,
        "actual_auc": 0.74,
        "n_permutations": 200,  # legacy alias not promoted
        "signal_genuine": True,
    }


def test_promoter_lifts_all_seven_scalar_keys_into_validation_metrics() -> None:
    metrics_result: dict = {"validation_metrics": {}}
    _promote_permutation_summary_to_validation_metrics(metrics_result, _full_perm_result())
    val = metrics_result["validation_metrics"]
    for key in _PERMUTATION_PROMOTED_KEYS:
        assert key in val, f"missing {key!r} in validation_metrics"
    assert val["permutation_null_p95"] == 0.572
    assert val["permutation_null_p99"] == 0.612
    assert val["permutation_n_permutations"] == 200
    assert val["permutation_pvalue"] == 0.02


def test_promoter_does_not_lift_legacy_alias_n_permutations() -> None:
    """The legacy `n_permutations` key is preserved on the perm-test sub-dict
    but NOT promoted into validation_metrics — only the new
    `permutation_n_permutations` key is. Prevents downstream confusion."""
    metrics_result: dict = {"validation_metrics": {}}
    _promote_permutation_summary_to_validation_metrics(metrics_result, _full_perm_result())
    val = metrics_result["validation_metrics"]
    assert "n_permutations" not in val  # legacy alias not promoted
    assert "permutation_n_permutations" in val


def test_promoter_creates_validation_metrics_when_absent() -> None:
    """`setdefault` semantics: caller need not pre-populate
    `validation_metrics` for promotion to succeed."""
    metrics_result: dict = {}
    _promote_permutation_summary_to_validation_metrics(metrics_result, _full_perm_result())
    assert "validation_metrics" in metrics_result
    assert metrics_result["validation_metrics"]["permutation_null_p95"] == 0.572


def test_promoter_preserves_other_validation_metrics_keys() -> None:
    """Promotion must not clobber pre-existing scalar keys (e.g., from CV
    promotion or earlier evaluator steps)."""
    metrics_result: dict = {
        "validation_metrics": {
            "roc_auc": 0.74,
            "cv_5fold_roc_auc_mean": 0.69,
        }
    }
    _promote_permutation_summary_to_validation_metrics(metrics_result, _full_perm_result())
    val = metrics_result["validation_metrics"]
    assert val["roc_auc"] == 0.74
    assert val["cv_5fold_roc_auc_mean"] == 0.69
    assert val["permutation_null_p95"] == 0.572


def test_promoter_promotes_none_when_keys_present_with_none_values() -> None:
    """Plan §3 Tier 1B step 1: degenerate perm runs (single-class y, no
    proba) emit None values; the promoter must lift them as None so
    downstream HBLP/T2.2 logic can distinguish 'perm test ran but
    degenerate' from 'perm test was never run' (key absent)."""
    metrics_result: dict = {"validation_metrics": {}}
    degenerate = {
        "permutation_pvalue": None,
        "permutation_null_p95": None,
        "permutation_null_p99": None,
        "permutation_n_permutations": 200,
        "signal_genuine": None,
    }
    _promote_permutation_summary_to_validation_metrics(metrics_result, degenerate)
    val = metrics_result["validation_metrics"]
    assert val["permutation_null_p95"] is None
    assert val["permutation_null_p99"] is None
    assert val["permutation_n_permutations"] == 200


def test_promoter_does_not_mutate_input_perm_result() -> None:
    """Codex MEDIUM-2: the promoter must NOT clobber or remove keys from
    the source `permutation_result` dict. The caller relies on the full
    sub-dict (including `actual_auc`, `signal_genuine`, legacy
    `n_permutations`) being intact at `metrics_result["permutation_test"]`
    after the promoter runs."""
    perm = _full_perm_result()
    perm_snapshot = dict(perm)
    metrics_result: dict = {"permutation_test": perm, "validation_metrics": {}}
    _promote_permutation_summary_to_validation_metrics(metrics_result, perm)
    assert perm == perm_snapshot, "promoter must not mutate input dict"
    # Sub-dict still has all original keys including unpromoted ones.
    assert metrics_result["permutation_test"] == perm_snapshot
    assert "actual_auc" in metrics_result["permutation_test"]
    assert "n_permutations" in metrics_result["permutation_test"]
    assert "signal_genuine" in metrics_result["permutation_test"]


def test_promoter_skips_keys_absent_from_perm_result() -> None:
    """Forward compat: if the perm result is missing a promoted key (e.g.,
    `permutation_n_effective` when an older caller is in flight), the
    promoter only lifts what's present — does NOT inject sentinel values."""
    metrics_result: dict = {"validation_metrics": {}}
    minimal = {
        "permutation_pvalue": 0.05,
        "permutation_null_p95": 0.55,
        "permutation_null_p99": 0.60,
        "permutation_n_permutations": 200,
        # `permutation_n_effective`, `permutation_auc_mean`, `..._std` absent.
    }
    _promote_permutation_summary_to_validation_metrics(metrics_result, minimal)
    val = metrics_result["validation_metrics"]
    assert "permutation_pvalue" in val
    assert "permutation_null_p95" in val
    assert "permutation_n_effective" not in val
    assert "permutation_auc_mean" not in val
