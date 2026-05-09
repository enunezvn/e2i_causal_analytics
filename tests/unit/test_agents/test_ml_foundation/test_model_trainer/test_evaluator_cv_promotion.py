"""Unit tests for ``_promote_cv_summary_to_validation_metrics`` (backlog #18).

Pins the contract: when 5-fold CV completes, scalar ``cv_<metric>_<stat>``
values from ``compute_stratified_cv`` get flattened into
``metrics_result["validation_metrics"]`` as ``cv_5fold_<metric>_<stat>``
keys so they survive the TIER0_E2E_JSON_OUT artifact's scalar-only filter
at ``scripts/run_tier0_test.py:5972-5976``.

Pre-fix: cv summary lived in ``metrics_result["cv_results"]`` (a sub-dict)
and got dropped by the artifact filter — downstream consumers stdout-scraped
the log line ``5-fold CV AUC: 0.7303±0.0566``.
"""

from __future__ import annotations

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _CV_PROMOTED_METRICS,
    _CV_PROMOTED_STATS,
    _promote_cv_summary_to_validation_metrics,
)


def _full_cv_result() -> dict:
    """A cv_result dict with all 4 metrics × 2 stats populated."""
    return {
        "cv_completed": True,
        "n_folds": 5,
        "cv_roc_auc_mean": 0.7303,
        "cv_roc_auc_std": 0.0566,
        "cv_roc_auc_folds": [0.71, 0.74, 0.73, 0.72, 0.74],
        "cv_pr_auc_mean": 0.4421,
        "cv_pr_auc_std": 0.0312,
        "cv_pr_auc_folds": [0.43, 0.45, 0.44, 0.43, 0.45],
        "cv_mcc_mean": 0.3855,
        "cv_mcc_std": 0.0291,
        "cv_mcc_folds": [0.37, 0.40, 0.39, 0.38, 0.40],
        "cv_f1_mean": 0.5102,
        "cv_f1_std": 0.0387,
        "cv_f1_folds": [0.49, 0.52, 0.51, 0.50, 0.53],
    }


def test_promotes_all_eight_scalar_keys():
    """All 4 metrics × 2 stats land in validation_metrics with cv_5fold_ prefix."""
    metrics_result: dict = {"validation_metrics": {}}
    _promote_cv_summary_to_validation_metrics(metrics_result, _full_cv_result())

    val = metrics_result["validation_metrics"]
    expected = {
        "cv_5fold_roc_auc_mean": 0.7303,
        "cv_5fold_roc_auc_std": 0.0566,
        "cv_5fold_pr_auc_mean": 0.4421,
        "cv_5fold_pr_auc_std": 0.0312,
        "cv_5fold_mcc_mean": 0.3855,
        "cv_5fold_mcc_std": 0.0291,
        "cv_5fold_f1_mean": 0.5102,
        "cv_5fold_f1_std": 0.0387,
    }
    for key, value in expected.items():
        assert key in val, f"missing {key!r} in validation_metrics"
        assert val[key] == value, f"{key}: expected {value}, got {val[key]}"


def test_does_not_promote_per_fold_lists():
    """``cv_<metric>_folds`` per-fold list is NOT promoted (would break the
    artifact's scalar-only filter)."""
    metrics_result: dict = {"validation_metrics": {}}
    _promote_cv_summary_to_validation_metrics(metrics_result, _full_cv_result())

    val = metrics_result["validation_metrics"]
    assert not any(k.endswith("_folds") for k in val), (
        f"validation_metrics should not contain per-fold lists; got: "
        f"{[k for k in val if k.endswith('_folds')]}"
    )


def test_does_not_promote_metadata_keys():
    """``cv_completed`` / ``n_folds`` are not metric scalars and stay out."""
    metrics_result: dict = {"validation_metrics": {}}
    _promote_cv_summary_to_validation_metrics(metrics_result, _full_cv_result())

    val = metrics_result["validation_metrics"]
    for key in ("cv_completed", "n_folds", "cv_5fold_completed", "cv_5fold_n_folds"):
        assert key not in val, f"{key!r} should not be in validation_metrics"


def test_partial_cv_result_only_promotes_present_keys():
    """If the cv_result is missing some metrics (e.g. mcc/f1), only present
    keys are promoted — no KeyError, no None placeholders."""
    metrics_result: dict = {"validation_metrics": {}}
    cv_result = {
        "cv_completed": True,
        "n_folds": 5,
        "cv_roc_auc_mean": 0.71,
        "cv_roc_auc_std": 0.04,
        # pr_auc / mcc / f1 absent
    }
    _promote_cv_summary_to_validation_metrics(metrics_result, cv_result)

    val = metrics_result["validation_metrics"]
    assert val == {
        "cv_5fold_roc_auc_mean": 0.71,
        "cv_5fold_roc_auc_std": 0.04,
    }


def test_creates_validation_metrics_subdict_when_missing():
    """If ``metrics_result`` has no ``validation_metrics`` key, the helper
    creates it (setdefault). Defensive against an empty pipeline state."""
    metrics_result: dict = {}
    _promote_cv_summary_to_validation_metrics(metrics_result, _full_cv_result())

    assert "validation_metrics" in metrics_result
    assert metrics_result["validation_metrics"]["cv_5fold_roc_auc_mean"] == 0.7303


def test_preserves_existing_validation_metrics_keys():
    """The helper adds cv_5fold_ keys without clobbering existing fields
    like ``roc_auc`` / ``pr_auc`` (the model's own validation metrics)."""
    metrics_result: dict = {
        "validation_metrics": {
            "roc_auc": 0.6592,
            "pr_auc": 0.41,
            "mcc": 0.32,
        }
    }
    _promote_cv_summary_to_validation_metrics(metrics_result, _full_cv_result())

    val = metrics_result["validation_metrics"]
    # Existing keys preserved:
    assert val["roc_auc"] == 0.6592
    assert val["pr_auc"] == 0.41
    assert val["mcc"] == 0.32
    # New keys added:
    assert val["cv_5fold_roc_auc_mean"] == 0.7303
    assert val["cv_5fold_pr_auc_mean"] == 0.4421
    assert val["cv_5fold_mcc_mean"] == 0.3855


def test_module_constants_reflect_promoted_set():
    """``_CV_PROMOTED_METRICS`` and ``_CV_PROMOTED_STATS`` are the source
    of truth for what gets promoted; pin them so a future addition (e.g.
    a new ``cv_brier_mean``) is a deliberate edit, not a side-effect."""
    assert _CV_PROMOTED_METRICS == ("roc_auc", "pr_auc", "mcc", "f1")
    assert _CV_PROMOTED_STATS == ("mean", "std")


def test_raises_when_n_folds_is_not_5():
    """Codex pass-1 MEDIUM-2: the ``cv_5fold_`` prefix is hardcoded against
    the fixed-5-fold CV at the runner's evaluator callsite. A different
    fold count would silently emit a misleading key name, so the helper
    fails loudly to surface the intent — both the prefix AND downstream
    JSON consumers must be updated in lockstep."""
    import pytest

    metrics_result: dict = {"validation_metrics": {}}
    cv_result_10fold = {**_full_cv_result(), "n_folds": 10}
    with pytest.raises(ValueError, match="n_folds=10"):
        _promote_cv_summary_to_validation_metrics(metrics_result, cv_result_10fold)


def test_does_not_raise_when_n_folds_missing():
    """If ``cv_result`` lacks ``n_folds`` (legacy / partial result), the
    helper passes through silently — the assertion is only meaningful when
    a fold count is explicitly recorded."""
    metrics_result: dict = {"validation_metrics": {}}
    cv_result_no_n_folds = {
        "cv_completed": True,
        "cv_roc_auc_mean": 0.71,
        "cv_roc_auc_std": 0.04,
        # n_folds intentionally absent
    }
    _promote_cv_summary_to_validation_metrics(metrics_result, cv_result_no_n_folds)
    assert metrics_result["validation_metrics"]["cv_5fold_roc_auc_mean"] == 0.71
