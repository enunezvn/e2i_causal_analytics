"""Unit tests for ``adaptive_success_criteria()`` (v3 — Option C).

Full design contract at ``.claude/plans/adaptive_success_criteria/01-design.md``.
v3 (post-deep-research) drops ``minimum_precision`` and ``minimum_f1`` per
Van Calster et al. 2025 (Lancet Digital Health) and replaces them with
``minimum_net_benefit_at_p_t`` (DCA-derived), ``minimum_mcc`` (sanity gate),
``maximum_calibration_slope_deviation`` and
``maximum_calibration_intercept_magnitude`` (calibration quality per van
Calster 2019). Keep this file in sync with the per-metric formulas and the
worked-example table in ``01-design.md``.
"""

from __future__ import annotations

from inspect import signature

import pytest


def test_adaptive_success_criteria_function_exists_and_signature_matches() -> None:
    """The function must exist with the exact canonical signature."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    sig = signature(adaptive_success_criteria)
    params = list(sig.parameters)
    assert params == [
        "n_samples",
        "prevalence",
        "baseline_auc",
        "feature_count",
        "regime",
        "deployment_intent",
    ], f"Signature drifted: got {params}"


def test_adaptive_returns_tuple_of_thresholds_and_skipped_set() -> None:
    """v3 contract: return is (thresholds_dict, skipped_set)."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    out = adaptive_success_criteria(900, 0.50, 0.50, 14, "clean")
    assert isinstance(out, tuple)
    assert len(out) == 2
    thresholds, skipped = out
    assert isinstance(thresholds, dict)
    assert isinstance(skipped, set)
    # Skipped names MUST NOT also appear in thresholds (v3 invariant).
    assert thresholds.keys().isdisjoint(skipped)


def test_adaptive_clean_regime_canonical_row() -> None:
    """Clean regime, N=900, prev=0.50: row 1 of the v3 worked-example table."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    thresholds, skipped = adaptive_success_criteria(
        n_samples=900,
        prevalence=0.50,
        baseline_auc=0.50,
        feature_count=14,
        regime="clean",
    )
    assert thresholds["minimum_auc"] == pytest.approx(0.75, abs=1e-6)
    assert thresholds["minimum_recall"] == pytest.approx(0.65, abs=1e-6)
    # v3: NB > 0 gate at the regime's p_t (audit value); threshold is 0.0
    assert thresholds["minimum_net_benefit_at_p_t"] == pytest.approx(0.0, abs=1e-6)
    # v3: MCC sanity gate (clean threshold per Chicco-Jurman 2020)
    assert thresholds["minimum_mcc"] == pytest.approx(0.45, abs=1e-6)
    # v3: van Calster 2019 calibration quality
    assert thresholds["maximum_calibration_slope_deviation"] == pytest.approx(0.15, abs=1e-6)
    assert thresholds["maximum_calibration_intercept_magnitude"] == pytest.approx(0.30, abs=1e-6)
    # v3 drops precision/f1 entirely
    assert "minimum_precision" not in thresholds
    assert "minimum_precision" not in skipped
    assert "minimum_f1" not in thresholds
    assert "minimum_f1" not in skipped
    # Lift fires at N=900, prev=0.50 (n_pos=450, 2*SE ≈ 0.047 < 0.10)
    assert thresholds["minimum_lift_over_baseline"] == pytest.approx(0.10, abs=1e-6)
    assert thresholds["maximum_calibration_error"] == pytest.approx(0.10, abs=1e-6)
    # Feature-density step function — 14/900 ≈ 1/64 ≤ 1/50 → 0.03
    assert thresholds["maximum_train_val_delta"] == pytest.approx(0.03, abs=1e-6)
    assert skipped == set()  # clean regime skips nothing in v3


def test_adaptive_default_regime_skips_auc() -> None:
    """Default regime: minimum_auc is in `skipped`, not in `thresholds`."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    thresholds, skipped = adaptive_success_criteria(
        n_samples=900,
        prevalence=0.30,
        baseline_auc=0.50,
        feature_count=14,
        regime="default",
    )
    assert "minimum_auc" not in thresholds
    assert "minimum_auc" in skipped
    assert thresholds["minimum_recall"] == pytest.approx(0.65, abs=1e-6)
    # v3: default-regime MCC threshold (between adverse 0.20 and clean 0.45)
    assert thresholds["minimum_mcc"] == pytest.approx(0.35, abs=1e-6)
    assert thresholds["minimum_net_benefit_at_p_t"] == pytest.approx(0.0, abs=1e-6)
    # v3 drops precision/f1
    assert "minimum_precision" not in thresholds
    assert "minimum_f1" not in thresholds


def test_adaptive_adverse_regime_skips_lift_only() -> None:
    """Adverse regime (prev=0.02): only lift is skipped in v3.

    In v2 precision and f1 were also skipped. v3 drops those gates entirely
    (Van Calster 2025), so adverse no longer reports them in `skipped` —
    they're simply absent from the v3 active gate set.
    """
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    thresholds, skipped = adaptive_success_criteria(
        n_samples=900,
        prevalence=0.02,
        baseline_auc=0.50,
        feature_count=14,
        regime="adverse",
    )
    assert thresholds["minimum_auc"] == pytest.approx(0.70, abs=1e-6)
    assert thresholds["minimum_recall"] == pytest.approx(0.50, abs=1e-6)
    # v3: adverse-regime gates fire, NB at p_t=0.05 ≡ precision > 0.05
    assert thresholds["minimum_net_benefit_at_p_t"] == pytest.approx(0.0, abs=1e-6)
    assert thresholds["minimum_mcc"] == pytest.approx(0.20, abs=1e-6)
    assert thresholds["maximum_calibration_slope_deviation"] == pytest.approx(0.15, abs=1e-6)
    assert thresholds["maximum_calibration_intercept_magnitude"] == pytest.approx(0.30, abs=1e-6)
    # n_pos = 18, SE = 0.5/sqrt(18) ≈ 0.118, 2*SE = 0.236 > 0.10 → skip
    assert "minimum_lift_over_baseline" not in thresholds
    assert "minimum_lift_over_baseline" in skipped
    # ECE always present
    assert thresholds["maximum_calibration_error"] == pytest.approx(0.10, abs=1e-6)
    # v3 drops precision/f1 — they are NEITHER in thresholds NOR in skipped
    assert "minimum_precision" not in thresholds
    assert "minimum_precision" not in skipped
    assert "minimum_f1" not in thresholds
    assert "minimum_f1" not in skipped


def test_adaptive_baseline_aware_clean_high_baseline() -> None:
    """Clean regime with baseline_auc=0.60 must clear baseline+0.20=0.80."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    thresholds, _ = adaptive_success_criteria(900, 0.50, 0.60, 14, "clean")
    assert thresholds["minimum_auc"] == pytest.approx(0.80, abs=1e-6)


def test_adaptive_n_threshold_for_ece_tightening() -> None:
    """N >= 1000 ⇒ max_ece = 0.05; below ⇒ 0.10."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    t_below, _ = adaptive_success_criteria(900, 0.50, 0.50, 14, "clean")
    t_at, _ = adaptive_success_criteria(1000, 0.50, 0.50, 14, "clean")
    t_above, _ = adaptive_success_criteria(5000, 0.50, 0.50, 14, "clean")
    assert t_below["maximum_calibration_error"] == pytest.approx(0.10, abs=1e-6)
    assert t_at["maximum_calibration_error"] == pytest.approx(0.05, abs=1e-6)
    assert t_above["maximum_calibration_error"] == pytest.approx(0.05, abs=1e-6)


@pytest.mark.parametrize(
    "n,features,expected",
    [
        (900, 14, 0.03),  # 14/900 ≈ 1/64 ≤ 1/50 → 0.03
        (5000, 30, 0.03),  # 30/5000 ≈ 1/167 ≤ 1/50 → 0.03
        (600, 20, 0.05),  # 20/600 ≈ 1/30, in (1/50, 1/30] → 0.05
        (300, 14, 0.07),  # 14/300 ≈ 1/21.4, in (1/30, 1/15] → 0.07
        (200, 20, 0.10),  # 20/200 = 1/10 > 1/15 → 0.10
    ],
)
def test_adaptive_train_val_delta_step_function(n: int, features: int, expected: float) -> None:
    """Feature-density step function across all four buckets."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    thresholds, _ = adaptive_success_criteria(n, 0.50, 0.50, features, "clean")
    assert thresholds["maximum_train_val_delta"] == pytest.approx(expected, abs=1e-6)


def test_adaptive_invalid_inputs_raise() -> None:
    """Out-of-range inputs must raise ValueError, not silently extrapolate."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    with pytest.raises(ValueError, match="n_samples"):
        adaptive_success_criteria(0, 0.5, 0.5, 14, "clean")
    with pytest.raises(ValueError, match="prevalence"):
        adaptive_success_criteria(900, 1.5, 0.5, 14, "clean")
    with pytest.raises(ValueError, match="prevalence"):
        adaptive_success_criteria(900, -0.1, 0.5, 14, "clean")
    with pytest.raises(ValueError, match="baseline_auc"):
        adaptive_success_criteria(900, 0.5, 1.5, 14, "clean")
    with pytest.raises(ValueError, match="feature_count"):
        adaptive_success_criteria(900, 0.5, 0.5, 0, "clean")


def test_adaptive_none_regime_treated_as_clean() -> None:
    """RWD callers pass regime=None; behavior must equal regime='clean'."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    none_t, none_s = adaptive_success_criteria(900, 0.50, 0.50, 14, None)
    clean_t, clean_s = adaptive_success_criteria(900, 0.50, 0.50, 14, "clean")
    assert none_t == clean_t
    assert none_s == clean_s


def test_adaptive_skipped_disjoint_from_thresholds_invariant() -> None:
    """v3 invariant: every key is in EXACTLY ONE of (thresholds, skipped)."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    for n, prev, regime in [
        (900, 0.50, "clean"),
        (900, 0.30, "default"),
        (900, 0.02, "adverse"),
        (5000, 0.10, None),
        (200, 0.10, None),
    ]:
        thresholds, skipped = adaptive_success_criteria(n, prev, 0.50, 14, regime)
        assert thresholds.keys().isdisjoint(skipped), (
            f"{regime} N={n} prev={prev}: {thresholds.keys() & skipped}"
        )


def test_v3_drops_precision_and_f1_from_thresholds_and_skipped() -> None:
    """v3 invariant: precision/f1 are GONE — never in either dict."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    for n, prev, regime in [
        (900, 0.50, "clean"),
        (900, 0.30, "default"),
        (900, 0.02, "adverse"),
        (200, 0.10, None),
    ]:
        thresholds, skipped = adaptive_success_criteria(n, prev, 0.50, 14, regime)
        for key in ("minimum_precision", "minimum_f1"):
            assert key not in thresholds, f"{regime} N={n} prev={prev}: v3 must not emit {key}"
            assert key not in skipped, (
                f"{regime} N={n} prev={prev}: v3 must not even mention {key} in skipped"
            )


def test_v3_calibration_gates_always_fire() -> None:
    """Calibration slope/intercept gates are regime-independent — they fire
    in every regime per van Calster 2019."""
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        adaptive_success_criteria,
    )

    for regime in ("clean", "default", "adverse"):
        prev = 0.02 if regime == "adverse" else (0.30 if regime == "default" else 0.50)
        thresholds, _ = adaptive_success_criteria(900, prev, 0.50, 14, regime)
        assert thresholds["maximum_calibration_slope_deviation"] == pytest.approx(0.15, abs=1e-6), (
            regime
        )
        assert thresholds["maximum_calibration_intercept_magnitude"] == pytest.approx(
            0.30, abs=1e-6
        ), regime
