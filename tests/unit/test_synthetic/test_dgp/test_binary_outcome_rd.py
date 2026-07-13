"""Phase 0: the general binary-outcome-with-recoverable-RD core."""

import numpy as np
import pytest

from src.ml.synthetic.dgp.treatment_arm import (
    _BIOLOGIC_EXPERIENCED_MULT,
    _BIOLOGIC_NAIVE_MULT,
    binary_outcome_rd,
    biologic_cate_modifier,
)


@pytest.mark.unit
def test_binary_outcome_rd_three_distinct_ordered_tau_in_band():
    rng = np.random.default_rng(7)
    n = 4000
    severity = rng.uniform(0, 10, n)
    segment = np.where(
        severity > 7, "high_severity", np.where(severity > 4, "medium_severity", "low_severity")
    )
    arm = (rng.random(n) < 0.5).astype(int)
    baseline = 0.10 * (severity - 5.0)
    cate_map = {"high_severity": 0.9, "medium_severity": 0.5, "low_severity": 0.2}

    y, tau_i = binary_outcome_rd(
        arm,
        baseline,
        segment,
        cate_map,
        rng,
        target_prevalence=0.35,
        noise_std=0.6,
    )

    distinct = sorted(set(np.round(tau_i, 6)))
    assert len(distinct) == 3
    hi = tau_i[segment == "high_severity"][0]
    md = tau_i[segment == "medium_severity"][0]
    lo = tau_i[segment == "low_severity"][0]
    assert hi > md > lo > 0
    assert 0.20 <= y.mean() <= 0.50


@pytest.mark.unit
def test_biologic_cate_modifier_constants_are_mean_preserving():
    """Phase 3 drift-lock: the biologic multipliers are the mean-preserving 2x
    spread (experienced attenuated, naive boosted) validated by the disproof."""
    assert _BIOLOGIC_EXPERIENCED_MULT < 1.0 < _BIOLOGIC_NAIVE_MULT
    # ~2x ratio (recoverable; a subtle <1.5x spread sign-flips per the disproof)
    assert round(_BIOLOGIC_NAIVE_MULT / _BIOLOGIC_EXPERIENCED_MULT, 1) == 2.0
    # mean-preserving at the generator's ~40% experienced prevalence
    assert abs(0.60 * _BIOLOGIC_NAIVE_MULT + 0.40 * _BIOLOGIC_EXPERIENCED_MULT - 1.0) < 1e-9
    m = biologic_cate_modifier(np.array([0, 1, 0, 1]))
    assert list(m) == [_BIOLOGIC_NAIVE_MULT, _BIOLOGIC_EXPERIENCED_MULT] * 2


@pytest.mark.unit
def test_cate_modifier_opens_recoverable_biologic_gap_without_shifting_mean():
    """cate_modifier yields segment x modifier cells (6 values here), experienced
    < naive within each segment, and — being mean-preserving — leaves the
    segment-marginal RD ~unchanged vs the unmodified call."""
    rng = np.random.default_rng(21)
    n = 6000
    severity = rng.uniform(0, 10, n)
    segment = np.where(
        severity > 7, "high_severity", np.where(severity > 4, "medium_severity", "low_severity")
    )
    arm = (rng.random(n) < 0.5).astype(int)
    baseline = 0.10 * (severity - 5.0)
    cate_map = {"high_severity": 0.9, "medium_severity": 0.5, "low_severity": 0.2}
    biologic = (rng.random(n) < 0.40).astype(int)
    modifier = biologic_cate_modifier(biologic)

    _, tau_plain = binary_outcome_rd(
        arm, baseline, segment, cate_map, np.random.default_rng(21), target_prevalence=0.35
    )
    _, tau_mod = binary_outcome_rd(
        arm,
        baseline,
        segment,
        cate_map,
        np.random.default_rng(21),
        target_prevalence=0.35,
        cate_modifier=modifier,
    )

    # 6 distinct cells (3 severity x 2 biologic), naive > experienced in each segment
    assert len(set(np.round(tau_mod, 6))) == 6
    for seg in ("high_severity", "medium_severity", "low_severity"):
        naive = tau_mod[(segment == seg) & (biologic == 0)][0]
        exp = tau_mod[(segment == seg) & (biologic == 1)][0]
        assert naive > exp, seg
    # mean-preserving: each segment's biologic-marginal RD ~= the unmodified RD
    for seg in ("high_severity", "medium_severity", "low_severity"):
        assert abs(tau_mod[segment == seg].mean() - tau_plain[segment == seg][0]) < 0.03, seg


@pytest.mark.unit
def test_recovery_probe_accepts_explicit_tuple_signature():
    # Signature-only smoke (no econml fit): the function must accept the new
    # keyword args without TypeError.
    import inspect

    from src.ml.synthetic.dgp import recovery_probe

    sig = inspect.signature(recovery_probe.recover_ate_and_cate)
    for p in ("treatment_col", "outcome_col", "confounders", "segment_col", "true_ate", "cate_map"):
        assert p in sig.parameters, f"{p} missing from recover_ate_and_cate signature"
