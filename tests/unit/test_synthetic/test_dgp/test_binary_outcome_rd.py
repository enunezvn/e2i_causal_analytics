"""Phase 0: the general binary-outcome-with-recoverable-RD core."""
import numpy as np
import pytest

from src.ml.synthetic.dgp.treatment_arm import binary_outcome_rd


@pytest.mark.unit
def test_binary_outcome_rd_three_distinct_ordered_tau_in_band():
    rng = np.random.default_rng(7)
    n = 4000
    severity = rng.uniform(0, 10, n)
    segment = np.where(severity > 7, "high_severity",
                       np.where(severity > 4, "medium_severity", "low_severity"))
    arm = (rng.random(n) < 0.5).astype(int)
    baseline = 0.10 * (severity - 5.0)
    cate_map = {"high_severity": 0.9, "medium_severity": 0.5, "low_severity": 0.2}

    y, tau_i = binary_outcome_rd(
        arm, baseline, segment, cate_map, rng,
        target_prevalence=0.35, noise_std=0.6,
    )

    distinct = sorted(set(np.round(tau_i, 6)))
    assert len(distinct) == 3
    hi = tau_i[segment == "high_severity"][0]
    md = tau_i[segment == "medium_severity"][0]
    lo = tau_i[segment == "low_severity"][0]
    assert hi > md > lo > 0
    assert 0.20 <= y.mean() <= 0.50


@pytest.mark.unit
def test_recovery_probe_accepts_explicit_tuple_signature():
    # Signature-only smoke (no econml fit): the function must accept the new
    # keyword args without TypeError.
    import inspect
    from src.ml.synthetic.dgp import recovery_probe

    sig = inspect.signature(recovery_probe.recover_ate_and_cate)
    for p in ("treatment_col", "outcome_col", "confounders", "segment_col",
              "true_ate", "cate_map"):
        assert p in sig.parameters, f"{p} missing from recover_ate_and_cate signature"
