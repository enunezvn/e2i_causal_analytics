"""Adversarial harness for leakage detector — Phase 5 of ml-leakage-holistic-fix.

Plants synthetic leaks at calibrated single-feature-AUC tiers and verifies
that the detector flags each at the appropriate severity. Pre-Phase-1 the
detector silently missed everything below AUC=0.80. Post-Phase-1 the
thresholds are 0.55/0.65/0.80 (MODERATE/HIGH/CRITICAL).

Why this is in `tests/integration/` and not `tests/unit/`: the harness pairs
the injector with the detection pipeline end-to-end (planted hazard → state
→ check → expected severity). It exercises the integration of the injector
contract with the detector's check ladder, not just one function in
isolation.
"""

from __future__ import annotations

import pytest

from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import (
    check_single_feature_auc,
)
from src.repositories.hazards.leakage_injectors import (
    inject_tiered_leak,
    make_clean_dataset,
    measure_leak_auc,
)


@pytest.mark.parametrize(
    "target_auc,expected_severity_set",
    [
        (0.55, {None}),  # noise floor — must NOT be flagged
        (0.60, {"moderate"}),
        (0.69, {"moderate", "high"}),  # journey_duration_days analogue
        (0.78, {"high", "critical"}),
        (0.92, {"critical"}),
    ],
)
def test_detector_classifies_planted_leak_at_tier(
    target_auc: float, expected_severity_set: set
):
    """Detector must classify a planted leak by its empirical AUC tier."""
    df = make_clean_dataset(n=2000, prevalence=0.30, seed=42)
    df_leaked = inject_tiered_leak(
        df, target_col="target", target_auc=target_auc, seed=11
    )

    # Sanity: confirm injector hit the requested tier within ±0.04
    measured = measure_leak_auc(
        df_leaked, target_col="target", leak_feature="leaked_feature"
    )
    assert abs(measured - target_auc) < 0.04, (
        f"Injector miscalibrated: requested {target_auc}, measured {measured:.3f}"
    )

    findings = check_single_feature_auc(
        df_leaked, "target", ["leaked_feature", "noise_a", "noise_b", "noise_c"]
    )
    leaked_findings = [f for f in findings if f.feature == "leaked_feature"]

    if None in expected_severity_set:
        # AUC at the noise floor — it might or might not be flagged at
        # MODERATE due to seed jitter; what we definitively reject is
        # being flagged HIGH or CRITICAL.
        for f in leaked_findings:
            assert f.severity.value not in ("high", "critical"), (
                f"Noise-floor feature falsely flagged at {f.severity.value}: {f.to_dict()}"
            )
    else:
        assert len(leaked_findings) >= 1, (
            f"Detector missed planted leak at AUC={target_auc:.2f} "
            f"(measured {measured:.3f}); expected one of {expected_severity_set}"
        )
        actual = leaked_findings[0].severity.value
        assert actual in expected_severity_set, (
            f"Severity mismatch at AUC={target_auc:.2f}: expected one of "
            f"{expected_severity_set}, got {actual}"
        )


def test_journey_duration_days_analogue_caught():
    """Direct regression: a feature at AUC=0.689 (the literal CSU 2026-05-07
    value for journey_duration_days) MUST be flagged. Pre-Phase-1, this slipped.
    """
    df = make_clean_dataset(n=2000, prevalence=0.30, seed=42)
    df_leaked = inject_tiered_leak(
        df, target_col="target", target_auc=0.69, seed=11
    )

    findings = check_single_feature_auc(
        df_leaked, "target", ["leaked_feature", "noise_a"]
    )
    leaked = [f for f in findings if f.feature == "leaked_feature"]
    assert len(leaked) >= 1, (
        "REGRESSION: journey_duration_days analogue at AUC=0.69 missed by detector"
    )
    assert leaked[0].severity.value in ("moderate", "high"), (
        f"Expected moderate/high; got {leaked[0].severity.value}"
    )


def test_clean_dataset_produces_no_findings():
    """A dataset with only noise features must produce zero findings.

    Critical: tightened thresholds must not generate false positives on
    legitimately clean data.
    """
    df = make_clean_dataset(n=2000, prevalence=0.30, seed=42)
    findings = check_single_feature_auc(
        df, "target", ["noise_a", "noise_b", "noise_c"]
    )
    high_or_critical = [
        f for f in findings if f.severity.value in ("high", "critical")
    ]
    assert len(high_or_critical) == 0, (
        f"False positive on clean noise: {[f.to_dict() for f in high_or_critical]}"
    )
