"""Plan v3 §3 Tier 1B step 2 + step 5 — HBLP variance-inflation +
derivation-lineage audit contract tests.

Pins:
  - `hblp_effective_z_threshold` — math (variance inflation + Layer 1 prior)
  - `hblp_classify` — severity-band routing under HBLP-effective thresholds
  - `lineage_audit_declared_path` — declared-path validity check + scope
    note (does NOT prove "no undetected leakage")

Plan §6 Tier 1B Gate B1: HBLP can ship as DIAGNOSTIC with these helpers
alone (no enforcement). Gate B2 (quality uplift claim) requires
pre-specified ΔAUC≥0.03 + ECE/2 + stability/0.7 — separate measurement.
"""

from __future__ import annotations

import pytest

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    HIGH_Z,
    MODERATE_Z,
    T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER,
    T2_1B_HBLP_VARIANCE_INFLATION_REFERENCE_N,
    hblp_classify,
    hblp_effective_z_threshold,
    lineage_audit_declared_path,
)

# --------------------------------------------------------------------------- #
# Module constants                                                            #
# --------------------------------------------------------------------------- #


class TestT21BConstants:
    def test_variance_inflation_reference_n(self) -> None:
        """Plan v3 §3 Tier 1B step 2: HBLP relaxes for n_positives < 50."""
        assert T2_1B_HBLP_VARIANCE_INFLATION_REFERENCE_N == 50

    def test_declared_safe_prior_multiplier(self) -> None:
        assert T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER == 1.5

    def test_base_thresholds_unchanged(self) -> None:
        """HBLP layers ON TOP of HIGH_Z=5σ / MODERATE_Z=3σ; legacy
        constants are unchanged."""
        assert HIGH_Z == 5.0
        assert MODERATE_Z == 3.0


# --------------------------------------------------------------------------- #
# hblp_effective_z_threshold — math                                           #
# --------------------------------------------------------------------------- #


class TestHblpEffectiveZThreshold:
    def test_at_reference_n_no_inflation(self) -> None:
        """At n_positives = reference_n (50), inflation factor = 1.0."""
        result = hblp_effective_z_threshold(50, layer_1_declared_safe=False)
        assert result == pytest.approx(5.0)

    def test_above_reference_n_no_tightening(self) -> None:
        """At n_positives > reference_n, factor would be < 1.0; the
        max(1.0, ...) clamp ensures HBLP NEVER tightens below 5σ."""
        for n in [100, 200, 1000]:
            result = hblp_effective_z_threshold(n, layer_1_declared_safe=False)
            assert result == pytest.approx(5.0), f"HBLP must not tighten at n_positives={n}"

    def test_low_n_inflates_proportional_to_sqrt(self) -> None:
        """Plan v3 §3 Tier 1B step 2: inflation = sqrt(50/n_positives).
        n=22 → factor = sqrt(50/22) ≈ 1.508; threshold = 5.0 * 1.508 ≈ 7.54."""
        result = hblp_effective_z_threshold(22, layer_1_declared_safe=False)
        expected = 5.0 * (50.0 / 22.0) ** 0.5
        assert result == pytest.approx(expected, rel=1e-9)
        # Sanity: at the Optum n=1294 anchor (n_train_pos~22), HBLP raises
        # the bar from 5σ to ~7.54σ.
        assert 7.5 < result < 7.6

    def test_layer_1_declared_safe_multiplies_by_prior(self) -> None:
        """When Layer 1 manifest cleared the feature, threshold becomes
        threshold * 1.5 (default declared_safe_prior_multiplier)."""
        # At reference_n: 5.0 * 1.0 * 1.5 = 7.5
        result = hblp_effective_z_threshold(50, layer_1_declared_safe=True)
        assert result == pytest.approx(7.5)

    def test_layer_1_safe_combines_with_variance_inflation(self) -> None:
        """At n=22 + declared_safe: 5.0 * sqrt(50/22) * 1.5 ≈ 11.31."""
        result = hblp_effective_z_threshold(22, layer_1_declared_safe=True)
        expected = 5.0 * (50.0 / 22.0) ** 0.5 * 1.5
        assert result == pytest.approx(expected, rel=1e-9)
        assert 11.2 < result < 11.4

    def test_n_positives_zero_returns_base(self) -> None:
        """Degenerate: n_positives=0 → no permutation signal possible.
        Helper returns base threshold (caller's responsibility to short-
        circuit further)."""
        result = hblp_effective_z_threshold(0, layer_1_declared_safe=False)
        assert result == 5.0

    def test_negative_n_positives_returns_base(self) -> None:
        result = hblp_effective_z_threshold(-1, layer_1_declared_safe=True)
        assert result == 5.0

    def test_custom_base_threshold_propagates(self) -> None:
        """Caller override: a stricter base threshold (e.g., 6σ for a
        regulatory cohort) propagates through HBLP."""
        result = hblp_effective_z_threshold(22, False, base_threshold=6.0)
        expected = 6.0 * (50.0 / 22.0) ** 0.5
        assert result == pytest.approx(expected, rel=1e-9)

    def test_custom_reference_n_changes_inflation_curve(self) -> None:
        """Reference_n=100 → at n=22, inflation = sqrt(100/22) ≈ 2.13."""
        result = hblp_effective_z_threshold(
            22,
            False,
            variance_inflation_reference_n=100,
        )
        expected = 5.0 * (100.0 / 22.0) ** 0.5
        assert result == pytest.approx(expected, rel=1e-9)

    def test_custom_declared_safe_multiplier(self) -> None:
        """Caller can tighten the prior (e.g., 2.0x) for a stricter cohort."""
        result = hblp_effective_z_threshold(50, True, declared_safe_prior_multiplier=2.0)
        assert result == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# hblp_classify — severity routing                                            #
# --------------------------------------------------------------------------- #


class TestHblpClassify:
    def test_severity_high_when_z_above_effective_high(self) -> None:
        # n=50, declared_safe=False → high_eff = 5.0
        result = hblp_classify(7.0, n_positives=50, layer_1_declared_safe=False)
        assert result["severity"] == "high"
        assert result["effective_high_threshold"] == pytest.approx(5.0)

    def test_severity_moderate_when_z_between_bands(self) -> None:
        result = hblp_classify(4.0, n_positives=50, layer_1_declared_safe=False)
        assert result["severity"] == "moderate"
        # band: 3.0 < z=4.0 ≤ 5.0
        assert result["effective_moderate_threshold"] == pytest.approx(3.0)

    def test_severity_info_when_z_below_moderate(self) -> None:
        result = hblp_classify(2.0, n_positives=50, layer_1_declared_safe=False)
        assert result["severity"] == "info"

    def test_low_n_relaxes_severity(self) -> None:
        """At n=22, z=6.0 would be 'high' under base 5σ but 'moderate'
        under HBLP-effective ~7.54σ. This is exactly the plan §3 Tier 1B
        step 2 fix for the Optum n=1294 over-drop."""
        legacy_severity_at_z6_n22 = (
            "high" if 6.0 > HIGH_Z else "moderate" if 6.0 > MODERATE_Z else "info"
        )
        assert legacy_severity_at_z6_n22 == "high"  # legacy would drop

        hblp_result = hblp_classify(6.0, n_positives=22, layer_1_declared_safe=False)
        # HBLP raises bar to ~7.54σ → 6.0 < 7.54 → MODERATE not HIGH
        assert hblp_result["severity"] == "moderate"
        assert hblp_result["hblp_relaxed"] is True

    def test_layer_1_declared_safe_further_relaxes(self) -> None:
        """At n=22 + declared_safe, the bar is ~11.31σ. z=8 → moderate."""
        result = hblp_classify(8.0, n_positives=22, layer_1_declared_safe=True)
        assert result["severity"] == "moderate"
        assert result["effective_high_threshold"] > 11.0

    def test_high_n_does_not_relax(self) -> None:
        """At n=200 (above reference_n), HBLP returns 5σ unchanged."""
        result = hblp_classify(7.0, n_positives=200, layer_1_declared_safe=False)
        assert result["severity"] == "high"
        assert result["hblp_relaxed"] is False
        assert result["effective_high_threshold"] == pytest.approx(5.0)

    def test_non_finite_z_classifies_as_info(self) -> None:
        for z in [float("nan"), float("inf"), float("-inf"), None, "string"]:
            result = hblp_classify(z, n_positives=50, layer_1_declared_safe=False)  # type: ignore[arg-type]
            assert result["severity"] == "info"

    def test_returns_canonical_keys(self) -> None:
        result = hblp_classify(5.5, n_positives=22, layer_1_declared_safe=True)
        for key in (
            "severity",
            "effective_high_threshold",
            "effective_moderate_threshold",
            "base_threshold",
            "variance_inflation_factor",
            "layer_1_factor",
            "hblp_relaxed",
            "n_positives",
            "layer_1_declared_safe",
            "rationale",
        ):
            assert key in result, f"missing key {key!r}"

    def test_rationale_explains_decision(self) -> None:
        result = hblp_classify(7.0, n_positives=50, layer_1_declared_safe=False)
        assert "5.00σ" in result["rationale"] or "5.0σ" in result["rationale"]
        assert "z=7.00σ" in result["rationale"]


# --------------------------------------------------------------------------- #
# lineage_audit_declared_path                                                 #
# --------------------------------------------------------------------------- #


class TestLineageAuditDeclaredPath:
    def test_unknown_data_source_returns_no_contract(self) -> None:
        result = lineage_audit_declared_path("feature_x", data_source="nonexistent_cohort")
        assert result["contract_found"] is False
        assert result["declared_path_valid"] is None
        assert "MANIFEST_SOURCES" in result["rationale"]

    def test_none_data_source_returns_no_contract(self) -> None:
        result = lineage_audit_declared_path("feature_x", data_source=None)
        assert result["contract_found"] is False
        assert result["declared_path_valid"] is None

    def test_unknown_feature_in_known_source_returns_no_contract(self) -> None:
        """When the source registry exists but doesn't have the named
        feature, the audit is inconclusive (declared_path_valid=None)."""
        # csu IS in MANIFEST_SOURCES but doesn't have an arbitrary feature.
        result = lineage_audit_declared_path("definitely_not_a_real_feature", data_source="csu")
        assert result["contract_found"] is False
        assert result["declared_path_valid"] is None

    def test_returns_canonical_keys(self) -> None:
        result = lineage_audit_declared_path("x", data_source="csu")
        for key in (
            "feature_name",
            "data_source",
            "contract_found",
            "knowable_at_reference",
            "declared_path_valid",
            "rationale",
        ):
            assert key in result

    def test_audit_rationale_documents_decision(self) -> None:
        """Even when contract_found is False, rationale explains why
        (so an operator triaging knows whether to update MANIFEST_SOURCES
        or accept the inconclusive verdict)."""
        result = lineage_audit_declared_path("x", data_source="bogus")
        assert "MANIFEST_SOURCES" in result["rationale"]
        assert "audit declared-path" in result["rationale"]


# --------------------------------------------------------------------------- #
# Plan v3 §3 Tier 1B step 3 — Leakage-injection regression smoke              #
# --------------------------------------------------------------------------- #


class TestLeakageInjectionRegression:
    """Plan §3 Tier 1B step 3 negative control: a synthetic leak inserted
    into the data MUST still produce severity=high under HBLP-effective
    thresholds. This guards against the HBLP relaxation accidentally
    becoming a 'mask the leak by inflating the threshold' regression.

    Scope: this is the unit-level smoke that confirms the HBLP math
    keeps strong z-scores classified as high. The full integration-level
    regression (insert leak into a real cohort, run pipeline end-to-end,
    assert HBLP catches it) is a separate test that requires the
    pipeline runner — deferred to follow-on PR.
    """

    def test_strong_leak_classified_high_at_low_n_with_layer_1_clear(self) -> None:
        """Even at n=22 + declared_safe (HBLP-effective threshold ~11.31σ),
        a STRONG leak (z=15σ) MUST be classified as severity=high. This
        guards against runaway HBLP relaxation."""
        result = hblp_classify(15.0, n_positives=22, layer_1_declared_safe=True)
        assert result["severity"] == "high"

    def test_strong_leak_classified_high_at_high_n(self) -> None:
        """At n=200 + not-declared-safe (HBLP-effective = 5σ), a leak at
        z=10 MUST be classified as severity=high."""
        result = hblp_classify(10.0, n_positives=200, layer_1_declared_safe=False)
        assert result["severity"] == "high"

    def test_borderline_signal_at_low_n_is_correctly_relaxed(self) -> None:
        """The intended HBLP behavior: at n=22 the moderate band shifts
        proportionally with the high band (3σ * 7.54/5 ≈ 4.52σ). A
        borderline z=6.5σ is between moderate=4.52 and high=7.54 — IS
        moderate, NOT high. (Under legacy fixed 5σ it would be HIGH and
        get auto-dropped, which is the over-drop the plan §3 anchor
        flagged on Optum n=1294.)"""
        result = hblp_classify(6.5, n_positives=22, layer_1_declared_safe=False)
        assert result["severity"] == "moderate"

    def test_relaxation_factor_capped_so_extreme_leaks_still_caught(self) -> None:
        """At n=1 + declared_safe, the relaxation could blow up to
        absurd values. Test that even at extreme inflation, a 50σ leak
        is still classified high (no infinite relaxation)."""
        # n=1: inflation = sqrt(50/1) ≈ 7.07; * 1.5 = 10.6; * 5 = ~53
        # So a z=60σ leak should still be 'high'.
        result = hblp_classify(60.0, n_positives=1, layer_1_declared_safe=True)
        assert result["severity"] == "high"
