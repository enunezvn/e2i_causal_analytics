"""Gaps G6 + G12: surface the regulatory authorization manifest + an
advisory-vs-enforced gate map in a human/audit-facing report.

The deployer builds a frozen v5-C1 RegulatoryDeploymentManifest (SHA256 payload
+ Gate-N1 audit), but it reached humans ONLY via a test-only env-gated JSON; the
console printed a different, simpler deployment_manifest. And the advisory-vs-
enforced status of the quality gates (only minimum_auc is enforced; honest-band /
perm-anchored-AUC / Layer-3 / T2.6 bands are advisory) lived only in scattered
code comments. This pins a pure formatter that renders both.
"""

from __future__ import annotations

from src.agents.ml_foundation.model_deployer.regulatory_report import (
    ADVISORY_VS_ENFORCED,
    format_regulatory_report,
)


def _full_manifest() -> dict:
    return {
        "cohort": "csu",
        "in_c1_scope": True,
        "t2_6c_authorization_status": "authorized",
        "t2_6c_authorization_reasons": [],
        "roc_auc": 0.83,
        "honest_auc_band": [0.62, 0.68],
        "auc_in_band": False,
        "permutation_pvalue": 0.001,
        "signal_genuineness_category": "genuine",
        "calibration_ece": 0.05,
        "calibrated_ece": 0.03,
        "calibration_method_resolved": "isotonic",
        "calibration_quality_category": "good",
        "cv_stability_std_over_mean": 0.04,
        "cv_stability_category": "stable",
        "feature_surface_count": 6,
        "regulatory_eligibility_audit_present": True,
        "regulatory_eligibility_audit_fingerprint": "abc123",
        "regulatory_eligible": True,
        "adapted_regulatory_candidate": True,
        "manifest_sha256": "deadbeefcafe",
        "emitted_at": "2026-06-01T00:00:00Z",
    }


def test_advisory_map_marks_minimum_auc_enforced_others_advisory() -> None:
    by_gate = {g: status for g, status, _note in ADVISORY_VS_ENFORCED}
    assert by_gate["minimum_auc"] == "ENFORCED"
    # Every other listed gate is advisory (T2.6c-pending).
    advisory = {g for g, s, _ in ADVISORY_VS_ENFORCED if s == "ADVISORY"}
    assert {"honest_auc_band", "layer3_ablation", "cv_stability"} <= advisory
    assert "minimum_auc" not in advisory


def test_report_surfaces_manifest_fields() -> None:
    report = format_regulatory_report(_full_manifest())
    # Authorization decision + the binding fingerprint must be visible.
    assert "authorized" in report.lower()
    assert "deadbeefcafe" in report  # manifest_sha256
    assert "csu" in report.lower()
    # The honest-band + in-band verdict (the advisory AUC signal) shows.
    assert "0.62" in report and "0.68" in report
    # The enforcement map is embedded.
    assert "ENFORCED" in report and "ADVISORY" in report
    assert "minimum_auc" in report


def test_report_handles_missing_manifest() -> None:
    """No manifest (deployer didn't run / promotion not reached) → a clear
    message, but the static enforcement map is still shown."""
    report = format_regulatory_report(None)
    assert "no regulatory" in report.lower() or "not available" in report.lower()
    # The advisory-vs-enforced map is static and useful regardless.
    assert "minimum_auc" in report
    assert "ENFORCED" in report


def test_report_flags_out_of_band_auc() -> None:
    m = _full_manifest()
    report = format_regulatory_report(m)
    # auc_in_band False must be surfaced (advisory, not blocking) so a reader
    # sees the AUC is outside the honest band even though it didn't block.
    assert "0.83" in report  # the roc_auc value
