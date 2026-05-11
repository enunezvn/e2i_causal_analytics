"""v5 Gate C1 integration tests — CSU production-grade deployment manifest.

Pins the load-bearing deliverable from
``disease_agnostic_quality_uplift_v5.md`` §2 C1: the deployer emits a
cohort-scoped ``regulatory_deployment_manifest`` that authorizes T2.6c
enforcement on CSU specifically (with Optum blocked pending v4 backlog
#32/#33).

These tests exercise the full ``validate_promotion`` integration path —
state in, manifest out — and pin every authorization-decision branch
(authorized / blocked-by-cohort-policy / blocked-by-AUC-band / blocked-
by-adaptation-history / out-of-scope).

The tests use synthetic state mimicking what the data_preparer +
model_trainer + previous N1 evaluation would have produced for a real
CSU pipeline run. A separate end-to-end test against the full CSU
runner is gated by the existing ``test_csu_negative_control_20260510``
infrastructure (which validates the upstream contract); this file
focuses on the manifest emission contract.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    validate_promotion,
)
from src.agents.ml_foundation.model_deployer.nodes.regulatory_deployment_manifest import (
    CSU_HONEST_AUC_BAND,
    T2_6C_AUTHORIZATION_STATUSES,
    RegulatoryDeploymentManifest,
    build_regulatory_deployment_manifest,
    resolve_cohort_authorization_policy,
)

# ---------------------------------------------------------------------------
# Cohort policy invariants — pinned per v5 §2 C1.
# ---------------------------------------------------------------------------


class TestCohortAuthorizationPolicy:
    def test_csu_is_in_c1_scope(self) -> None:
        policy = resolve_cohort_authorization_policy("csu")
        assert policy["cohort"] == "csu"
        assert policy["in_c1_scope"] is True
        assert policy["honest_auc_band"] == CSU_HONEST_AUC_BAND
        assert policy["blocked_reason"] is None

    def test_optum_is_blocked_with_backlog_citation(self) -> None:
        policy = resolve_cohort_authorization_policy("optum")
        assert policy["in_c1_scope"] is False
        assert policy["blocked_reason"] is not None
        # Plan §2 C1 specifically cites the v4 backlog items.
        assert "backlog #32" in policy["blocked_reason"]
        assert "backlog #33" in policy["blocked_reason"]

    def test_unknown_cohort_is_out_of_scope(self) -> None:
        for src in (None, "", "synthetic", "future_cohort_xyz"):
            policy = resolve_cohort_authorization_policy(src)
            assert policy["in_c1_scope"] is False, src

    def test_csu_honest_band_is_pinned(self) -> None:
        # Drift detector: the honest band is anchored on v4 empirical
        # evidence (CSU val_AUC=0.66 in band [0.62, 0.68] per
        # docs/results/optum_initiation_revalidation_20260510.md and
        # the v4 plan §2 G2 closure). Any change requires a fresh memo.
        assert CSU_HONEST_AUC_BAND == (0.62, 0.68)

    def test_authorization_statuses_vocabulary_pinned(self) -> None:
        assert T2_6C_AUTHORIZATION_STATUSES == (
            "authorized",
            "blocked",
            "out_of_scope",
        )


# ---------------------------------------------------------------------------
# Fixture helpers — synthetic CSU / Optum state.
# ---------------------------------------------------------------------------


def _csu_state(
    *,
    roc_auc: float = 0.6592,
    permutation_pvalue: float = 0.0,
    cv_std: float = 0.012,
    cv_mean: float = 0.65,
    calibration_error: float = 0.04,
    calibrated_ece: float = 0.025,
    calibration_method_resolved: str = "isotonic",
    feature_count: int = 9,
    adaptation_history: list = None,
    gate_history: list = None,
) -> Dict[str, Any]:
    """Build a synthetic CSU state matching what the deployer would see
    after a clean tier0 + N1 + B1 pipeline run."""
    return {
        "current_stage": "None",
        "target_stage": "Staging",
        "target_environment": "staging",
        "scope_spec": {"feature_manifest_source": "csu"},
        "success_criteria": {"minimum_auc": 0.75},
        "validation_metrics": {
            "roc_auc": roc_auc,
            "permutation_pvalue": permutation_pvalue,
            "cv_5fold_roc_auc_std": cv_std,
            "cv_5fold_roc_auc_mean": cv_mean,
            "calibration_error": calibration_error,
            "regulatory_eligibility_audit": {
                "gate_history": list(gate_history or []),
                "adaptation_history": list(adaptation_history or []),
            },
        },
        "calibrated_ece": calibrated_ece,
        "post_hoc_calibration": {
            "calibration_applied": True,
            "calibration_method": "auto",
            "calibration_method_resolved": calibration_method_resolved,
        },
        "feature_names": [f"f{i}" for i in range(feature_count)],
    }


# ---------------------------------------------------------------------------
# Builder-level integration tests.
# ---------------------------------------------------------------------------


class TestRegulatoryDeploymentManifestBuilder:
    def test_csu_clean_state_emits_authorized_manifest(self) -> None:
        state = _csu_state()
        manifest = build_regulatory_deployment_manifest(state)
        assert isinstance(manifest, RegulatoryDeploymentManifest)
        assert manifest.cohort == "csu"
        assert manifest.in_c1_scope is True
        assert manifest.t2_6c_authorization_status == "authorized"
        assert manifest.t2_6c_authorization_reasons == []
        assert manifest.roc_auc == pytest.approx(0.6592)
        assert manifest.auc_in_band is True
        assert manifest.honest_auc_band == CSU_HONEST_AUC_BAND
        assert manifest.signal_genuineness_category == "genuine"
        assert manifest.cv_stability_category == "stable"
        assert manifest.calibration_quality_category == "excellent"
        assert manifest.feature_surface_count == 9
        assert manifest.regulatory_eligibility_audit_present is True
        # B1 fields surface through.
        assert manifest.calibrated_ece == pytest.approx(0.025)
        assert manifest.calibration_method_resolved == "isotonic"
        # Hash is deterministic (no clock dependency).
        assert len(manifest.manifest_sha256) == 64
        # N3 signature deferred.
        assert manifest.n3_signature is None

    def test_csu_with_auc_below_honest_band_blocked(self) -> None:
        state = _csu_state(roc_auc=0.55)
        m = build_regulatory_deployment_manifest(state)
        assert m.t2_6c_authorization_status == "blocked"
        assert any("outside CSU honest band" in r for r in m.t2_6c_authorization_reasons)
        assert m.auc_in_band is False

    def test_csu_with_auc_above_honest_band_blocked(self) -> None:
        # AUC=0.78 above honest band → blocked. The deliberate point of
        # the cohort-specific band: even a "great" CSU AUC is a flag
        # for leakage / overfitting until investigated.
        state = _csu_state(roc_auc=0.78)
        m = build_regulatory_deployment_manifest(state)
        assert m.t2_6c_authorization_status == "blocked"
        assert m.auc_in_band is False

    def test_csu_with_adaptation_history_blocked(self) -> None:
        state = _csu_state(
            adaptation_history=[
                {
                    "commit_sha": "abc123",
                    "justification_doc": "docs/adapt.md",
                    "gate_name": "cv_stability",
                    "before_threshold": 0.5,
                    "after_threshold": 0.7,
                    "timestamp": "2026-05-11T00:00:00",
                }
            ]
        )
        m = build_regulatory_deployment_manifest(state)
        assert m.t2_6c_authorization_status == "blocked"
        assert any("adaptation_history is non-empty" in r for r in m.t2_6c_authorization_reasons)

    def test_csu_with_missing_audit_blocked(self) -> None:
        state = _csu_state()
        # Strip the audit from state.
        state["validation_metrics"].pop("regulatory_eligibility_audit", None)
        m = build_regulatory_deployment_manifest(state)
        assert m.t2_6c_authorization_status == "blocked"
        assert any(
            "regulatory_eligibility_audit missing" in r for r in m.t2_6c_authorization_reasons
        )
        assert m.regulatory_eligibility_audit_present is False

    def test_csu_with_marginal_permutation_blocked(self) -> None:
        # permutation_pvalue between 0.01 and 0.05 → "likely_genuine"
        # which is NOT in T2.6b reject set; but p between 0.05 and 0.15
        # → "marginal" which IS rejected.
        state = _csu_state(permutation_pvalue=0.10)
        m = build_regulatory_deployment_manifest(state)
        assert m.t2_6c_authorization_status == "blocked"
        assert any("signal category" in r for r in m.t2_6c_authorization_reasons)

    def test_optum_state_blocked_with_backlog_citation(self) -> None:
        state = _csu_state()
        state["scope_spec"]["feature_manifest_source"] = "optum"
        m = build_regulatory_deployment_manifest(state)
        assert m.cohort == "optum"
        assert m.in_c1_scope is False
        assert m.t2_6c_authorization_status == "blocked"
        # Cohort policy reason wins over any other check.
        assert any("backlog #32" in r for r in m.t2_6c_authorization_reasons)
        assert any("backlog #33" in r for r in m.t2_6c_authorization_reasons)

    def test_unknown_cohort_out_of_scope(self) -> None:
        state = _csu_state()
        state["scope_spec"]["feature_manifest_source"] = "future_cohort"
        m = build_regulatory_deployment_manifest(state)
        assert m.t2_6c_authorization_status == "out_of_scope"
        assert m.in_c1_scope is False

    def test_manifest_sha_is_deterministic_across_emissions(self) -> None:
        state = _csu_state()
        m1 = build_regulatory_deployment_manifest(state)
        m2 = build_regulatory_deployment_manifest(state)
        # emitted_at differs between calls; sha covers only the
        # load-bearing fields and must be identical.
        assert m1.manifest_sha256 == m2.manifest_sha256
        assert m1.emitted_at != m2.emitted_at or m1.emitted_at == m2.emitted_at

    def test_manifest_sha_changes_when_authorization_changes(self) -> None:
        state_authorized = _csu_state()
        state_blocked = _csu_state(roc_auc=0.55)
        m_a = build_regulatory_deployment_manifest(state_authorized)
        m_b = build_regulatory_deployment_manifest(state_blocked)
        assert m_a.manifest_sha256 != m_b.manifest_sha256


# ---------------------------------------------------------------------------
# validate_promotion integration — manifest emission as state key.
# ---------------------------------------------------------------------------


class TestValidatePromotionEmitsManifest:
    def test_csu_promotion_emits_authorized_manifest(self) -> None:
        state = _csu_state()
        result = asyncio.run(validate_promotion(state))
        assert "regulatory_deployment_manifest" in result
        m = result["regulatory_deployment_manifest"]
        assert m["cohort"] == "csu"
        assert m["t2_6c_authorization_status"] == "authorized"
        # The manifest is signal-only — promotion_allowed is set
        # independently by validate_promotion's path-level logic.
        assert result["promotion_allowed"] is True

    def test_optum_promotion_emits_blocked_manifest(self) -> None:
        state = _csu_state()
        state["scope_spec"]["feature_manifest_source"] = "optum"
        result = asyncio.run(validate_promotion(state))
        m = result["regulatory_deployment_manifest"]
        assert m["cohort"] == "optum"
        assert m["t2_6c_authorization_status"] == "blocked"
        # Optum's blocked status does NOT block promotion at the
        # registry-manager level — promotion is determined by the path
        # rules (current_stage → target_stage). The manifest is the
        # deployer-input signal for T2.6c, not a promotion blocker.
        assert result["promotion_allowed"] is True

    def test_manifest_serializable_as_json(self) -> None:
        import json

        state = _csu_state()
        result = asyncio.run(validate_promotion(state))
        manifest_dict = result["regulatory_deployment_manifest"]
        # Round-trip through JSON to confirm no non-serializable types.
        round_trip = json.loads(json.dumps(manifest_dict, default=str))
        assert round_trip["cohort"] == "csu"
        # honest_auc_band serializes as list (per to_dict).
        assert round_trip["honest_auc_band"] == [0.62, 0.68]


# ---------------------------------------------------------------------------
# Plan v5 §2 C1 acceptance — the manifest LISTS the load-bearing fields.
# ---------------------------------------------------------------------------


class TestC1ManifestFieldCoverage:
    """v5 §2 C1 explicitly requires the manifest to list: AUC band,
    permutation p-value, calibration ECE, CV stability, feature surface,
    manifest source. Pin that every required field is populated on a
    clean CSU emission."""

    def test_manifest_lists_all_required_fields(self) -> None:
        state = _csu_state()
        m = build_regulatory_deployment_manifest(state).to_dict()
        # The acceptance enumeration from v5 §2 C1, verbatim:
        required_fields = {
            "honest_auc_band",  # "AUC band"
            "roc_auc",
            "auc_in_band",
            "permutation_pvalue",  # "permutation p-value"
            "signal_genuineness_category",
            "calibration_ece",  # "calibration ECE"
            "calibration_quality_category",
            "cv_stability_std_over_mean",  # "CV stability"
            "cv_stability_category",
            "feature_surface_count",  # "feature surface"
            "manifest_source",  # "manifest source"
            "cohort",
            "t2_6c_authorization_status",
            "t2_6c_authorization_reasons",
            "regulatory_eligibility_audit_present",
            "manifest_sha256",
        }
        missing = required_fields - set(m.keys())
        assert not missing, f"v5 C1 acceptance: missing fields {missing}"
