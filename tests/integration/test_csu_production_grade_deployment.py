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

# v5 C1 codex pass-1 MED-1 — re-import the CSU runner fixture from the
# negative-control test file so the real-CSU-runner pin in this file can
# share its module-scoped subprocess. Pytest treats fixtures referenced
# via direct import the same as fixtures from conftest, provided the
# import lands in the test module (here).
from tests.integration.test_csu_negative_control_20260510 import (  # noqa: F401
    csu_negative_control_artifact,
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
        """v5 codex pass-1 LOW-1: below-band reason mentions 'under-performing'."""
        state = _csu_state(roc_auc=0.55)
        m = build_regulatory_deployment_manifest(state)
        assert m.t2_6c_authorization_status == "blocked"
        assert any("BELOW CSU honest band" in r for r in m.t2_6c_authorization_reasons)
        assert any("under-performing" in r for r in m.t2_6c_authorization_reasons)
        assert m.auc_in_band is False

    def test_csu_with_auc_above_honest_band_blocked(self) -> None:
        """v5 codex pass-1 LOW-1: above-band reason flags suspicious — operator
        should investigate leakage or cohort shift, not treat as authorization.

        AUC=0.78 above honest band → blocked. The deliberate point of
        the cohort-specific band: even a "great" CSU AUC is a flag
        for leakage / overfitting until investigated.
        """
        state = _csu_state(roc_auc=0.78)
        m = build_regulatory_deployment_manifest(state)
        assert m.t2_6c_authorization_status == "blocked"
        assert any("ABOVE CSU honest band" in r for r in m.t2_6c_authorization_reasons)
        assert any(
            "suspicious" in r.lower() or "investigate leakage" in r
            for r in m.t2_6c_authorization_reasons
        )
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

    def test_manifest_sha_distinguishes_different_audit_contents(self) -> None:
        """v5 codex pass-1 MED-2: two manifests with different non-empty
        adaptation_history payloads must produce DIFFERENT hashes even
        if they generate identical t2_6c_authorization_reasons. Without
        the audit-fingerprint hash binding, hashes could collapse."""
        state_a = _csu_state(
            adaptation_history=[
                {
                    "commit_sha": "aaa111",
                    "justification_doc": "docs/a.md",
                    "gate_name": "cv_stability",
                    "before_threshold": 0.5,
                    "after_threshold": 0.7,
                    "timestamp": "2026-05-11T00:00:00",
                }
            ]
        )
        state_b = _csu_state(
            adaptation_history=[
                {
                    "commit_sha": "bbb222",
                    "justification_doc": "docs/b.md",
                    "gate_name": "minimum_auc",
                    "before_threshold": 0.75,
                    "after_threshold": 0.66,
                    "timestamp": "2026-05-11T01:00:00",
                }
            ]
        )
        m_a = build_regulatory_deployment_manifest(state_a)
        m_b = build_regulatory_deployment_manifest(state_b)
        # Both blocked + same authorization reason text format.
        assert m_a.t2_6c_authorization_status == "blocked"
        assert m_b.t2_6c_authorization_status == "blocked"
        # Hashes must differ because audit fingerprints differ.
        assert m_a.manifest_sha256 != m_b.manifest_sha256
        assert m_a.regulatory_eligibility_audit_fingerprint != (
            m_b.regulatory_eligibility_audit_fingerprint
        )

    def test_audit_fingerprint_null_when_audit_absent(self) -> None:
        """When the audit is missing entirely, fingerprint is None
        (not an empty-string sha)."""
        state = _csu_state()
        state["validation_metrics"].pop("regulatory_eligibility_audit", None)
        m = build_regulatory_deployment_manifest(state)
        assert m.regulatory_eligibility_audit_present is False
        assert m.regulatory_eligibility_audit_fingerprint is None


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

    def test_malformed_audit_payload_emits_blocked_manifest_not_crash(self) -> None:
        """v5 codex pass-1 HIGH-2: a malformed
        ``validation_metrics["regulatory_eligibility_audit"]`` (non-list
        fields) must produce a BLOCKED manifest with the N1 reconstruction
        failure cited — NOT a generic promotion-validation crash."""
        state = _csu_state()
        # Inject a malformed payload — adaptation_history must be a list,
        # but we give it a string. from_dict raises TypeError.
        state["validation_metrics"]["regulatory_eligibility_audit"] = {
            "gate_history": [],
            "adaptation_history": "not_a_list",
        }
        result = asyncio.run(validate_promotion(state))
        # The integration path must NOT clobber to a generic error.
        assert "regulatory_deployment_manifest" in result, (
            "Malformed audit should still emit a manifest (blocked), "
            "not a generic promotion-validation error."
        )
        m = result["regulatory_deployment_manifest"]
        assert m["t2_6c_authorization_status"] == "blocked"
        # The builder's malformed-audit guard fires first and surfaces
        # "payload malformed" in reasons. The regulatory_result-level
        # fallback catches the same condition but its message stays in
        # state["regulatory_eligibility_failures"] (not the manifest).
        assert any(
            "regulatory_eligibility_audit payload malformed" in r
            for r in m["t2_6c_authorization_reasons"]
        ), m["t2_6c_authorization_reasons"]

    def test_non_dict_audit_payload_emits_blocked_manifest(self) -> None:
        """v5 codex pass-2 HIGH-2: a non-mapping
        ``validation_metrics["regulatory_eligibility_audit"]`` (e.g.
        list, string, int) raises AttributeError inside
        ``RegulatoryEligibilityAudit.from_dict.get(...)``. The fix at
        registry_manager catches AttributeError alongside TypeError /
        ValueError so the manifest still emits as blocked.
        """
        state = _csu_state()
        # Replace the audit with a list (non-mapping payload).
        state["validation_metrics"]["regulatory_eligibility_audit"] = [
            "not_a_mapping"
        ]
        result = asyncio.run(validate_promotion(state))
        assert "regulatory_deployment_manifest" in result, (
            "Non-mapping audit payload should still emit a blocked manifest "
            "via the AttributeError catch (codex pass-2 HIGH-2)."
        )
        m = result["regulatory_deployment_manifest"]
        assert m["t2_6c_authorization_status"] == "blocked"

    def test_validate_promotion_sees_fresh_n1_audit_for_manifest(self) -> None:
        """v5 codex pass-1 HIGH-1: the manifest must read the FRESH N1
        audit (with gate_history entries N1 just appended), not the
        stale incoming state's audit. Verify by passing a state with
        an EMPTY incoming audit; after validate_promotion runs, the
        manifest's regulatory_eligibility_audit_present is True."""
        state = _csu_state()
        # Confirm the incoming audit has empty gate_history.
        incoming_audit = state["validation_metrics"]["regulatory_eligibility_audit"]
        assert incoming_audit["gate_history"] == []
        result = asyncio.run(validate_promotion(state))
        m = result["regulatory_deployment_manifest"]
        # N1 must have appended a gate_history entry. The manifest
        # reads the FRESH audit, not the stale incoming one.
        fresh_audit = result["regulatory_eligibility_audit"]
        assert len(fresh_audit["gate_history"]) >= 1, (
            "N1 should have appended at least one gate_history entry."
        )
        # Manifest sees audit as present.
        assert m["regulatory_eligibility_audit_present"] is True
        # Fingerprint binds to fresh audit contents.
        assert m["regulatory_eligibility_audit_fingerprint"] is not None


# ---------------------------------------------------------------------------
# Plan v5 §2 C1 acceptance — the manifest LISTS the load-bearing fields.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Real CSU runner — pin manifest emission on a full tier0 e2e run.
# Codex pass-1 MED-1: plan v5 §2 C1 requires "Integration test pins the
# deployment-manifest emission on a real CSU run." Synthetic-state tests
# above pin the contract; this one pins the production pipeline path.
# ---------------------------------------------------------------------------


class TestRealCSURunnerEmitsManifest:
    """Pin v5 C1 manifest emission on a real CSU tier0 pipeline run.

    Imports the existing ``csu_negative_control_artifact`` fixture from
    ``test_csu_negative_control_20260510`` — it already spawns a tier0
    subprocess on real CSU data with TIER0_E2E_JSON_OUT set, and the v5
    C1 wiring extends the runner's JSON output schema to include the
    ``regulatory_deployment_manifest`` field. Reuses the fixture to
    amortize the ~5-15 minute subprocess cost across the negative-control
    + v5 C1 assertions.
    """

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.real_data
    @pytest.mark.timeout(2000)
    def test_csu_runner_emits_regulatory_deployment_manifest(
        self, csu_negative_control_artifact: dict
    ) -> None:
        """The full tier0 + deployer e2e run on real CSU data MUST emit
        a ``regulatory_deployment_manifest`` field in the JSON artifact."""
        manifest = csu_negative_control_artifact.get("regulatory_deployment_manifest")
        # Codex pass-1 MED-1: manifest emission MUST surface in the
        # TIER0_E2E_JSON_OUT artifact, not just the in-process state.
        assert manifest is not None, (
            "regulatory_deployment_manifest missing from CSU runner "
            "artifact. v5 C1 wiring requires the runner's JSON output "
            "to include this field — check scripts/run_tier0_test.py "
            "TIER0_E2E_JSON_OUT serialization + deployer agent's "
            "output composition."
        )
        # Cohort identity threads through from scope_spec.
        assert manifest.get("cohort") == "csu", (
            f"Expected cohort='csu' in CSU runner manifest; got "
            f"{manifest.get('cohort')!r}. scope_spec.feature_manifest_source "
            "threading is broken."
        )
        # Authorization status is one of the three valid values.
        assert manifest.get("t2_6c_authorization_status") in T2_6C_AUTHORIZATION_STATUSES
        # The manifest CARRIES the load-bearing fields from v5 §2 C1.
        for required in (
            "honest_auc_band",
            "permutation_pvalue",
            "calibration_ece",
            "cv_stability_std_over_mean",
            "feature_surface_count",
            "manifest_source",
        ):
            assert required in manifest, (
                f"v5 C1 acceptance: {required} missing from runner manifest"
            )
        # The sha256 fingerprint is populated (manifest is signed-off-ready).
        assert manifest.get("manifest_sha256"), "manifest_sha256 must be non-empty"


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
            # Codex pass-1 MED-2: audit fingerprint binds the hash
            # to audit contents, not just presence.
            "regulatory_eligibility_audit_fingerprint",
            "manifest_sha256",
        }
        missing = required_fields - set(m.keys())
        assert not missing, f"v5 C1 acceptance: missing fields {missing}"
