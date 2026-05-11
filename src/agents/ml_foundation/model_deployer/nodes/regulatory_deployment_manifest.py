"""Gate C1 (plan v5 §2) — CSU regulatory deployment manifest emission.

The load-bearing v5 deliverable: a cohort-scoped deployment manifest the
deployer can consume to authorize T2.6c enforcement on CSU specifically.
Optum remains MARGINAL/blocked pending v4 backlog #32 (Candidate B
external-reviewer onboarding) and #33 (Optum cohort expansion to
n_pos ≥ 150).

This module is NOT a parallel audit trail — it builds ON TOP of the
existing Gate N1 ``RegulatoryEligibilityAudit`` + T2.6a deployer-input
metrics. The manifest IS the cohort-aware deployment-authorization
payload the operator attaches to a deployment PR; the underlying
audit + threshold-provenance machinery is unchanged.

Per plan v5 §2 C1 acceptance:

  * CSU pipeline run produces a ``regulatory_eligibility_audit``
    artifact per N1's contract — REUSES existing N1 wiring.
  * The C1 manifest emitted here lists: AUC band, permutation
    p-value, calibration ECE, CV stability, feature surface,
    manifest source.
  * Deployer can consume this to authorize T2.6c enforcement on
    CSU; Optum-flagged state produces a manifest with
    ``t2_6c_authorization_status="blocked"`` + reason citing the
    relevant v4 backlog item.
  * N3 INTERIM precedent applies — cryptographic-signature
    infrastructure remains deferred to ``v4-N3-signature-infra``.

Plan reference: ``.claude/plans/disease_agnostic_quality_uplift_v5.md`` §2 C1.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional

from src.agents.ml_foundation.model_deployer.regulatory_audit import (
    RegulatoryEligibilityAudit,
)

# Public API surface — the dataclass + builder + cohort policy.
__all__ = [
    "CSU_HONEST_AUC_BAND",
    "OPTUM_HONEST_AUC_BAND",
    "T2_6C_AUTHORIZATION_STATUSES",
    "RegulatoryDeploymentManifest",
    "build_regulatory_deployment_manifest",
    "resolve_cohort_authorization_policy",
]


# --------------------------------------------------------------------------- #
# Cohort-specific honest-AUC bands.                                            #
#                                                                              #
# These are the literature-anchored bands the model must land in to be         #
# considered for T2.6c enforcement. Outside the band → manifest fails        #
# closed regardless of how the deployer-input categories look.                 #
# --------------------------------------------------------------------------- #

CSU_HONEST_AUC_BAND: tuple[float, float] = (0.62, 0.68)
"""CSU treatment_initiated honest band per v4 empirical anchor."""

OPTUM_HONEST_AUC_BAND: tuple[float, float] = (0.62, 0.68)
"""Optum initiation honest band; same domain-typical literature anchor."""


# --------------------------------------------------------------------------- #
# T2.6c authorization-status vocabulary.                                       #
# --------------------------------------------------------------------------- #

T2_6C_AUTHORIZATION_STATUSES: tuple[str, ...] = (
    "authorized",  # Cohort cleared every C1 acceptance gate.
    "blocked",  # Cohort failed one or more C1 acceptance gates.
    "out_of_scope",  # Cohort is not the C1 deliverable target.
)

_T2_6C_AUTHORIZATION_STATUS_T = Literal["authorized", "blocked", "out_of_scope"]


# --------------------------------------------------------------------------- #
# Cohort policy resolution.                                                    #
# --------------------------------------------------------------------------- #


def resolve_cohort_authorization_policy(
    cohort_manifest_source: Optional[str],
) -> Dict[str, Any]:
    """Return the cohort-scoped authorization policy.

    v5 §2 C1 establishes CSU as the load-bearing deployment target and
    blocks Optum pending v4 backlog #32/#33. Unknown / missing cohort
    manifest sources are treated as ``out_of_scope`` — the C1 deliverable
    is cohort-aware by design.

    Args:
        cohort_manifest_source: ``"csu"``, ``"optum"``, or any other
            value (including ``None``).

    Returns:
        A dict with::

            "cohort":          the normalized cohort identifier
                               ("csu" | "optum" | "<unknown>").
            "in_c1_scope":     whether v5 C1 targets this cohort.
            "honest_auc_band": the (lo, hi) tuple to enforce, or None
                               if cohort is out of C1 scope.
            "blocked_reason":  None when in C1 scope and cohort is the
                               deliverable target; otherwise a
                               human-readable string citing the
                               specific gating constraint (e.g.
                               "v4 backlog #32 + #33" for Optum).

    Pure function — no I/O, no state mutation.
    """
    norm = (cohort_manifest_source or "").strip().lower()
    if norm == "csu":
        return {
            "cohort": "csu",
            "in_c1_scope": True,
            "honest_auc_band": CSU_HONEST_AUC_BAND,
            "blocked_reason": None,
        }
    if norm == "optum":
        return {
            "cohort": "optum",
            "in_c1_scope": False,
            "honest_auc_band": OPTUM_HONEST_AUC_BAND,
            "blocked_reason": (
                "Optum is MARGINAL: permutation p>0.05 (RANDOM) at default "
                "window n=1294. Authorization requires (a) cohort expansion "
                "to n_pos ≥ 150 (v4 backlog #33) OR (b) external-reviewer "
                "signoff on relaxed PRE/POST window n=1697 (v4 backlog #32). "
                "Neither lands inside v5 C1 scope."
            ),
        }
    return {
        "cohort": norm or "<unknown>",
        "in_c1_scope": False,
        "honest_auc_band": None,
        "blocked_reason": (
            f"Cohort manifest_source={cohort_manifest_source!r} is out of v5 "
            "C1 scope. The C1 deliverable is CSU-specific; other cohorts "
            "require their own production-grade deployment plan."
        ),
    }


# --------------------------------------------------------------------------- #
# Manifest dataclass.                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegulatoryDeploymentManifest:
    """v5 C1 deployment manifest — cohort-scoped T2.6c authorization payload.

    The manifest IS the load-bearing artifact the operator attaches to a
    deployment PR. Frozen dataclass so once built, fields are immutable;
    the manifest captures a point-in-time snapshot of every quality signal
    the deployer must consume before granting T2.6c enforcement.

    Fields:

    * ``cohort`` — normalized cohort identifier (``"csu"`` | ``"optum"`` |
      ``"<unknown>"``).
    * ``manifest_source`` — original ``scope_spec.feature_manifest_source``
      value (raw, for audit trail).
    * ``in_c1_scope`` — whether v5 C1 targets this cohort. Optum is False
      pending v4 backlog #32/#33.
    * ``t2_6c_authorization_status`` — one of ``"authorized"``,
      ``"blocked"``, ``"out_of_scope"``.
    * ``t2_6c_authorization_reasons`` — human-readable list of decision
      drivers (empty list when authorized).
    * ``roc_auc`` — held-out test AUC (float or None).
    * ``honest_auc_band`` — (lo, hi) tuple the AUC must land in.
    * ``auc_in_band`` — bool indicating whether ``roc_auc`` lands in
      ``honest_auc_band``.
    * ``permutation_pvalue`` — Layer 3 perm-null p-value.
    * ``signal_genuineness_category`` — categorical band from T2.6a
      (``"genuine" | "likely_genuine" | "marginal" | "random" |
      "degenerate"``).
    * ``calibration_ece`` — pre-calibration ECE.
    * ``calibrated_ece`` — post-calibration ECE (B1 wired). May be None
      if calibration was skipped.
    * ``calibration_method_resolved`` — B1 audit-trail field (e.g.,
      ``"isotonic"`` / ``"sigmoid"``). None when calibration skipped.
    * ``calibration_quality_category`` — categorical band from T2.6a.
    * ``cv_stability_std_over_mean`` — CV std/mean ratio.
    * ``cv_stability_category`` — categorical band from T2.6a.
    * ``feature_surface_count`` — number of retained features at the
      time of training (data-preparer output).
    * ``regulatory_eligibility_audit_present`` — bool indicating
      whether an N1 audit was found in state.
    * ``regulatory_eligible`` — N1 verdict (True iff all preconditions).
    * ``adapted_regulatory_candidate`` — N1 candidate verdict.
    * ``manifest_sha256`` — sha256 of the canonical JSON form for
      deployment-PR attachment.
    * ``emitted_at`` — UTC ISO-8601 timestamp.
    * ``n3_signature`` — None per N3 INTERIM precedent (deferred to
      ``v4-N3-signature-infra``).
    """

    cohort: str
    manifest_source: Optional[str]
    in_c1_scope: bool
    t2_6c_authorization_status: _T2_6C_AUTHORIZATION_STATUS_T
    t2_6c_authorization_reasons: List[str]
    roc_auc: Optional[float]
    honest_auc_band: Optional[tuple[float, float]]
    auc_in_band: Optional[bool]
    permutation_pvalue: Optional[float]
    signal_genuineness_category: Optional[str]
    calibration_ece: Optional[float]
    calibrated_ece: Optional[float]
    calibration_method_resolved: Optional[str]
    calibration_quality_category: Optional[str]
    cv_stability_std_over_mean: Optional[float]
    cv_stability_category: Optional[str]
    feature_surface_count: Optional[int]
    regulatory_eligibility_audit_present: bool
    regulatory_eligible: bool
    adapted_regulatory_candidate: bool
    manifest_sha256: str = field(default="")
    emitted_at: str = field(default="")
    n3_signature: Optional[str] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep-copy JSON-serializable snapshot."""
        out = asdict(self)
        # Tuple → list for JSON-friendly shape (frozen tuples don't
        # serialize round-trip through json.loads → list).
        if out["honest_auc_band"] is not None:
            out["honest_auc_band"] = list(out["honest_auc_band"])
        return copy.deepcopy(out)


# --------------------------------------------------------------------------- #
# Builder.                                                                     #
# --------------------------------------------------------------------------- #


def _extract_roc_auc(validation_metrics: Mapping[str, Any]) -> Optional[float]:
    """Read held-out test AUC. validation_metrics may use either
    ``roc_auc`` (modern producer key) or ``auc_roc`` (canonical schema
    name) — both are accepted at MetricsSchema construction so the
    deployer must read both."""
    value = validation_metrics.get("roc_auc")
    if value is None:
        value = validation_metrics.get("auc_roc")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_cohort_manifest_source(state: Mapping[str, Any]) -> Optional[str]:
    """Find ``scope_spec.feature_manifest_source`` on state.

    The scope_spec lives on data_preparer state and is threaded through
    later agents; in the deployer it surfaces either directly on
    ``state["scope_spec"]`` (when the deployer ran in the same pipeline)
    OR via a flattened ``state["feature_manifest_source"]`` (for
    standalone invocations / checkpoint replays).
    """
    scope_spec = state.get("scope_spec")
    if hasattr(scope_spec, "model_dump"):
        scope_spec = scope_spec.model_dump()
    if isinstance(scope_spec, dict):
        ms = scope_spec.get("feature_manifest_source")
        if ms is not None:
            return str(ms)
    ms = state.get("feature_manifest_source")
    return str(ms) if ms is not None else None


def _extract_feature_surface_count(state: Mapping[str, Any]) -> Optional[int]:
    """Number of retained features at training time.

    Reads ``state["feature_names"]`` (data_preparer output) when present,
    falling back to ``state["validation_metrics"]["n_features"]`` when
    available. Returns None when neither is set.
    """
    feature_names = state.get("feature_names")
    if isinstance(feature_names, (list, tuple)):
        return len(feature_names)
    vm = state.get("validation_metrics") or {}
    if hasattr(vm, "model_dump"):
        vm = vm.model_dump()
    if isinstance(vm, dict):
        n = vm.get("n_features")
        if n is not None:
            try:
                return int(n)
            except (TypeError, ValueError):
                return None
    return None


def _hash_manifest(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON form (sort_keys=True, no whitespace).

    Hash excludes the ``manifest_sha256``, ``emitted_at``, and
    ``n3_signature`` fields — those are emitted-time metadata. Hashing
    only the load-bearing payload means two manifests built from the
    same validation_metrics + audit produce the same hash regardless of
    when they were emitted.
    """
    canonical = {
        k: v
        for k, v in payload.items()
        if k not in ("manifest_sha256", "emitted_at", "n3_signature")
    }
    payload_str = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def build_regulatory_deployment_manifest(
    state: Mapping[str, Any],
) -> RegulatoryDeploymentManifest:
    """Build the v5 C1 deployment manifest from agent state.

    Composes the load-bearing fields from existing sources:

      * Cohort + manifest source: ``state["scope_spec"]["feature_manifest_source"]``.
      * AUC: ``state["validation_metrics"]["roc_auc"]`` (or ``auc_roc``).
      * T2.6a categories: re-computed from ``state["validation_metrics"]``
        + ``state["calibration_error"]`` via ``compute_deployer_input_metrics``.
      * N1 verdict + audit presence: from
        ``state["regulatory_eligible"]`` and
        ``state["validation_metrics"]["regulatory_eligibility_audit"]``.
      * Calibration B1 fields: from
        ``state["post_hoc_calibration"]["calibration_method_resolved"]``
        + ``state["calibrated_ece"]``.

    The T2.6c authorization decision is built deterministically from:

      1. Cohort policy (CSU in scope; Optum + others out).
      2. AUC band check (auc_in_band must be True — uses cohort-
         specific honest band, NOT the universal N1 literature anchor
         of 0.75 which CSU's AUC=0.66 cannot clear by design).
      3. T2.6a categories (signal genuine, calibration acceptable,
         cv-stability not in T2.6b reject sets).
      4. N1 audit-trail cleanliness: ``regulatory_eligibility_audit``
         present AND its ``adaptation_history`` is empty. This is the
         load-bearing N1 contract: no adaptive threshold relaxation
         happened during the model's lifecycle. NOTE: this is a
         WEAKER signal than ``regulatory_eligible=True`` (which also
         requires the universal 0.75 literature anchor to clear) —
         C1 deliberately operates against the cohort honest band
         (CSU [0.62, 0.68]) per v5 §2 C1.

    If ANY check fails → status="blocked" with the failing reason(s)
    enumerated in ``t2_6c_authorization_reasons``. CSU-scope + all
    checks passing → status="authorized". Non-CSU cohorts →
    status="out_of_scope".

    The manifest is built deterministically from state — no I/O, no
    randomness. Safe to call multiple times; identical state produces
    identical manifests (sha256 stable across re-emission).
    """
    # Avoid a circular import — ``registry_manager`` already imports
    # ``regulatory_audit``, and this module imports ``registry_manager``
    # only for the deployer-input metric reducers.
    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
        T2_6B_CALIBRATION_QUALITY_REJECT_CATEGORIES,
        T2_6B_CV_STABILITY_REJECT_CATEGORIES,
        T2_6B_SIGNAL_GENUINENESS_REJECT_CATEGORIES,
        compute_deployer_input_metrics,
    )

    validation_metrics_raw = state.get("validation_metrics") or {}
    if hasattr(validation_metrics_raw, "model_dump"):
        validation_metrics_raw = validation_metrics_raw.model_dump()
    if not isinstance(validation_metrics_raw, dict):
        validation_metrics_raw = {}

    calibration_error = state.get("calibration_error")
    if calibration_error is None:
        calibration_error = validation_metrics_raw.get("calibration_error")

    t26 = compute_deployer_input_metrics(
        validation_metrics_raw,
        calibration_error=calibration_error,
    )

    cohort_manifest_source = _extract_cohort_manifest_source(state)
    cohort_policy = resolve_cohort_authorization_policy(cohort_manifest_source)

    roc_auc = _extract_roc_auc(validation_metrics_raw)
    honest_band = cohort_policy["honest_auc_band"]
    auc_in_band: Optional[bool]
    if roc_auc is None or honest_band is None:
        auc_in_band = None
    else:
        lo, hi = honest_band
        auc_in_band = bool(lo <= roc_auc <= hi)

    # B1 audit-trail fields surface the resolved post-hoc calibration
    # method ("isotonic" | "sigmoid") and the calibrated ECE.
    cal_info = state.get("post_hoc_calibration") or {}
    if not isinstance(cal_info, dict):
        cal_info = {}
    calibration_method_resolved = cal_info.get("calibration_method_resolved") or cal_info.get(
        "calibration_method"
    )
    calibrated_ece = state.get("calibrated_ece")
    if calibrated_ece is None and isinstance(validation_metrics_raw, dict):
        calibrated_ece = validation_metrics_raw.get("calibrated_ece")

    audit_payload = validation_metrics_raw.get("regulatory_eligibility_audit")
    audit_present = audit_payload is not None
    # Sanity-check the payload shape; mirrors registry_manager's loader
    # contract. A non-dict / malformed payload counts as "present but
    # unusable" — we surface a blocked manifest rather than crash.
    if audit_present and isinstance(audit_payload, dict):
        try:
            RegulatoryEligibilityAudit.from_dict(audit_payload)
            audit_well_formed = True
        except (TypeError, ValueError):
            audit_well_formed = False
    else:
        audit_well_formed = False if audit_present else False

    regulatory_eligible = bool(state.get("regulatory_eligible", False))
    adapted_candidate = bool(state.get("adapted_regulatory_candidate", False))

    feature_surface_count = _extract_feature_surface_count(state)

    # Decision tree — accumulate failure reasons; status follows from
    # the cohort policy + the cleared / not-cleared decision.
    failure_reasons: List[str] = []

    if not cohort_policy["in_c1_scope"]:
        # Out of v5 C1 scope OR blocked-by-design (e.g., Optum).
        blocked_reason = cohort_policy["blocked_reason"]
        if blocked_reason:
            failure_reasons.append(blocked_reason)
        # Differentiate: known-blocked cohorts (Optum) → "blocked",
        # everything else → "out_of_scope".
        status: _T2_6C_AUTHORIZATION_STATUS_T = (
            "blocked" if cohort_policy["cohort"] == "optum" else "out_of_scope"
        )
    else:
        # In CSU scope — run the gating checks.
        if roc_auc is None:
            failure_reasons.append(
                "AUC missing from validation_metrics; cannot evaluate honest band."
            )
        elif not auc_in_band:
            lo, hi = honest_band  # type: ignore[misc]
            failure_reasons.append(
                f"AUC={roc_auc:.4f} outside CSU honest band [{lo:.2f}, {hi:.2f}]."
            )

        sig_cat = t26.get("signal_genuineness_category")
        if sig_cat in T2_6B_SIGNAL_GENUINENESS_REJECT_CATEGORIES:
            failure_reasons.append(f"Permutation null signal category={sig_cat!r} (T2.6b reject).")

        cal_cat = t26.get("calibration_quality_category")
        if cal_cat in T2_6B_CALIBRATION_QUALITY_REJECT_CATEGORIES:
            failure_reasons.append(f"Calibration quality category={cal_cat!r} (T2.6b reject).")

        cv_cat = t26.get("cv_stability_category")
        if cv_cat in T2_6B_CV_STABILITY_REJECT_CATEGORIES:
            failure_reasons.append(f"CV stability category={cv_cat!r} (T2.6b reject).")

        # v5 C1 — N1 contract for C1's authorization gate: the audit
        # MUST be present AND well-formed AND its adaptation_history
        # MUST be empty. This is a weaker check than
        # ``regulatory_eligible=True`` (which also requires the universal
        # 0.75 literature anchor to clear) — C1 deliberately operates
        # against the cohort honest band (CSU [0.62, 0.68]) per v5 §2 C1.
        if not audit_present:
            failure_reasons.append(
                "regulatory_eligibility_audit missing on state; N1 contract not satisfied."
            )
        elif not audit_well_formed:
            failure_reasons.append(
                "regulatory_eligibility_audit payload malformed; cannot verify N1 contract."
            )
        else:
            try:
                audit_obj = RegulatoryEligibilityAudit.from_dict(
                    audit_payload if isinstance(audit_payload, dict) else {}
                )
                if audit_obj.adaptation_history:
                    failure_reasons.append(
                        "regulatory_eligibility_audit.adaptation_history is "
                        "non-empty: adaptive threshold relaxation occurred "
                        "during the model's lifecycle. v5 C1 requires a "
                        "clean N1 audit trail (no adaptations) before T2.6c "
                        "authorization."
                    )
            except (TypeError, ValueError) as exc:  # noqa: BLE001
                failure_reasons.append(f"regulatory_eligibility_audit reconstruction failed: {exc}")

        status = "authorized" if not failure_reasons else "blocked"

    # Build pre-hash payload, then compute sha256.
    pre_hash_payload: Dict[str, Any] = {
        "cohort": cohort_policy["cohort"],
        "manifest_source": cohort_manifest_source,
        "in_c1_scope": cohort_policy["in_c1_scope"],
        "t2_6c_authorization_status": status,
        "t2_6c_authorization_reasons": failure_reasons,
        "roc_auc": roc_auc,
        "honest_auc_band": list(honest_band) if honest_band else None,
        "auc_in_band": auc_in_band,
        "permutation_pvalue": t26.get("signal_genuineness_pvalue"),
        "signal_genuineness_category": t26.get("signal_genuineness_category"),
        "calibration_ece": t26.get("calibration_quality_ece"),
        "calibrated_ece": float(calibrated_ece)
        if calibrated_ece is not None and isinstance(calibrated_ece, (int, float))
        else None,
        "calibration_method_resolved": (
            str(calibration_method_resolved) if calibration_method_resolved is not None else None
        ),
        "calibration_quality_category": t26.get("calibration_quality_category"),
        "cv_stability_std_over_mean": t26.get("cv_stability_std_over_mean"),
        "cv_stability_category": t26.get("cv_stability_category"),
        "feature_surface_count": feature_surface_count,
        "regulatory_eligibility_audit_present": audit_present,
        "regulatory_eligible": regulatory_eligible,
        "adapted_regulatory_candidate": adapted_candidate,
    }

    manifest_sha256 = _hash_manifest(pre_hash_payload)
    emitted_at = datetime.now(tz=timezone.utc).isoformat()

    return RegulatoryDeploymentManifest(
        cohort=pre_hash_payload["cohort"],
        manifest_source=pre_hash_payload["manifest_source"],
        in_c1_scope=pre_hash_payload["in_c1_scope"],
        t2_6c_authorization_status=pre_hash_payload["t2_6c_authorization_status"],
        t2_6c_authorization_reasons=list(pre_hash_payload["t2_6c_authorization_reasons"]),
        roc_auc=pre_hash_payload["roc_auc"],
        honest_auc_band=tuple(pre_hash_payload["honest_auc_band"])  # type: ignore[arg-type]
        if pre_hash_payload["honest_auc_band"]
        else None,
        auc_in_band=pre_hash_payload["auc_in_band"],
        permutation_pvalue=pre_hash_payload["permutation_pvalue"],
        signal_genuineness_category=pre_hash_payload["signal_genuineness_category"],
        calibration_ece=pre_hash_payload["calibration_ece"],
        calibrated_ece=pre_hash_payload["calibrated_ece"],
        calibration_method_resolved=pre_hash_payload["calibration_method_resolved"],
        calibration_quality_category=pre_hash_payload["calibration_quality_category"],
        cv_stability_std_over_mean=pre_hash_payload["cv_stability_std_over_mean"],
        cv_stability_category=pre_hash_payload["cv_stability_category"],
        feature_surface_count=pre_hash_payload["feature_surface_count"],
        regulatory_eligibility_audit_present=pre_hash_payload[
            "regulatory_eligibility_audit_present"
        ],
        regulatory_eligible=pre_hash_payload["regulatory_eligible"],
        adapted_regulatory_candidate=pre_hash_payload["adapted_regulatory_candidate"],
        manifest_sha256=manifest_sha256,
        emitted_at=emitted_at,
        n3_signature=None,
    )
