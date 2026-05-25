"""LangGraph assembly for data_preparer agent.

This module assembles the data preparation pipeline using LangGraph.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from src.data.role_attribution import derive_role_attributions

from .nodes import (
    adaptive_validity_check,
    audit_sampling_frame,
    compute_baseline_metrics,
    detect_leakage,
    engineer_features_node,
    kg_role_enrichment,
    load_data,
    register_features_in_feast,
    review_and_remediate_leakage,
    review_and_remediate_qc,
    run_ge_validation,
    run_quality_checks,
    run_schema_validation,
    run_sufficiency_check,
    transform_data,
)
from .nodes.adaptive_validity_check import _resolve_manifest_features
from .state import DataPreparerState

logger = logging.getLogger(__name__)

# Maximum remediation attempts before giving up
MAX_REMEDIATION_ATTEMPTS = 2

from .nodes.leakage_remediation import MAX_LEAKAGE_REMEDIATION_ATTEMPTS


def _derive_role_attributions_safely(state: Any) -> list[Dict[str, Any]]:
    """Derive ``RoleAttribution`` rows from ``adaptive_verdicts`` + the
    resolved manifest's ``FeatureContract`` registry.

    Phase 1 of Issue #237 reframe (plan
    ``.claude/plans/causal_role_propagation_FINAL.md`` §1.2/§1.3).

    Failure-mode: any exception (unknown manifest source, malformed
    state) returns ``[]`` and logs a WARNING. The producer is additive;
    a bug here must NEVER cascade into the existing QC gate. The
    sidecar still writes (with ``role_attributions=[]``), and Phase 2
    interprets an empty attribution list as "no source attests any
    role; gate everything through the C1 default predicate".
    """
    try:
        verdicts = list(state.get("adaptive_verdicts") or [])
        scope_spec = state.get("scope_spec") or {}
        # ``scope_spec`` may be either a raw dict (legacy callers, tests)
        # or a ``ScopeSpecSchema`` pydantic BaseModel (the typed scope
        # contract, BaseAgentSchema). BaseAgentSchema provides a dict-
        # compat ``.get`` shim, but pydantic instances also expose
        # attribute access; the helper accepts both shapes so the
        # producer works regardless of whether the caller constructed
        # scope_spec via schema or via dict literal. Codex iter-0 HIGH
        # fix: ``isinstance(scope_spec, dict)`` would silently skip the
        # manifest path for the typed-schema code path.
        manifest_source = None
        if isinstance(scope_spec, dict):
            manifest_source = scope_spec.get("feature_manifest_source")
        else:
            getter = getattr(scope_spec, "get", None)
            if callable(getter):
                manifest_source = getter("feature_manifest_source")
            if manifest_source is None:
                manifest_source = getattr(scope_spec, "feature_manifest_source", None)
        feature_contracts: Dict[str, Any] = {}
        if isinstance(manifest_source, str) and manifest_source:
            contracts_list = _resolve_manifest_features(manifest_source)
            if contracts_list:
                feature_contracts = {c.name: c for c in contracts_list}
        # Cast to list[dict] for the sidecar payload (TypedDict is a
        # dict at runtime; this is just a type-system formality).
        return [dict(a) for a in derive_role_attributions(verdicts, feature_contracts)]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "derive_role_attributions failed; emitting empty role_attributions list. "
            "Sidecar will record role_attributions=[]. Cause: %s",
            exc,
        )
        return []


def write_adaptive_verdicts_sidecar(state: Dict[str, Any]) -> Path | None:
    """Write the adaptive-validity audit trail to a JSON sidecar.

    Writes when ADAPTIVE_VALIDITY_ARTIFACTS_DIR is set in the environment AND
    the state has at least one verdict. Otherwise no-ops (silently skipped in
    unit tests, which generally do not configure an artifacts dir).

    Sidecar contents are INTENDED to persist even when the evaluator is
    disabled (Plan layer4_evaluator_audit_consumer.md, codex review MED-1).
    The non-evaluator fields (severity, remediation, z_score, p_value,
    delta_auc, layer routing, etc.) are useful for post-hoc Layer-4 audit
    independent of the Haiku evaluator. When
    ``ADAPTIVE_VALIDITY_EVALUATOR_ENABLED`` is unset, the 5 evaluator_*
    keys are all ``None`` and downstream consumers (e.g.
    ``scripts/curate_compile_set_candidates.py``) correctly skip those
    records via the ``evaluator_satisfied is not False`` predicate.

    Writes are atomic: payload is staged to ``<sidecar>.tmp`` then
    ``Path.replace()``d into place so an interrupted run never leaves a
    half-written JSON on the volume (codex review LOW-9).

    The serialized verdicts carry an empirical ``p_value`` field whose floor
    is ``1 / n_permutations`` (default 200). A persisted ``p_value=0.0``
    therefore means ``< 1/n_permutations``, NOT exact zero (backlog #11.e).
    Severity routing in the producer uses ``z_score``, so this rounding is
    purely informational for downstream sidecar consumers.

    Layer-4 evaluator audit-only fields (Plan
    ``.claude/plans/layer4_evaluator_audit_signal.md``):
    ``evaluator_satisfied``, ``evaluator_rationale_complete``,
    ``evaluator_missed_considerations``, ``evaluator_notes``,
    ``evaluator_model``. All five are ``None`` when
    ``ADAPTIVE_VALIDITY_EVALUATOR_ENABLED`` is unset, when the worker
    verdict was ``None``, when the evaluator failed, or when no LLM
    verdict was supplied for this feature. Consumers must treat these
    keys as audit-only; the orchestrator does not gate or override on
    them. ``evaluator_missed_considerations`` is serialized as a tuple
    by the producer; downstream JSON readers receive a Python list.

    Issue #241 — Layer-4 evaluator telemetry: ``evaluator_latency_ms``
    (float, milliseconds), ``evaluator_input_tokens`` (int),
    ``evaluator_output_tokens`` (int), ``evaluator_cost_usd`` (float).
    Same nullability semantics as the 5 audit keys above, plus an
    additional partial-telemetry case: ``evaluator_latency_ms`` may be
    non-``None`` while ``evaluator_input_tokens`` /
    ``evaluator_output_tokens`` / ``evaluator_cost_usd`` are ``None``
    when the underlying DSPy LM emitted no usage block (cache hit,
    stub LM in tests, etc.). Cost is computed at write time using the
    pinned Haiku rate constants in ``src/data/causal_role_evaluator.py``
    (``HAIKU_INPUT_USD_PER_MTOK`` / ``HAIKU_OUTPUT_USD_PER_MTOK``);
    operators bumping those constants surface the change in the
    pricing-pin unit test. The telemetry keys are audit-only — never
    consumed by the orchestrator.

    Args:
        state: DataPreparerState dict-like with adaptive_verdicts.

    Returns:
        Path to the written sidecar, or None when no write occurred.
    """
    artifacts_dir = os.environ.get("ADAPTIVE_VALIDITY_ARTIFACTS_DIR")
    verdicts = state.get("adaptive_verdicts") or []
    if not artifacts_dir or not verdicts:
        return None
    try:
        base = Path(artifacts_dir) / str(state.get("experiment_id") or "anon")
        base.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sidecar = base / f"adaptive_verdicts_{ts}.json"
        # Phase 1 of causal-role propagation (Issue #237): the producer
        # writes a typed ``role_attributions`` list alongside the existing
        # ``adaptive_verdicts``. Reader sidecar contract: a feature's
        # role_attribution carries ``{feature, causal_role, source,
        # evaluator_satisfied, evaluator_model}``. Phase 2 acts on this;
        # in Phase 1 the field is audit-only. ``state.get`` falls back to
        # ``[]`` for pre-Phase-1 callers (forward-compat with the
        # data_preparer state shape; the field is declared on
        # ``DataPreparerState`` as ``Optional[List[Dict[str, Any]]]``).
        role_attributions = state.get("role_attributions") or []
        payload = {
            # Schema-version pin (Issue #235): ``major.minor`` string. Bump the
            # major on any breaking change to the sidecar payload shape; bump
            # the minor on additive forward-compatible changes. The reader
            # (``src/data/audit_sidecar_reader.py``) WARNs on missing or
            # unknown-major schema_version values.
            #
            # 1.1 (Phase 1, Issue #237): additive ``role_attributions``
            # list. Reader pins MAJOR=1; minor bumps do not WARN.
            # 1.2 (Issue #240 Stage 1): additive shadow promotion keys
            # (would_promote_severity / would_flag_for_review /
            # rationale_incomplete_flag) per the minor-bump-on-additive
            # policy above. Still MAJOR=1.
            # 1.3 (Issue #240 Stage 3): additive env-gated soft-gate keys
            # (gate_rule_fired / worker_severity_pre_gate). Still MAJOR=1.
            # 1.4 (Issue #501 / #240): additive leakage × role cross-check
            # shadow key (would_flag_role_leak_disagreement). Still MAJOR=1.
            "schema_version": "1.4",
            "experiment_id": state.get("experiment_id"),
            "data_source": state.get("data_source"),
            "written_at": ts,
            "leakage_severity": state.get("leakage_severity"),
            "leaked_features": state.get("leaked_features", []),
            "adaptive_flagged_features": state.get("adaptive_flagged_features", []),
            "adaptive_verdicts": verdicts,
            "role_attributions": role_attributions,
        }
        # Atomic write: stage to .tmp then rename so an interrupted run
        # never leaves a half-written JSON (codex review LOW-9).
        sidecar_tmp = sidecar.with_suffix(".json.tmp")
        sidecar_tmp.write_text(json.dumps(payload, indent=2, default=str))
        sidecar_tmp.replace(sidecar)
        logger.info(
            "Wrote adaptive-validity audit trail to %s (verdicts=%d)",
            sidecar,
            len(verdicts),
        )
        return sidecar
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write adaptive-validity sidecar: %s", exc)
        return None


def create_data_preparer_graph() -> StateGraph:  # type: ignore[type-arg]
    """Create the data_preparer LangGraph.

    The graph executes the following pipeline:
    1. load_data - Load and split data from Supabase using MLDataLoader
    2. audit_sampling_frame - Advisory drift audit vs scope_spec.deployment_reference
    3. run_schema_validation - Pandera schema validation (fast, ~10ms)
    4. run_quality_checks - Validate data quality (completeness, validity, etc.)
    5. run_ge_validation - Great Expectations validation (business rules)
    6. detect_leakage - Check for data leakage (temporal, target, train-test)
    7. transform_data - Encode, scale, and impute features
    8. register_features_in_feast - Register features in Feast feature store
    9. compute_baseline_metrics - Compute baseline metrics from train split
    10. finalize_output - Generate final output and QC gate decision
    11. qc_remediation - LLM-assisted review and remediation if QC fails

    QC Remediation Loop:
    - If QC gate fails, routes to LLM-assisted remediation review
    - Analyzes root causes using Claude
    - Attempts automatic fixes (imputation, type conversion, etc.)
    - Re-runs quality checks if fixes are applied
    - Maximum 2 remediation attempts before final failure

    Validation Pipeline Order:
    - Pandera: Fast schema checks (types, nullability, enums)
    - Quality Checker: 5 dimension scoring (completeness, validity, etc.)
    - Great Expectations: Business rules and statistical checks

    Feast Integration:
    - Features registered after transformation (ready for point-in-time retrieval)
    - Freshness check included in registration (QC validation)
    - Non-blocking: failures generate warnings, not errors

    Returns:
        StateGraph ready to be compiled
    """
    # Create the graph
    graph = StateGraph(DataPreparerState)

    # Add nodes
    graph.add_node("load_data", load_data)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("audit_sampling_frame", audit_sampling_frame)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("run_schema_validation", run_schema_validation)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("run_quality_checks", run_quality_checks)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("run_ge_validation", run_ge_validation)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("engineer_features", engineer_features_node)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("detect_leakage", detect_leakage)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("adaptive_validity_check", adaptive_validity_check)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("leakage_remediation", review_and_remediate_leakage)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("transform_data", transform_data)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("register_features_in_feast", register_features_in_feast)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("compute_baseline_metrics", compute_baseline_metrics)  # type: ignore[type-var,arg-type,call-overload]
    # Phase 1 of data-sufficiency diagnostics. Sits between
    # ``compute_baseline_metrics`` and ``kg_role_enrichment`` so it has
    # access to baseline statistics (target_rate, feature_stats) when
    # computing the verdict. HARD_FAIL / blocking SOFT_FAIL appends to
    # ``blocking_issues`` which the existing QC gate at
    # ``finalize_output`` picks up.
    graph.add_node("sufficiency_check", run_sufficiency_check)  # type: ignore[type-var,arg-type,call-overload]
    # Phase 6 of causal-role propagation (Issue #237). Sits between
    # ``compute_baseline_metrics`` and ``finalize_output`` to reconcile
    # LLM-source role attributions against the Phase-6 FalkorDB
    # ``(:Feature)`` nodes. Non-blocking: a graph outage or malformed
    # state passes the input ``role_attributions`` through unchanged.
    graph.add_node("kg_role_enrichment", kg_role_enrichment)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[type-var,arg-type,call-overload]
    graph.add_node("qc_remediation", review_and_remediate_qc)  # type: ignore[type-var,arg-type,call-overload]

    # Define edges (sequential execution with QC remediation loop)
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "audit_sampling_frame")
    graph.add_edge("audit_sampling_frame", "run_schema_validation")
    graph.add_edge("run_schema_validation", "run_quality_checks")
    graph.add_edge("run_quality_checks", "run_ge_validation")
    # v5 Gate B3: engineer_features runs AFTER GE validation (on base
    # schema) and BEFORE detect_leakage / adaptive_validity_check so the
    # engineered columns are audited by Layer 3 alongside base features.
    # Gated on state["enable_feature_engineering"] (default False); when
    # False the node returns an empty patch and the pipeline is
    # behaviorally identical to its pre-B3 form.
    graph.add_edge("run_ge_validation", "engineer_features")
    graph.add_edge("engineer_features", "detect_leakage")

    # Layer 5 wiring: detect_leakage emits hardcoded findings; adaptive_validity_check
    # then runs Layer 3 (data-derived adversarial discriminator) on every numeric
    # feature, augmenting leaked_features and (only) escalating leakage_severity.
    # Routing decisions read the merged severity, so any adaptive escalation feeds
    # the existing leakage_remediation flow.
    graph.add_edge("detect_leakage", "adaptive_validity_check")

    # Conditional edge: route to remediation if critical/high leakage detected
    graph.add_conditional_edges(
        "adaptive_validity_check",
        _route_after_leakage_detection,
        {
            "remediate": "leakage_remediation",
            "continue": "transform_data",
        },
    )

    # After remediation: re-check, continue, or halt
    graph.add_conditional_edges(
        "leakage_remediation",
        _route_after_leakage_remediation,
        {
            "recheck": "detect_leakage",
            "continue": "transform_data",
            "end": END,
        },
    )

    graph.add_edge("transform_data", "register_features_in_feast")
    graph.add_edge("register_features_in_feast", "compute_baseline_metrics")
    # Phase 6 wiring (Issue #237 plan §6.3): the direct edge from
    # ``compute_baseline_metrics`` to ``finalize_output`` is replaced
    # with a two-hop path through ``kg_role_enrichment``. Operating
    # post-baseline avoids restructuring the leakage-remediation
    # conditional and lets the enrichment node act on the final
    # post-transform feature set.
    # Phase 1 data-sufficiency: inserted between
    # ``compute_baseline_metrics`` and ``kg_role_enrichment`` so the
    # sufficiency verdict can use baseline statistics.
    graph.add_edge("compute_baseline_metrics", "sufficiency_check")
    graph.add_edge("sufficiency_check", "kg_role_enrichment")
    graph.add_edge("kg_role_enrichment", "finalize_output")

    # Conditional edge: after finalize_output, check if QC passed
    graph.add_conditional_edges(
        "finalize_output",
        _route_after_finalize,
        {
            "end": END,
            "remediate": "qc_remediation",
        },
    )

    # Conditional edge: after remediation, either retry validation or end
    graph.add_conditional_edges(
        "qc_remediation",
        _route_after_remediation,
        {
            "retry": "run_quality_checks",
            "end": END,
        },
    )

    return graph


def _route_after_finalize(state: DataPreparerState) -> Literal["end", "remediate"]:
    """Route after finalize_output based on QC gate result.

    Args:
        state: Current agent state

    Returns:
        "end" if QC passed, "remediate" if QC failed
    """
    gate_passed = state.get("gate_passed", False)
    qc_status = state.get("qc_status", "unknown")

    # Accept both "passed" and "warning" as valid statuses
    # "warning" indicates non-blocking issues (e.g., expected nulls in optional columns)
    if gate_passed and qc_status in ("passed", "warning"):
        logger.info(f"QC gate passed (status={qc_status}), proceeding to end")
        return "end"
    else:
        logger.info(
            f"QC gate failed (status={qc_status}, passed={gate_passed}), "
            "routing to remediation review"
        )
        return "remediate"


def _route_after_remediation(state: DataPreparerState) -> Literal["retry", "end"]:
    """Route after remediation based on result.

    Args:
        state: Current agent state

    Returns:
        "retry" if remediation was applied and revalidation needed, "end" otherwise
    """
    remediation_status = state.get("remediation_status", "unknown")
    requires_revalidation = state.get("requires_revalidation", False)
    remediation_attempts = state.get("remediation_attempts", 0)

    if remediation_status == "applied" and requires_revalidation:
        if remediation_attempts < MAX_REMEDIATION_ATTEMPTS:
            logger.info(
                f"Remediation applied, retrying validation "
                f"(attempt {remediation_attempts + 1}/{MAX_REMEDIATION_ATTEMPTS})"
            )
            return "retry"

    logger.info(f"Remediation complete with status: {remediation_status}")
    return "end"


def _route_after_leakage_detection(
    state: DataPreparerState,
) -> Literal["remediate", "continue"]:
    """Route after detect_leakage based on severity.

    CRITICAL or HIGH leakage triggers the LLM-assisted remediation node.
    Lower severities pass through to transform_data.
    Also guards against re-entry if max remediation attempts exhausted.

    Args:
        state: Current agent state

    Returns:
        "remediate" if severity warrants intervention, "continue" otherwise
    """
    severity = state.get("leakage_severity", "none")
    attempts = state.get("leakage_remediation_attempts", 0)
    if severity in ("critical", "high") and attempts < MAX_LEAKAGE_REMEDIATION_ATTEMPTS:
        logger.info(
            f"Leakage severity '{severity}' detected (attempt {attempts + 1}) "
            "— routing to remediation"
        )
        return "remediate"
    return "continue"


def _route_after_leakage_remediation(
    state: DataPreparerState,
) -> Literal["recheck", "continue", "end"]:
    """Route after leakage remediation.

    - If remediation was applied and viable, re-check via detect_leakage
    - If remediation found no viable features, halt the pipeline
    - Otherwise continue to transform_data

    Args:
        state: Current agent state

    Returns:
        "recheck", "continue", or "end"
    """
    status = state.get("leakage_remediation_status", "not_needed")
    viable = state.get("leakage_remediation_viable", True)
    attempts = state.get("leakage_remediation_attempts", 0)

    if not viable:
        logger.warning("Leakage remediation found no viable features — halting pipeline")
        return "end"

    if (
        status == "applied"
        and state.get("requires_leakage_revalidation")
        and attempts < MAX_LEAKAGE_REMEDIATION_ATTEMPTS
    ):
        logger.info(
            f"Leakage remediation applied, re-checking "
            f"(attempt {attempts}/{MAX_LEAKAGE_REMEDIATION_ATTEMPTS})"
        )
        return "recheck"

    return "continue"


async def finalize_output(state: DataPreparerState) -> Dict[str, Any]:
    """Finalize output and make QC gate decision.

    This node:
    1. Aggregates all QC results
    2. Computes data readiness
    3. Makes the CRITICAL QC gate decision
    4. Prepares final output

    The QC gate blocks downstream training if:
    - QC status is "failed"
    - There are blocking issues
    - Overall QC score < 0.80

    Args:
        state: Current agent state

    Returns:
        Updated state with final outputs
    """
    logger.info(f"Finalizing output for experiment {state['experiment_id']}")

    try:
        # === QC GATE DECISION ===
        qc_status = state.get("qc_status", "unknown")
        overall_score = state.get("overall_score")
        blocking_issues = list(state.get("blocking_issues", []) or [])

        # Re-promote sampling-frame audit's blocking entry (Phase-1 Task 1.3).
        # ``run_quality_checks`` overwrites ``blocking_issues`` with a fresh
        # local list, so the audit's earlier append (from
        # ``audit_sampling_frame``) is lost by the time we reach the gate.
        # Re-derive it from the audit report here so the gate decision is
        # durable across intermediate node overwrites.
        sampling_frame_report = state.get("sampling_frame_audit_report") or {}
        sampling_frame_blocking_detail = sampling_frame_report.get("blocking_detail")
        if sampling_frame_blocking_detail:
            sf_message = sampling_frame_blocking_detail.get(
                "message", "Sampling-frame drift exceeds blocking threshold"
            )
            sf_blocking_entry = f"sampling_frame_drift: {sf_message}"
            if sf_blocking_entry not in blocking_issues:
                blocking_issues.append(sf_blocking_entry)

        # Apply gate logic (from tier0-contracts.md)
        # Gate passes if qc_status is "passed" OR "warning" (with score threshold)
        # "warning" allows expected issues (e.g., nulls in optional columns for active patients)
        # "failed" always blocks (blocking issues like data leakage)
        gate_passed = True

        # CRITICAL: Gate fails if QC status is "failed" or unknown
        # "warning" is acceptable if score meets threshold (checked below)
        if qc_status not in ("passed", "warning"):
            gate_passed = False
            logger.warning(
                f"QC gate BLOCKED: qc_status='{qc_status}' (must be 'passed' or 'warning')"
            )

        if blocking_issues:
            gate_passed = False
            logger.warning(f"QC gate BLOCKED: {len(blocking_issues)} blocking issues")

        # CRITICAL: Gate fails if overall_score is None or below threshold
        if overall_score is None:
            gate_passed = False
            logger.warning("QC gate BLOCKED: overall_score is None (QC checks may not have run)")
        elif overall_score < 0.80:
            gate_passed = False
            logger.warning(f"QC gate BLOCKED: score {overall_score:.2f} < 0.80")

        # === DATA READINESS ===
        train_df = state.get("train_df")
        validation_df = state.get("validation_df")
        test_df = state.get("test_df")
        holdout_df = state.get("holdout_df")

        train_samples = len(train_df) if train_df is not None else 0
        validation_samples = len(validation_df) if validation_df is not None else 0
        test_samples = len(test_df) if test_df is not None else 0
        holdout_samples = len(holdout_df) if holdout_df is not None else 0
        total_samples = train_samples + validation_samples + test_samples + holdout_samples

        # Available features
        available_features = list(train_df.columns) if train_df is not None else []

        # Missing required features
        scope_spec = state.get("scope_spec", {})
        required_features = scope_spec.get("required_features", [])
        missing_required_features = [f for f in required_features if f not in available_features]

        # Data is ready if QC passed and no missing required features
        qc_passed = gate_passed
        is_ready = qc_passed and len(missing_required_features) == 0

        # Blockers (same as blocking_issues)
        blockers = (blocking_issues or []).copy()
        if missing_required_features:
            blockers.append(f"Missing required features: {', '.join(missing_required_features)}")

        # Phase 1 of causal-role propagation (Issue #237 reframe).
        # Derive typed RoleAttribution rows from adaptive_verdicts + the
        # resolved manifest's FeatureContracts. The list is persisted to
        # the sidecar via write_adaptive_verdicts_sidecar and is
        # audit-only in this phase (Phase 2 is the first consumer).
        # Failure here is non-blocking — propagation is additive and a
        # producer-side bug must never block the existing QC gate.
        role_attributions = _derive_role_attributions_safely(state)

        # Update state. ``blocking_issues`` is propagated explicitly so that
        # the sampling-frame audit's re-promoted entry (if any) survives into
        # the final state — otherwise ``run_quality_checks``' fresh list
        # remains the last-write-wins value.
        updates = {
            "gate_passed": gate_passed,
            "qc_passed": qc_passed,
            "qc_score": overall_score,
            "is_ready": is_ready,
            "total_samples": total_samples,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "holdout_samples": holdout_samples,
            "available_features": available_features,
            "missing_required_features": missing_required_features,
            "blockers": blockers,
            "blocking_issues": blocking_issues,
            "role_attributions": role_attributions,
        }

        # Persist adaptive-validity audit trail to a JSON sidecar so the
        # per-feature verdicts survive outside the run state. The sidecar
        # writer reads ``state.role_attributions`` directly, so seed it
        # in the dict-shaped pydantic state via ``__setitem__`` (the
        # BaseAgentSchema dict-compat shim accepts it).
        state["role_attributions"] = role_attributions  # type: ignore[index]
        write_adaptive_verdicts_sidecar(state)

        logger.info(
            f"Data preparation completed: gate_passed={gate_passed}, "
            f"is_ready={is_ready}, total_samples={total_samples}"
        )

        return updates

    except Exception as e:
        logger.error(f"Finalize output failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "finalize_output_error",
            "gate_passed": False,
            "qc_passed": False,
            "is_ready": False,
            "blockers": [f"Finalization error: {str(e)}"],
        }
