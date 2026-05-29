"""Scope specification builder for scope_definer.

This module builds the complete ScopeSpec from business requirements.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from src.utils.sufficiency_defaults import (
    DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR,
    DEFAULT_MDE_BINARY_RELATIVE,
    DEFAULT_MDE_CONTINUOUS_COHENS_D,
)

logger = logging.getLogger(__name__)


def _normalise_prediction_timestamp(value: Any) -> Optional[str]:
    """Coerce a prediction_timestamp input to an ISO 8601 string.

    Accepts ``datetime``, ``pd.Timestamp``, or a string parseable by
    ``pd.Timestamp`` (ISO-8601 and the broader set of forms pandas accepts)
    and returns the normalised string form for stable storage in
    ``scope_spec``. Returns ``None`` when the input is missing or empty so
    downstream agents can distinguish "not provided" from "explicit
    timestamp".

    Block 1B-M2: this helper is strict-validating. Permissive ``str(value)``
    coercion was a footgun in temporal pipelines — silent type drift would
    corrupt lag windows / rolling means without surfacing a single warning.
    Unknown types and unparseable strings now raise ``TypeError`` so the
    drift fails loud at the scope_definer boundary.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    # Late import to keep scope_builder dependency-light.
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dep elsewhere
        pd = None  # type: ignore[assignment]
    if pd is not None and isinstance(value, pd.Timestamp):
        # pd.Timestamp.isoformat is typed as Any in pandas-stubs, so cast.
        return cast(str, value.isoformat())
    if isinstance(value, str):
        if pd is not None:
            try:
                pd.Timestamp(value)
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    f"prediction_timestamp string is not parseable by pd.Timestamp: {value!r}"
                ) from exc
        return value
    raise TypeError(
        f"prediction_timestamp must be datetime, pd.Timestamp, ISO-8601 str, "
        f"or None; got {type(value).__name__}: {value!r}"
    )


def _build_scope_sufficiency(
    state: Dict[str, Any],
    *,
    problem_type: str,
) -> Optional[Dict[str, Any]]:
    """Build the `scope_spec.sufficiency` payload — F4 / D1 implementation.

    D1 of the data-sufficiency rollout plan
    (`docs/superpowers/plans/2026-05-22-data-sufficiency-diagnostics-rollout.md`)
    requires `scope_spec.target_mde` to be "optional with a data-driven default
    + LOUD WARNING when defaulted". Before this fix, scope_builder didn't set
    `sufficiency` at all — every defaulted MDE looked identical to a user-
    supplied one in the audit chain, and the "loud warning" only fired in
    DataPreparer (downstream of where the decision was made) for ALL
    data-driven defaults (warning fatigue).

    Resolution hierarchy (mirrors the runtime resolver):

    1. ``state['sufficiency']['target_mde']`` (user override) →
       ``target_mde_source='user_override'``.
    2. Data-driven default — for binary classification at scope time we know
       ``state['baseline_rate']`` if the caller / cohort runner pre-set it;
       for continuous we know ``state['sigma_outcome']`` if pre-set.
       Otherwise this branch defers — the DataPreparer's resolver will
       compute from data once train_df is loaded → ``computed_from_data``.
    3. Literature default (`DEFAULT_MDE_*` from sufficiency_defaults) →
       ``target_mde_source='literature_default'`` with a LOUD WARN.

    We ONLY warn on case (3). Cases (1) and (2) carry their source through
    the audit chain via `target_mde_source`; they don't need a log warning
    because the source field IS the audit signal (no archaeology required).

    For causal_inference: defer to the DataPreparer because sigma estimates
    aren't available until the data is loaded; pass through any user override
    only.

    Returns:
        A dict with the populated `sufficiency` fields, or None if neither
        user override nor scope-time defaults apply. The caller writes this
        into `scope_spec['sufficiency']`.
    """
    user_suff = state.get("sufficiency") or {}
    if user_suff and not isinstance(user_suff, dict):
        # Pydantic SufficiencyConfig — normalize to dict so we can blend
        # without losing fields.
        if hasattr(user_suff, "model_dump"):
            user_suff = user_suff.model_dump(exclude_none=True)
        else:
            logger.warning(
                "state['sufficiency'] has unknown type %s; ignoring.",
                type(user_suff).__name__,
            )
            user_suff = {}

    user_mde = user_suff.get("target_mde")
    # User override path (1). Just stamp the source and pass through.
    if user_mde is not None:
        merged: Dict[str, Any] = dict(user_suff)
        merged["target_mde_source"] = "user_override"
        return merged

    # Data-driven path (2) — only kicks in when scope-time signals exist.
    # The runtime resolver in DataPreparer also computes this from train_df
    # if scope-time signals are absent; we mark `computed_from_data` here
    # only when we actually computed the value at scope-time, otherwise we
    # leave target_mde unset and let the resolver fill in downstream.
    if problem_type == "binary_classification":
        baseline_rate = state.get("baseline_rate")
        if isinstance(baseline_rate, (int, float)) and 0.0 < baseline_rate < 1.0:
            relative_mde = DEFAULT_MDE_BINARY_RELATIVE * float(baseline_rate)
            computed_mde = max(DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR, relative_mde)
            merged = dict(user_suff)
            merged["target_mde"] = computed_mde
            merged["target_mde_source"] = "computed_from_data"
            return merged
    elif problem_type == "regression":
        sigma_outcome = state.get("sigma_outcome")
        if isinstance(sigma_outcome, (int, float)) and sigma_outcome > 0:
            computed_mde_continuous = 0.5 * float(sigma_outcome)  # Cohen medium
            merged = dict(user_suff)
            merged["target_mde"] = computed_mde_continuous
            merged["target_mde_source"] = "computed_from_data"
            return merged

    # Causal-inference deferral path (R2.4): the round-1 implementation
    # contradicted its own docstring (lines 95-97: "For causal_inference:
    # defer to the DataPreparer because sigma estimates aren't available
    # until the data is loaded; pass through any user override only.").
    # The fall-through path below was writing a literature_default for
    # causal too — combined with the R2.3 resolver bug (which re-stamped
    # any non-None target_mde as user_override) every causal scope ended
    # up with a fake "user_override" audit chain. Round-2 fix: for
    # causal_inference + no user override, defer to the resolver inside
    # sufficiency_check (which has access to baseline_rate from loaded
    # data). Set ``target_mde_source = None`` explicitly so any consumer
    # that inspects the field sees an explicit "deferred", not a missing
    # key that could be mistaken for "never considered". No WARN at scope
    # time — the resolver will warn IFF the deferral still leaves
    # ``target_mde`` defaulted after data load.
    if problem_type == "causal_inference":
        merged = dict(user_suff)
        merged["target_mde_source"] = None
        return merged if merged else None

    # Literature default path (3) — only used when we can't compute from
    # data AND the runtime resolver also won't be able to (no train_df).
    # For binary/continuous we let the runtime resolver handle this so a
    # late-arriving signal still routes through `computed_from_data`. We
    # only set a scope-time literature default for the rare case where the
    # caller is constructing a fully-detached scope_spec with no downstream
    # data preparer (e.g. config-only validation). Per the rollout plan, the
    # WARN fires here so audit logs surface "no data, no override → fell
    # through to literature".
    if problem_type == "binary_classification":
        literature_mde = DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR
    elif problem_type == "regression":
        literature_mde = DEFAULT_MDE_CONTINUOUS_COHENS_D
    else:
        # multiclass / time_series — no literature MDE convention applies
        # at scope-time. Return user_suff unchanged (may be empty).
        return dict(user_suff) if user_suff else None

    merged = dict(user_suff)
    merged["target_mde"] = literature_mde
    merged["target_mde_source"] = "literature_default"
    logger.warning(
        "scope_spec.sufficiency.target_mde not specified and no scope-time "
        "data signal (baseline_rate / sigma_outcome) available; falling back "
        "to literature default %.3f for problem_type=%s. Set "
        "scope_spec.sufficiency.target_mde or pre-compute baseline_rate / "
        "sigma_outcome to silence this warning.",
        literature_mde,
        problem_type,
    )
    return merged


_COST_MATRIX_KEYS = ("tp", "fp", "fn", "tn")


def _validate_cost_matrix(value: Any) -> Optional[Dict[str, float]]:
    """Coerce and validate a cost_matrix input.

    Accepts a dict with all four confusion-matrix keys (``tp``/``fp``/
    ``fn``/``tn``) mapped to numeric values. Returns ``None`` when the
    input is missing or empty so downstream code can treat "no cost
    matrix configured" the same as "skip business_utility".

    Raises ``ValueError`` on malformed input (missing keys or non-numeric
    values) — fail-loud at the scope_definer boundary rather than
    silently dropping a misconfigured matrix that would defeat the
    purpose of the business_utility metric.
    """
    if value is None or value == {}:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            f"cost_matrix must be a dict with keys {_COST_MATRIX_KEYS}, got {type(value).__name__}"
        )
    missing = [k for k in _COST_MATRIX_KEYS if k not in value]
    if missing:
        raise ValueError(
            f"cost_matrix missing required keys: {missing}. All four of "
            f"{_COST_MATRIX_KEYS} are required."
        )
    coerced: Dict[str, float] = {}
    for k in _COST_MATRIX_KEYS:
        v = value[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"cost_matrix['{k}'] must be int or float, got {type(v).__name__}")
        coerced[k] = float(v)
    return coerced


async def build_scope_spec(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build complete ScopeSpec from inferred problem details.

    Creates a complete ML experiment specification including:
    - Experiment identification
    - Problem type and target
    - Population criteria
    - Feature requirements
    - Constraints

    Args:
        state: ScopeDefinerState with inferred problem type, target variable,
               and business context

    Returns:
        Dictionary with scope_spec, experiment_id, experiment_name
    """
    # Generate unique experiment ID
    brand = state.get("brand", "unknown")
    region = state.get("region", "all")
    timestamp = datetime.now(tz=None).strftime("%Y%m%d%H%M%S")
    # Add UUID suffix for uniqueness even within same second
    uuid_suffix = uuid.uuid4().hex[:6]
    experiment_id = f"exp_{brand.lower()[:4]}_{region.lower()[:2]}_{timestamp}_{uuid_suffix}"

    # Generate experiment name
    target_outcome = state.get("target_outcome", "ML Model")
    experiment_name = f"{brand} - {target_outcome}"

    # Get inferred problem details
    problem_type = state.get("inferred_problem_type", "binary_classification")
    prediction_target = state.get("inferred_target_variable", "target")
    prediction_horizon = state.get("prediction_horizon_days", 30)

    # Define population criteria
    target_population = _define_target_population(state)
    inclusion_criteria = _define_inclusion_criteria(state)
    exclusion_criteria = _define_exclusion_criteria(state)

    # Define feature requirements
    required_features = _define_required_features(state)
    excluded_features = _define_excluded_features(state)
    feature_categories = _define_feature_categories(state)

    # Define constraints
    regulatory_constraints = ["HIPAA", "GDPR"]
    ethical_constraints = ["no_protected_attributes", "no_race_features", "no_direct_pii"]
    technical_constraints = ["inference_latency_<100ms", "model_size_<1GB"]

    # Determine minimum samples
    minimum_samples = _calculate_minimum_samples(problem_type)

    # Block 1B: forward the optional inference cutoff timestamp from the
    # business spec so downstream agents (Tier 1-5) share a single anchor.
    prediction_timestamp = _normalise_prediction_timestamp(state.get("prediction_timestamp"))

    # Block 5 (finding #10): forward the optional cost matrix so the
    # evaluator can compute a business_utility metric driven by the chosen
    # decision threshold and the per-outcome dollar value the business
    # assigns to each confusion-matrix cell. None means "skip the metric".
    cost_matrix = _validate_cost_matrix(state.get("cost_matrix"))

    # F4 / D1 (PR #462 hotfix): build scope-level `sufficiency` payload so
    # the audit chain records the target_mde provenance at the scope
    # boundary, not just at the DataPreparer runtime resolver. Per the
    # rollout plan a data-driven default is preferred; user override always
    # wins; literature fallback emits a LOUD WARN.
    sufficiency = _build_scope_sufficiency(state, problem_type=problem_type)

    # Build complete ScopeSpec.
    # Naming guard (Block 1B-M6): the two ``prediction_*`` fields below are
    # NOT redundant — ``prediction_horizon_days`` is a *duration* (e.g. "30
    # days lookahead from the cutoff"), while ``prediction_timestamp`` is the
    # *anchor* (the inference cutoff itself, e.g. ``2026-04-26T00:00``).
    # Together they define a half-open prediction window
    # ``[prediction_timestamp, prediction_timestamp + horizon)``.
    scope_spec = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "problem_type": problem_type,
        "prediction_target": prediction_target,
        "prediction_horizon_days": prediction_horizon,
        "prediction_timestamp": prediction_timestamp,
        "cost_matrix": cost_matrix,
        "target_population": target_population,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "required_features": required_features,
        "excluded_features": excluded_features,
        "feature_categories": feature_categories,
        "regulatory_constraints": regulatory_constraints,
        "ethical_constraints": ethical_constraints,
        "technical_constraints": technical_constraints,
        "minimum_samples": minimum_samples,
        "brand": brand,
        "region": region,
        "use_case": state.get("use_case", "commercial_targeting"),
        "created_by": "scope_definer",
        "created_at": datetime.now(tz=None).isoformat(),
    }

    # F4 / D1: only attach when we actually produced a payload (user override
    # OR scope-time data signal OR literature fallback). Empty/None case
    # leaves the field absent — the DataPreparer's runtime resolver still
    # has a chance to compute from data once train_df is loaded.
    if sufficiency is not None:
        scope_spec["sufficiency"] = sufficiency

    # Layer 5 manifest opt-in: forward the resolved cohort manifest source
    # (set upstream by the pipeline from data_source / explicit override) so
    # adaptive_validity_check consults the matching FeatureContract registry
    # for layer="1" verdicts. Only attach when set — leaving it absent
    # preserves the cross-cohort false-positive guard for synthetic / research
    # regimes that never resolved a manifest.
    feature_manifest_source = state.get("feature_manifest_source")
    if feature_manifest_source is not None:
        scope_spec["feature_manifest_source"] = feature_manifest_source

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "scope_spec": scope_spec,
    }


def _define_target_population(state: Dict[str, Any]) -> str:
    """Define target population description."""
    brand = state.get("brand", "")
    state.get("inferred_problem_type", "")

    # Brand-specific populations
    if "remibrutinib" in brand.lower() or "csu" in brand.lower():
        return "HCPs treating Chronic Spontaneous Urticaria patients"
    elif "fabhalta" in brand.lower() or "pnh" in brand.lower():
        return "HCPs treating Paroxysmal Nocturnal Hemoglobinuria patients"
    elif "kisqali" in brand.lower() or "breast" in brand.lower():
        return "Oncologists treating HR+/HER2- breast cancer patients"

    # Generic
    return "HCPs with relevant patient population"


def _define_inclusion_criteria(state: Dict[str, Any]) -> List[str]:
    """Define data inclusion criteria."""
    brand = state.get("brand", "")

    criteria = ["hcp_is_active", "has_patient_data", "recent_activity_90days"]

    # Brand-specific criteria
    if "remibrutinib" in brand.lower():
        criteria.append("specialty_in_dermatology_or_allergy")
    elif "fabhalta" in brand.lower():
        criteria.append("specialty_in_hematology")
    elif "kisqali" in brand.lower():
        criteria.append("specialty_in_oncology")

    return criteria


def _define_exclusion_criteria(state: Dict[str, Any]) -> List[str]:
    """Define data exclusion criteria."""
    return ["test_accounts", "invalid_data", "duplicate_records", "missing_required_fields"]


def _define_required_features(state: Dict[str, Any]) -> List[str]:
    """Define required features based on problem type."""
    problem_type = state.get("inferred_problem_type", "")
    candidate_features = state.get("candidate_features", [])

    if candidate_features:
        return cast(List[str], candidate_features)

    # Default feature sets by problem type
    base_features = [
        "hcp_specialty",
        "patient_count",
        "prescription_history",
        "brand_affinity_score",
    ]

    if problem_type == "regression":
        base_features.extend(["historical_prescription_volume", "market_share"])
    elif problem_type == "binary_classification":
        base_features.extend(["engagement_score", "channel_response_rate"])

    return base_features


def _define_excluded_features(state: Dict[str, Any]) -> List[str]:
    """Define features to exclude (PII, leakage risks, pipeline metadata)."""
    return [
        # PII
        "hcp_name",
        "hcp_npi",
        "patient_name",
        "patient_ssn",
        "exact_address",
        "phone_number",
        "email_address",
        # Temporal leakage
        "future_prescription_data",
        # Pipeline construction metadata (never clinical predictors)
        "index_date",
        "lookback_start_date",
        "prediction_end_date",
        "cohort_entry_date",
        "observation_end_date",
        # Structural target-correlation by construction (NOT lookback-fixable). Added
        # 2026-05-06 by tier0_quality_remediation_arc Shard B (B.5 Option 2):
        # data_quality_score is np.random.uniform-bucketed by archetype in
        # convert_csu_rwd.py:824-830, and archetype is target-correlated by
        # construction. Per the n=9,607 lookback sweep (4 windows {30,90,180,365}),
        # data_quality_score AUC stays at ~0.82 in every window — PARTIAL-COLLAPSE.
        # See `.claude/state/quality_arc_b_csu_close_20260506.md`.
        "data_quality_score",
    ]


def _define_feature_categories(state: Dict[str, Any]) -> List[str]:
    """Define feature categories for this problem."""
    return [
        "demographics",
        "prescription_history",
        "engagement",
        "market_dynamics",
        "brand_affinity",
    ]


def _initial_min_samples_estimate(problem_type: str) -> int:
    """Advisory rule-of-thumb sample-size estimate by problem type.

    NOTE: this is an ADVISORY value, not a gating threshold. The
    authoritative sufficiency check lives in DataPreparer's
    ``sufficiency_check`` node (Phase 1 of the data-sufficiency
    diagnostics rollout), which has access to actual data
    characteristics (n_features, minority prevalence, outcome variance)
    that this function cannot see at scope-definition time.

    These per-problem-type defaults were originally written as gating
    thresholds but were never enforced anywhere downstream (handoff
    protocol declared the rule but did not check it). Retained as a
    cheap upstream hint that the operator can sanity-check the
    minimum row count against before the data is even loaded.

    Citations live in src/utils/sufficiency_defaults.py — the
    ``sufficiency_check`` node delegates threshold resolution to
    ``src/utils/sufficiency_resolver.py`` which carries the literature
    references (Vergouwe 2007, Riley 2020, etc.).

    Args:
        problem_type: One of binary_classification, regression,
            multiclass_classification, causal_inference, time_series.

    Returns:
        Advisory minimum n. Always positive (so the handoff-protocol
        check ``minimum_samples > 0`` is satisfied).
    """
    if problem_type == "binary_classification":
        return 500
    elif problem_type == "regression":
        return 300
    elif problem_type == "multiclass_classification":
        return 1000
    elif problem_type == "causal_inference":
        return 1000
    elif problem_type == "time_series":
        return 500

    return 500


# Backward-compat alias for callers that haven't migrated to the new name.
# Slated for removal once all in-tree callers and tests are updated.
_calculate_minimum_samples = _initial_min_samples_estimate
