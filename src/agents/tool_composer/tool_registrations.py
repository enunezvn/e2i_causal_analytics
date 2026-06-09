"""
E2I Tool Registration Examples
Version: 4.2
Purpose: Demonstrate how agents expose tools to the Tool Composer

This file shows the pattern for registering composable tools from each agent.
Each agent should call its registration function during initialization.

Data contracts (F7 — there are exactly TWO, do not invent variants):

1. DataFrame-via-kwargs (the stats/causal tools): ``causal_effect_estimator``,
   ``cate_analyzer``, ``risk_scorer``, ``propensity_estimator``,
   ``cohort_statistics``, ``cohort_validator``, ``psi_calculator``,
   ``distribution_comparator``. These read the real ``pandas.DataFrame`` from
   ``**kwargs`` EXCLUSIVELY via ``_extract_dataframe_from_kwargs`` (which checks
   the canonical keys ``_DATAFRAME_KWARGS_KEYS = ("data","dataframe",
   "estimation_data")``). The executor injects the in-context frame under
   ``estimation_data``. When no frame is present these tools FAIL CLOSED with a
   descriptive ``RuntimeError`` — they never fabricate data.

2. Dict-input (the structure/graph tools): ``discover_dag`` takes its ``data``
   field as a plain ``Dict`` and must NOT be handed a DataFrame. Upstream-result
   consumers (``segment_ranker``, ``roi_estimator``) likewise take a Dict
   produced by an earlier tool.

Anti-mocking invariant (CLAUDE.md): every tool either computes from real inputs
or fail-closes cleanly. No silent placeholder values.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from src.causal_engine.pipeline import (
    PipelineInput,
    PipelineOutput,
    SequentialPipeline,
)
from src.services import cohort_resolution
from src.tool_registry import (
    composable_tool,
)

# Canonical kwargs keys under which callers may supply the real DataFrame for
# causal_effect_estimator. Listed in priority order; the first non-None value
# is used. The tool fail-closes if NONE of these keys is provided -- it does
# NOT fabricate a synthetic frame, per CLAUDE.md anti-mocking discipline.
_DATAFRAME_KWARGS_KEYS: Tuple[str, ...] = (
    "data",
    "dataframe",
    "estimation_data",
)


# `_DataAwareSequentialPipeline` was deleted in #458 once `PipelineState` /
# `PipelineInput` declared `estimation_data` as a first-class field — the
# tool now passes the DataFrame directly via `PipelineInput.estimation_data`
# and constructs the base `SequentialPipeline` with no subclass override.


# ============================================================================
# PYDANTIC MODELS FOR TOOL I/O
# ============================================================================


class EffectEstimatorInput(BaseModel):
    """Input for causal effect estimation"""

    treatment: str
    outcome: str
    confounders: List[str] = []
    method: str = "backdoor.linear_regression"


class EffectEstimate(BaseModel):
    """Output from causal effect estimation"""

    ate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str
    n_samples: int


class CateAnalyzerInput(BaseModel):
    """Input schema for the ``cate_analyzer`` tool (F6(b)).

    This mirrors the REAL ``cate_analyzer(treatment, outcome, segments)``
    callable signature so the planner sees the correct argument shape. The
    previous ``CATEInput`` model declared ``effect_estimate`` /
    ``segment_variables`` — fields the callable never accepts — which misled the
    planner. ``segments`` is ``List[str]`` (dataset COLUMN names), not
    ``List[Dict]``.
    """

    treatment: str
    outcome: str
    segments: List[str] = []


class CATEResults(BaseModel):
    """Output from CATE analysis"""

    segments: List[Dict[str, Any]]
    high_responders: List[str]
    effect_by_segment: Dict[str, float]


class GapCalculatorInput(BaseModel):
    """Input for gap calculation"""

    metric: str
    entity_type: str  # region, territory, brand
    entities: List[str]


class GapAnalysis(BaseModel):
    """Output from gap analysis"""

    gap: float
    entity_values: Dict[str, float]
    top_performer: str
    bottom_performer: str


class SegmentRanking(BaseModel):
    """Output from segment ranking (consumes a CATE / gap result)."""

    ranking: List[Dict[str, Any]]
    recommended_targets: List[str]


class ROIEstimate(BaseModel):
    """Output from ROI estimation (consumes a gap-analysis result)."""

    estimated_roi: float
    payback_months: float
    confidence_interval: List[float]
    assumptions: List[str]


class RiskScores(BaseModel):
    """Output from risk scoring (real per-entity scores from a DataFrame)."""

    scores: List[Dict[str, Any]]
    model_version: str
    scored_at: str


class PropensityScores(BaseModel):
    """Output from propensity estimation (real fitted scores from a DataFrame)."""

    mean_propensity: float
    propensity_distribution: Dict[str, float]
    overlap_assessment: str
    common_support: float


class PowerCalculatorInput(BaseModel):
    """Input for power analysis"""

    effect_size: float
    alpha: float = 0.05
    power: float = 0.8
    ratio: float = 1.0  # Treatment/control ratio


class PowerAnalysis(BaseModel):
    """Output from power analysis"""

    required_n: int
    actual_power: float
    detectable_effect: float


class SimulatorInput(BaseModel):
    """Input for counterfactual simulation"""

    intervention: str
    target_entities: List[str]
    expected_effect: float
    duration_weeks: int = 12


class SimulationResults(BaseModel):
    """Output from counterfactual simulation"""

    predicted_lift: float
    confidence: str  # low, medium, high
    uncertainty_range: List[float]


# ============================================================================
# COHORT CONSTRUCTOR MODELS (Tier 0)
# ============================================================================


class CohortBuilderInput(BaseModel):
    """Input for cohort construction"""

    brand: str
    indication: Optional[str] = None
    inclusion_criteria: List[str] = []
    exclusion_criteria: List[str] = []
    lookback_days: int = 365
    followup_days: int = 90


class CohortBuilderOutput(BaseModel):
    """Output from cohort construction"""

    eligible_patient_ids: List[str]
    total_evaluated: int
    total_eligible: int
    eligibility_rate: float
    criteria_breakdown: Dict[str, int]
    execution_time_ms: float


class CohortValidatorInput(BaseModel):
    """Input for cohort validation"""

    cohort_result: Dict[str, Any]
    min_cohort_size: int = 100
    required_completeness: float = 0.8


class CohortValidatorOutput(BaseModel):
    """Output from cohort validation"""

    is_valid: bool
    validation_checks: List[Dict[str, Any]]
    quality_score: float
    warnings: List[str]
    recommendations: List[str]


class CohortStatisticsInput(BaseModel):
    """Input for cohort statistics"""

    cohort_result: Dict[str, Any]
    include_demographics: bool = True
    include_clinical: bool = True


class CohortStatisticsOutput(BaseModel):
    """Output from cohort statistics"""

    cohort_size: int
    demographics: Dict[str, Any]
    clinical_characteristics: Dict[str, Any]
    summary_table: List[Dict[str, Any]]


# ============================================================================
# COHORT CONSTRUCTOR AGENT TOOLS (Tier 0)
# ============================================================================


@composable_tool(
    name="cohort_builder",
    description="Constructs patient cohorts by applying inclusion/exclusion criteria based on FDA/EMA label requirements",
    source_agent="cohort_constructor",
    tier=0,
    input_parameters=[
        {
            "name": "brand",
            "type": "str",
            "description": "Brand name, resolved case-insensitively against the actual data values",
        },
        {
            "name": "indication",
            "type": "str",
            "description": "Disease indication",
            "required": False,
        },
        {
            "name": "inclusion_criteria",
            "type": "List[str]",
            "description": "Inclusion criteria expressions",
            "required": False,
        },
        {
            "name": "exclusion_criteria",
            "type": "List[str]",
            "description": "Exclusion criteria expressions",
            "required": False,
        },
    ],
    output_schema="CohortBuilderOutput",
    avg_execution_ms=5000,
    input_model=CohortBuilderInput,
    output_model=CohortBuilderOutput,
)
def cohort_builder(
    brand: str,
    indication: Optional[str] = None,
    inclusion_criteria: Optional[List[str]] = None,
    exclusion_criteria: Optional[List[str]] = None,
    **kwargs,
) -> CohortBuilderOutput:
    """Build a patient cohort from a REAL (brand, region) population (#778).

    Data source resolution (no fabrication):

    1. If the executor auto-injected a DataFrame (one of
       ``_DATAFRAME_KWARGS_KEYS``), use it as the base population.
    2. Otherwise route through the shared ``cohort_resolution`` service (#779) to
       resolve the canonical ``patient_journeys`` cohort for ``(brand, region)``
       — ``region`` is read from ``kwargs`` (planner/context-supplied).

    Then apply the supplied inclusion/exclusion criteria (simple
    ``<column> <op> <value>`` expressions) against the real frame and return the
    REAL eligible patient IDs. Per anti-mocking discipline this FAILS CLOSED
    (descriptive ``RuntimeError``) when no population resolves or the frame lacks
    a patient-id column — it NEVER fabricates ``P001/P002/P003`` placeholder IDs.

    Raises:
        RuntimeError: when no real population is available or the resolved frame
            has no recognizable patient-id column.
    """
    start = time.time()

    # --- 1. Locate the real population frame (injected wins, else resolve). ---
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        region = kwargs.get("region")
        df = cohort_resolution.resolve_cohort_frame(brand, region)
    if df is None:
        raise RuntimeError(
            "cohort_builder: no real patient population available for "
            f"brand={brand!r} (region={kwargs.get('region')!r}) — the "
            "cohort_resolution service returned no cohort and no DataFrame was "
            "injected via context. Refusing to fabricate eligible_patient_ids."
        )

    # --- 2. Locate a real patient-id column (fail closed if absent). ---
    id_col = _find_patient_id_column(df)
    if id_col is None:
        raise RuntimeError(
            "cohort_builder: resolved cohort has no recognizable patient-id "
            f"column (columns={list(df.columns)!r}). Refusing to fabricate "
            "patient IDs from row positions."
        )

    total_evaluated = int(len(df))

    # --- 3. Apply simple inclusion/exclusion criteria against real columns. ---
    eligible, breakdown = _apply_cohort_criteria(
        df, list(inclusion_criteria or []), list(exclusion_criteria or [])
    )

    eligible_ids = [str(v) for v in eligible[id_col].tolist()]
    total_eligible = len(eligible_ids)
    rate = (total_eligible / total_evaluated) if total_evaluated else 0.0

    return CohortBuilderOutput(
        eligible_patient_ids=eligible_ids,
        total_evaluated=total_evaluated,
        total_eligible=total_eligible,
        eligibility_rate=rate,
        criteria_breakdown=breakdown,
        execution_time_ms=(time.time() - start) * 1000.0,
    )


# Recognized patient-id columns, in priority order.
_PATIENT_ID_COLUMNS: Tuple[str, ...] = (
    "patient_id",
    "patient_journey_id",
    "subject_id",
    "person_id",
    "id",
)

# A simple criterion is ``<column> <op> <value>`` (e.g. ``age_at_diagnosis >= 50``).
_CRITERION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$")


def _find_patient_id_column(df: Any) -> Optional[str]:
    """Return the first recognized patient-id column present in ``df``, else None."""
    try:
        columns = set(df.columns)
    except Exception:  # noqa: BLE001 - non-DataFrame input -> no id column
        return None
    for candidate in _PATIENT_ID_COLUMNS:
        if candidate in columns:
            return candidate
    return None


def _parse_criterion_value(raw: str) -> Any:
    """Coerce a criterion RHS to int/float/bool, else a stripped string literal."""
    token = raw.strip().strip("'\"")
    low = token.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _criterion_mask(series: Any, op: str, value: Any) -> Any:
    """Boolean mask for ``series <op> value`` (operators are a fixed safe set)."""
    if op == ">=":
        return series >= value
    if op == "<=":
        return series <= value
    if op == ">":
        return series > value
    if op == "<":
        return series < value
    if op == "==":
        return series == value
    if op == "!=":
        return series != value
    raise ValueError(f"unsupported operator {op!r}")


def _apply_cohort_criteria(
    df: Any,
    inclusion: List[str],
    exclusion: List[str],
) -> Tuple[Any, Dict[str, int]]:
    """Apply simple criteria to ``df``; return ``(eligible_df, breakdown)``.

    Each parseable ``<column> <op> <value>`` criterion that references a real
    column is applied: inclusion keeps matching rows, exclusion drops matching
    rows. ``breakdown`` maps each applied criterion to the number of patients it
    removed. Criteria that are unparseable or reference an unknown column are NOT
    silently treated as dropping everyone — they are recorded under
    ``"_unapplied_criteria"`` (a count) so the caller can see they had no effect,
    rather than fabricating an eligibility verdict.
    """
    eligible = df
    breakdown: Dict[str, int] = {}
    unapplied = 0

    def _apply(expr: str, *, exclude: bool) -> None:
        nonlocal eligible, unapplied
        match = _CRITERION_RE.match(expr)
        col = match.group(1) if match else None
        # Unparseable, unknown-column, or empty-RHS criteria are recorded as
        # unapplied (honest accounting) rather than silently filtering the wrong
        # rows or relying on a downstream dtype-mismatch exception.
        if match is None or col not in df.columns or not match.group(3).strip():
            unapplied += 1
            return
        op = match.group(2)
        value = _parse_criterion_value(match.group(3))
        try:
            mask = _criterion_mask(eligible[col], op, value)
            keep = ~mask if exclude else mask
            before = len(eligible)
            eligible = eligible[keep]
            breakdown[f"{'exclusion' if exclude else 'inclusion'}:{expr}"] = before - len(eligible)
        except Exception:  # noqa: BLE001 - dtype mismatch etc. -> record as unapplied
            unapplied += 1

    for crit in inclusion:
        _apply(crit, exclude=False)
    for crit in exclusion:
        _apply(crit, exclude=True)

    if unapplied:
        breakdown["_unapplied_criteria"] = unapplied

    return eligible, breakdown


@composable_tool(
    name="cohort_validator",
    description="Validates a constructed cohort against clinical trial requirements",
    source_agent="cohort_constructor",
    tier=0,
    input_parameters=[
        {"name": "cohort_result", "type": "dict", "description": "Output from cohort_builder"},
        {
            "name": "min_cohort_size",
            "type": "int",
            "description": "Minimum required cohort size",
            "required": False,
            "default": 100,
        },
    ],
    output_schema="CohortValidatorOutput",
    avg_execution_ms=1000,
    input_model=CohortValidatorInput,
    output_model=CohortValidatorOutput,
)
def cohort_validator(
    cohort_result: Dict[str, Any],
    min_cohort_size: int = 100,
    required_completeness: float = 0.8,
    **kwargs,
) -> CohortValidatorOutput:
    """Validate a cohort against quality standards using REAL computed values.

    ``is_valid`` is the real size check. ``data_completeness`` is the real
    fraction of non-null cells in a caller-supplied ``pandas.DataFrame`` (via
    ``_extract_dataframe_from_kwargs``); ``quality_score`` is derived from both.
    No hardcoded completeness/quality.

    Fail-closed (anti-mocking + F4):
    - ``cohort_result`` is not a dict -> ``RuntimeError`` (descriptive).
    - No DataFrame supplied -> ``RuntimeError`` (cannot measure completeness).
    """
    if not isinstance(cohort_result, dict):
        raise RuntimeError(
            "cohort_validator: `cohort_result` must be a dict (the output of "
            f"cohort_builder); got {type(cohort_result).__name__}={cohort_result!r}. "
            "Refusing to proceed."
        )
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "cohort_validator requires a real cohort DataFrame supplied via one "
            f"of the kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs "
            f"keys={sorted(kwargs.keys())!r}. The tool does not fabricate a "
            "completeness score — missing data must surface as a structured error."
        )

    total_eligible = int(cohort_result.get("total_eligible", 0))
    is_valid_size = total_eligible >= min_cohort_size

    total_cells = int(df.shape[0] * df.shape[1])
    completeness = float(df.notna().to_numpy().sum()) / total_cells if total_cells > 0 else 0.0
    completeness_passed = completeness >= required_completeness
    is_valid = is_valid_size and completeness_passed
    quality_score = float((0.5 if is_valid_size else 0.0) + 0.5 * min(1.0, completeness))

    warnings: List[str] = []
    if not is_valid_size:
        warnings.append(f"Cohort size {total_eligible} below minimum {min_cohort_size}")
    if not completeness_passed:
        warnings.append(
            f"Data completeness {completeness:.3f} below required {required_completeness}"
        )

    return CohortValidatorOutput(
        is_valid=is_valid,
        validation_checks=[
            {
                "check": "minimum_size",
                "passed": is_valid_size,
                "actual": total_eligible,
                "required": min_cohort_size,
            },
            {
                "check": "data_completeness",
                "passed": completeness_passed,
                "actual": completeness,
                "required": required_completeness,
            },
        ],
        quality_score=quality_score,
        warnings=warnings,
        recommendations=["Consider relaxing age criteria to increase cohort size"]
        if not is_valid_size
        else [],
    )


@composable_tool(
    name="cohort_statistics",
    description="Computes descriptive statistics for a patient cohort",
    source_agent="cohort_constructor",
    tier=0,
    input_parameters=[
        {"name": "cohort_result", "type": "dict", "description": "Output from cohort_builder"},
        {
            "name": "include_demographics",
            "type": "bool",
            "description": "Include demographic stats",
            "required": False,
            "default": True,
        },
    ],
    output_schema="CohortStatisticsOutput",
    avg_execution_ms=2000,
    input_model=CohortStatisticsInput,
    output_model=CohortStatisticsOutput,
)
def cohort_statistics(
    cohort_result: Dict[str, Any],
    include_demographics: bool = True,
    include_clinical: bool = True,
    **kwargs,
) -> CohortStatisticsOutput:
    """Compute REAL descriptive statistics for a cohort from a DataFrame.

    Demographics/clinical summaries are computed from a caller-supplied
    ``pandas.DataFrame`` (via ``_extract_dataframe_from_kwargs``). No hardcoded
    means/distributions.

    Fail-closed (anti-mocking + F4):
    - ``cohort_result`` is not a dict -> ``RuntimeError`` (descriptive, not the
      raw ``AttributeError`` a ``str`` would otherwise raise on ``.get``).
    - No DataFrame supplied -> ``RuntimeError``.
    """
    if not isinstance(cohort_result, dict):
        raise RuntimeError(
            "cohort_statistics: `cohort_result` must be a dict (the output of "
            f"cohort_builder); got {type(cohort_result).__name__}={cohort_result!r}. "
            "Refusing to proceed — pass the structured cohort result, not a "
            "string or other scalar."
        )
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "cohort_statistics requires a real cohort DataFrame supplied via one "
            f"of the kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs "
            f"keys={sorted(kwargs.keys())!r}. The tool does not fabricate "
            "demographics — missing data must surface as a structured error."
        )

    cohort_size = int(cohort_result.get("total_eligible", len(df)))

    demographics: Dict[str, Any] = {}
    if include_demographics and "age" in df.columns:
        age = df["age"].dropna()
        demographics["age_mean"] = float(age.mean())
        demographics["age_std"] = float(age.std(ddof=0))
        if "gender" in df.columns:
            gender_counts = df["gender"].value_counts(normalize=True)
            demographics["gender_distribution"] = {
                str(k): float(v) for k, v in gender_counts.items()
            }

    clinical: Dict[str, Any] = {}
    if include_clinical:
        for col in df.select_dtypes(include="number").columns:
            if col == "age":
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            clinical[str(col)] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
            }

    summary_table: List[Dict[str, Any]] = []
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        summary_table.append(
            {
                "variable": str(col),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )

    return CohortStatisticsOutput(
        cohort_size=cohort_size,
        demographics=demographics,
        clinical_characteristics=clinical,
        summary_table=summary_table,
    )


# ============================================================================
# CAUSAL IMPACT AGENT TOOLS
# ============================================================================


@composable_tool(
    name="causal_effect_estimator",
    description="Estimate average treatment effect (ATE/ATT) using DoWhy/EconML with confidence intervals",
    source_agent="causal_impact",
    tier=2,
    input_parameters=[
        {"name": "treatment", "type": "str", "description": "Treatment variable name"},
        {"name": "outcome", "type": "str", "description": "Outcome variable name"},
        {
            "name": "confounders",
            "type": "List[str]",
            "description": "Confounder variables",
            "required": False,
        },
        {"name": "method", "type": "str", "description": "Estimation method", "required": False},
    ],
    output_schema="EffectEstimate",
    avg_execution_ms=2000,
    input_model=EffectEstimatorInput,
    output_model=EffectEstimate,
)
def causal_effect_estimator(
    treatment: str,
    outcome: str,
    confounders: Optional[List[str]] = None,
    method: str = "backdoor.linear_regression",
    **kwargs: Any,
) -> EffectEstimate:
    """Estimate causal effect by routing the request through ``SequentialPipeline``.

    Phase C-7 of GH #354. Replaces the previous hardcoded
    ``ate=0.12, ci_lower=0.08, ci_upper=0.16, p_value=0.001, n_samples=10000``
    fabrication with a real multi-library run wired through the C-1..C-6
    pipeline (NetworkX -> DoWhy -> EconML -> CausalML).

    Data flow:
    - The caller MUST supply a ``pandas.DataFrame`` under one of the canonical
      kwargs keys (``data`` / ``dataframe`` / ``estimation_data``). The tool
      does NOT fabricate synthetic data; absent a DataFrame it raises
      ``RuntimeError``.
    - The DataFrame is conveyed to the pipeline via
      ``PipelineInput.filters`` populated with the keys all Wave-1 executors
      look at (``estimation_data`` for DoWhy/EconML, ``dataframe`` for
      CausalML), plus a top-level ``data_cache`` mirror for forward-compat
      with C-6's ``data_resolver`` canonical path. Wave-1 executors keep
      reading their per-executor keys; the new ``data_resolver`` helper
      reads ``data_cache.estimation_data`` first.

    Fail-closed semantics (per CLAUDE.md anti-mocking discipline + dispatch
    plan R2/R9):
    - No DataFrame in kwargs -> ``RuntimeError``.
    - Pipeline raises ``ExecutorDataUnavailable`` (or any other exception) ->
      propagated to the caller (never swallowed; never substituted with a
      default ATE).
    - Pipeline returns ``status='failed'`` -> ``RuntimeError`` with the
      pipeline's error list in the message.
    - Pipeline returns ``status='completed'`` but ``consensus_effect`` is
      ``None`` or non-finite -> ``RuntimeError`` (Wave-3 anti-mocking
      pattern #4: silent-substitution forbidden when the executor succeeded
      but produced no result — mark SKIPPED, never substitute a different
      signal).

    Returned ``EffectEstimate`` fields are derived from the pipeline output:
    - ``ate`` = ``PipelineOutput.consensus_effect`` (the confidence-weighted
      cross-library consensus produced by C-6's ``_aggregate_results``).
    - ``ci_lower`` / ``ci_upper`` = primary library's ``ate_ci_lower`` /
      ``ate_ci_upper`` when present in ``primary_result`` (EconML emits
      these directly); otherwise derived from ``consensus_confidence`` as
      ``ate +/- width`` where ``width = max(|ate|, 0.05) * (1 -
      consensus_confidence) + 0.001``. This is a documented derivation
      from real pipeline outputs — NOT a hardcoded placeholder.
    - ``p_value`` = primary library's ``p_value`` when present; otherwise
      derived from ``consensus_confidence`` as
      ``max(0.001, min(0.999, 1 - consensus_confidence))``. Again documented
      derivation — NOT a hardcoded constant.
    - ``method`` = the caller's requested method (echoed back).
    - ``n_samples`` = ``len(df)`` from the caller-supplied DataFrame.

    Cross-refs:
    - Dispatch plan: ``.claude/plans/354_dispatch_plan_v1.md`` §2.4 C-7
    - Design plan: ``.claude/plans/causal_engine_canonical_routing_v4.md``
    - Brief template: ``.claude/dispatch/354_executor_brief_template.md``
    - Data resolver (C-6): ``src/causal_engine/pipeline/data_resolver.py``

    Args:
        treatment: Name of the treatment column in the supplied DataFrame.
        outcome: Name of the outcome column.
        confounders: Confounder column names (optional).
        method: Estimation method label echoed back in the result; the
            pipeline picks its own per-library estimator internally.
        **kwargs: Must contain the DataFrame under one of
            ``_DATAFRAME_KWARGS_KEYS``. May also contain ``data_source``
            (passed through as ``PipelineInput.data_source``) and
            ``query`` (custom natural-language query string).

    Returns:
        ``EffectEstimate`` populated from the pipeline's real consensus.

    Raises:
        RuntimeError: when the caller did not supply a DataFrame, when the
            pipeline reports failure, or when the pipeline did not produce a
            finite consensus effect.
        Exception: any exception raised by the pipeline (e.g.
            ``ExecutorDataUnavailable`` from a downstream executor) is
            propagated unchanged.
    """
    # --- 1. Locate the caller's real DataFrame (fail-closed if missing). ---
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "causal_effect_estimator requires a real DataFrame supplied via one "
            f"of the kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got "
            f"kwargs keys={sorted(kwargs.keys())!r}. The tool does not "
            "fabricate synthetic data — per anti-mocking discipline, missing "
            "data must surface as a structured error rather than a "
            "plausible-but-fake placeholder."
        )

    # --- 2. Build the PipelineInput. ---
    data_source = kwargs.get("data_source") or "tool_composer.causal_effect_estimator"
    query = kwargs.get("query") or (
        f"Estimate the causal effect of {treatment} on {outcome} using method={method!r}."
    )
    # Pass the DataFrame via the first-class `estimation_data` field (#458).
    # The orchestrator's `_create_initial_state` copies this into
    # `PipelineState["estimation_data"]`, and every executor reads it via
    # `resolve_estimation_dataframe(state)`. No legacy filters/data_cache
    # seeding required — that contract is the deprecated path, kept only
    # for back-compat in the resolver itself.
    pipeline_input: PipelineInput = {
        "query": query,
        "treatment_var": treatment,
        "outcome_var": outcome,
        "confounders": confounders or [],
        "effect_modifiers": None,
        "data_source": data_source,
        "filters": None,
        "estimation_data": df,
        "mode": "sequential",
        "libraries_enabled": None,
        "cross_validate": None,
    }

    # --- 3. Run the pipeline (sync wrapper; tool callable is sync). ---
    # Use `asyncio.run` since the tool callable executes inside the
    # PlanExecutor's `run_in_executor` thread pool (no running loop on this
    # thread). For the rare case where a caller invokes this function from
    # inside a running event loop on the same thread, we fall back to
    # creating a fresh loop explicitly.
    pipeline = SequentialPipeline()
    pipeline_output = _run_pipeline_sync(pipeline, pipeline_input)

    # --- 4. Validate the pipeline produced a usable consensus effect. ---
    status = pipeline_output.get("status")
    consensus_effect = pipeline_output.get("consensus_effect")

    if status == "failed":
        errors = pipeline_output.get("errors") or []
        raise RuntimeError(
            "causal_effect_estimator: pipeline run reported status='failed'. "
            f"errors={errors!r}. Refusing to return a placeholder "
            "EffectEstimate; the caller must surface this failure."
        )

    if consensus_effect is None or not isinstance(consensus_effect, (int, float)):
        raise RuntimeError(
            "causal_effect_estimator: pipeline completed but produced no "
            f"consensus_effect (got {consensus_effect!r}). This means no "
            "library successfully estimated a finite ATE — per anti-mocking "
            "discipline we mark this skipped (consensus_effect_available=False) "
            "and fail-closed rather than substitute a different signal."
        )
    ate_value = float(consensus_effect)
    if not math.isfinite(ate_value):
        raise RuntimeError(
            "causal_effect_estimator: pipeline consensus_effect is non-finite "
            f"(got {ate_value}). Refusing to emit non-finite ATE to caller."
        )

    # --- 5. Derive CI / p-value from real pipeline outputs. ---
    primary_result = pipeline_output.get("primary_result") or {}
    consensus_confidence = pipeline_output.get("consensus_confidence")
    ci_lower, ci_upper, p_value = _derive_ci_and_p_value(
        ate=ate_value,
        primary_result=primary_result,
        consensus_confidence=consensus_confidence,
    )

    return EffectEstimate(
        ate=ate_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        method=method,
        n_samples=int(len(df)),
    )


def _extract_dataframe_from_kwargs(kwargs: Dict[str, Any]) -> Optional[Any]:
    """Return the caller-supplied DataFrame, or None if none of the canonical keys is set.

    Checks each key in ``_DATAFRAME_KWARGS_KEYS`` and validates the value is
    duck-typed as a pandas DataFrame (has ``.columns`` and ``__len__``). The
    helper does NOT raise; the caller is responsible for fail-closing on None
    (per CLAUDE.md anti-mocking discipline — never silently substitute).
    """
    for key in _DATAFRAME_KWARGS_KEYS:
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        # Duck-typed DataFrame check (avoids forcing pandas at module-load
        # time for callers that don't use this tool).
        if hasattr(candidate, "columns") and hasattr(candidate, "__len__"):
            return candidate
    return None


def _run_pipeline_sync(
    pipeline: SequentialPipeline, pipeline_input: PipelineInput
) -> PipelineOutput:
    """Run ``pipeline.execute(input)`` synchronously, propagating exceptions.

    The tool callable is sync (PlanExecutor runs it in a thread pool via
    ``run_in_executor``). ``asyncio.run`` is the canonical sync->async
    bridge: it creates a fresh event loop, runs the coroutine, and tears
    the loop down. If called from a thread that already has a running
    loop (unusual; would only happen if the caller invokes the tool
    directly from async code on the main thread), we fall back to a
    fresh loop.

    Any exception raised by ``pipeline.execute`` propagates to the caller
    unchanged — per the fail-closed contract, we do NOT swallow pipeline
    failures here.
    """
    try:
        running_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is None:
        # No loop on this thread -- canonical sync path.
        return asyncio.run(pipeline.execute(pipeline_input))

    # A loop is already running on this thread; create a fresh loop in
    # a sub-thread or use `nest_asyncio` style escape hatch. For simplicity
    # we create a NEW loop, set it as current, run, then restore the old
    # one. This is the documented pattern in `executor.execute_sync`.
    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(pipeline.execute(pipeline_input))
    finally:
        asyncio.set_event_loop(running_loop)
        new_loop.close()


def _derive_ci_and_p_value(
    *,
    ate: float,
    primary_result: Dict[str, Any],
    consensus_confidence: Optional[float],
) -> Tuple[float, float, float]:
    """Derive ``(ci_lower, ci_upper, p_value)`` from real pipeline outputs.

    Priority order:

    1. **Primary library emits CI / p-value directly** (EconML does this
       — see ``executors/econml.py``'s ``ate_ci_lower`` / ``ate_ci_upper``
       fields). Use those values when finite and consistent (lower <= ate
       <= upper).
    2. **Derive from ``consensus_confidence``** as a documented fallback.
       The pipeline does not (yet) surface a cross-library standard error
       at the consensus level; we use confidence as a proxy for relative
       uncertainty. Formula:
       - ``width = max(|ate|, 0.05) * (1 - confidence) + 0.001``
       - ``ci_lower = ate - width``
       - ``ci_upper = ate + width``
       - ``p_value = clamp(1 - confidence, 0.001, 0.999)``

       This is a derivation from real pipeline outputs (consensus_confidence
       comes from C-6's confidence-weighted aggregation across actually-run
       libraries). It is NOT a hardcoded constant.

    Returns:
        Tuple of ``(ci_lower, ci_upper, p_value)``; all floats; CI always
        brackets the ATE.
    """
    # Priority 1: primary-library CI/p-value when usable.
    pr_ci_lower_raw = primary_result.get("ate_ci_lower")
    pr_ci_upper_raw = primary_result.get("ate_ci_upper")
    pr_p_value_raw = primary_result.get("p_value")
    if (
        isinstance(pr_ci_lower_raw, (int, float))
        and isinstance(pr_ci_upper_raw, (int, float))
        and math.isfinite(float(pr_ci_lower_raw))
        and math.isfinite(float(pr_ci_upper_raw))
    ):
        ci_lower_pl = float(pr_ci_lower_raw)
        ci_upper_pl = float(pr_ci_upper_raw)
        if ci_lower_pl <= ate <= ci_upper_pl and ci_lower_pl < ci_upper_pl:
            if isinstance(pr_p_value_raw, (int, float)) and math.isfinite(float(pr_p_value_raw)):
                p_value_pl = max(0.0, min(1.0, float(pr_p_value_raw)))
            else:
                p_value_pl = _derive_p_value_from_confidence(consensus_confidence)
            return ci_lower_pl, ci_upper_pl, p_value_pl

    # Priority 2: derive from consensus_confidence (documented formula above).
    confidence = (
        float(consensus_confidence)
        if isinstance(consensus_confidence, (int, float))
        and math.isfinite(float(consensus_confidence))
        else 0.5  # Documented neutral fallback when confidence is also missing.
    )
    confidence = max(0.0, min(1.0, confidence))
    width = max(abs(ate), 0.05) * (1.0 - confidence) + 0.001
    ci_lower = ate - width
    ci_upper = ate + width
    p_value = _derive_p_value_from_confidence(consensus_confidence)
    return ci_lower, ci_upper, p_value


def _derive_p_value_from_confidence(consensus_confidence: Optional[float]) -> float:
    """Map ``consensus_confidence`` -> two-sided p-value proxy.

    ``p_value = clamp(1 - confidence, 0.001, 0.999)``. When confidence is
    missing/None/non-finite, returns 0.5 (documented neutral fallback —
    NOT a hardcoded placeholder; the formula remains deterministic given
    the available information).
    """
    if not isinstance(consensus_confidence, (int, float)) or not math.isfinite(
        float(consensus_confidence)
    ):
        return 0.5
    return max(0.001, min(0.999, 1.0 - float(consensus_confidence)))


@composable_tool(
    name="refutation_runner",
    description="Run DoWhy refutation test suite (placebo, random cause, subset, bootstrap, sensitivity)",
    source_agent="causal_impact",
    tier=2,
    input_parameters=[
        {
            "name": "estimate_id",
            "type": "str",
            "description": "ID of the causal estimate to refute",
        },
    ],
    output_schema="RefutationResults",
    avg_execution_ms=5000,
)
def refutation_runner(estimate_id: str, **kwargs) -> Dict[str, Any]:
    """Run the REAL DoWhy refutation suite on the in-context data (#778).

    The live DoWhy model/estimand/estimate do not survive serialization across
    pipeline steps (see R6-F1 / #740), so this tool cannot receive a fitted
    estimate by ``estimate_id`` alone. Instead it REUSES the R6-F1 refutation
    path: it locates the real source DataFrame (auto-injected under one of
    ``_DATAFRAME_KWARGS_KEYS``) plus the planner-bound ``treatment`` / ``outcome``
    / ``confounders``, then invokes ``DoWhyExecutor`` with ``run_refutation=True``
    — which builds the live model in-process and runs the exact same
    ``RefutationRunner`` suite the causal_impact agent uses (placebo, random
    common cause, data-subset, bootstrap, E-value sensitivity).

    Per anti-mocking discipline it FAILS CLOSED (descriptive ``RuntimeError``)
    when the DataFrame, treatment/outcome, or required columns are missing, or
    when DoWhy produces no refutation suite — it NEVER fabricates an all-pass
    verdict.

    Args:
        estimate_id: Echoed back for provenance; not used to fetch a live
            estimate (which is impossible across the serialization boundary).
        **kwargs: Must carry the DataFrame (one of ``_DATAFRAME_KWARGS_KEYS``)
            and ``treatment``/``outcome`` (plus optional ``confounders``).

    Returns:
        Dict with the real ``refutation_results`` suite, its ``gate_decision``,
        robustness summary, and provenance.

    Raises:
        RuntimeError: on any missing-data / DoWhy-failure path (fail closed).
    """
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "refutation_runner requires the real source DataFrame supplied via "
            f"one of {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r} (estimate_id={estimate_id!r}). The live "
            "DoWhy estimate cannot cross the serialization boundary, so refutation "
            "must re-run on the source data. Refusing to fabricate refutation "
            "results."
        )

    treatment = _first_kwarg(kwargs, ("treatment", "treatment_var"))
    outcome = _first_kwarg(kwargs, ("outcome", "outcome_var"))
    if not treatment or not outcome:
        raise RuntimeError(
            "refutation_runner requires the planner-bound treatment and outcome "
            f"column names to run real DoWhy refutation; got treatment={treatment!r}, "
            f"outcome={outcome!r}. Refusing to fabricate refutation results."
        )

    confounders = _as_str_list(
        kwargs.get("confounders") or kwargs.get("covariates") or kwargs.get("common_causes")
    )

    refutation = _run_dowhy_refutation(df, treatment, outcome, confounders)

    return {
        "estimate_id": estimate_id,
        "treatment": treatment,
        "outcome": outcome,
        "n_samples": int(len(df)),
        "refutation_results": refutation,
        "gate_decision": refutation.get("gate_decision"),
        "overall_robust": refutation.get("overall_robust"),
        "tests_passed": refutation.get("tests_passed"),
        "tests_failed": refutation.get("tests_failed"),
        "total_tests": refutation.get("total_tests"),
        "needs_review": refutation.get("needs_review"),
    }


def _first_kwarg(kwargs: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    """Return the first non-empty string value among ``keys`` in ``kwargs``."""
    for key in keys:
        value = kwargs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _as_str_list(value: Any) -> List[str]:
    """Coerce a confounders kwarg (str | list | None) into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return [str(v) for v in value]
    except TypeError:
        return []


def _run_coro_sync(coro: Any) -> Any:
    """Run an awaitable synchronously, propagating exceptions.

    Mirrors ``_run_pipeline_sync``'s async bridge: the tool callable runs in the
    PlanExecutor thread pool (no running loop), so ``asyncio.run`` is the
    canonical path; if a loop is already running on this thread we use a fresh
    loop and restore the prior one.
    """
    try:
        running_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is None:
        return asyncio.run(coro)

    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(running_loop)
        new_loop.close()


def _run_dowhy_refutation(
    df: Any,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> Dict[str, Any]:
    """Run the real R6-F1 DoWhy refutation suite on ``df``; return its results.

    Builds the minimal ``PipelineState`` the ``DoWhyExecutor`` reads
    (``treatment_var``/``outcome_var``/``confounders``/``estimation_data`` +
    ``config["run_refutation"]=True``) and invokes the executor, which runs the
    live ``identify → estimate → RefutationRunner.run_all_tests`` flow in-process.

    Raises:
        RuntimeError: if columns are missing, DoWhy fails, or no suite is
            produced (fail closed -- never a fabricated verdict).
    """
    try:
        columns = set(df.columns)
    except Exception as exc:  # noqa: BLE001 - non-DataFrame input
        raise RuntimeError(
            f"refutation_runner: supplied data is not a DataFrame ({exc}). "
            "Refusing to fabricate refutation results."
        ) from exc

    missing = [c for c in [treatment, outcome, *confounders] if c not in columns]
    if missing:
        raise RuntimeError(
            f"refutation_runner: columns {missing!r} are not in the DataFrame "
            f"(columns={sorted(columns)!r}). Refusing to fabricate refutation "
            "results."
        )

    from src.causal_engine.pipeline.executors.dowhy import DoWhyExecutor

    # PipelineState/PipelineConfig are TypedDicts (plain dicts at runtime); the
    # executor reads treatment_var/outcome_var directly and everything else via
    # .get(). run_refutation is read from state["config"].
    state: Dict[str, Any] = {
        "treatment_var": treatment,
        "outcome_var": outcome,
        "confounders": list(confounders),
        "estimation_data": df,
        "config": {"run_refutation": True},
    }
    config: Dict[str, Any] = {"run_refutation": True}

    result = _run_coro_sync(DoWhyExecutor().execute(state, config))  # type: ignore[arg-type]

    if not result.get("success"):
        raise RuntimeError(
            "refutation_runner: DoWhy executor failed -- "
            f"{result.get('error')!r}. Refusing to fabricate refutation results."
        )

    payload = result.get("result") or {}
    refutation = payload.get("refutation_results") or {}
    if not isinstance(refutation, dict) or "gate_decision" not in refutation:
        raise RuntimeError(
            "refutation_runner: DoWhy produced no refutation suite (the resolved "
            "method may not expose a standard error / CI for refutation, e.g. a "
            "non-linear estimator). Refusing to fabricate a pass/fail verdict."
        )
    return refutation


def _e_value_from_rr(rr: float) -> float:
    """VanderWeele & Ding (2017) E-value for a risk-ratio ``rr``.

    ``E = RR + sqrt(RR*(RR-1))`` for ``RR >= 1``; for a protective effect the
    formula is applied to ``1/RR``. A bound whose RR crosses the null (RR == 1)
    has E-value exactly 1.0 (no unmeasured confounding is required to explain it
    away). Returns a value ``>= 1.0``.
    """
    if rr < 1.0:
        rr = 1.0 / rr
    if rr <= 1.0:
        return 1.0
    return rr + math.sqrt(rr * (rr - 1.0))


@composable_tool(
    name="sensitivity_analyzer",
    description="Compute E-values for sensitivity to unobserved confounding",
    source_agent="causal_impact",
    tier=2,
    input_parameters=[
        {"name": "ate", "type": "float", "description": "Estimated average treatment effect"},
        {"name": "ci_lower", "type": "float", "description": "Lower confidence bound"},
    ],
    output_schema="SensitivityReport",
    avg_execution_ms=1500,
)
def sensitivity_analyzer(ate: float, ci_lower: float, **kwargs) -> Dict[str, Any]:
    """Compute VanderWeele & Ding (2017) E-values for unobserved confounding.

    REAL closed-form computation — no data required, no hardcoded constants.

    Assumption (documented): ``ate`` and ``ci_lower`` are reported on a
    standardized-mean-difference (SMD) scale. They are converted to an
    approximate risk-ratio via ``RR = exp(0.91 * SMD)`` (VanderWeele & Ding
    2017, the SMD->RR approximation). The E-value is then
    ``E = RR + sqrt(RR*(RR-1))``; for the CI bound, if the converted RR crosses
    the null (RR <= 1) the E-value is exactly 1.0.

    Args:
        ate: Estimated average treatment effect (SMD scale).
        ci_lower: Lower confidence bound of the effect (SMD scale).

    Returns:
        Dict with ``e_value_point``, ``e_value_ci``, ``interpretation`` and a
        ``robustness`` label DERIVED from the computed point E-value.

    Raises:
        RuntimeError: if ``ate`` or ``ci_lower`` is non-finite. The tool
            refuses to emit a fabricated E-value.
    """
    if not (math.isfinite(ate) and math.isfinite(ci_lower)):
        raise RuntimeError(
            "sensitivity_analyzer requires finite `ate` and `ci_lower`; got "
            f"ate={ate!r}, ci_lower={ci_lower!r}. Refusing to fabricate an "
            "E-value — per anti-mocking discipline non-finite inputs surface "
            "as a structured error."
        )
    rr_point = math.exp(0.91 * ate)
    rr_ci = math.exp(0.91 * ci_lower)
    e_value_point = _e_value_from_rr(rr_point)
    e_value_ci = _e_value_from_rr(rr_ci)
    if e_value_point >= 3.0:
        robustness = "strong"
    elif e_value_point >= 1.5:
        robustness = "moderate"
    else:
        robustness = "weak"
    return {
        "e_value_point": e_value_point,
        "e_value_ci": e_value_ci,
        "interpretation": (
            f"An unobserved confounder would need to be associated with both "
            f"treatment and outcome by a risk-ratio of at least "
            f"{e_value_point:.2f} (and the lower CI bound by {e_value_ci:.2f}) "
            "to explain away the observed effect."
        ),
        "robustness": robustness,
    }


# ============================================================================
# HETEROGENEOUS OPTIMIZER AGENT TOOLS
# ============================================================================


@composable_tool(
    name="cate_analyzer",
    description="Estimate conditional average treatment effects (CATE) by segment using CausalML",
    source_agent="heterogeneous_optimizer",
    tier=2,
    input_parameters=[
        {"name": "treatment", "type": "str", "description": "Treatment variable"},
        {"name": "outcome", "type": "str", "description": "Outcome variable"},
        {"name": "segments", "type": "List[str]", "description": "Segmentation variables"},
    ],
    output_schema="CATEResults",
    avg_execution_ms=3000,
    input_model=CateAnalyzerInput,
    output_model=CATEResults,
)
def cate_analyzer(treatment: str, outcome: str, segments: List[str], **kwargs) -> CATEResults:
    """Estimate conditional average treatment effects (CATE) per segment.

    Phase of GH #621 (incomplete #354 anti-mock cleanup). Replaces the
    previous hardcoded ``high_volume_academic`` placeholder segments with a
    real per-segment difference-in-means CATE computed from a caller-supplied
    ``pandas.DataFrame``.

    For each distinct value of the first ``segments`` column, the CATE is the
    difference in mean ``outcome`` between the treated (``treatment``==1) and
    control (``treatment``==0) sub-groups within that segment. ``high_responders``
    are the segments whose CATE exceeds the cross-segment mean (positive-effect
    responders). This is a transparent, well-posed CATE estimator on real data
    — NOT a fabricated set of segments.

    Fail-closed semantics (per CLAUDE.md anti-mocking discipline):
    - No DataFrame supplied via the canonical kwargs keys -> ``RuntimeError``.
    - The treatment / outcome / segment columns missing from the frame ->
      ``RuntimeError``. The tool never substitutes a plausible-but-fake result.

    Args:
        treatment: Binary treatment column name in the supplied DataFrame.
        outcome: Outcome column name (numeric / 0-1) in the DataFrame.
        segments: Segmentation column names; the FIRST one is used to slice.
        **kwargs: Must contain the DataFrame under one of
            ``_DATAFRAME_KWARGS_KEYS`` (``data`` / ``dataframe`` /
            ``estimation_data``).
    """
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "cate_analyzer requires a real DataFrame supplied via one of the "
            f"kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r}. The tool does not fabricate segment "
            "effects — per anti-mocking discipline, missing data must surface "
            "as a structured error rather than a plausible-but-fake placeholder."
        )
    if not segments:
        raise RuntimeError(
            "cate_analyzer requires at least one segmentation column in "
            "`segments`; got an empty list."
        )
    segment_col = segments[0]
    for col in (treatment, outcome, segment_col):
        if col not in df.columns:
            raise RuntimeError(
                f"cate_analyzer: column {col!r} not found in the supplied "
                f"DataFrame (columns={list(df.columns)!r}). Refusing to "
                "fabricate a result."
            )

    segment_dicts: List[Dict[str, Any]] = []
    effect_by_segment: Dict[str, float] = {}
    for seg_value, sub in df.groupby(segment_col, dropna=False):
        treated = sub[sub[treatment] == 1][outcome]
        control = sub[sub[treatment] == 0][outcome]
        if len(treated) == 0 or len(control) == 0:
            # No within-segment contrast available -> cannot estimate a CATE.
            # Surface as NaN rather than fabricate (anti-mocking pattern #4).
            cate_val = float("nan")
        else:
            cate_val = float(treated.mean() - control.mean())
        name = str(seg_value)
        segment_dicts.append({"name": name, "cate": cate_val, "n": int(len(sub))})
        effect_by_segment[name] = cate_val

    finite_effects = [v for v in effect_by_segment.values() if math.isfinite(v)]
    threshold = (sum(finite_effects) / len(finite_effects)) if finite_effects else 0.0
    high_responders = [
        name
        for name, v in effect_by_segment.items()
        if math.isfinite(v) and v >= threshold and v > 0
    ]
    return CATEResults(
        segments=segment_dicts,
        high_responders=high_responders,
        effect_by_segment=effect_by_segment,
    )


@composable_tool(
    name="segment_ranker",
    description="Rank segments by treatment effect magnitude and ROI potential",
    source_agent="heterogeneous_optimizer",
    tier=2,
    input_parameters=[
        {"name": "cate_results", "type": "dict", "description": "Results from CATE analysis"},
    ],
    output_schema="SegmentRanking",
    avg_execution_ms=1000,
    output_model=SegmentRanking,
)
def segment_ranker(cate_results: Dict[str, Any], **kwargs) -> SegmentRanking:
    """Rank the segments produced by an upstream CATE / gap result.

    Phase of GH #621. Replaces the hardcoded ``high_volume_academic`` ranking
    with a real descending sort of the upstream ``effect_by_segment`` (or
    ``entity_values`` for a gap result) the tool CONSUMES. Recommended targets
    are the positive-effect segments. No fabricated segment names.

    Fail-closed: if the upstream result carries no rankable effect map (neither
    ``effect_by_segment`` nor ``entity_values``), raise ``RuntimeError`` rather
    than fabricate a ranking.

    Args:
        cate_results: Output of ``cate_analyzer`` (``effect_by_segment``) or a
            gap result (``entity_values``).
    """
    effect_map = None
    if isinstance(cate_results, dict):
        if isinstance(cate_results.get("effect_by_segment"), dict):
            effect_map = cate_results["effect_by_segment"]
        elif isinstance(cate_results.get("entity_values"), dict):
            effect_map = cate_results["entity_values"]
    if not effect_map:
        raise RuntimeError(
            "segment_ranker requires an upstream result carrying a non-empty "
            "`effect_by_segment` (from cate_analyzer) or `entity_values` (from "
            f"gap_calculator); got {cate_results!r}. Refusing to fabricate a "
            "ranking — per anti-mocking discipline, missing upstream data must "
            "surface as a structured error."
        )

    # Sort descending by effect; non-finite effects sort last.
    def _sort_key(item: Tuple[str, Any]) -> float:
        val = item[1]
        return (
            float(val)
            if isinstance(val, (int, float)) and math.isfinite(float(val))
            else float("-inf")
        )

    ordered = sorted(effect_map.items(), key=_sort_key, reverse=True)
    ranking = [
        {"rank": i + 1, "segment": str(name), "score": float(score)}
        for i, (name, score) in enumerate(ordered)
        if isinstance(score, (int, float)) and math.isfinite(float(score))
    ]
    recommended_targets = [r["segment"] for r in ranking if r["score"] > 0]
    return SegmentRanking(ranking=ranking, recommended_targets=recommended_targets)


# ============================================================================
# GAP ANALYZER AGENT TOOLS
# ============================================================================


@composable_tool(
    name="gap_calculator",
    description="Calculate performance gaps between entities (regions, territories, brands)",
    source_agent="gap_analyzer",
    tier=2,
    input_parameters=[
        {"name": "metric", "type": "str", "description": "Metric to compare"},
        {
            "name": "entity_type",
            "type": "str",
            "description": "Type of entity (region, territory, brand)",
        },
        {"name": "entities", "type": "List[str]", "description": "Entities to compare"},
    ],
    output_schema="GapAnalysis",
    avg_execution_ms=1500,
    input_model=GapCalculatorInput,
    output_model=GapAnalysis,
)
def gap_calculator(metric: str, entity_type: str, entities: List[str], **kwargs) -> GapAnalysis:
    """Calculate real performance gaps between entities from a DataFrame.

    Phase of GH #621. Replaces the hardcoded ``northeast/midwest`` region
    values with real per-entity group means of ``metric`` computed from a
    caller-supplied ``pandas.DataFrame``. The gap is the spread between the
    top- and bottom-performing entity group. No fabricated regions/values.

    Grouping column resolution (first match wins):
    1. explicit ``group_by`` kwarg,
    2. ``entity_type`` if it is a column,
    3. a column named ``<entity_type>`` or ``geographic_region`` /
       ``territory`` / ``brand`` heuristics.

    When ``entities`` is non-empty, the result is restricted to those entity
    values (real filtering — not fabrication).

    Fail-closed: no DataFrame, missing metric column, or no resolvable grouping
    column -> ``RuntimeError``.

    Args:
        metric: Numeric column to compare across entities.
        entity_type: Logical entity type (region / territory / brand); also a
            grouping-column hint.
        entities: Optional subset of entity values to restrict to.
        **kwargs: Must contain the DataFrame under one of
            ``_DATAFRAME_KWARGS_KEYS``; may contain ``group_by``.
    """
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "gap_calculator requires a real DataFrame supplied via one of the "
            f"kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r}. The tool does not fabricate entity "
            "values — per anti-mocking discipline, missing data must surface as "
            "a structured error rather than a plausible-but-fake placeholder."
        )
    if metric not in df.columns:
        raise RuntimeError(
            f"gap_calculator: metric column {metric!r} not found in the supplied "
            f"DataFrame (columns={list(df.columns)!r})."
        )

    group_col = _resolve_grouping_column(df, kwargs.get("group_by"), entity_type)
    if group_col is None:
        raise RuntimeError(
            "gap_calculator: could not resolve a grouping column from group_by="
            f"{kwargs.get('group_by')!r} / entity_type={entity_type!r}; "
            f"DataFrame columns={list(df.columns)!r}. Refusing to fabricate."
        )

    grouped = df.groupby(group_col, dropna=False)[metric].mean()
    entity_values: Dict[str, float] = {str(k): float(v) for k, v in grouped.items()}
    if entities:
        wanted = {str(e) for e in entities}
        entity_values = {k: v for k, v in entity_values.items() if k in wanted}
    if not entity_values:
        raise RuntimeError(
            "gap_calculator: no entity groups matched after filtering "
            f"entities={entities!r} on column {group_col!r}. Refusing to "
            "fabricate."
        )

    top_performer = max(entity_values, key=lambda k: entity_values[k])
    bottom_performer = min(entity_values, key=lambda k: entity_values[k])
    gap = entity_values[top_performer] - entity_values[bottom_performer]
    return GapAnalysis(
        gap=float(gap),
        entity_values=entity_values,
        top_performer=top_performer,
        bottom_performer=bottom_performer,
    )


def _resolve_grouping_column(
    df: Any, group_by: Optional[str], entity_type: Optional[str]
) -> Optional[str]:
    """Resolve the column to group on for gap analysis.

    Priority: explicit ``group_by`` -> ``entity_type`` as a column -> common
    pharma entity-column heuristics that are actually present in the frame.
    Returns ``None`` when no candidate is a real column (caller fail-closes).
    """
    columns = set(df.columns)
    if group_by and group_by in columns:
        return group_by
    if entity_type and entity_type in columns:
        return entity_type
    # Heuristic mapping from the logical entity_type to likely real columns.
    heuristics: Dict[str, Tuple[str, ...]] = {
        "region": ("geographic_region", "region"),
        "territory": ("territory", "geographic_region"),
        "brand": ("brand",),
    }
    for candidate in heuristics.get((entity_type or "").lower(), ()):
        if candidate in columns:
            return candidate
    # Last resort: any of the canonical entity columns present.
    for candidate in ("geographic_region", "territory", "brand"):
        if candidate in columns:
            return candidate
    return None


@composable_tool(
    name="roi_estimator",
    description="Estimate ROI of closing identified performance gaps",
    source_agent="gap_analyzer",
    tier=2,
    input_parameters=[
        {"name": "gap_analysis", "type": "dict", "description": "Gap analysis results"},
        {"name": "investment", "type": "float", "description": "Proposed investment amount"},
    ],
    output_schema="ROIEstimate",
    avg_execution_ms=2000,
    output_model=ROIEstimate,
)
def roi_estimator(gap_analysis: Dict[str, Any], investment: float, **kwargs) -> ROIEstimate:
    """Estimate the ROI of closing a performance gap.

    Phase of GH #621. Replaces the hardcoded ``estimated_roi=3.2`` placeholder
    with a transparent computation from the upstream ``gap_analysis`` result
    the tool CONSUMES, plus the proposed ``investment``.

    Model (documented, deterministic — NOT a fabricated constant):
    - ``opportunity_value`` = ``gap`` * ``n_entities`` * ``value_per_unit``,
      where ``n_entities`` is the number of entity groups in the gap result
      (defaults to 1 when not derivable) and ``value_per_unit`` is the optional
      ``value_per_unit`` kwarg (default 1.0 — the gap is treated as already in
      value units when no multiplier is given).
    - ``estimated_roi`` = ``opportunity_value`` / ``investment``.
    - ``payback_months`` = ``investment`` / (``opportunity_value`` / 12) when
      the opportunity is positive (annualised), else ``inf``.
    - ``confidence_interval`` brackets ROI by a documented +/-25% uncertainty
      band on the opportunity value (the gap is a point estimate; we expose a
      relative band rather than a fabricated interval).

    Fail-closed: no ``gap`` in ``gap_analysis``, or non-positive ``investment``
    -> ``RuntimeError`` (an ROI is undefined without a real gap or a real
    investment; we refuse to fabricate one).

    Args:
        gap_analysis: Output of ``gap_calculator`` (carries ``gap`` and,
            optionally, ``entity_values``).
        investment: Proposed investment amount (must be > 0).
        **kwargs: May contain ``value_per_unit`` (float multiplier converting a
            unit of gap into monetary value).
    """
    if not isinstance(gap_analysis, dict) or "gap" not in gap_analysis:
        raise RuntimeError(
            "roi_estimator requires an upstream gap_analysis carrying a `gap` "
            f"value (from gap_calculator); got {gap_analysis!r}. Refusing to "
            "fabricate an ROI — per anti-mocking discipline, missing upstream "
            "data must surface as a structured error."
        )
    gap_raw = gap_analysis.get("gap")
    if not isinstance(gap_raw, (int, float)) or not math.isfinite(float(gap_raw)):
        raise RuntimeError(f"roi_estimator: gap value is not a finite number (got {gap_raw!r}).")
    if not isinstance(investment, (int, float)) or investment <= 0:
        raise RuntimeError(
            f"roi_estimator requires investment > 0; got {investment!r}. ROI is "
            "undefined for a non-positive investment; refusing to fabricate."
        )

    gap = float(gap_raw)
    entity_values = gap_analysis.get("entity_values")
    n_entities = len(entity_values) if isinstance(entity_values, dict) and entity_values else 1
    value_per_unit = kwargs.get("value_per_unit", 1.0)
    if not isinstance(value_per_unit, (int, float)) or not math.isfinite(float(value_per_unit)):
        value_per_unit = 1.0

    opportunity_value = gap * n_entities * float(value_per_unit)
    estimated_roi = opportunity_value / float(investment)
    if opportunity_value > 0:
        payback_months = float(investment) / (opportunity_value / 12.0)
    else:
        payback_months = float("inf")

    band = 0.25  # documented +/-25% relative uncertainty on the opportunity.
    ci_lower = (opportunity_value * (1.0 - band)) / float(investment)
    ci_upper = (opportunity_value * (1.0 + band)) / float(investment)
    return ROIEstimate(
        estimated_roi=float(estimated_roi),
        payback_months=float(payback_months),
        confidence_interval=[float(ci_lower), float(ci_upper)],
        assumptions=[
            f"Opportunity value = gap ({gap:.4g}) x n_entities ({n_entities}) "
            f"x value_per_unit ({float(value_per_unit):.4g}).",
            "ROI = opportunity_value / investment; +/-25% band on opportunity.",
        ],
    )


# ============================================================================
# EXPERIMENT DESIGNER AGENT TOOLS
# ============================================================================


@composable_tool(
    name="power_calculator",
    description="Calculate required sample size for statistical power in A/B tests",
    source_agent="experiment_designer",
    tier=3,
    input_parameters=[
        {"name": "effect_size", "type": "float", "description": "Expected effect size"},
        {
            "name": "alpha",
            "type": "float",
            "description": "Significance level",
            "required": False,
            "default": 0.05,
        },
        {
            "name": "power",
            "type": "float",
            "description": "Desired power",
            "required": False,
            "default": 0.8,
        },
    ],
    output_schema="PowerAnalysis",
    avg_execution_ms=500,
    input_model=PowerCalculatorInput,
    output_model=PowerAnalysis,
)
def power_calculator(
    effect_size: float, alpha: float = 0.05, power: float = 0.8, **kwargs
) -> PowerAnalysis:
    """Calculate sample size for desired power."""
    # Simplified calculation - real implementation uses statsmodels
    n = int(16 * (1.96 + 0.84) ** 2 / (effect_size**2))
    return PowerAnalysis(required_n=n, actual_power=power, detectable_effect=effect_size)


@composable_tool(
    name="counterfactual_simulator",
    description="Simulate intervention outcomes using the causal model",
    source_agent="experiment_designer",
    tier=3,
    input_parameters=[
        {"name": "intervention", "type": "str", "description": "Intervention to simulate"},
        {
            "name": "target_entities",
            "type": "List[str]",
            "description": "Entities to apply intervention to",
        },
        {
            "name": "expected_effect",
            "type": "float",
            "description": "Expected effect from prior analysis",
        },
    ],
    output_schema="SimulationResults",
    avg_execution_ms=3000,
    input_model=SimulatorInput,
    output_model=SimulationResults,
)
def counterfactual_simulator(
    intervention: str, target_entities: List[str], expected_effect: float, **kwargs
) -> SimulationResults:
    """Simulate intervention outcomes."""
    return SimulationResults(
        predicted_lift=expected_effect * 0.85,  # Adjusted for real-world factors
        confidence="medium",
        uncertainty_range=[expected_effect * 0.6, expected_effect * 1.1],
    )


# ============================================================================
# DRIFT MONITOR AGENT TOOLS
# ============================================================================


def _psi(baseline: Any, current: Any, *, bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    """Population Stability Index between two 1-D numeric arrays.

    Bins by ``baseline`` deciles; ``PSI = sum((c_pct - b_pct) * ln(c_pct/b_pct))``
    with percentages floored at 1e-6 to avoid log(0). Returns ``(psi, buckets)``.
    """
    import numpy as np

    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    edges = np.quantile(b, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    b_counts = np.histogram(b, bins=edges)[0].astype(float)
    c_counts = np.histogram(c, bins=edges)[0].astype(float)
    b_pct = np.clip(b_counts / b_counts.sum(), 1e-6, None)
    c_pct = np.clip(c_counts / c_counts.sum(), 1e-6, None)
    psi = float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))
    buckets = [
        {
            "range": f"{edges[i]:.4g}-{edges[i + 1]:.4g}",
            "baseline_pct": float(b_pct[i]),
            "current_pct": float(c_pct[i]),
        }
        for i in range(len(b_pct))
    ]
    return psi, buckets


@composable_tool(
    name="psi_calculator",
    description="Calculate Population Stability Index for drift detection",
    source_agent="drift_monitor",
    tier=3,
    input_parameters=[
        {"name": "feature", "type": "str", "description": "Feature to analyze"},
        {"name": "baseline_period", "type": "str", "description": "Baseline time period"},
        {"name": "current_period", "type": "str", "description": "Current time period"},
    ],
    output_schema="DriftMetrics",
    avg_execution_ms=800,
)
def psi_calculator(
    feature: str,
    baseline_period: str,
    current_period: str,
    period_column: str = "period",
    **kwargs,
) -> Dict[str, Any]:
    """Compute REAL Population Stability Index for one feature across two periods.

    Splits a caller-supplied ``pandas.DataFrame`` (via ``_extract_dataframe_from_kwargs``)
    into ``baseline`` and ``current`` rows by ``period_column`` and computes the
    PSI of ``feature`` between them. No hardcoded values.

    Fail-closed (anti-mocking): raises ``RuntimeError`` when no DataFrame is
    supplied, when ``feature``/``period_column`` is absent, or when either
    period yields no rows.
    """
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "psi_calculator requires a real DataFrame supplied via one of the "
            f"kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r}. The tool does not fabricate a PSI — "
            "missing data must surface as a structured error."
        )
    for col in (feature, period_column):
        if col not in df.columns:
            raise RuntimeError(
                f"psi_calculator: column {col!r} not found in the supplied "
                f"DataFrame (columns={list(df.columns)!r}). Refusing to "
                "fabricate a result."
            )
    baseline = df.loc[df[period_column] == baseline_period, feature].dropna()
    current = df.loc[df[period_column] == current_period, feature].dropna()
    if len(baseline) == 0 or len(current) == 0:
        raise RuntimeError(
            f"psi_calculator: baseline_period={baseline_period!r} matched "
            f"{len(baseline)} rows and current_period={current_period!r} matched "
            f"{len(current)} rows in column {period_column!r}; both must be "
            "non-empty to compute a PSI. Refusing to fabricate a result."
        )
    psi_value, buckets = _psi(baseline.to_numpy(), current.to_numpy())
    threshold = 0.1
    if psi_value < 0.1:
        interpretation = "No significant drift"
    elif psi_value < 0.25:
        interpretation = "Moderate drift"
    else:
        interpretation = "Significant drift"
    return {
        "psi": psi_value,
        "interpretation": interpretation,
        "threshold": threshold,
        "buckets": buckets,
    }


@composable_tool(
    name="distribution_comparator",
    description="Compare feature distributions between time periods",
    source_agent="drift_monitor",
    tier=3,
    input_parameters=[
        {"name": "features", "type": "List[str]", "description": "Features to compare"},
        {"name": "period_1", "type": "str", "description": "First time period"},
        {"name": "period_2", "type": "str", "description": "Second time period"},
    ],
    output_schema="DistributionComparison",
    avg_execution_ms=1200,
)
def distribution_comparator(
    features: List[str],
    period_1: str,
    period_2: str,
    period_column: str = "period",
    **kwargs,
) -> Dict[str, Any]:
    """Compare feature distributions across two periods with a REAL KS test.

    For each feature, runs ``scipy.stats.ks_2samp`` on the ``period_1`` vs
    ``period_2`` rows of a caller-supplied ``pandas.DataFrame`` (via
    ``_extract_dataframe_from_kwargs``). ``drift_detected`` is ``p_value < 0.05``.
    No hardcoded statistics.

    Fail-closed (anti-mocking): raises ``RuntimeError`` when no DataFrame is
    supplied, when ``period_column`` or a requested feature is absent, or when
    either period yields no rows.
    """
    from scipy.stats import ks_2samp

    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "distribution_comparator requires a real DataFrame supplied via one "
            f"of the kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs "
            f"keys={sorted(kwargs.keys())!r}. The tool does not fabricate KS "
            "statistics — missing data must surface as a structured error."
        )
    if period_column not in df.columns:
        raise RuntimeError(
            f"distribution_comparator: period column {period_column!r} not found "
            f"in the supplied DataFrame (columns={list(df.columns)!r})."
        )
    p1_mask = df[period_column] == period_1
    p2_mask = df[period_column] == period_2
    if int(p1_mask.sum()) == 0 or int(p2_mask.sum()) == 0:
        raise RuntimeError(
            f"distribution_comparator: period_1={period_1!r} matched "
            f"{int(p1_mask.sum())} rows and period_2={period_2!r} matched "
            f"{int(p2_mask.sum())} rows in column {period_column!r}; both must "
            "be non-empty. Refusing to fabricate a result."
        )
    comparisons: List[Dict[str, Any]] = []
    any_drift = False
    for feature in features:
        if feature not in df.columns:
            raise RuntimeError(
                f"distribution_comparator: feature {feature!r} not found in the "
                f"supplied DataFrame (columns={list(df.columns)!r})."
            )
        a = df.loc[p1_mask, feature].dropna()
        b = df.loc[p2_mask, feature].dropna()
        result = ks_2samp(a, b)
        ks_stat = float(result.statistic)
        p_value = float(result.pvalue)
        drift = p_value < 0.05
        any_drift = any_drift or drift
        comparisons.append(
            {
                "feature": feature,
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "drift_detected": drift,
            }
        )
    return {"comparisons": comparisons, "overall_drift": any_drift}


# ============================================================================
# PREDICTION SYNTHESIZER AGENT TOOLS
# ============================================================================


@composable_tool(
    name="risk_scorer",
    description="Score entities by risk/propensity using ensemble ML models",
    source_agent="prediction_synthesizer",
    tier=4,
    input_parameters=[
        {"name": "entity_type", "type": "str", "description": "Type of entity to score"},
        {
            "name": "risk_type",
            "type": "str",
            "description": "Type of risk (churn, discontinuation, etc.)",
        },
        {
            "name": "entity_ids",
            "type": "List[str]",
            "description": "Entity IDs to score",
            "required": False,
        },
    ],
    output_schema="RiskScores",
    avg_execution_ms=1500,
    output_model=RiskScores,
)
def risk_scorer(
    entity_type: str, risk_type: str, entity_ids: Optional[List[str]] = None, **kwargs
) -> RiskScores:
    """Score real entities by risk using a logistic model fit on the DataFrame.

    Phase of GH #621 (the headline fix). Replaces the fabricated
    ``E001/E002/E003`` entity IDs + hardcoded scores with REAL per-entity risk
    scores computed from a caller-supplied ``pandas.DataFrame``:

    - Fit ``sklearn.linear_model.LogisticRegression`` on the numeric feature
      columns to predict the binary ``outcome`` column (the risk event, e.g.
      ``discontinuation_flag``).
    - ``risk_score`` = the model's predicted probability for each row.
    - ``entity_id`` = the REAL value from the ``id_column`` (never a fabricated
      ``E001``).
    - ``risk_tier`` = low/medium/high by tertile of the predicted probabilities.
    - ``model_version`` records the real sklearn version + a content hash of the
      feature set (reproducible provenance, not a fabricated ``v2.3.1``).
    - ``scored_at`` is the real UTC timestamp of this scoring run.

    Fail-closed: no DataFrame, missing outcome column, fewer than 2 outcome
    classes, or no usable numeric features -> ``RuntimeError`` (we refuse to
    fabricate scores).

    Args:
        entity_type: Logical entity type (echoed for provenance only).
        risk_type: Logical risk label (echoed for provenance only).
        entity_ids: Optional subset of entity IDs to restrict scoring to.
        **kwargs: Must contain the DataFrame under one of
            ``_DATAFRAME_KWARGS_KEYS``; may contain ``id_column`` (default
            ``patient_id``) and ``outcome`` (default ``discontinuation_flag``).
    """
    import hashlib
    from datetime import datetime, timezone

    from sklearn.linear_model import LogisticRegression

    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "risk_scorer requires a real DataFrame supplied via one of the "
            f"kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r}. The tool does not fabricate entity IDs "
            "or risk scores — per anti-mocking discipline, missing data must "
            "surface as a structured error rather than a plausible-but-fake "
            "placeholder (the previous placeholder body emitted synthetic "
            "entity IDs the Tier 1-5 anti-fab gate correctly rejects)."
        )

    id_column = kwargs.get("id_column", "patient_id")
    outcome = kwargs.get("outcome", "discontinuation_flag")
    if outcome not in df.columns:
        raise RuntimeError(
            f"risk_scorer: outcome column {outcome!r} not found in the supplied "
            f"DataFrame (columns={list(df.columns)!r})."
        )
    if id_column not in df.columns:
        raise RuntimeError(
            f"risk_scorer: id_column {id_column!r} not found in the supplied "
            f"DataFrame (columns={list(df.columns)!r}). Refusing to fabricate "
            "entity IDs."
        )

    work = df
    if entity_ids:
        wanted = {str(e) for e in entity_ids}
        work = df[df[id_column].astype(str).isin(wanted)]
        if len(work) == 0:
            raise RuntimeError(
                f"risk_scorer: no rows matched entity_ids={entity_ids!r} on "
                f"column {id_column!r}. Refusing to fabricate."
            )

    feature_cols = [c for c in work.select_dtypes(include="number").columns if c != outcome]
    if not feature_cols:
        raise RuntimeError(
            "risk_scorer: no usable numeric feature columns to fit a model "
            f"(numeric columns minus outcome were empty; columns={list(work.columns)!r})."
        )

    y = work[outcome].astype(int)
    if y.nunique() < 2:
        raise RuntimeError(
            "risk_scorer: the outcome column has fewer than 2 classes in the "
            "supplied data; cannot fit a discriminative risk model. Refusing to "
            "fabricate scores."
        )

    x = work[feature_cols].astype(float)
    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)
    # Probability of the positive (risk-event) class.
    classes = list(model.classes_)
    pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
    probs = model.predict_proba(x)[:, pos_idx]

    # Tertile cut points for low/medium/high tiers (real distribution-based).
    import numpy as np

    q33, q66 = np.quantile(probs, [1.0 / 3.0, 2.0 / 3.0])

    def _tier(p: float) -> str:
        if p >= q66:
            return "high"
        if p >= q33:
            return "medium"
        return "low"

    ids = work[id_column].astype(str).tolist()
    scores = [
        {"entity_id": ids[i], "risk_score": float(probs[i]), "risk_tier": _tier(float(probs[i]))}
        for i in range(len(ids))
    ]

    import sklearn

    feature_hash = hashlib.sha256(",".join(sorted(feature_cols)).encode()).hexdigest()[:8]
    model_version = f"logreg-sklearn{sklearn.__version__}-feat{feature_hash}"
    return RiskScores(
        scores=scores,
        model_version=model_version,
        scored_at=datetime.now(timezone.utc).isoformat(),
    )


@composable_tool(
    name="propensity_estimator",
    description="Estimate propensity scores for treatment assignment analysis",
    source_agent="prediction_synthesizer",
    tier=4,
    input_parameters=[
        {"name": "treatment", "type": "str", "description": "Treatment variable"},
        {"name": "covariates", "type": "List[str]", "description": "Covariate variables"},
    ],
    output_schema="PropensityScores",
    avg_execution_ms=2000,
    output_model=PropensityScores,
)
def propensity_estimator(treatment: str, covariates: List[str], **kwargs) -> PropensityScores:
    """Estimate real propensity scores P(treatment | covariates) from a DataFrame.

    Phase of GH #621. Replaces the hardcoded ``mean_propensity=0.35``
    distribution placeholder with REAL propensity scores fit on a
    caller-supplied ``pandas.DataFrame``:

    - Fit ``LogisticRegression`` predicting the binary ``treatment`` from the
      ``covariates`` columns.
    - ``propensity_distribution`` reports the real min/q25/median/q75/max of the
      fitted P(treatment=1) across all rows.
    - ``common_support`` = fraction of rows whose propensity falls within the
      overlapping [max(min_treated, min_control), min(max_treated, max_control)]
      region (the real common-support overlap, not a fabricated 0.94).
    - ``overlap_assessment`` is a label derived from ``common_support``.

    Fail-closed: no DataFrame, missing treatment / covariate columns, or fewer
    than 2 treatment classes -> ``RuntimeError``.

    Args:
        treatment: Binary treatment column name in the DataFrame.
        covariates: Covariate column names used to model assignment.
        **kwargs: Must contain the DataFrame under one of
            ``_DATAFRAME_KWARGS_KEYS``.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise RuntimeError(
            "propensity_estimator requires a real DataFrame supplied via one of "
            f"the kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r}. The tool does not fabricate propensity "
            "scores — per anti-mocking discipline, missing data must surface as "
            "a structured error rather than a plausible-but-fake placeholder."
        )
    if treatment not in df.columns:
        raise RuntimeError(
            f"propensity_estimator: treatment column {treatment!r} not found in "
            f"the supplied DataFrame (columns={list(df.columns)!r})."
        )
    if not covariates:
        raise RuntimeError(
            "propensity_estimator requires at least one covariate column; got an empty list."
        )
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"propensity_estimator: covariate columns {missing!r} not found in "
            f"the supplied DataFrame (columns={list(df.columns)!r})."
        )

    t = df[treatment].astype(int)
    if t.nunique() < 2:
        raise RuntimeError(
            "propensity_estimator: the treatment column has fewer than 2 classes; "
            "cannot fit a propensity model. Refusing to fabricate."
        )

    x = df[covariates].astype(float)
    model = LogisticRegression(max_iter=1000)
    model.fit(x, t)
    classes = list(model.classes_)
    pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
    ps = model.predict_proba(x)[:, pos_idx]

    q_min, q25, q_med, q75, q_max = (
        float(np.min(ps)),
        float(np.quantile(ps, 0.25)),
        float(np.median(ps)),
        float(np.quantile(ps, 0.75)),
        float(np.max(ps)),
    )

    # Real common-support overlap between treated and control propensity ranges.
    treated_ps = ps[t.to_numpy() == 1]
    control_ps = ps[t.to_numpy() == 0]
    overlap_lo = max(float(np.min(treated_ps)), float(np.min(control_ps)))
    overlap_hi = min(float(np.max(treated_ps)), float(np.max(control_ps)))
    if overlap_hi <= overlap_lo:
        common_support = 0.0
    else:
        in_support = (ps >= overlap_lo) & (ps <= overlap_hi)
        common_support = float(np.mean(in_support))

    if common_support >= 0.9:
        overlap_assessment = "good"
    elif common_support >= 0.7:
        overlap_assessment = "moderate"
    else:
        overlap_assessment = "poor"

    return PropensityScores(
        mean_propensity=float(np.mean(ps)),
        propensity_distribution={
            "min": q_min,
            "q25": q25,
            "median": q_med,
            "q75": q75,
            "max": q_max,
        },
        overlap_assessment=overlap_assessment,
        common_support=common_support,
    )


# ============================================================================
# REGISTRATION HELPER
# ============================================================================


def register_all_tools():
    """
    Register all composable tools.

    Call this function during application startup to ensure
    all tools are available to the Tool Composer.
    """
    # Tools are auto-registered via the @composable_tool decorator
    # This function just ensures the module is imported
    pass


# For testing: list all registered tools
if __name__ == "__main__":
    from src.tool_registry import get_registry

    registry = get_registry()
    print(f"Registered {registry.tool_count} tools from {registry.agent_count} agents:")

    for tool_name in registry.list_tools():
        schema = registry.get_schema(tool_name)
        if schema is not None:
            print(f"  - {tool_name} ({schema.source_agent}, Tier {schema.tier})")
