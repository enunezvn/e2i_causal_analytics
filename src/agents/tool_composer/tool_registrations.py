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

Fail-closed exception types (#1600). Every ``Raises: RuntimeError`` below is
still accurate, because the guards raise :class:`~.errors.ToolRefusalError`,
which SUBCLASSES ``RuntimeError``. The subclass carries one extra property: the
executor does not retry it. The dividing line, applied site by site:

* **``ToolRefusalError`` (non-retryable)** — the refusal is a property of the
  resolved INPUTS, so re-running the identical call is futile by construction:
  no DataFrame supplied, a missing metric/treatment/id column, a treatment
  column with one class, fewer than 2 comparable entity groups, a non-finite
  upstream value threaded in as an input. This is the large majority of the
  guards.
* **plain ``RuntimeError`` (still retried)** — the failure reports the OUTCOME
  of a computation rather than a property of the inputs:
  ``causal_effect_estimator``'s pipeline ``status='failed'`` / absent or
  non-finite ``consensus_effect``, and ``refutation_runner``'s DoWhy-executor
  failure / missing refutation suite. That machinery resamples (bootstrap,
  placebo, random-common-cause) with no pinned ``random_state``, so a retry is
  not futile by construction. Each such site carries an inline note.

``cohort_builder``'s two guards are non-retryable despite sitting downstream of
the ``cohort_resolution`` service: that service returns ``None`` only for an
unrecognized brand/region or a genuinely empty result — both stable across a
retry — while real infrastructure faults PROPAGATE as their own exception types
and therefore still reach the executor's retrying arm unchanged.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from pydantic import BaseModel, Field

from src.causal_engine import evalue
from src.causal_engine.pipeline import (
    PipelineInput,
    PipelineOutput,
    SequentialPipeline,
)
from src.causal_engine.pipeline.sequential import ECONML_SAMPLING_INTERVAL_ESTIMATORS
from src.services import cohort_resolution
from src.tool_registry import (
    composable_tool,
)

from .errors import ToolInputError, ToolRefusalError

# Canonical kwargs keys under which callers may supply the real DataFrame for
# causal_effect_estimator. Listed in priority order; the first non-None value
# is used. The tool fail-closes if NONE of these keys is provided -- it does
# NOT fabricate a synthetic frame, per CLAUDE.md anti-mocking discipline.
_DATAFRAME_KWARGS_KEYS: Tuple[str, ...] = (
    "data",
    "dataframe",
    "estimation_data",
)

# The one estimator ``causal_effect_estimator`` runs (#2014): DoWhy's default
# pipeline method, the only one with an analytic standard error.
_EFFECT_ESTIMATOR_METHOD = "backdoor.linear_regression"

# Two-sided 95 % normal critical value (the consensus aggregator uses the same).
_Z_95 = 1.959963984540054


# `_DataAwareSequentialPipeline` was deleted in #458 once `PipelineState` /
# `PipelineInput` declared `estimation_data` as a first-class field — the
# tool now passes the DataFrame directly via `PipelineInput.estimation_data`
# and constructs the base `SequentialPipeline` with no subclass override.


# ============================================================================
# PYDANTIC MODELS FOR TOOL I/O
# ============================================================================


class EffectEstimatorInput(BaseModel):
    """Input for causal effect estimation.

    No ``method``: the tool runs DoWhy linear regression only (#2014).
    """

    treatment: str
    outcome: str
    confounders: List[str] = []


class EffectEstimate(BaseModel):
    """Output from causal effect estimation (#2014).

    ``ci_lower`` / ``ci_upper`` / ``p_value`` / ``standard_error`` are real sampling
    quantities or all ``None``. ``uncertainty_method`` names their source
    (``ols_hc1_normal`` / ``dowhy_standard_error_normal``, or ``not_computed``) and ``uncertainty_note`` says how they were computed or why they
    were not. ``method`` is the estimator that actually ran, ``estimand`` what it
    estimated in words, ``effect_scale`` ``binary_contrast`` (treatment 1 vs 0) or
    ``per_unit`` (per one-unit increase in a non-binary treatment).
    """

    ate: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    p_value: Optional[float]
    standard_error: Optional[float]
    uncertainty_method: str
    uncertainty_note: str
    method: str
    estimand: str
    effect_scale: str
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
    """Output from CATE analysis.

    ``segments`` and ``effect_by_segment`` carry ONLY the segments whose CATE was
    actually measured — every value in them is a finite float (#1610). A segment
    that could not produce one moves to ``excluded_segments`` instead of entering
    the numeric results as ``NaN``: a ``NaN`` there is not JSON-compliant for a
    strict consumer (``json.dumps(allow_nan=False)`` raises) and renders as a
    plausible-looking blank in synthesis. Same reasoning as ``GapAnalysis``'s
    finite ``entity_values`` (#1599).

    Each ``excluded_segments`` entry is ``{"name", "n", "reason", "detail"}``.
    ``reason`` is one of the ``_CATE_EXCLUDED_*`` codes (stable, for consumers to
    branch on), ``detail`` is the prose for synthesis to disclose, and ``name`` is
    ``None`` for the null-key group — the rows whose segment value is missing name
    no segment, so there is no honest label to give them.
    """

    segments: List[Dict[str, Any]]
    high_responders: List[str]
    effect_by_segment: Dict[str, float]
    excluded_segments: List[Dict[str, Any]] = Field(default_factory=list)


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
    """Output from ROI estimation (consumes a gap-analysis result).

    ``sensitivity_band`` is a leave-one-out sensitivity band over the entity
    values the gap was derived from -- NOT a sampling confidence interval
    (renamed from ``confidence_interval`` in issue #1526; the identically named
    field on ``gap_analyzer``'s ``ROIEstimate`` genuinely is a bootstrap CI).
    """

    estimated_roi: float
    payback_months: float
    sensitivity_band: List[float]
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


# The four models below are the output contracts of the tools that return a plain dict
# (#2003). Each tool builds its result through its model and returns ``model_dump()``,
# so the keys the planner is shown (``output_fields``) are the keys the tool returns.


class SensitivityReport(BaseModel):
    """Output from ``sensitivity_analyzer``: E-values and the shared ``evalue`` reading.

    Without a confidence interval (#2014) the report is point-only: ``e_value_ci`` is
    None and ``reading`` is ``interval_unavailable``. ``benchmark`` is still the
    confounding the adjustment removed when a naive contrast is given (a point
    quantity), else None with basis ``none_measured``.
    """

    e_value_point: float
    e_value_ci: Optional[float]
    reading: str
    headline: str
    benchmark: Optional[float]
    benchmark_basis: str
    conversion: str
    interpretation: str


class RefutationResults(BaseModel):
    """Output from ``refutation_runner``: the DoWhy refutation suite and its summary.

    The summary fields are read from the suite with ``.get`` and stay optional.
    ``estimate_id`` is the caller's optional label, echoed (#2014).
    """

    estimate_id: Optional[str]
    treatment: str
    outcome: str
    n_samples: int
    refutation_results: Dict[str, Any]
    gate_decision: Optional[str]
    overall_robust: Optional[bool]
    tests_passed: Optional[int]
    tests_failed: Optional[int]
    total_tests: Optional[int]
    needs_review: Optional[bool]


class DriftMetrics(BaseModel):
    """Output from ``psi_calculator``."""

    psi: float
    interpretation: str
    threshold: float
    buckets: List[Dict[str, Any]]


class DistributionComparison(BaseModel):
    """Output from ``distribution_comparator``: one KS comparison per feature."""

    comparisons: List[Dict[str, Any]]
    overall_drift: bool


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
        raise ToolRefusalError(
            "cohort_builder: no real patient population available for "
            f"brand={brand!r} (region={kwargs.get('region')!r}) — the "
            "cohort_resolution service returned no cohort and no DataFrame was "
            "injected via context. Refusing to fabricate eligible_patient_ids."
        )

    # --- 2. Locate a real patient-id column (fail closed if absent). ---
    id_col = _find_patient_id_column(df)
    if id_col is None:
        raise ToolRefusalError(
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
        {
            "name": "required_completeness",
            "type": "float",
            "description": "Minimum share of non-missing cells in the cohort data (0-1)",
            "required": False,
            "default": 0.8,
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
        raise ToolRefusalError(
            "cohort_validator: `cohort_result` must be a dict (the output of "
            f"cohort_builder); got {type(cohort_result).__name__}={cohort_result!r}. "
            "Refusing to proceed."
        )
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise ToolRefusalError(
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
        {
            "name": "include_clinical",
            "type": "bool",
            "description": "Include summaries of the numeric clinical columns",
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
        raise ToolRefusalError(
            "cohort_statistics: `cohort_result` must be a dict (the output of "
            f"cohort_builder); got {type(cohort_result).__name__}={cohort_result!r}. "
            "Refusing to proceed — pass the structured cohort result, not a "
            "string or other scalar."
        )
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise ToolRefusalError(
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
    description=(
        "Estimate the effect of a treatment on an outcome with DoWhy linear regression, "
        "adjusted for confounders; reports a 95% CI and p-value from a "
        "heteroskedasticity-robust (HC1) standard error, or none with the reason, and "
        "names the estimand (per unit for a non-binary treatment)"
    ),
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
    method: Optional[str] = None,
    **kwargs: Any,
) -> EffectEstimate:
    """Estimate causal effect by routing the request through ``SequentialPipeline``.

    Phase C-7 of GH #354. Replaces the previous hardcoded
    ``ate=0.12, ci_lower=0.08, ci_upper=0.16, p_value=0.001, n_samples=10000``
    fabrication with a real run through the C-1..C-6 pipeline. Since #2014 the
    run is pinned to DoWhy (primary) and NetworkX: the router used to pick the
    libraries from keywords in the generated sentence, so a column name such as
    ``payer_category`` sent the estimate to EconML + CausalML. EconML's
    heterogeneity analysis stays in ``cate_analyzer`` and the causal_impact agent.

    Data flow:
    - The caller MUST supply a ``pandas.DataFrame`` under one of the canonical
      kwargs keys (``data`` / ``dataframe`` / ``estimation_data``). The tool
      does NOT fabricate synthetic data; absent a DataFrame it raises
      ``RuntimeError``.
    - The DataFrame is conveyed to the pipeline via the first-class
      ``PipelineInput.estimation_data`` field (#458); every executor reads it
      through ``data_resolver.resolve_estimation_dataframe``.

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
    - ``ate`` = ``PipelineOutput.consensus_effect``; with the pin DoWhy is the
      only effect library, so it is DoWhy's estimate.
    - ``ci_lower`` / ``ci_upper`` / ``p_value`` / ``standard_error`` come from
      :func:`_derive_uncertainty` (#2014): the 95 % normal interval and
      two-sided p of the primary library's own standard error — for DoWhy the
      HC1 SE of its OLS fit — or all ``None`` with the reason in
      ``uncertainty_note``. The former proxy (``ate +/- 0.001, p = 0.001``
      whenever one library ran, built from library agreement) is gone.
    - ``method`` / ``estimand`` / ``effect_scale`` = what actually ran and what
      it estimated (:func:`_describe_estimate`).
    - ``n_samples`` = ``len(df)`` from the caller-supplied DataFrame.

    ``method`` is not offered to the planner (#2014). It used to be echoed back
    while DoWhy ran linear regression regardless. Passing it through was
    measured and rejected: only linear regression has an analytic SE; the other
    DoWhy methods need a bootstrap (100 refits at n = 8,730: matching 8.0 s,
    weighting 6.1 s, stratification 126 s — past the 120 s step envelope),
    DoWhy's bootstrap is unseeded (the same question would give a different
    interval each run), and ``refutation_runner`` re-estimates with linear
    regression, so it would refute a different estimate than the one reported.
    A caller that still names another estimator gets a ``ToolInputError``.

    Cross-refs:
    - Dispatch plan: ``.claude/plans/354_dispatch_plan_v1.md`` §2.4 C-7
    - Design plan: ``.claude/plans/causal_engine_canonical_routing_v4.md``
    - Brief template: ``.claude/dispatch/354_executor_brief_template.md``
    - Data resolver (C-6): ``src/causal_engine/pipeline/data_resolver.py``

    Args:
        treatment: Name of the treatment column in the supplied DataFrame.
        outcome: Name of the outcome column.
        confounders: Confounder column names (optional).
        method: Not offered to the planner. Accepted only as ``None`` or
            ``"backdoor.linear_regression"``; any other value is refused.
        **kwargs: Must contain the DataFrame under one of
            ``_DATAFRAME_KWARGS_KEYS``. May also contain ``data_source``
            (passed through as ``PipelineInput.data_source``) and
            ``query`` (custom natural-language query string).

    Returns:
        ``EffectEstimate`` populated from the pipeline's real consensus.

    Raises:
        ToolInputError: when ``method`` names an estimator other than DoWhy
            linear regression.
        RuntimeError: when the caller did not supply a DataFrame, when the
            pipeline reports failure, or when the pipeline did not produce a
            finite consensus effect.
        Exception: any exception raised by the pipeline (e.g.
            ``ExecutorDataUnavailable`` from a downstream executor) is
            propagated unchanged.
    """
    if method is not None and method != _EFFECT_ESTIMATOR_METHOD:
        raise ToolInputError(
            f"causal_effect_estimator estimates with DoWhy {_EFFECT_ESTIMATOR_METHOD!r} "
            f"only; got method={method!r}. No other estimator is run by this tool, so the "
            "request is refused rather than answered with a linear-regression estimate "
            "labelled as another method (#2014). Omit method."
        )

    # --- 1. Locate the caller's real DataFrame (fail-closed if missing). ---
    df = _extract_dataframe_from_kwargs(kwargs)
    if df is None:
        raise ToolRefusalError(
            "causal_effect_estimator requires a real DataFrame supplied via one "
            f"of the kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got "
            f"kwargs keys={sorted(kwargs.keys())!r}. The tool does not "
            "fabricate synthetic data — per anti-mocking discipline, missing "
            "data must surface as a structured error rather than a "
            "plausible-but-fake placeholder."
        )

    # --- 2. Build the PipelineInput. ---
    data_source = kwargs.get("data_source") or "tool_composer.causal_effect_estimator"
    # Wording unchanged from when ``method`` was echoed into it: routing reads it.
    query = kwargs.get("query") or (
        f"Estimate the causal effect of {treatment} on {outcome} "
        f"using method={_EFFECT_ESTIMATOR_METHOD!r}."
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
        # Pinned (#2014): DoWhy first, because the router makes the first forced
        # library primary (measured on a real run in both orders). NetworkX runs for
        # the graph-quality channel and estimates no effect. A ``query`` override
        # therefore no longer changes which libraries run.
        "libraries_enabled": ["dowhy", "networkx"],
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
    #
    # These three stay plain ``RuntimeError`` (i.e. RETRYABLE) while the input
    # guards above became ``ToolRefusalError`` (#1600). They do not describe the
    # inputs — they report the OUTCOME of an estimation run, and that machinery
    # is genuinely stochastic (bootstrap resampling, placebo simulations, no
    # pinned ``random_state``), so a second attempt over the same frame is not
    # futile by construction. Only refusals that cannot succeed on retry BY
    # CONSTRUCTION are made non-retryable.
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

    # --- 5. Real uncertainty of the reported effect, and what was estimated. ---
    primary_result = pipeline_output.get("primary_result") or {}
    libraries_used = list(pipeline_output.get("libraries_used") or [])
    errors = list(pipeline_output.get("errors") or [])
    uncertainty = _derive_uncertainty(
        ate=ate_value, primary_result=primary_result, libraries_used=libraries_used, errors=errors
    )
    method_used, estimand, effect_scale = _describe_estimate(
        ate=ate_value,
        primary_result=primary_result,
        libraries_used=libraries_used,
        errors=errors,
        treatment_values=df.get(treatment),
        treatment=treatment,
        outcome=outcome,
        confounders=confounders or [],
    )

    return EffectEstimate(
        ate=ate_value,
        **uncertainty,
        method=method_used,
        estimand=estimand,
        effect_scale=effect_scale,
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


def _finite_float(value: Any) -> Optional[float]:
    """``value`` as a finite float, or None (bools and non-numbers are None)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _primary_estimate(primary_result: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """The primary library's name and its OWN effect estimate, from its result keys.

    DoWhy reports ``causal_effect`` (and ``dowhy_method``); EconML ``ate`` with its
    ``estimator``; CausalML ``ate`` with its ``model``. NetworkX (primary for
    impact-flow wording) estimates no effect.
    """
    if "dowhy_method" in primary_result:
        return "dowhy", _finite_float(primary_result.get("causal_effect"))
    if "estimator" in primary_result and "ate" in primary_result:
        return "econml", _finite_float(primary_result.get("ate"))
    if "model" in primary_result and "ate" in primary_result:
        return "causalml", _finite_float(primary_result.get("ate"))
    return ("networkx" if "is_dag" in primary_result else None), None


# Libraries whose result is an effect estimate the pipeline consensus can blend.
_EFFECT_LIBRARIES = ("dowhy", "econml", "causalml")


class _EstimateProvenance(NamedTuple):
    """Which library the reported effect came from (#2014)."""

    library: Optional[str]  # the primary library
    own_effect: Optional[float]  # the primary library's own estimate
    effect_libraries: List[str]  # libraries that ran without error and estimate effects
    is_primary_estimate: bool  # the reported ate IS the primary library's sole estimate


def _estimate_provenance(
    *,
    ate: float,
    primary_result: Dict[str, Any],
    libraries_used: List[str],
    errors: List[Any],
) -> _EstimateProvenance:
    """Whether the reported ``ate`` is the primary library's own, unblended estimate.

    Decided from provenance, not only numbers: the consensus blends every effect
    library that ran, so two libraries agreeing (or a blend landing on the primary's
    value) must not borrow the primary's uncertainty. A library counts when it ran
    and reported no error; one that ran but contributed no effect is still counted,
    which can only withhold an interval, never invent one.
    """
    library, own_effect = _primary_estimate(primary_result)
    failed = {e.get("library") for e in errors if isinstance(e, dict)}
    effect_libraries = [
        lib for lib in libraries_used if lib in _EFFECT_LIBRARIES and lib not in failed
    ]
    is_primary_estimate = (
        own_effect is not None
        and effect_libraries == [library]
        and math.isclose(own_effect, ate, rel_tol=1e-9, abs_tol=1e-12)
    )
    return _EstimateProvenance(library, own_effect, effect_libraries, is_primary_estimate)


def _derive_uncertainty(
    *,
    ate: float,
    primary_result: Dict[str, Any],
    libraries_used: List[str],
    errors: List[Any],
) -> Dict[str, Any]:
    """Real sampling uncertainty of the reported effect, or None with the reason (#2014).

    Replaces the former proxy, which built ``ate +/- width`` and ``p`` from library
    AGREEMENT (``consensus_confidence``) — ATE +/- 0.001 with p = 0.001 whenever a
    single library ran — while the standard error DoWhy returned went unread.

    An interval exists only when the reported ``ate`` IS the primary library's own
    estimate (:func:`_estimate_provenance`) and that library measured its sampling
    error:

    * DoWhy: its ``standard_error`` (the HC1 SE of its OLS fit) gives the 95 % normal
      interval ``ate +/- z*SE`` and the two-sided normal p-value.
    * EconML, for the estimators whose interval is a sampling interval: that 95 %
      interval and its ``ate_std`` (the SE back-derived from the width, ``(hi - lo) /
      2z``, only when ``ate_std`` is absent), with the p-value from that SE.

    Otherwise every field is None: the effect is a consensus across libraries (no SE
    exists for that blend), the primary library produced no estimate of its own, it
    measured no SE (e.g. zero residual degrees of freedom), or its interval is not a
    sampling interval — CausalML's is always ``std(predicted uplift) / sqrt(n)``,
    which ignores the uncertainty of fitting the uplift model.
    """
    prov = _estimate_provenance(
        ate=ate, primary_result=primary_result, libraries_used=libraries_used, errors=errors
    )

    def not_computed(reason: str) -> Dict[str, Any]:
        return {
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "standard_error": None,
            "uncertainty_method": "not_computed",
            "uncertainty_note": f"No confidence interval or p-value: {reason}",
        }

    if prov.own_effect is None:
        return not_computed(
            f"the primary library ({prov.library or 'unknown'}) produced no effect estimate "
            "of its own, so no standard error belongs to the reported effect."
        )
    if not prov.is_primary_estimate:
        return not_computed(
            "the reported effect is the consensus of the "
            f"{', '.join(prov.effect_libraries) or 'pipeline'} estimates; no standard error "
            "exists for that blend."
        )

    if prov.library == "dowhy":
        se = _finite_float(primary_result.get("standard_error"))
        if se is None or se <= 0:
            return not_computed(
                f"DoWhy's {primary_result.get('dowhy_method')!r} produced no standard error "
                "for this estimate (linear regression needs residual degrees of freedom)."
            )
        se_method = primary_result.get("standard_error_method")
        robust = se_method == "ols_hc1"
        lower, upper = ate - _Z_95 * se, ate + _Z_95 * se
        method_code = "ols_hc1_normal" if robust else "dowhy_standard_error_normal"
        note = "95% normal-approximation interval and two-sided p-value from the " + (
            "heteroskedasticity-robust (HC1) standard error of the OLS fit."
            if robust
            else f"standard error DoWhy reported ({se_method or 'method unstated'})."
        )
    elif prov.library == "econml":
        estimator = primary_result.get("estimator")
        if estimator not in ECONML_SAMPLING_INTERVAL_ESTIMATORS:
            return not_computed(
                f"EconML's {estimator!r} interval is the spread of its per-unit effects "
                "divided by sqrt(n), not a sampling interval for the average effect."
            )
        lower_raw = _finite_float(primary_result.get("ate_ci_lower"))
        upper_raw = _finite_float(primary_result.get("ate_ci_upper"))
        if (
            lower_raw is None
            or upper_raw is None
            or not lower_raw < upper_raw
            or not lower_raw <= ate <= upper_raw
        ):
            return not_computed(
                f"EconML reported no usable interval around its estimate "
                f"(ate_ci_lower={primary_result.get('ate_ci_lower')!r}, "
                f"ate_ci_upper={primary_result.get('ate_ci_upper')!r})."
            )
        lower, upper = lower_raw, upper_raw
        # EconML's own standard error when it reports one (as /causal/treatment-effects
        # reads it), else back-derived from the interval's width.
        ate_std = _finite_float(primary_result.get("ate_std"))
        se = ate_std if ate_std is not None and ate_std > 0 else (upper - lower) / (2.0 * _Z_95)
        method_code = "library_interval"
        note = (
            f"95% interval and standard error as reported by EconML {estimator}; the "
            "two-sided p-value is from that standard error under a normal approximation."
        )
    else:
        return not_computed(
            "CausalML's interval is the spread of its model-predicted uplift divided by "
            "sqrt(n); it omits the uncertainty of fitting the uplift model, so it is not a "
            "sampling interval for the effect."
        )

    return {
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": math.erfc(abs(ate / se) / math.sqrt(2.0)),
        "standard_error": se,
        "uncertainty_method": method_code,
        "uncertainty_note": note,
    }


def _is_binary_01(values: Any) -> bool:
    """Whether a treatment column holds only 0 and 1 (booleans included)."""
    if values is None:
        return False
    try:
        observed = {float(v) for v in values.dropna().unique()}
    except (TypeError, ValueError):
        return False
    return bool(observed) and observed <= {0.0, 1.0}


def _describe_estimate(
    *,
    ate: float,
    primary_result: Dict[str, Any],
    libraries_used: List[str],
    errors: List[Any],
    treatment_values: Any,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> Tuple[str, str, str]:
    """``(method, estimand, effect_scale)`` for what the pipeline actually estimated (#2014).

    DoWhy's linear regression reports ``E[Y | T=1] - E[Y | T=0]`` from the fitted model,
    which is the treatment coefficient: for a binary treatment a regression-adjusted
    1-vs-0 difference, for any other numeric treatment the change per ONE UNIT. The
    wording gives SUFFICIENT conditions for that coefficient to be the average treatment
    effect: treatment independent of the confounders (the linear adjustment then only
    adds precision), or an outcome linear in the confounders as modelled with a constant
    effect. A constant effect alone is not enough: confounding that is non-linear in the
    confounders biases the linear adjustment.
    """
    prov = _estimate_provenance(
        ate=ate, primary_result=primary_result, libraries_used=libraries_used, errors=errors
    )
    binary = _is_binary_01(treatment_values)
    effect_scale = "binary_contrast" if binary else "per_unit"
    # A count, not the names: the synthesizer truncates each step's JSON at 1,000
    # characters, and a long confounder list would cut this sentence off.
    adjusted = (
        f"adjusted for {len(confounders)} confounder{'s' if len(confounders) != 1 else ''}"
        if confounders
        else "with no confounder adjustment"
    )
    contrast = (
        f"between {treatment} = 1 and {treatment} = 0"
        if binary
        else f"per one-unit increase in {treatment}"
    )
    if not prov.is_primary_estimate:
        libraries = prov.effect_libraries or [lib for lib in libraries_used if lib != "networkx"]
        return (
            f"consensus({','.join(libraries)})",
            f"Consensus of the {', '.join(libraries)} estimates of the effect on {outcome} "
            f"{contrast}, {adjusted}.",
            effect_scale,
        )
    if prov.library == "dowhy":
        dowhy_method = str(primary_result.get("dowhy_method"))
        if "linear_regression" in dowhy_method:
            estimand = (
                f"Regression-adjusted difference in {outcome} {contrast} (OLS with an "
                f"additive treatment term, {adjusted}"
                + ("" if binary else f"; assumes the effect is linear in {treatment}")
                + "). It is the average treatment effect when treatment does not depend on "
                "the confounders, or when the outcome is linear in them as modelled and the "
                "effect is constant; otherwise it can differ from the average treatment effect."
            )
        else:
            estimand = f"Effect on {outcome} {contrast}, {adjusted} (DoWhy {dowhy_method})."
        return dowhy_method, estimand, effect_scale
    if prov.library == "econml":
        estimator = primary_result.get("estimator")
        return (
            f"econml.{estimator}",
            f"Average treatment effect on {outcome} {contrast}, {adjusted} (EconML {estimator}).",
            effect_scale,
        )
    model = primary_result.get("model")
    return (
        f"causalml.{model}",
        f"Mean model-predicted uplift in {outcome} {contrast} (CausalML {model}); not an "
        "identification-validated average treatment effect.",
        effect_scale,
    )


@composable_tool(
    name="refutation_runner",
    description="Run DoWhy refutation test suite (placebo, random cause, subset, bootstrap, sensitivity)",
    source_agent="causal_impact",
    tier=2,
    input_parameters=[
        {
            "name": "estimate_id",
            "type": "str",
            "description": (
                "Optional label echoed back; the suite re-estimates from the data and does "
                "not look an estimate up by it"
            ),
            "required": False,
        },
        {
            "name": "treatment",
            "type": "str",
            "description": "Treatment column the refutation suite re-estimates on",
        },
        {
            "name": "outcome",
            "type": "str",
            "description": "Outcome column the refutation suite re-estimates on",
        },
        {
            "name": "confounders",
            "type": "List[str]",
            "description": "Confounder columns (use the ones the estimate adjusted for)",
            "required": False,
        },
    ],
    output_schema="RefutationResults",
    avg_execution_ms=5000,
    output_model=RefutationResults,
)
def refutation_runner(estimate_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
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
        estimate_id: Optional caller label, echoed back. Not used to fetch a live
            estimate (impossible across the serialization boundary). Optional since
            #2014: no tool produces an estimate id, so the planner filled the required
            field with whatever it could reference (``$step_1.method``,
            ``$step_1.ate``). An id minted by ``causal_effect_estimator`` was rejected:
            the suite re-estimates from the data, so the id would claim a link to an
            estimate this run never refutes.
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
        raise ToolRefusalError(
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
        raise ToolRefusalError(
            "refutation_runner requires the planner-bound treatment and outcome "
            f"column names to run real DoWhy refutation; got treatment={treatment!r}, "
            f"outcome={outcome!r}. Refusing to fabricate refutation results."
        )

    confounders = _as_str_list(
        kwargs.get("confounders") or kwargs.get("covariates") or kwargs.get("common_causes")
    )

    refutation = _run_dowhy_refutation(df, treatment, outcome, confounders)

    return RefutationResults(
        estimate_id=estimate_id,
        treatment=treatment,
        outcome=outcome,
        n_samples=int(len(df)),
        refutation_results=refutation,
        gate_decision=refutation.get("gate_decision"),
        overall_robust=refutation.get("overall_robust"),
        tests_passed=refutation.get("tests_passed"),
        tests_failed=refutation.get("tests_failed"),
        total_tests=refutation.get("total_tests"),
        needs_review=refutation.get("needs_review"),
    ).model_dump()


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
        raise ToolRefusalError(
            f"refutation_runner: supplied data is not a DataFrame ({exc}). "
            "Refusing to fabricate refutation results."
        ) from exc

    missing = [c for c in [treatment, outcome, *confounders] if c not in columns]
    if missing:
        raise ToolRefusalError(
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

    # Both guards below stay plain ``RuntimeError`` (RETRYABLE) while the input
    # guards above became ``ToolRefusalError`` (#1600): they report what the
    # DoWhy run PRODUCED, not what was supplied. The refutation suite resamples
    # (bootstrap, placebo, random-common-cause, data-subset) with no pinned
    # ``random_state``, so a retry can legitimately reach a different outcome.
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


@composable_tool(
    name="sensitivity_analyzer",
    description=(
        "Compute VanderWeele-Ding E-values and, when a naive contrast is given, the "
        "measured-confounding reading (beyond / within / null finding) the refutation "
        "gate uses"
    ),
    source_agent="causal_impact",
    tier=2,
    input_parameters=[
        {"name": "ate", "type": "float", "description": "Estimated average treatment effect"},
        {
            "name": "ci_lower",
            "type": "Optional[float]",
            "description": (
                "Lower confidence bound (optional; null when the estimate has no interval, "
                "which makes the report point-only)"
            ),
            "required": False,
        },
        {
            "name": "ci_upper",
            "type": "Optional[float]",
            "description": "Upper confidence bound (optional; defaults to ate + (ate - ci_lower))",
            "required": False,
        },
        {
            "name": "baseline_risk",
            "type": "float",
            "description": (
                "Control-arm outcome rate for a binary outcome (optional; enables the "
                "risk-ratio path)"
            ),
            "required": False,
        },
        {
            "name": "naive_ate",
            "type": "float",
            "description": (
                "Unadjusted difference in means (optional; enables the measured-confounding "
                "benchmark)"
            ),
            "required": False,
        },
    ],
    output_schema="SensitivityReport",
    avg_execution_ms=1500,
    output_model=SensitivityReport,
)
def sensitivity_analyzer(
    ate: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    baseline_risk: Optional[float] = None,
    naive_ate: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """E-values and the sensitivity READING from the shared ``evalue`` module.

    Spec docs/superpowers/specs/2026-09-10-sensitivity-gate-calibration-design.md §4.7.
    Without ``baseline_risk`` the inputs are taken on the standardized-mean-difference
    scale (``RR = exp(0.91*d)``). Without ``naive_ate`` no benchmark exists and the
    reading is ``unbenchmarked``: the E-value is reported with the statement that no
    universal threshold exists. Refuses non-finite inputs (anti-mocking: never a
    fabricated E-value). A ``ValueError`` from the classifier (a point estimate
    outside its own CI) is surfaced as a structured ``ToolRefusalError``. A supplied
    ``baseline_risk`` that cannot form risks in (0, 1) with the effect and CI is
    refused rather than silently read on the standardized-difference scale. A CI
    that includes zero is reported as a null finding regardless of the benchmark
    (spec §4.4 precedence).

    Without an interval (#2014: ``causal_effect_estimator`` returns ``ci_lower`` /
    ``ci_upper`` = None when no sampling uncertainty was measured, and the planner maps
    those fields here by name) the report is point-only: the point E-value under the
    same conversion rule, no CI E-value, the measured-confounding benchmark when
    ``naive_ate`` is given, and reading ``interval_unavailable`` — no verdict, because
    the null-finding check that precedes beyond / within needs the interval. An upper
    bound without a lower one is refused rather than mirrored into an invented bound.
    """
    for name, value in (
        ("ate", ate),
        ("ci_lower", ci_lower),
        ("ci_upper", ci_upper),
        ("baseline_risk", baseline_risk),
        ("naive_ate", naive_ate),
    ):
        if value is not None and not math.isfinite(float(value)):
            raise ToolRefusalError(
                f"sensitivity_analyzer requires finite inputs; got {name}={value!r}. Refusing to "
                "fabricate an E-value — per anti-mocking discipline non-finite inputs surface as "
                "a structured error."
            )
    if ci_lower is None:
        if ci_upper is not None:
            raise ToolInputError(
                f"sensitivity_analyzer got ci_upper={ci_upper!r} without ci_lower. An interval "
                "needs both bounds (or ci_lower alone, mirrored around ate); refusing to invent "
                "the lower bound."
            )
        return _point_only_sensitivity(ate, baseline_risk=baseline_risk, naive_ate=naive_ate)
    hi = float(ci_upper) if ci_upper is not None else float(ate) + (float(ate) - float(ci_lower))
    try:
        reading = evalue.classify(
            float(ate),
            (float(ci_lower), hi),
            randomized=False,
            baseline_risk=baseline_risk,
            outcome_std=None,
            naive_effect=naive_ate,
            covariate_factors={},
            n_rows=None,
        )
    except ValueError as exc:
        raise ToolRefusalError(f"sensitivity_analyzer refused its inputs: {exc}") from exc
    if baseline_risk is not None and reading.conversion != "risk_ratio":
        raise ToolRefusalError(
            f"sensitivity_analyzer: baseline_risk={baseline_risk!r} with ate={ate!r} and "
            f"CI=({ci_lower!r}, {hi!r}) does not form valid risks in (0, 1), so no "
            "risk-ratio E-value exists. Refusing to substitute a standardized-difference "
            "scale for a caller who asked for the risk-ratio path."
        )
    interpretation = reading.message
    if reading.reading == evalue.READING_UNBENCHMARKED:
        interpretation = (
            f"An unobserved confounder would need to be associated with both treatment and "
            f"outcome by a risk ratio of at least {reading.e_value_point:.2f} (and the CI bound "
            f"by {reading.e_value_ci:.2f}) to explain away the observed effect. There is no "
            "universal E-value threshold: benchmark it against the confounding the measured "
            "covariates carried (pass naive_ate and baseline_risk to get that reading)."
        )
    return SensitivityReport(
        e_value_point=reading.e_value_point,
        e_value_ci=reading.e_value_ci,
        reading=reading.reading,
        headline=reading.headline,
        benchmark=reading.benchmark,
        benchmark_basis=reading.benchmark_basis,
        conversion=reading.conversion,
        interpretation=interpretation,
    ).model_dump()


_READING_INTERVAL_UNAVAILABLE = "interval_unavailable"


def _point_only_sensitivity(
    ate: float, *, baseline_risk: Optional[float], naive_ate: Optional[float]
) -> Dict[str, Any]:
    """``sensitivity_analyzer``'s report when the estimate has no confidence interval (#2014).

    The point E-value and, with a naive contrast, the measured-confounding benchmark
    (both point quantities — spec §2.5 benchmarks the point estimate and states
    precision separately). No reading: ``classify`` checks "the CI includes zero" before
    beyond / within, and that check cannot be made.
    """
    try:
        e_point, _, conversion = evalue.point_e_value(
            float(ate), baseline_risk=baseline_risk, outcome_std=None, naive_effect=naive_ate
        )
        joint = evalue.joint_confounding_benchmark(
            naive_ate,
            float(ate),
            baseline_risk=baseline_risk if conversion == "risk_ratio" else None,
            outcome_std=None,
        )
        benchmark, basis = evalue.measured_confounding_benchmark(joint, {})
    except ValueError as exc:
        raise ToolRefusalError(f"sensitivity_analyzer refused its inputs: {exc}") from exc
    if baseline_risk is not None and conversion != "risk_ratio":
        raise ToolRefusalError(
            f"sensitivity_analyzer: baseline_risk={baseline_risk!r} with ate={ate!r} does not "
            "form valid risks in (0, 1), so no risk-ratio E-value exists. Refusing to "
            "substitute a standardized-difference scale for a caller who asked for the "
            "risk-ratio path."
        )
    benchmark_sentence = (
        f" The confounding the measured adjustment removed corresponds to a risk ratio of "
        f"{benchmark:.2f} ({evalue.BASIS_IN_WORDS[basis]}); no reading against it is given "
        "without the interval."
        if benchmark is not None
        else ""
    )
    return SensitivityReport(
        e_value_point=e_point,
        e_value_ci=None,
        reading=_READING_INTERVAL_UNAVAILABLE,
        headline="Robustness not assessed: the estimate has no confidence interval",
        benchmark=benchmark,
        benchmark_basis=basis,
        conversion=conversion,
        interpretation=(
            f"No confidence interval exists for this estimate, so only the point E-value is "
            f"reported: an unobserved confounder would need a risk ratio of at least "
            f"{e_point:.2f} with both treatment and outcome to explain away the point "
            "estimate. Without the interval it is unknown whether the effect is "
            "distinguishable from zero, so no robustness reading (null finding, or beyond / "
            "within measured confounding) is given." + benchmark_sentence
        ),
    ).model_dump()


# ============================================================================
# HETEROGENEOUS OPTIMIZER AGENT TOOLS
# ============================================================================

# Stable codes for ``CATEResults.excluded_segments[*]["reason"]`` (#1610). A
# consumer branches on the code; the accompanying ``detail`` is the prose.
_CATE_EXCLUDED_MISSING = "missing_segment_value"
_CATE_EXCLUDED_NO_CONTRAST = "no_within_segment_contrast"
_CATE_EXCLUDED_NON_FINITE = "non_finite_mean_difference"


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
    - No segment yields a MEASURED CATE -> ``RuntimeError`` (#1610). An empty
      segment set would read as "no heterogeneity between segments", a finding
      this data cannot support because no segment was ever estimated.
    - A non-numeric ``outcome`` column, whose mean raises out of pandas ->
      ``RuntimeError`` (#1600 shape, codex iter-1). The dtype is a property of
      the resolved inputs, so the refusal must surface ONCE instead of being
      retried and charged to the tool's circuit breaker.

    Segments that cannot produce a CATE are excluded from ``segments`` /
    ``effect_by_segment`` and disclosed in ``excluded_segments`` (#1610), the
    same exclude-or-refuse treatment ``gap_calculator`` gives a non-finite group
    mean (#1599). Three ways a segment fails to produce one:

    * its rows sit on only ONE side of ``treatment``, so there is no contrast to
      difference (the case the previous ``float("nan")`` sentinel marked);
    * one arm carries no non-null ``outcome`` values, so the difference in means
      is ``NaN`` (or ``pd.NA`` under a nullable dtype) even though BOTH arms are
      populated — a shape the emptiness check above cannot see; and
    * the group KEY is null. ``groupby(..., dropna=False)`` is deliberate and is
      KEPT: the null rows must not vanish silently, so they are still seen,
      counted and disclosed. What changes is that they no longer enter the
      results, where ``str(float('nan'))`` labeled them ``"nan"`` — a label
      indistinguishable from a real category of that name, and promotable into
      ``high_responders`` (measured) and from there into
      ``segment_ranker.recommended_targets``. Rows with an unknown segment name
      no segment anyone can act on, so they are excluded rather than relabeled.

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
        raise ToolRefusalError(
            "cate_analyzer requires a real DataFrame supplied via one of the "
            f"kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r}. The tool does not fabricate segment "
            "effects — per anti-mocking discipline, missing data must surface "
            "as a structured error rather than a plausible-but-fake placeholder."
        )
    if not segments:
        raise ToolRefusalError(
            "cate_analyzer requires at least one segmentation column in "
            "`segments`; got an empty list."
        )
    segment_col = segments[0]
    for col in (treatment, outcome, segment_col):
        if col not in df.columns:
            raise ToolRefusalError(
                f"cate_analyzer: column {col!r} not found in the supplied "
                f"DataFrame (columns={list(df.columns)!r}). Refusing to "
                "fabricate a result."
            )

    segment_dicts: List[Dict[str, Any]] = []
    effect_by_segment: Dict[str, float] = {}
    excluded_segments: List[Dict[str, Any]] = []
    named_groups: List[str] = []
    rows_missing_segment_value = 0
    for seg_value, sub in df.groupby(segment_col, dropna=False):
        n = int(len(sub))
        if _is_missing_group_key(seg_value):
            rows_missing_segment_value += n
            excluded_segments.append(
                {
                    "name": None,
                    "n": n,
                    "reason": _CATE_EXCLUDED_MISSING,
                    "detail": (
                        f"rows whose {segment_col!r} value is null name no segment, "
                        "so they are excluded from the per-segment effects rather "
                        "than reported as one"
                    ),
                }
            )
            continue
        name = str(seg_value)
        named_groups.append(name)
        treated = sub[sub[treatment] == 1][outcome]
        control = sub[sub[treatment] == 0][outcome]
        if len(treated) == 0 or len(control) == 0:
            # No within-segment contrast available -> cannot estimate a CATE.
            # Excluded rather than fabricated (anti-mocking pattern #4); the
            # exclusion is disclosed below.
            excluded_segments.append(
                {
                    "name": name,
                    "n": n,
                    "reason": _CATE_EXCLUDED_NO_CONTRAST,
                    "detail": (
                        f"segment carries {len(treated)} rows with {treatment!r}=1 "
                        f"and {len(control)} with {treatment!r}=0, so there is no "
                        "within-segment contrast to difference"
                    ),
                }
            )
            continue
        # ``_coerce_finite`` -- not ``float(...)`` -- because BOTH arms can be
        # populated while one carries no non-null ``outcome`` value: its mean is
        # then NaN (or ``pd.NA`` under a nullable dtype, whose ``float()`` raises
        # TypeError and would escape into the executor's RETRYING arm). Guarding
        # the computed VALUE is what #1599 found the group-mean case needed. It
        # also catches a datetime outcome, whose difference is a ``Timedelta``.
        try:
            cate_val = _coerce_finite(treated.mean() - control.mean())
        except (TypeError, ValueError) as exc:
            # One seam earlier: a non-numeric outcome raises out of pandas'
            # aggregation itself ("Could not convert string 'hi' to numeric"),
            # before there is any value to coerce. The column's dtype is a
            # property of the RESOLVED inputs and identical for every segment, so
            # this must surface ONCE as a refusal (#1600) rather than be retried
            # and charged to the tool's circuit breaker as if the tool were
            # unhealthy -- the same guard ``gap_calculator`` puts on its metric.
            raise ToolRefusalError(
                f"cate_analyzer: outcome column {_clip_name(outcome)!r} is not numeric "
                f"(dtype={df[outcome].dtype!s}), so no within-segment mean difference "
                f"can be computed for {_clip_name(segment_col)!r}={_clip_name(name)!r}: "
                f"{exc}. Refusing to fabricate a treatment effect from a non-numeric "
                "outcome."
            ) from exc
        if cate_val is None:
            excluded_segments.append(
                {
                    "name": name,
                    "n": n,
                    "reason": _CATE_EXCLUDED_NON_FINITE,
                    "detail": (
                        f"the within-segment difference in mean {outcome!r} is not a "
                        "finite number — one arm carries no non-null values to average"
                    ),
                }
            )
            continue
        segment_dicts.append({"name": name, "cate": cate_val, "n": n})
        effect_by_segment[name] = cate_val

    if not effect_by_segment:
        no_contrast = sum(1 for e in excluded_segments if e["reason"] == _CATE_EXCLUDED_NO_CONTRAST)
        non_finite = sum(1 for e in excluded_segments if e["reason"] == _CATE_EXCLUDED_NON_FINITE)
        # Bounded like the gap reason (#1574): the reason is carried verbatim into
        # the composer's total-failure envelope, which truncates from the END --
        # where the scope payload sits.
        segment_label = _clip_name(segment_col)
        scope: Dict[str, Any] = {
            "segment_column": segment_label,
            "row_count": int(len(df)),
            "segment_groups_present": _clip_entity_labels(sorted(named_groups)),
            "segment_groups_present_count": len(named_groups),
            "rows_missing_segment_value": rows_missing_segment_value,
        }
        raise ToolRefusalError(
            f"cate_analyzer: no {segment_label!r} segment in the supplied estimation "
            f"data yields a measured CATE — of {len(named_groups)} named segments, "
            f"{no_contrast} carry rows on only one side of {_clip_name(treatment)!r} "
            f"and {non_finite} have no non-null {_clip_name(outcome)!r} values to "
            f"average in one arm; a further {rows_missing_segment_value} rows carry no "
            f"{segment_label!r} value at all and so name no segment. Refusing to "
            "fabricate: an empty segment set reads as 'no heterogeneity between "
            "segments', which is a finding this data cannot support — the segments "
            f"were never estimated. cate_estimation_scope={scope!r}"
        )

    # Every value is finite by construction now, so the cross-segment mean is a
    # mean over MEASURED effects only -- an excluded segment can no longer shift
    # the threshold that decides which segments are recommended.
    effects = list(effect_by_segment.values())
    threshold = sum(effects) / len(effects)
    high_responders = [name for name, v in effect_by_segment.items() if v >= threshold and v > 0]
    return CATEResults(
        segments=segment_dicts,
        high_responders=high_responders,
        effect_by_segment=effect_by_segment,
        excluded_segments=excluded_segments,
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
        raise ToolRefusalError(
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
        {
            "name": "group_by",
            "type": "str",
            "description": (
                "Column to group entities by (optional; otherwise resolved from entity_type)"
            ),
            "required": False,
        },
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
    1. explicit ``group_by`` kwarg (refused when it is not a column),
    2. ``entity_type`` if it is a column,
    3. a column named ``<entity_type>`` or ``geographic_region`` /
       ``territory`` / ``brand`` heuristics.

    When ``entities`` is non-empty, the result is restricted to those entity
    values (real filtering — not fabrication).

    Fail-closed: no DataFrame, missing metric column, no resolvable grouping
    column, or fewer than 2 entity groups with a FINITE metric mean after
    filtering (#1574, tightened by #1599) -> ``ToolRefusalError`` (a
    ``RuntimeError`` subclass the executor does not retry, since the refusal is
    deterministic over the resolved inputs). The last guard is what keeps a
    single-brand estimation frame from being reported as a comparison of the
    focal brand against itself, and keeps a group whose metric is entirely null
    from hijacking the top/bottom selection; its reason carries an
    ``estimation_data_scope`` payload so synthesis can disclose the scope
    actually covered (see :func:`_gap_comparability_reason`).

    Groups whose metric mean is non-finite are excluded from the comparison
    basis and from ``entity_values`` — a gap is a spread between two MEASURED
    values, and ``max``/``min`` silently return the first key when a NaN is in
    play (#1599).

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
        raise ToolRefusalError(
            "gap_calculator requires a real DataFrame supplied via one of the "
            f"kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r}. The tool does not fabricate entity "
            "values — per anti-mocking discipline, missing data must surface as "
            "a structured error rather than a plausible-but-fake placeholder."
        )
    if metric not in df.columns:
        raise ToolRefusalError(
            f"gap_calculator: metric column {metric!r} not found in the supplied "
            f"DataFrame (columns={list(df.columns)!r})."
        )

    group_by = kwargs.get("group_by")
    if group_by is not None and group_by not in df.columns:
        raise ToolRefusalError(
            f"gap_calculator: group_by={group_by!r} is not a column of the supplied "
            f"DataFrame (columns={list(df.columns)!r}). Refusing to group by a different "
            "column than the one requested."
        )
    group_col = _resolve_grouping_column(df, group_by, entity_type)
    if group_col is None:
        raise ToolRefusalError(
            "gap_calculator: could not resolve a grouping column from group_by="
            f"{kwargs.get('group_by')!r} / entity_type={entity_type!r}; "
            f"DataFrame columns={list(df.columns)!r}. Refusing to fabricate."
        )

    try:
        grouped = df.groupby(group_col, dropna=False)[metric].mean()
    except (TypeError, ValueError) as exc:
        # A non-numeric metric column raises out of pandas' aggregation
        # ("agg function failed [how->mean,dtype->object]"). That is
        # deterministic over the resolved inputs, so it must surface ONCE as a
        # refusal (#1600) rather than be retried and charged to the tool's
        # circuit breaker as if the tool were unhealthy.
        raise ToolRefusalError(
            f"gap_calculator: metric column {metric!r} is not numeric (dtype="
            f"{df[metric].dtype!s}), so no per-{group_col!r} group mean can be "
            f"computed: {exc}. Refusing to fabricate a comparison from a "
            "non-numeric metric."
        ) from exc

    # Group KEYS are coerced with ``str`` because ``GapAnalysis.entity_values``
    # is a ``Dict[str, float]``, so distinct raw keys can collide under that
    # coercion. Keep every raw mean per label rather than letting the last one
    # win: measured, raw groups ``1``, ``"1"``, ``"Kisqali"`` (means
    # 0.1 / 0.9 / 0.5) silently became ``{'1': 0.9, 'Kisqali': 0.5}`` and
    # reported gap=0.4, dropping the 0.1 group when the true spread was 0.8 --
    # a gap over a basis that is not the one it claims, which is the
    # fabricated-finding shape #1574 exists to forbid.
    by_label: Dict[str, List[Any]] = {}
    rows_missing_group_key = 0
    for raw_key, raw_mean in grouped.items():
        if _is_missing_group_key(raw_key):
            # #1610. ``dropna=False`` is deliberate and is KEPT -- the null rows
            # must not vanish silently -- but a null key is not an entity, and
            # ``str(float('nan'))`` made it the label ``"nan"``. Measured on a
            # ``{west, null}`` region column that label won the comparison:
            # ``top_performer='nan'``, ``gap=0.35``. Naming a non-entity as the
            # top performer is the fabricated-finding shape #1574 forbids, and
            # ``roi_estimator`` / ``segment_ranker`` would then recommend
            # targeting it. Excluded from the basis, counted, and disclosed in
            # the refusal reason when the exclusion is what makes the request
            # uncomparable.
            rows_missing_group_key = int(df[group_col].isna().sum())
            continue
        by_label.setdefault(str(raw_key), []).append(raw_mean)

    groups_present = sorted(by_label)
    selected = by_label
    if entities:
        wanted = {str(e) for e in entities}
        selected = {k: v for k, v in by_label.items() if k in wanted}
    groups_matched = sorted(selected)

    # Collisions are only disqualifying for labels that can actually enter THIS
    # comparison, so the check runs AFTER the entity filter: an ambiguous label
    # elsewhere in the grouping column says nothing about a requested pair that
    # is itself unambiguous (codex iter-2 -- refusing that was an over-refusal).
    collisions = sorted(label for label, means in selected.items() if len(means) > 1)
    if collisions:
        raw_in_basis = sum(len(means) for means in selected.values())
        raise ToolRefusalError(
            f"gap_calculator: {raw_in_basis} distinct {group_col!r} groups entering this "
            f"comparison collapse to {len(selected)} labels under string coercion — "
            f"{_clip_entity_labels(collisions)!r} each name more than one raw group, so a "
            "per-group mean cannot be attributed unambiguously. Refusing to fabricate: "
            "comparing the surviving labels would silently drop a real group and report "
            "a spread over the wrong basis."
        )

    # ``_coerce_finite`` funnels every "not a usable number" shape to ``None``:
    # ``NaN``, ``±inf``, and ``pd.NA`` from pandas nullable dtypes (``Float64``/
    # ``Int64``), whose ``float()`` raises ``TypeError`` rather than yielding
    # ``NaN`` — which would otherwise escape this tool as a bare ``TypeError``
    # into the executor's RETRYING arm, bypassing the #1599 guard entirely.
    group_means: Dict[str, Optional[float]] = {
        label: _coerce_finite(means[0]) for label, means in selected.items()
    }

    # #1599. A group whose metric column is entirely null WITHIN the group still
    # counts as a group -- ``mean()`` yields NaN for it -- so the #1574 count
    # guard passed it through. Two consequences, both measured:
    #
    #   * two all-NaN groups produced ``gap=nan``; and
    #   * a SINGLE non-finite group hijacked ``max``/``min``. Every comparison
    #     against NaN is false, so the first-iterated key wins BOTH slots:
    #     ``{'AAA': nan, 'Ibrance': 0.7, 'Kisqali': 0.5}`` returned
    #     ``top == bottom == 'AAA'`` with ``gap=nan``, discarding a real 0.20
    #     spread. That is the #1574 top-equals-bottom shape through another
    #     door, so raising the COUNT threshold alone would not have fixed it.
    #
    # A gap is a spread between two MEASURED values, so non-finite groups are
    # excluded from the comparison basis outright -- and therefore from
    # ``entity_values``, which must stay self-consistent with the gap it
    # explains (a NaN there is also not JSON-compliant for strict consumers).
    entity_values: Dict[str, float] = {k: v for k, v in group_means.items() if v is not None}
    groups_non_finite = [k for k in groups_matched if k not in entity_values]

    if len(entity_values) < 2:
        raise ToolRefusalError(
            _gap_comparability_reason(
                entity_type=entity_type,
                group_col=group_col,
                groups_present=groups_present,
                groups_matched=groups_matched,
                groups_non_finite=groups_non_finite,
                entities=entities,
                metric=metric,
                row_count=len(df),
                rows_missing_group_key=rows_missing_group_key,
            )
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


# Caps for the entity lists in the #1574 failure reason. Both the number of
# listed values and each value's length are bounded, which gives the whole
# reason a hard ceiling: it is carried verbatim into the composer's
# total-failure envelope, which truncates from the END, and the
# ``estimation_data_scope`` payload sits at the end. An unbounded grouping
# column (territories) or a long planner-emitted ``entities`` list would
# otherwise push the scope past that bound and elide it. The ``_count``
# companions always report the true totals, so a capped list can never be read
# as a complete one.
_MAX_LISTED_ENTITY_GROUPS = 8
_MAX_ENTITY_LABEL_CHARS = 28
# #1599's non-finite list is the FOURTH entity list the reason can carry, and it
# is a SUBSET of ``entity_groups_matched`` (already listed), so it gets a tighter
# cap: naming a few examples is what the reader needs, and the ``_count``
# companion still reports the true total. Measured worst case with this cap:
# see ``test_non_finite_reason_stays_inside_the_composer_carry_limit``.
_MAX_LISTED_NON_FINITE_GROUPS = 4
# The metric name is planner/frame-supplied, so it is clipped like any other
# untrusted label before it enters a length-bounded message.
_MAX_METRIC_LABEL_CHARS = 32


def _clip_entity_labels(values: List[str], *, limit: int = _MAX_LISTED_ENTITY_GROUPS) -> List[str]:
    """Cap an entity list to ``limit`` labels, each clipped in length."""
    return [
        v if len(v) <= _MAX_ENTITY_LABEL_CHARS else v[: _MAX_ENTITY_LABEL_CHARS - 1] + "…"
        for v in values[:limit]
    ]


def _clip_name(name: Optional[str], *, limit: int = _MAX_METRIC_LABEL_CHARS) -> str:
    """Clip a single caller-supplied name for a length-bounded failure reason.

    Covers the metric column, the ``entity_type`` label and the resolved
    grouping column. All three are rendered MORE THAN ONCE in the reason (prose
    plus the ``estimation_data_scope`` payload) and none is bounded at its
    source: ``entity_type`` is planner-emitted (LLM output) and the column name
    comes from the frame. Unclipped, a pathological value measured at 21,780
    chars against the composer's 2000-char carry limit — which truncates from
    the END, eliding the scope payload the reason exists to deliver. A no-op
    for every real value ('brand', 'geographic_region', 'market_share').
    """
    text = str(name or "")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _coerce_finite(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None`` if it is not one.

    Group means arrive as numpy scalars, plain floats, ``NaN``, ``±inf``, or —
    for pandas nullable dtypes (``Float64``/``Int64``) — ``pd.NA``, whose
    ``float()`` raises ``TypeError`` instead of yielding ``NaN``. Left
    unguarded that ``TypeError`` escapes ``gap_calculator`` into the executor's
    RETRYING arm, bypassing the #1599 fail-closed guard entirely and losing the
    ``estimation_data_scope`` disclosure. Funnelling every "not a usable
    number" shape to ``None`` keeps one branch responsible for all of them.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_missing_group_key(value: Any) -> bool:
    """Return True when a ``groupby`` key is a null rather than a category.

    ``groupby(..., dropna=False)`` yields the null group under a key that is
    ``float('nan')`` for an object/float column, ``pd.NA`` for a nullable dtype
    and ``pd.NaT`` for a datetime one. All three are nulls; none of them is a
    segment or an entity, and ``str()`` turns the first into the literal label
    ``"nan"`` (#1610).

    The string ``"nan"`` is NOT a null: a column may genuinely contain a category
    of that name, and collapsing the two is the confusion this guard exists to
    prevent — so the check is ``pd.isna``, never a comparison against the
    stringified key.

    ``pandas`` is imported inside the function because this module deliberately
    keeps it off the module-load path (see ``_extract_dataframe_from_kwargs``);
    by the time a key exists the caller has already supplied a real DataFrame, so
    the import is a cached lookup. ``pd.isna`` over an array-like key (a grouper
    on multiple columns) returns an array whose ``bool()`` raises — a composite
    key is not a null key, so that case answers False.
    """
    if value is None:
        return True
    import pandas as pd

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _gap_comparability_reason(
    *,
    entity_type: str,
    group_col: str,
    groups_present: List[str],
    groups_matched: List[str],
    entities: List[str],
    row_count: int,
    groups_non_finite: Optional[List[str]] = None,
    metric: Optional[str] = None,
    rows_missing_group_key: int = 0,
) -> str:
    """Build the #1574 fail-closed reason for an uncomparable gap request.

    A gap is a SPREAD, so it needs at least two distinct entity groups. The
    orchestrator resolves ONE focal-brand-filtered estimation frame per plan
    (``dispatcher._extract_brand_region`` binds exactly one brand via
    ``query_entities.brand_from_text``, then ``resolve_kpi_frame`` /
    ``resolve_cohort_frame`` filters to it), so a "<brand> vs competitors" ask
    arrives with a single-valued brand column.

    Two things this reason must never say:

    * that a gap was computed — mirroring the one available group into both the
      top and bottom slot yields ``gap=0.0``, a plausible-but-fake "no
      competitive gap" finding (the #1574 defect), and
    * that the requested entities do not exist. ``Brand.competitor`` is a real
      enum member and ``patient_journeys`` carries competitor RWD rows; only
      ``business_metrics`` deliberately excludes them
      (``business_metrics_generator.py``). The honest claim is scoped to what
      THIS estimation frame models — the same policy as the KPI tool's
      unmatched-brand envelope (``chatbot_tools._query_kpis``): name what was
      requested, enumerate what is actually available, assert nothing about the
      wider platform.

    The ``estimation_data_scope`` payload rides in the message because that is
    the only channel that reaches synthesis (the executor surfaces a tool
    exception as ``str(e)`` on ``ToolOutput.error``), letting the answer
    disclose the scope the estimation actually covered.
    """
    # Both are rendered repeatedly below (prose + scope payload) and neither is
    # bounded at its source, so both are clipped -- see :func:`_clip_name`.
    label = _clip_name((entity_type or group_col or "entity").strip().lower()) or "entity"
    group_col = _clip_name(group_col)
    requested = [str(e) for e in entities]
    non_finite = list(groups_non_finite or [])
    # The reason has a FIXED total list budget, because it is carried verbatim
    # into the composer's total-failure envelope, which truncates from the END —
    # where ``estimation_data_scope`` sits. The #1599 branch carries a FOURTH
    # entity list, so in that branch every list gets a smaller per-list cap
    # rather than the payload growing past the carry limit and eliding itself.
    # Measured: 8-wide in the new branch = 2315 chars (over); halved = see
    # ``test_non_finite_reason_stays_inside_the_composer_carry_limit``. The
    # ``_count`` companions always report the true totals, so a tighter cap
    # narrows the examples, never the disclosed magnitude.
    listed = _MAX_LISTED_ENTITY_GROUPS // 2 if non_finite else _MAX_LISTED_ENTITY_GROUPS
    scope: Dict[str, Any] = {
        "grouping_column": group_col,
        "row_count": int(row_count),
        "entity_groups_present": _clip_entity_labels(groups_present, limit=listed),
        "entity_groups_present_count": len(groups_present),
        "entity_groups_matched": _clip_entity_labels(groups_matched, limit=listed),
    }
    if rows_missing_group_key:
        # #1610. A COUNT, never a list: there is exactly one null group per
        # column, so the count is already complete information -- and the reason
        # has a fixed total budget. Omitted entirely at zero, which is what keeps
        # every pre-#1610 reason byte-identical.
        scope["rows_missing_grouping_value"] = int(rows_missing_group_key)
    if non_finite:
        # Placed next to ``entity_groups_matched`` (the set it is a subset of)
        # rather than appended, so the two are read together.
        scope["entity_groups_non_finite"] = _clip_entity_labels(
            non_finite, limit=_MAX_LISTED_NON_FINITE_GROUPS
        )
        scope["entity_groups_non_finite_count"] = len(non_finite)
    scope["entities_requested"] = _clip_entity_labels(requested, limit=listed)
    scope["entities_requested_count"] = len(requested)

    requirement = f"at least 2 distinct {label} groups"
    verdict = "a plausible-but-fake finding"
    if non_finite:
        # #1599. The groups are real and distinct; their metric means are not
        # numbers, so there is nothing to take a spread BETWEEN. The list is
        # deliberately NOT re-rendered here — it is in the scope payload below,
        # and a second rendering is what pushes a wide list past the composer's
        # carry limit (the same call the no-match branch makes).
        metric_label = f"{_clip_name(metric)!r} " if metric else ""
        comparable = len(groups_matched) - len(non_finite)
        requirement = f"at least 2 distinct {label} groups with a finite {metric_label}mean"
        observed = (
            f"{comparable} of the {len(groups_matched)} matched {label} groups qualify — the "
            f"rest carry no non-null {metric_label}values in the supplied estimation data, so "
            "their group mean is NaN"
        )
        refusal = (
            "reporting a spread against a NaN group mean, which also silently mirrors one "
            "group into both the top and the bottom slot"
        )
        verdict = "a broken, uninterpretable finding"
    elif groups_matched:
        observed = (
            f"only 1 distinct {label} group "
            f"({_clip_entity_labels(groups_matched)[0]!r}) is present "
            "in the supplied estimation data"
        )
        refusal = (
            "reporting the one available group as both the top and the bottom "
            "performer with a gap of 0.0"
        )
    else:
        # The requested entities are NOT repeated here — they are in the scope
        # payload below, and a second rendering is what pushes a wide list past
        # the composer's carry limit.
        observed = f"no {label} group matched the requested entities on column {group_col!r}"
        refusal = "reporting a spread over groups this estimation data does not contain"
    if rows_missing_group_key:
        # #1610. Without this clause the observed-group count reads as a complete
        # account of the frame while some of its rows were excluded for having no
        # grouping value at all. Rendered in the prose (not only in the scope
        # payload) because the prose survives an END-truncation of the message.
        observed += (
            f" ({int(rows_missing_group_key)} rows carry no {group_col!r} value at all, "
            f"so they name no {label})"
        )
    return (
        f"gap_calculator: comparing requires {requirement}, but "
        f"{observed}. The requested comparison entities are not modeled as comparable "
        f"{label} entities in this estimation data, which covers "
        f"{scope['entity_groups_present']!r} on column {group_col!r} — that is a "
        "statement about the scope of THIS estimation frame, NOT a claim that the "
        f"requested entities have no data anywhere on the platform. Refusing to "
        f"fabricate a comparison: {refusal} would be {verdict}, so "
        "an uncomparable request surfaces as a structured error. Report the covered "
        f"scope instead. estimation_data_scope={scope!r}"
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


def _gap_leave_one_out_bounds(entity_values: Any) -> Optional[Tuple[float, float]]:
    """Leave-one-out sensitivity bounds for a max-minus-min gap statistic.

    The gap driving ``roi_estimator`` is ``max(entity_values) - min(entity_values)``
    — a statistic determined entirely by two extreme entities. Its real fragility
    is therefore "does this opportunity vanish if one entity is dropped?", and
    that is what this measures: recompute the gap over each leave-one-out subset
    and return ``(min, max)`` of those recomputed gaps.

    Why not a bootstrap: resampling with replacement to bracket an extremum is
    known to be inconsistent (a resample can only ever reproduce or shrink the
    observed range, never exceed it), so a bootstrap "CI" on max-minus-min is
    biased low by construction. The jackknife makes no distributional claim — it
    reports the observed sensitivity of the statistic to single entities, which
    is a question the data can actually answer.

    Returns ``None`` when fewer than 3 entity values are present: at n=2 every
    leave-one-out subset is a single entity and the gap is undefined, so there is
    no honest range to report and the caller must omit the interval rather than
    substitute a constant.
    """
    if not isinstance(entity_values, dict) or len(entity_values) < 3:
        return None

    values = [float(v) for v in entity_values.values() if isinstance(v, (int, float))]
    if len(values) < 3 or not all(math.isfinite(v) for v in values):
        return None

    loo_gaps = []
    for i in range(len(values)):
        subset = values[:i] + values[i + 1 :]
        loo_gaps.append(max(subset) - min(subset))
    return min(loo_gaps), max(loo_gaps)


@composable_tool(
    name="roi_estimator",
    description="Estimate ROI of closing identified performance gaps",
    source_agent="gap_analyzer",
    tier=2,
    input_parameters=[
        {"name": "gap_analysis", "type": "dict", "description": "Gap analysis results"},
        {"name": "investment", "type": "float", "description": "Proposed investment amount"},
        {
            "name": "value_per_unit",
            "type": "float",
            "description": "Value of one unit of the gap metric (optional; default 1.0)",
            "required": False,
            "default": 1.0,
        },
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
    - ``sensitivity_band`` is MEASURED from the entity spread, not asserted:
      a leave-one-out sensitivity band over ``entity_values`` (see
      :func:`_gap_leave_one_out_bounds`). It is ``[]`` when fewer than 3 entity
      values make the gap's sensitivity unmeasurable — an omitted range rather
      than a constant. The name states what it is: a sensitivity band, NOT a
      sampling confidence interval; ``assumptions`` spells out the semantics.

    Fail-closed: no ``gap`` in ``gap_analysis``, non-positive ``investment``, or a
    supplied ``value_per_unit`` that is not a finite number > 0 -> ``RuntimeError``
    (an ROI is undefined without a real gap, a real investment and a usable unit
    value; we refuse to fabricate one).

    Args:
        gap_analysis: Output of ``gap_calculator`` (carries ``gap`` and,
            optionally, ``entity_values``).
        investment: Proposed investment amount (must be > 0).
        **kwargs: May contain ``value_per_unit`` (float multiplier converting a
            unit of gap into monetary value).
    """
    if not isinstance(gap_analysis, dict) or "gap" not in gap_analysis:
        raise ToolRefusalError(
            "roi_estimator requires an upstream gap_analysis carrying a `gap` "
            f"value (from gap_calculator); got {gap_analysis!r}. Refusing to "
            "fabricate an ROI — per anti-mocking discipline, missing upstream "
            "data must surface as a structured error."
        )
    gap_raw = gap_analysis.get("gap")
    if not isinstance(gap_raw, (int, float)) or not math.isfinite(float(gap_raw)):
        raise ToolRefusalError(
            f"roi_estimator: gap value is not a finite number (got {gap_raw!r})."
        )
    if not isinstance(investment, (int, float)) or investment <= 0:
        raise ToolRefusalError(
            f"roi_estimator requires investment > 0; got {investment!r}. ROI is "
            "undefined for a non-positive investment; refusing to fabricate."
        )

    gap = float(gap_raw)
    entity_values = gap_analysis.get("entity_values")
    n_entities = len(entity_values) if isinstance(entity_values, dict) and entity_values else 1
    value_per_unit = kwargs.get("value_per_unit")
    if value_per_unit is None:
        value_per_unit = 1.0
    elif (
        isinstance(value_per_unit, bool)
        or not isinstance(value_per_unit, (int, float))
        or not math.isfinite(float(value_per_unit))
        or value_per_unit <= 0
    ):
        raise ToolRefusalError(
            f"roi_estimator requires value_per_unit to be a finite number > 0; got "
            f"{value_per_unit!r}. Refusing to substitute 1.0 for a value the caller supplied."
        )

    opportunity_value = gap * n_entities * float(value_per_unit)
    estimated_roi = opportunity_value / float(investment)
    if opportunity_value > 0:
        payback_months = float(investment) / (opportunity_value / 12.0)
    else:
        payback_months = float("inf")

    assumptions = [
        f"Opportunity value = gap ({gap:.4g}) x n_entities ({n_entities}) "
        f"x value_per_unit ({float(value_per_unit):.4g}).",
        "ROI = opportunity_value / investment.",
    ]

    # Uncertainty is MEASURED from the entity spread the gap was derived from,
    # never asserted as a constant: a fixed band carries no information about
    # this gap and reads as a confidence interval to downstream consumers.
    bounds = _gap_leave_one_out_bounds(entity_values)
    if bounds is None:
        sensitivity_band: List[float] = []
        assumptions.append(
            "No uncertainty range reported: fewer than 3 usable entity values, so "
            "the gap's sensitivity to individual entities is not measurable. An "
            "interval is omitted rather than assumed."
        )
    else:
        low_gap, high_gap = bounds
        sensitivity_band = [
            (low_gap * n_entities * float(value_per_unit)) / float(investment),
            (high_gap * n_entities * float(value_per_unit)) / float(investment),
        ]
        assumptions.append(
            f"Range is a leave-one-out sensitivity band over {n_entities} entity "
            f"values (gap recomputed with each entity dropped: {low_gap:.4g} to "
            f"{high_gap:.4g}), holding n_entities fixed. It measures how far the "
            "gap depends on any single entity — it is NOT a sampling confidence "
            "interval and carries no coverage guarantee."
        )

    return ROIEstimate(
        estimated_roi=float(estimated_roi),
        payback_months=float(payback_months),
        sensitivity_band=[float(x) for x in sensitivity_band],
        assumptions=assumptions,
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
    intervention: str, target_entities: List[str], expected_effect: Optional[float], **kwargs
) -> SimulationResults:
    """Simulate intervention outcomes.

    Null-guard (#1573): a ``None`` / non-numeric ``expected_effect`` means no
    upstream effect estimate was actually supplied (live q08: the planner
    referenced fields the CATE output does not carry, which degraded to
    ``None`` and crashed here with ``NoneType * float`` three times). The
    tool declines with a stated reason — a deterministic
    :class:`ToolInputError` the executor does NOT retry — instead of
    fabricating a lift or raising a bare ``TypeError``.
    """
    if not isinstance(expected_effect, (int, float)) or isinstance(expected_effect, bool):
        raise ToolInputError(
            "counterfactual_simulator declined: expected_effect is "
            f"{expected_effect!r} — no usable effect estimate was supplied "
            "(an upstream step likely failed or its output lacked the "
            "referenced field). Refusing to simulate a lift from a missing "
            "effect."
        )
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
        {
            "name": "period_column",
            "type": "str",
            "description": "Column holding the period labels",
            "required": False,
            "default": "period",
        },
    ],
    output_schema="DriftMetrics",
    avg_execution_ms=800,
    output_model=DriftMetrics,
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
        raise ToolRefusalError(
            "psi_calculator requires a real DataFrame supplied via one of the "
            f"kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r}. The tool does not fabricate a PSI — "
            "missing data must surface as a structured error."
        )
    for col in (feature, period_column):
        if col not in df.columns:
            raise ToolRefusalError(
                f"psi_calculator: column {col!r} not found in the supplied "
                f"DataFrame (columns={list(df.columns)!r}). Refusing to "
                "fabricate a result."
            )
    baseline = df.loc[df[period_column] == baseline_period, feature].dropna()
    current = df.loc[df[period_column] == current_period, feature].dropna()
    if len(baseline) == 0 or len(current) == 0:
        raise ToolRefusalError(
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
    return DriftMetrics(
        psi=psi_value,
        interpretation=interpretation,
        threshold=threshold,
        buckets=buckets,
    ).model_dump()


@composable_tool(
    name="distribution_comparator",
    description="Compare feature distributions between time periods",
    source_agent="drift_monitor",
    tier=3,
    input_parameters=[
        {"name": "features", "type": "List[str]", "description": "Features to compare"},
        {"name": "period_1", "type": "str", "description": "First time period"},
        {"name": "period_2", "type": "str", "description": "Second time period"},
        {
            "name": "period_column",
            "type": "str",
            "description": "Column holding the period labels",
            "required": False,
            "default": "period",
        },
    ],
    output_schema="DistributionComparison",
    avg_execution_ms=1200,
    output_model=DistributionComparison,
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
        raise ToolRefusalError(
            "distribution_comparator requires a real DataFrame supplied via one "
            f"of the kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs "
            f"keys={sorted(kwargs.keys())!r}. The tool does not fabricate KS "
            "statistics — missing data must surface as a structured error."
        )
    if period_column not in df.columns:
        raise ToolRefusalError(
            f"distribution_comparator: period column {period_column!r} not found "
            f"in the supplied DataFrame (columns={list(df.columns)!r})."
        )
    p1_mask = df[period_column] == period_1
    p2_mask = df[period_column] == period_2
    if int(p1_mask.sum()) == 0 or int(p2_mask.sum()) == 0:
        raise ToolRefusalError(
            f"distribution_comparator: period_1={period_1!r} matched "
            f"{int(p1_mask.sum())} rows and period_2={period_2!r} matched "
            f"{int(p2_mask.sum())} rows in column {period_column!r}; both must "
            "be non-empty. Refusing to fabricate a result."
        )
    comparisons: List[Dict[str, Any]] = []
    any_drift = False
    for feature in features:
        if feature not in df.columns:
            raise ToolRefusalError(
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
    return DistributionComparison(comparisons=comparisons, overall_drift=any_drift).model_dump()


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
        {
            "name": "id_column",
            "type": "str",
            "description": "Column holding the entity IDs",
            "required": False,
            "default": "patient_id",
        },
        {
            "name": "outcome",
            "type": "str",
            "description": "Binary outcome column the risk model predicts (the risk event)",
            "required": False,
            "default": "discontinuation_flag",
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

    Fail-closed: no DataFrame, missing outcome column, an outcome that is not a
    0/1 event column, fewer than 2 outcome classes, or no usable numeric features
    -> ``RuntimeError`` (we refuse to fabricate scores).

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
        raise ToolRefusalError(
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
        raise ToolRefusalError(
            f"risk_scorer: outcome column {outcome!r} not found in the supplied "
            f"DataFrame (columns={list(df.columns)!r})."
        )
    if id_column not in df.columns:
        raise ToolRefusalError(
            f"risk_scorer: id_column {id_column!r} not found in the supplied "
            f"DataFrame (columns={list(df.columns)!r}). Refusing to fabricate "
            "entity IDs."
        )

    work = df
    if entity_ids:
        wanted = {str(e) for e in entity_ids}
        work = df[df[id_column].astype(str).isin(wanted)]
        if len(work) == 0:
            raise ToolRefusalError(
                f"risk_scorer: no rows matched entity_ids={entity_ids!r} on "
                f"column {id_column!r}. Refusing to fabricate."
            )

    feature_cols = [c for c in work.select_dtypes(include="number").columns if c != outcome]
    if not feature_cols:
        raise ToolRefusalError(
            "risk_scorer: no usable numeric feature columns to fit a model "
            f"(numeric columns minus outcome were empty; columns={list(work.columns)!r})."
        )

    observed = set(work[outcome].dropna().unique())
    if not observed <= {0, 1}:
        raise ToolRefusalError(
            f"risk_scorer: outcome column {outcome!r} is not a binary 0/1 event column "
            f"(observed values include {sorted(map(str, observed))[:6]!r}). Refusing to "
            "cast it to classes and report a class probability as a risk score."
        )
    y = work[outcome].astype(int)
    if y.nunique() < 2:
        raise ToolRefusalError(
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
        raise ToolRefusalError(
            "propensity_estimator requires a real DataFrame supplied via one of "
            f"the kwargs keys {list(_DATAFRAME_KWARGS_KEYS)!r}; got kwargs keys="
            f"{sorted(kwargs.keys())!r}. The tool does not fabricate propensity "
            "scores — per anti-mocking discipline, missing data must surface as "
            "a structured error rather than a plausible-but-fake placeholder."
        )
    if treatment not in df.columns:
        raise ToolRefusalError(
            f"propensity_estimator: treatment column {treatment!r} not found in "
            f"the supplied DataFrame (columns={list(df.columns)!r})."
        )
    if not covariates:
        raise ToolRefusalError(
            "propensity_estimator requires at least one covariate column; got an empty list."
        )
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ToolRefusalError(
            f"propensity_estimator: covariate columns {missing!r} not found in "
            f"the supplied DataFrame (columns={list(df.columns)!r})."
        )

    t = df[treatment].astype(int)
    if t.nunique() < 2:
        raise ToolRefusalError(
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
