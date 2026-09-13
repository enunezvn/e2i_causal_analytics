"""
E2I Causal Inference API
========================

FastAPI endpoints for causal inference capabilities.

Phase B10: Causal API endpoints for:
- Hierarchical analysis (EconML within CausalML segments)
- Library routing (DoWhy, EconML, CausalML, NetworkX)
- Multi-library pipelines (sequential, parallel)
- Cross-validation between libraries

Endpoints:
- /causal/hierarchical/analyze: Run hierarchical CATE analysis
- /causal/hierarchical/{analysis_id}: Get analysis results
- /causal/route: Route query to appropriate library
- /causal/pipeline/sequential: Run sequential multi-library pipeline
- /causal/pipeline/parallel: Run parallel multi-library analysis
- /causal/validate: Run cross-library validation
- /causal/estimators: List available estimators
- /causal/health: Health check for causal engine

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import asyncio
import contextlib
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    cast,
)

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query

if TYPE_CHECKING:
    from src.repositories.causal_path import CausalPathRepository

from src.api.dependencies.auth import require_analyst, require_viewer
from src.api.dependencies.compute import HeavyComputeSaturated, heavy_compute_slot
from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.errors import user_safe_503_detail
from src.api.models.graph import (
    CausalChainResponse,
    EntityType,
    GraphNode,
    GraphPath,
    GraphRelationship,
    RelationshipType,
)
from src.api.schemas.causal import (
    AGENT_FORCEABLE_ESTIMATORS,
    AgentCausalAnalysisRequest,
    AgentCausalAnalysisResponse,
    AggregationMethod,
    AnalysisStatus,
    CausalAnalysisHistoryItem,
    CausalAnalysisHistoryResponse,
    CausalBrandsResponse,
    CausalDAGModel,
    CausalHealthResponse,
    CausalLibrary,
    CausalVariablesResponse,
    ClinicalContext,
    CrossValidationRequest,
    CrossValidationResponse,
    DiscoveredEffect,
    DiscoverEffectsRequest,
    DiscoverEffectsResponse,
    DiscoverQuestion,
    DiscoverQuestionSelection,
    DiscoverQuestionsResponse,
    EdgeProvenanceModel,
    EstimationDataResponse,
    EstimatorCandidate,
    EstimatorComparison,
    EstimatorInfo,
    EstimatorListResponse,
    HierarchicalAnalysisRequest,
    HierarchicalAnalysisResponse,
    NestedCIResult,
    ParallelPipelineRequest,
    ParallelPipelineResponse,
    PipelineMode,
    PipelineStageResult,
    ProposedQuestion,
    ProposeQuestionsResponse,
    QuestionType,
    RefutationSummary,
    RefutationTestDetail,
    RouteQueryRequest,
    RouteQueryResponse,
    SegmentationMethod,
    SegmentCATEResult,
    SequentialPipelineRequest,
    SequentialPipelineResponse,
    TreatmentEffectResponse,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.causal.stats import z_score_for_confidence

# #354 C-8: real-pipeline wiring (replaces 503-default short-circuit in
# non-demo mode). Imported lazily-safely; the LibraryExecutor implementations
# inside ParallelPipeline / SequentialPipeline themselves guard their backend
# dependencies (dowhy/econml/causalml/networkx availability), so importing
# the orchestrator classes is cheap.
from src.causal_engine.pipeline.parallel import ParallelPipeline
from src.causal_engine.pipeline.router import (
    LibraryRouter,
)
from src.causal_engine.pipeline.router import (
    QuestionType as RouterQuestionType,
)
from src.causal_engine.pipeline.sequential import SequentialPipeline
from src.causal_engine.pipeline.state import (
    PipelineInput,
    PipelineOutput,
    PipelineState,
)
from src.insights.robustness_phrase import gate_verdict_phrase

# #931: the health check's analysis-activity fields and the Analysis History tab
# read REAL completed causal-analysis events from episodic_memories (the
# canonical store written by the causal_impact agent's
# ``causal_analysis_completed`` episodic hook). Reuse the episodic repository
# rather than issuing raw SQL from the route. Imported at module level so the
# read functions are patchable in tests as ``causal.count_memories_by_type`` /
# ``causal.get_recent_memories``.
from src.memory.episodic_memory import count_memories_by_type, get_recent_memories
from src.repositories.provenance import apply_provenance_filter, deployment_includes_synthetic
from src.utils.redaction import redact_query

from . import agent, catalog, hierarchical
from ._common import (  # noqa: F401  moved here by the #1991 debt-4 split
    _AGENT_HARD_TIMEOUT_S,
    _CAUSAL_JOB_TTL_SECONDS,
    _CYCLE_IRRELEVANT_WARNING,
    _DATA_REQUIRED_LIBRARIES,
    _GENERIC_500_DETAIL,
    _NO_REAL_DATA_BACKEND_DETAIL,
    _NO_RESOLVABLE_DATA_DETAIL,
    _NON_DAG_STRUCTURAL_WARNING,
    _REFUTATION_COMPUTE_BUDGET_S,
    _ROBUSTNESS_BLOCK_WARNING,
    _ROBUSTNESS_REVIEW_WARNING,
    _ROBUSTNESS_UNVALIDATED_WARNING,
    CAUSAL_COMPLETED_EVENT_TYPE,
    _as_float,
    _as_optional_float,
    _dowhy_interval,
    _opt_float,
    _parse_occurred_at,
    _resolve_pipeline_dataframe,
    _te_pvalue_from_z,
)
from .agent import _agent_analysis_store, _run_agent_analysis_task
from .catalog import _adjusted_partial_corr, _get_clinical_context_service
from .datasets import (  # noqa: F401  moved here by the #1991 debt-4 split
    _ADVANCED_LINE_STAGES,
    _ALL_CLINICAL_COVARIATES,
    _BRAND_CLINICAL_COVARIATES,
    _CAUSAL_BRAND_COLUMN,
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_FILL_ZERO_OUTCOMES,
    _CAUSAL_NEGATIVE_CONTROL_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_NUMERIC_DERIVATIONS,
    _CAUSAL_PHYSICAL_TABLE,
    _COLUMN_DEFINITIONS,
    _COLUMN_LABELS,
    _DEFAULT_CAUSAL_DATASET,
    _DISCOVERY_ROW_CAP,
    _JOIN_DATASETS,
    _NBA_JOINED_COVARIATES,
    _UNCONTROLLED_UAS7_THRESHOLD,
    _UNIVERSAL_COVARIATES,
    _brand_scoped_covariates,
    _column_label,
    _derive_is_accepted,
    _derive_is_advanced_line,
    _derive_is_prior_c5,
    _derive_is_uncontrolled_csu,
    _derive_presence,
    _is_randomized_treatment,
    _list_dataset_brands,
    _negative_control_outcome,
)
from .loaders import (  # noqa: F401  moved here by the #1991 debt-4 split
    _NBA_BASELINE_CATEGORICALS,
    _NBA_JOIN_MAX_PAGES,
    _TE_MAX_PAGES,
    _TE_PAGE_SIZE,
    _coerce_estimation_row,
    _get_causal_path_repo,
    _load_agent_estimation_frame,
    _load_hcp_adoption_join_frame,
    _load_hcp_profile_centrality,
    _load_nba_triggers_join_frame,
    _load_patient_baseline_rows,
    _load_trigger_question_rows,
    _one_hot_categoricals,
    _require_covariate_role,
    _resolve_requested_baselines,
    _te_paged_select,
    _te_paged_select_all_brands,
)

logger = logging.getLogger(__name__)


# #931 (review M1): /causal/health is a PUBLIC, unauthenticated endpoint the
# dashboard polls every ~30s. The activity fields now read episodic_memories,
# so memoize the result for a short window to keep repeated/unauthenticated
# polls from amplifying into two DB reads each. The cache holds the REAL value
# (or the honest fallback) — it never serves a fabricated number.
_ACTIVITY_CACHE_TTL_SECONDS = 30.0
_activity_cache: dict[str, Any] = {"expires_at": 0.0, "value": (0, None)}


router = APIRouter(
    prefix="/causal",
    tags=["Causal Inference"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

for _sub in (hierarchical, catalog, agent):
    router.include_router(_sub.router)


# =============================================================================
# IN-MEMORY STORAGE (for demo - replace with database in production)
# =============================================================================

_pipeline_cache: Dict[str, Dict[str, Any]] = {}
_validation_cache: Dict[str, CrossValidationResponse] = {}


async def _prerank_signal(dataset: str, q: "_CandidateQuestion") -> float:
    """Cheap FWL screen for one question; 0.0 when undefined / unloadable."""
    try:
        df, select_cols = await _load_agent_estimation_frame(
            dataset=dataset,
            treatment_var=q.treatment,
            outcome_var=q.outcome,
            covariates=q.adjustment_set,
            limit=1500,
            brand=q.brand,
        )
    except HTTPException:
        return 0.0
    # Use the loader's EXPANDED columns (categoricals like geographic_region are
    # one-hot dummies in ``df``, not their raw name) so the FWL screen indexes real
    # frame columns instead of KeyError-ing on the raw categorical in adjustment_set.
    cov = [c for c in select_cols if c not in (q.treatment, q.outcome)]
    pc = _adjusted_partial_corr(df, q.treatment, q.outcome, cov)
    return abs(pc) if pc is not None else 0.0


async def _prerank_questions(
    dataset: str, questions: List["_CandidateQuestion"]
) -> List["_CandidateQuestion"]:
    """Order candidates by descending data-driven association so strong effects
    validate first (the leaderboard fills progressively)."""
    scored = await asyncio.gather(*[_prerank_signal(dataset, q) for q in questions])
    return [
        q for _, q in sorted(zip(scored, questions, strict=True), key=lambda p: p[0], reverse=True)
    ]


# =============================================================================
# DISCOVER EFFECTS — validated-effects leaderboard (async submit -> poll)
# =============================================================================

# Cross-worker job store (Redis-backed; mirrors _agent_analysis_store). Each job
# runs the agent for a set of candidate questions and ranks the VALIDATED effects.
_discover_effects_store: DurableJobStore["DiscoverEffectsResponse"] = DurableJobStore(
    "causal:discover_effects", DiscoverEffectsResponse, ttl_seconds=_CAUSAL_JOB_TTL_SECONDS
)
# A discover job ends in exactly one of these; a cancel on any of them is a no-op.
_TERMINAL_DISCOVERY_STATUSES = frozenset({"completed", "cancelled", "failed"})
# Sidecar marker (see DurableJobStore.set_marker) the cancel route raises and the
# background task polls at every question boundary. A BackgroundTask cannot be
# signalled from another request (or worker) any other way.
_DISCOVERY_CANCEL_MARKER = "cancel"
# Liveness heartbeat (see DurableJobStore.touch_marker): the task re-stamps this
# sidecar every INTERVAL while it runs; a poll that finds a non-terminal row
# whose stamp is older than TTL (or absent) knows the task is gone — the API
# restarted (every deploy), gunicorn recycled the worker (--max-requests), or
# the task crashed — and repairs the row to `failed` instead of leaving it
# `running` until the 8h TTL with the FE polling forever. Why not a startup
# sweep: prod runs 2 workers, and a freshly (re)started worker cannot tell
# whether the OTHER worker's jobs are still alive; the heartbeat can. TTL = the
# gunicorn worker timeout (docker/Dockerfile --timeout 120): an event loop
# stalled that long is killed anyway, so a gap that long means the worker is
# gone, never a live run.
_DISCOVERY_ALIVE_MARKER = "alive"
_DISCOVERY_HEARTBEAT_INTERVAL_SECONDS: float = 15.0
_DISCOVERY_HEARTBEAT_TTL_SECONDS: int = 120

# Complementary outcomes are 1 - each other (persistent_180d vs discontinued_180d);
# running both is redundant, so one is skipped to dedupe the leaderboard.
_COMPLEMENT_OUTCOMES_SKIP = {"discontinued_180d"}


class _CandidateQuestion(NamedTuple):
    treatment: str
    outcome: str
    brand: Optional[str]
    adjustment_set: List[str]


async def _discover_candidate_questions(
    dataset: str, brand: Optional[str]
) -> List[_CandidateQuestion]:
    """SSOT-derived leaderboard questions (replaces the hand-curated cross-product).

    Reads distinct (treatment, outcome, brand) from ``causal_paths`` and attaches
    each row's modeled ``confounders_controlled``, intersected with the dataset's
    numeric AND categorical allowlists (P1b/D5: ``geographic_region`` is categorical,
    so it now enters the adjustment set as its RAW name here — the loader one-hot
    EXPANDS it into ``geographic_region=<level>`` dummies downstream). The
    column-allowlist security gate in ``_load_agent_estimation_frame`` is unchanged
    — this replaces ENUMERATION only, not validation."""
    spec = _CAUSAL_DATASET_SPECS[dataset]
    numeric = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())
    categorical = _CAUSAL_CATEGORICAL_COLUMNS.get(dataset, set())
    allowed_cov = set(spec["covariate"]) & (numeric | categorical)
    repo = await _get_causal_path_repo()
    rows = await repo.get_distinct_questions(brand=brand, include_synthetic=True)
    out: List[_CandidateQuestion] = []
    for r in rows:
        t, o = r["treatment"], r["outcome"]
        if t == o or o in _COMPLEMENT_OUTCOMES_SKIP:
            continue
        # Grain-scope guard (shared with later phases): causal_paths has no `grain`
        # column and other grains will share the table, so restrict to questions
        # whose treatment AND outcome are in THIS dataset's spec. No-op for patient.
        if t not in spec["treatment"] or o not in spec["outcome"]:
            continue
        adj = _brand_scoped_covariates(
            [c for c in r.get("confounders", []) if c in allowed_cov and c not in (t, o)],
            brand,
        )
        out.append(
            _CandidateQuestion(treatment=t, outcome=o, brand=r.get("brand"), adjustment_set=adj)
        )
    return out


def _effect_confidence_score(gate_decision: Optional[str], significant: bool) -> float:
    """Map the robustness gate + significance to a 0-1 ranking signal."""
    base = {"proceed": 0.6, "review": 0.35}.get(gate_decision or "", 0.1)
    return min(1.0, base + (0.3 if significant else 0.0))


def _effect_status_from_gate(ate: Optional[float], gate: Optional[str], resp_status: str) -> str:
    """Honest leaderboard status. A run that produced an estimate is reported by
    its robustness verdict (completed/needs_review/blocked) — only a run that
    produced NO estimate is 'failed'. This separates 'the gate blocked it'
    (computed, worth inspecting) from 'it could not run'."""
    if ate is None:
        return "failed" if resp_status not in {"pending", "running"} else resp_status
    if gate == "proceed":
        return "completed"
    if gate == "review":
        return "needs_review"
    if gate == "block":
        return "blocked"
    return resp_status


def _effect_summary(
    treatment: str,
    outcome: str,
    ate: Optional[float],
    gate: Optional[str],
    significant: bool,
    tests: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """One-line plain-language reading of a validated effect. None until estimated.

    #1868: the verdict phrase follows the per-test outcomes — "survived all"
    is reserved for an all-PASSED suite; a proceed gate with warnings or
    non-critical failures says so."""
    if ate is None:
        return None
    direction = "raises" if ate > 0 else "lowers" if ate < 0 else "does not change"
    verdict = gate_verdict_phrase(gate, tests) or "robustness unknown"
    sig = "statistically significant" if significant else "not statistically significant"
    # Curated display labels, never the raw column: the leaderboard row above
    # this sentence renders the label, and the two must read as one name.
    return (
        f"{_column_label(treatment)} {direction} {_column_label(outcome)} "
        f"by {ate:+.3f} — {verdict}, {sig}."
    )


def _effect_from_agent_response(
    treatment: str,
    outcome: str,
    resp: "AgentCausalAnalysisResponse",
    analysis_id: str,
    question: Optional[_CandidateQuestion] = None,
) -> DiscoveredEffect:
    gate = resp.refutation.gate_decision if resp.refutation else None
    gate_tests = (
        [t.model_dump() for t in resp.refutation.tests]
        if resp.refutation and resp.refutation.tests
        else None
    )
    return DiscoveredEffect(
        treatment=treatment,
        outcome=outcome,
        brand=question.brand if question is not None else None,
        adjustment_set=list(question.adjustment_set) if question is not None else [],
        status=_effect_status_from_gate(resp.ate, gate, resp.status),
        ate=resp.ate,
        ate_ci_lower=resp.ate_ci_lower,
        ate_ci_upper=resp.ate_ci_upper,
        p_value=resp.p_value,
        statistical_significance=bool(resp.statistical_significance),
        selected_estimator=resp.selected_estimator,
        gate_decision=gate,
        confidence_score=_effect_confidence_score(gate, bool(resp.statistical_significance)),
        impact=abs(resp.ate) if resp.ate is not None else None,
        n_rows=resp.n_rows,
        summary=_effect_summary(
            treatment, outcome, resp.ate, gate, bool(resp.statistical_significance), gate_tests
        ),
        analysis_id=analysis_id,
    )


async def _attach_clinical_context(effect: DiscoveredEffect) -> None:
    """Best-effort: attach clinical context for THIS row's analysis (brand +
    treatment -> outcome) to a completed leaderboard row. FAIL-OPEN — any failure
    (unknown brand, API down) leaves ``clinical_context=None`` and never disrupts the
    row or the discover job. Skips rows without a brand or without an estimate. The
    brand-level fan-out is cached per (brand, disease) so the many candidate rows of
    one brand trigger a single live fan-out; only the literature citation varies per
    analysis (#1763) and is cached per composed query."""
    if not effect.brand or effect.ate is None:
        return
    try:
        payload = await asyncio.to_thread(
            _get_clinical_context_service().get_context,
            effect.brand,
            effect.outcome,
            treatment=effect.treatment,
        )
        effect.clinical_context = ClinicalContext.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — best-effort; context never fails a row
        logger.debug(
            "discover-effects: clinical context unavailable for %s/%s: %s",
            effect.brand,
            effect.outcome,
            exc,
        )


def _rank_effects(effects: List[DiscoveredEffect]) -> List[DiscoveredEffect]:
    """Rank by confidence (gate + significance) then impact (|ate|). Not-yet-run
    questions (score 0) sort last."""
    return sorted(
        effects,
        key=lambda e: (e.confidence_score, e.impact if e.impact is not None else -1.0),
        reverse=True,
    )


def _pending_effect(q: _CandidateQuestion, status: str) -> DiscoveredEffect:
    """A not-yet-validated leaderboard cell (pending/running/failed) carrying the
    SSOT brand + modeled adjustment set but no estimate yet."""
    return DiscoveredEffect(
        treatment=q.treatment,
        outcome=q.outcome,
        brand=q.brand,
        adjustment_set=list(q.adjustment_set),
        status=status,
    )


def _interrupt_effect(e: DiscoveredEffect) -> DiscoveredEffect:
    """Honest terminal row for a question the run never finished: the one in
    flight is `failed` (it could not run to an estimate), a queued one is
    `cancelled` (it never got its turn). Finished rows are untouched."""
    if e.status == "running":
        return e.model_copy(update={"status": "failed"})
    if e.status == "pending":
        return e.model_copy(update={"status": "cancelled"})
    return e


async def _touch_discovery_heartbeat(job_id: str) -> None:
    await _discover_effects_store.touch_marker(
        job_id, _DISCOVERY_ALIVE_MARKER, ttl_seconds=_DISCOVERY_HEARTBEAT_TTL_SECONDS
    )


async def _discovery_is_alive(job_id: str) -> bool:
    age = await _discover_effects_store.marker_age_seconds(job_id, _DISCOVERY_ALIVE_MARKER)
    return age is not None and age <= _DISCOVERY_HEARTBEAT_TTL_SECONDS


async def _repair_if_orphaned(job: DiscoverEffectsResponse) -> DiscoverEffectsResponse:
    """Read-repair for an orphaned run. A non-terminal row whose task no longer
    beats is over: mark it `failed` with the reason, keep every finished row,
    close the unfinished ones honestly, and PERSIST it so every later poll (on
    any worker) agrees and the FE stops polling. Terminal rows are returned
    as-is — their task is gone because it finished."""
    if job.status in _TERMINAL_DISCOVERY_STATUSES or await _discovery_is_alive(job.job_id):
        return job
    repaired = job.model_copy(
        update={
            "status": "failed",
            "error": (
                "The discovery run was interrupted (the API restarted or its worker "
                f"was recycled) after {job.completed}/{job.total} questions; re-run "
                "discovery to continue."
            ),
            "effects": [_interrupt_effect(e) for e in job.effects],
        }
    )
    logger.warning(
        f"discover-effects {job.job_id}: no heartbeat for a `{job.status}` row "
        f"({job.completed}/{job.total}); repaired to failed"
    )
    await _discover_effects_store.set(job.job_id, repaired)
    return repaired


async def _run_discover_effects_task(
    job_id: str,
    dataset: str,
    questions: List[_CandidateQuestion],
    data_source: str,
    brand: Optional[str] = None,
) -> None:
    """Background: validate each candidate question with the causal_impact agent
    (serial — each acquires the heavy-compute slot), updating the cached job after
    each so the FE leaderboard fills in progressively, ranked by confidence+impact.

    Questions are SSOT-derived (see ``_discover_candidate_questions``): each carries
    its own brand and modeled adjustment set. ``brand`` (the request-level filter)
    is a fallback when a question's row has no brand."""
    # Keyed pending effects we mutate in place across the run. The (treatment,
    # outcome, brand) triple keys the dict because the SSOT can carry the same
    # (treatment, outcome) for several brands.
    effects: Dict[tuple, DiscoveredEffect] = {
        (q.treatment, q.outcome, q.brand): _pending_effect(q, "pending") for q in questions
    }

    async def _publish(
        status: str, completed: int, cancel_requested: bool = False, error: Optional[str] = None
    ) -> None:
        await _discover_effects_store.set(
            job_id,
            DiscoverEffectsResponse(
                job_id=job_id,
                status=status,
                dataset=dataset,
                brand=brand,
                total=len(questions),
                completed=completed,
                cancel_requested=cancel_requested,
                error=error,
                effects=_rank_effects(list(effects.values())),
            ),
        )

    async def _boundary_stop_reason() -> Optional[str]:
        """Why the run must stop at this question boundary, or None to go on.

        ``"cancel"``: the sidecar marker is the primary, race-free signal. The
        row flag is the fallback for a degraded cancel: the route's marker SET
        can fail transiently on ITS worker (the marker then lives only in that
        process's memory, invisible here) while its row write still reaches
        Redis and the route has already answered 200. Honouring the flag too
        means the API never acknowledges a cancel this run then ignores.

        ``"repaired"``: a poll on some worker already closed this row as
        ``failed`` — it found no live heartbeat (see ``_repair_if_orphaned``),
        e.g. because THIS worker's beats could not reach Redis for the whole
        budget. The row is terminal and the page has stopped polling; publishing
        again would resurrect a run nobody is watching and burn minutes per
        question for nothing. The repaired row stands.
        """
        if await _discover_effects_store.has_marker(job_id, _DISCOVERY_CANCEL_MARKER):
            return "cancel"
        row = await _discover_effects_store.get(job_id)
        if row is None:
            return None
        if row.cancel_requested:
            return "cancel"
        if row.status == "failed":
            return "repaired"
        return None

    def _log_repaired(completed: int) -> None:
        logger.warning(
            f"discover-effects {job_id}: a poll already closed this row as failed "
            f"(no live heartbeat seen); stopping after {completed}/{len(questions)} "
            "without publishing"
        )

    async def _stop_cancelled(completed: int) -> None:
        # Honest terminal rows for the questions that never ran: status only —
        # no estimate, no summary, nothing fabricated. Finished rows are kept.
        for k, e in effects.items():
            if e.status == "pending":
                effects[k] = e.model_copy(update={"status": "cancelled"})
        await _publish("cancelled", completed, cancel_requested=True)

    async def _beat() -> None:
        # Liveness heartbeat (see _repair_if_orphaned). The estimators run in a
        # worker thread, so the loop is free to beat right through a question;
        # polls on ANY worker read the stamp's age. A failed touch must never end
        # the beat — a silently stopped heartbeat would declare a live run dead.
        while True:
            try:
                await _touch_discovery_heartbeat(job_id)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"discover-effects {job_id}: heartbeat touch failed: {e}")
            await asyncio.sleep(_DISCOVERY_HEARTBEAT_INTERVAL_SECONDS)

    # First beat inline, BEFORE anything else: a created task only runs once this
    # coroutine suspends, and nothing here is guaranteed to (the pre-rank, the
    # store, a stubbed loader may all complete without yielding). The submit
    # route stamps the row too, but the task must not rely on it.
    await _touch_discovery_heartbeat(job_id)
    heartbeat = asyncio.create_task(_beat())
    completed = 0
    try:
        questions = await _prerank_questions(dataset, questions)
        for q in questions:
            # Cooperative cancel, honoured at question boundaries only: the agent's
            # estimators run synchronously inside the question and cannot be
            # interrupted, so a cancel lands after the in-flight question finishes.
            reason = await _boundary_stop_reason()
            if reason == "cancel":
                await _stop_cancelled(completed)
                return
            if reason == "repaired":
                _log_repaired(completed)
                return
            t, o = q.treatment, q.outcome
            key = (t, o, q.brand)
            # Fallback to the request-level brand filter when the SSOT row has no
            # brand. NOTE: this fallback scopes the DATA LOAD only; the effect's
            # ``brand`` label stays q.brand (None here). Harmless for patient grain —
            # the reseed populates brand per causal_paths row — but later grains that
            # share this table must revisit whether the fallback brand should be the
            # displayed label.
            q_brand = q.brand or brand
            effects[key] = _pending_effect(q, "running")
            await _publish("running", completed)
            # #2007: fetch the declared negative-control outcome as a PASSTHROUGH
            # column exactly like the submit endpoint does — ``_run_agent_analysis_task``
            # splits it off into ``data_cache["negative_control_data"]`` only when it is
            # a column of ``df``. Live cert 2026-09-11 (job 457b345f on 903b7addc): without
            # this every discovery row read SKIPPED ``negative_control_column_missing``.
            negative_control = _negative_control_outcome(dataset, t, o)
            try:
                df, select_cols = await _load_agent_estimation_frame(
                    dataset=dataset,
                    treatment_var=t,
                    outcome_var=o,
                    covariates=q.adjustment_set,
                    limit=_DISCOVERY_ROW_CAP,
                    brand=q_brand,
                    passthrough_columns=[negative_control] if negative_control else None,
                )
                # The loader EXPANDS categorical covariates (e.g. geographic_region)
                # into one-hot dummies; the agent run must adjust on the resolved frame
                # columns (the dummy names), not the raw categorical. Derive them from
                # the loader's returned column list, excluding treatment/outcome.
                resolved_cov = [c for c in select_cols if c not in (t, o)]
                aid = str(uuid.uuid4())
                req = AgentCausalAnalysisRequest(
                    treatment_var=t,
                    outcome_var=o,
                    dataset=dataset,
                    limit=_DISCOVERY_ROW_CAP,
                    auto_discover=True,
                    brand=q_brand,
                )
                await _agent_analysis_store.set(
                    aid,
                    AgentCausalAnalysisResponse(
                        analysis_id=aid,
                        status="pending",
                        treatment_var=t,
                        outcome_var=o,
                        dataset=dataset,
                        n_rows=int(df.shape[0]),
                        data_source=data_source,
                        dag=CausalDAGModel(),
                        statistical_significance=False,
                        refutation=RefutationSummary(),
                        latency_ms=0,
                    ),
                )
                await _run_agent_analysis_task(aid, req, df, resolved_cov, data_source)
                resp = await _agent_analysis_store.get(aid)
                if resp is None:
                    raise RuntimeError(f"agent analysis {aid} produced no cached result")
                effects[key] = _effect_from_agent_response(t, o, resp, aid, question=q)
            except HTTPException as e:
                # Fail-closed: a question with no usable data is marked failed, not faked.
                logger.warning(f"discover-effects: {t}->{o} failed-closed: {e.detail}")
                effects[key] = _pending_effect(q, "failed")
            except Exception as e:  # noqa: BLE001
                logger.error(f"discover-effects: {t}->{o} errored: {e}", exc_info=True)
                effects[key] = _pending_effect(q, "failed")
            # Best-effort clinical context for the freshly-built row (no-op for failed/
            # pending rows; fail-open so it never disrupts the leaderboard).
            await _attach_clinical_context(effects[key])
            completed += 1
            reason = await _boundary_stop_reason()
            if reason == "repaired":
                _log_repaired(completed)
                return
            if completed == len(questions):
                # A cancel landing after the last question is still a completed run.
                await _publish("completed", completed)
            elif reason == "cancel":
                await _stop_cancelled(completed)
                return
            else:
                await _publish("running", completed)
    except Exception as e:  # noqa: BLE001
        # An error OUTSIDE a question (the pre-rank, a publish, ...) used to end
        # the task silently and leave the row `running` until the TTL. Close it
        # honestly instead: finished rows kept, the rest interrupted, reason set.
        logger.error(f"discover-effects {job_id}: run crashed: {e}", exc_info=True)
        for k, eff in effects.items():
            effects[k] = _interrupt_effect(eff)
        await _publish(
            "failed",
            completed,
            error=(
                f"The discovery run crashed before finishing ({type(e).__name__}: {e}); "
                "re-run discovery to continue."
            ),
        )
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat


async def _resolve_discovery_scope(
    dataset: str, brand: Optional[str]
) -> tuple[Optional[str], List[_CandidateQuestion]]:
    """Validate a (dataset, brand) discovery scope and enumerate its SSOT candidate
    questions — shared by the question list and the run submit so both see the
    same set (404 unknown dataset, 400 unknown brand)."""
    spec = _CAUSAL_DATASET_SPECS.get(dataset)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown causal dataset '{dataset}'. "
                f"Known datasets: {sorted(_CAUSAL_DATASET_SPECS)}"
            ),
        )
    brand = brand or None
    if brand is not None:
        available = await _list_dataset_brands(dataset)
        if available and brand not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown brand '{brand}' for dataset '{dataset}'. Known: {available}",
            )
    return brand, await _discover_candidate_questions(dataset, brand)


def _select_discovery_questions(
    candidates: List[_CandidateQuestion],
    selection: Optional[List[DiscoverQuestionSelection]],
) -> List[_CandidateQuestion]:
    """The run set: every SSOT candidate, or the user's SUBSET of them.

    A selection is matched on (treatment, outcome, brand) against the candidates
    so the column-allowlist gate stays authoritative — a pair outside the SSOT is
    a 400, never a run. Duplicates collapse; the SSOT row (its modeled adjustment
    set) is what runs, never the request's echo. Request order is kept (the
    pre-rank reorders it later, exactly as for a full run)."""
    if selection is None:
        return candidates
    if not selection:
        raise HTTPException(
            status_code=400,
            detail=(
                "Select at least one question to discover, or omit `questions` to "
                "run every candidate."
            ),
        )
    by_key = {(c.treatment, c.outcome, c.brand or None): c for c in candidates}
    chosen: List[_CandidateQuestion] = []
    seen: set = set()
    unknown: List[str] = []
    for sel in selection:
        key = (sel.treatment, sel.outcome, sel.brand or None)
        cand = by_key.get(key)
        if cand is None:
            unknown.append(
                f"{sel.treatment} -> {sel.outcome}" + (f" [{sel.brand}]" if sel.brand else "")
            )
        elif key not in seen:
            seen.add(key)
            chosen.append(cand)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown discovery question(s) for this dataset/brand: {unknown}. "
                "Selections must come from GET /causal/discover-effects/questions."
            ),
        )
    return chosen


@router.get(
    "/discover-effects/questions",
    response_model=DiscoverQuestionsResponse,
    summary="List the SSOT candidate questions a discover-effects run would validate",
    operation_id="list_discover_causal_questions",
)
async def list_discover_causal_questions(
    dataset: str = Query(_DEFAULT_CAUSAL_DATASET, description="Gold-standard dataset"),
    brand: Optional[str] = Query(
        None,
        description="Optional brand scope — same semantics as POST /causal/discover-effects.",
    ),
    user: Dict[str, Any] = Depends(require_viewer),
) -> DiscoverQuestionsResponse:
    """What ``POST /causal/discover-effects`` WOULD run for this scope, with the
    curated display labels, so the user can pick a subset up front (each question
    is minutes of agent time). Declared BEFORE ``GET /discover-effects/{job_id}``
    so the literal ``questions`` segment is never captured as a job id."""
    brand, questions = await _resolve_discovery_scope(dataset, brand)
    return DiscoverQuestionsResponse(
        dataset=dataset,
        brand=brand,
        questions=[
            DiscoverQuestion(
                treatment=q.treatment,
                outcome=q.outcome,
                brand=q.brand,
                treatment_label=_column_label(q.treatment),
                outcome_label=_column_label(q.outcome),
                adjustment_set=list(q.adjustment_set),
            )
            for q in questions
        ],
    )


@router.post(
    "/discover-effects",
    response_model=DiscoverEffectsResponse,
    summary="Discover & rank the agent's VALIDATED causal effects (async submit -> poll)",
    operation_id="discover_causal_effects",
)
async def discover_causal_effects(
    background_tasks: BackgroundTasks,
    dataset: str = Query(_DEFAULT_CAUSAL_DATASET, description="Gold-standard dataset"),
    brand: Optional[str] = Query(
        None,
        description=(
            "Optional brand to scope the cohort to (e.g. Kisqali). None = all "
            "brands. The candidate questions are unchanged; only the rows the "
            "agent estimates on are subset to this brand."
        ),
    ),
    body: Optional[DiscoverEffectsRequest] = Body(
        None,
        description=(
            "Optional. `questions` names a SUBSET of GET /causal/discover-effects/"
            "questions to run; omit it (or send `{}`) to run every candidate."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> DiscoverEffectsResponse:
    """Run the causal_impact agent across the dataset's candidate questions and
    rank the VALIDATED effects (discovered DAG + estimator + refutation gate) by
    confidence then impact. Heavy (minutes per effect) -> async: returns a pending
    job; poll ``GET /causal/discover-effects/{job_id}``. Fail-closed per question.

    ``brand`` (optional) scopes the cohort to one brand — a plain row subset, so
    each candidate is validated on that brand's patients only. ``questions``
    (optional body) restricts the run to a subset of the SSOT candidates; stop a
    run early with ``POST /causal/discover-effects/{job_id}/cancel``.
    """
    brand, candidates = await _resolve_discovery_scope(dataset, brand)
    questions = _select_discovery_questions(candidates, body.questions if body else None)
    job_id = str(uuid.uuid4())
    data_source = "synthetic" if deployment_includes_synthetic() else "database"
    initial = DiscoverEffectsResponse(
        job_id=job_id,
        status="pending",
        dataset=dataset,
        brand=brand,
        total=len(questions),
        completed=0,
        effects=[
            DiscoveredEffect(
                treatment=q.treatment,
                outcome=q.outcome,
                brand=q.brand,
                adjustment_set=list(q.adjustment_set),
                status="pending",
            )
            for q in questions
        ],
    )
    await _discover_effects_store.set(job_id, initial)
    # Alive from the moment the row exists: a poll landing before the task's
    # first beat must not declare a brand-new job dead.
    await _touch_discovery_heartbeat(job_id)
    background_tasks.add_task(
        _run_discover_effects_task, job_id, dataset, questions, data_source, brand
    )
    return initial


@router.get(
    "/discover-effects/{job_id}",
    response_model=DiscoverEffectsResponse,
    summary="Poll a discover-effects job",
    operation_id="get_discover_causal_effects",
)
async def get_discover_causal_effects(
    job_id: str,
    user: Dict[str, Any] = Depends(require_viewer),
) -> DiscoverEffectsResponse:
    job = await _discover_effects_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown discover-effects job '{job_id}'")
    return await _repair_if_orphaned(job)


@router.post(
    "/discover-effects/{job_id}/cancel",
    response_model=DiscoverEffectsResponse,
    summary="Stop a discover-effects job at its next question boundary",
    operation_id="cancel_discover_causal_effects",
)
async def cancel_discover_causal_effects(
    job_id: str,
    user: Dict[str, Any] = Depends(require_analyst),
) -> DiscoverEffectsResponse:
    """Cooperative cancel. The question in flight finishes (its estimators run
    synchronously and cannot be interrupted — up to a few minutes); the run then
    stops, keeps every finished row and marks the unrun ones ``cancelled``.
    Idempotent, and a no-op on a job that already ended — including one whose
    task died (reported `failed`, never "stopping after the current question":
    there is no current question)."""
    job = await _discover_effects_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown discover-effects job '{job_id}'")
    job = await _repair_if_orphaned(job)
    if job.status in _TERMINAL_DISCOVERY_STATUSES:
        return job
    # The marker is the signal the task honours; it is raised FIRST so the task
    # can never miss it. The row is then re-read and echoed with the flag so a
    # poll reflects the request before the next boundary. (The task's own next
    # publish may briefly overwrite the flag on the row — never the marker.)
    await _discover_effects_store.set_marker(job_id, _DISCOVERY_CANCEL_MARKER)
    current = await _discover_effects_store.get(job_id) or job
    if current.status in _TERMINAL_DISCOVERY_STATUSES:
        return current
    flagged = current.model_copy(update={"cancel_requested": True})
    await _discover_effects_store.set(job_id, flagged)
    return flagged


# =============================================================================
# PIPELINE ENDPOINTS
# =============================================================================


@router.post(
    "/pipeline/sequential",
    response_model=SequentialPipelineResponse,
    summary="Run sequential multi-library pipeline",
    operation_id="run_sequential_pipeline",
)
async def run_sequential_pipeline(
    request: SequentialPipelineRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(default=False, description="Run asynchronously"),
    demo_mode: bool = Query(
        default=False,
        description=(
            "If true, return pinned-zero placeholder results labeled with "
            "is_demo=true (for UI demonstrations only). Default is false: "
            "the endpoint runs real estimator selection or fails with 503."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> SequentialPipelineResponse:
    """
    Run sequential multi-library pipeline.

    Executes causal analysis stages in sequence:
    NetworkX → DoWhy → EconML → CausalML

    Each stage can pass results to the next for validation and refinement.

    Args:
        request: Pipeline configuration
        background_tasks: FastAPI background tasks
        async_mode: If True, runs asynchronously
        demo_mode: If True, return clearly-labeled placeholder values

    Returns:
        SequentialPipelineResponse with stage results and consensus
    """
    pipeline_id = str(uuid.uuid4())
    time.time()

    logger.info(
        f"Sequential pipeline requested: {pipeline_id}",
        extra={
            "pipeline_id": pipeline_id,
            "stages": len(request.stages),
            "libraries": [s.library.value for s in request.stages],
            "demo_mode": demo_mode,
        },
    )

    if async_mode:
        # Return pending response
        pending_response = SequentialPipelineResponse(
            pipeline_id=pipeline_id,
            status=AnalysisStatus.PENDING,
            stages_completed=0,
            stages_total=len(request.stages),
            stage_results=[],
            consensus_effect=None,
            consensus_ci_lower=None,
            consensus_ci_upper=None,
            confidence_level=request.confidence_level,
            library_agreement_score=None,
            effect_estimate_variance=None,
            total_latency_ms=0,
            created_at=datetime.now(timezone.utc),
            warnings=[],
        )
        _pipeline_cache[pipeline_id] = pending_response.model_dump()
        background_tasks.add_task(_run_sequential_pipeline_task, pipeline_id, request, demo_mode)
        return pending_response

    # Synchronous execution
    try:
        result = await _execute_sequential_pipeline(pipeline_id, request, demo_mode=demo_mode)
        _pipeline_cache[pipeline_id] = result.model_dump()
        return result
    except HTTPException:
        raise
    except HeavyComputeSaturated:
        # Reject fast under load — surfaced as 503 + Retry-After by the app
        # exception handler (OOM guard). HeavyComputeSaturated is NOT an
        # HTTPException, so this must precede the broad handler below to avoid
        # being swallowed into a 500.
        raise
    except Exception as e:
        logger.error(f"Sequential pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from e


async def _run_sequential_pipeline_task(
    pipeline_id: str,
    request: SequentialPipelineRequest,
    demo_mode: bool = False,
) -> None:
    """Background task for sequential pipeline."""
    try:
        result = await _execute_sequential_pipeline(pipeline_id, request, demo_mode=demo_mode)
        _pipeline_cache[pipeline_id] = result.model_dump()
    except Exception as e:
        # Log the raw error server-side (with traceback); the cached FAILED
        # record is later returned to clients, so it must carry only a generic
        # message, not raw exception text.
        logger.error(f"Background sequential pipeline failed: {e}", exc_info=True)
        _pipeline_cache[pipeline_id] = SequentialPipelineResponse(
            pipeline_id=pipeline_id,
            status=AnalysisStatus.FAILED,
            stages_completed=0,
            stages_total=len(request.stages),
            stage_results=[],
            consensus_effect=None,
            consensus_ci_lower=None,
            consensus_ci_upper=None,
            confidence_level=request.confidence_level,
            library_agreement_score=None,
            effect_estimate_variance=None,
            total_latency_ms=0,
            created_at=datetime.now(timezone.utc),
            warnings=["Pipeline failed due to an internal error."],
        ).model_dump()


# =============================================================================
# #354 C-8: real-pipeline wiring helpers
# =============================================================================


def _build_pipeline_input_sequential(
    request: SequentialPipelineRequest,
    *,
    libraries_enabled: Optional[List[str]] = None,
) -> PipelineInput:
    """Construct a PipelineInput for SequentialPipeline.execute() from request.

    The DataFrame is conveyed via the first-class ``PipelineInput.estimation_data``
    field (#458). The orchestrator copies it into ``state["estimation_data"]``
    and every executor resolves it via ``resolve_estimation_dataframe(state)``.
    ``request.filters`` (which carries inline-record passthrough and any
    DoWhy method override) is forwarded unchanged.
    """
    df = _resolve_pipeline_dataframe(request.filters)
    request_filters: Dict[str, Any] = dict(request.filters or {})

    return PipelineInput(
        query=(
            f"Sequential pipeline: treatment={request.treatment_var}, outcome={request.outcome_var}"
        ),
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        confounders=list(request.covariates),
        effect_modifiers=None,
        data_source=request.data_source,
        filters=request_filters,
        estimation_data=df,
        mode="sequential",
        libraries_enabled=libraries_enabled,
        cross_validate=None,
        # R6-F1 (#740): opt-in real-refutation flag → state["config"]["run_refutation"].
        run_refutation=request.run_refutation,
    )


def _build_pipeline_input_parallel(
    request: ParallelPipelineRequest,
) -> PipelineInput:
    """Construct a PipelineInput for ParallelPipeline.execute() from request."""
    df = _resolve_pipeline_dataframe(request.filters)
    request_filters: Dict[str, Any] = dict(request.filters or {})

    return PipelineInput(
        query=(
            f"Parallel pipeline: treatment={request.treatment_var}, outcome={request.outcome_var}"
        ),
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        confounders=list(request.covariates),
        effect_modifiers=None,
        data_source=request.data_source,
        filters=request_filters,
        estimation_data=df,
        mode="parallel",
        libraries_enabled=[lib.value for lib in request.libraries],
        cross_validate=None,
        # R6-F1 (#740): opt-in real-refutation flag → state["config"]["run_refutation"].
        run_refutation=request.run_refutation,
    )


class _SurfaceCSequentialPipeline(SequentialPipeline):
    """SequentialPipeline subclass for Surface C wiring.

    Provides one C-8-specific extension on top of the base orchestrator:

    **Per-library result capture** (``self.last_state``). The base
    ``execute()`` returns ``PipelineOutput`` which only carries the
    primary library's full payload. The C-8 response builder needs every
    executed library's payload (to populate per-library stage results /
    library_results without dropping data) — so we capture the final
    state in ``_create_output`` for the adapter to read.

    DataFrame injection into ``state["data_cache"]`` was a separate concern
    handled by an earlier ``dataframe=`` constructor kwarg + a
    ``_create_initial_state`` override. That mechanism is gone as of #458:
    the DataFrame now travels through ``PipelineInput.estimation_data`` and
    the orchestrator copies it into ``state["estimation_data"]`` itself,
    so this subclass no longer touches initial state.
    """

    def __init__(self, *, fail_fast: bool = False) -> None:
        super().__init__(fail_fast=fail_fast)
        self.last_state: Optional[PipelineState] = None

    def _create_output(self, state: PipelineState) -> PipelineOutput:
        # Capture the final state so the adapter can read per-library results
        # (state["<lib>_result"]) — PipelineOutput.primary_result only carries
        # the primary library's payload, which would drop non-primary library
        # data from the API response.
        self.last_state = state
        return super()._create_output(state)


class _SurfaceCParallelPipeline(ParallelPipeline):
    """ParallelPipeline subclass that mirrors ``_SurfaceCSequentialPipeline``.

    Captures ``self.last_state`` for per-library result extraction; DataFrame
    conveyance is via ``PipelineInput.estimation_data`` (#458), not a
    constructor kwarg.
    """

    def __init__(
        self,
        *,
        max_parallel: int = 4,
        fail_fast: bool = False,
    ) -> None:
        super().__init__(max_parallel=max_parallel, fail_fast=fail_fast)
        self.last_state: Optional[PipelineState] = None

    def _create_output(self, state: PipelineState) -> PipelineOutput:
        self.last_state = state
        return super()._create_output(state)


async def _run_real_sequential_pipeline(
    pipeline_id: str,
    request: SequentialPipelineRequest,
) -> SequentialPipelineResponse:
    """Invoke the wired SequentialPipeline.execute() and adapt to the API response.

    Fail-closed contract:
        - If no library produced a successful result (every executor returned
          ``success=False`` because no DataFrame was resolvable from state),
          raises ``HTTPException(503)`` — the honest signal that the data
          backend is absent for this request.
        - If at least one library executes successfully, returns a real
          ``SequentialPipelineResponse`` constructed from per-library state
          (no hardcoded values).
    """
    pipeline = _SurfaceCSequentialPipeline(
        fail_fast=request.stop_on_failure,
    )
    libraries_enabled = [stage.library.value for stage in request.stages]
    pipeline_input = _build_pipeline_input_sequential(request, libraries_enabled=libraries_enabled)
    output = await pipeline.execute(pipeline_input)
    return _sequential_output_to_response(pipeline_id, request, output, state=pipeline.last_state)


async def _run_real_parallel_pipeline(
    pipeline_id: str,
    request: ParallelPipelineRequest,
) -> ParallelPipelineResponse:
    """Invoke the wired ParallelPipeline.execute() and adapt to the API response."""
    pipeline = _SurfaceCParallelPipeline(
        max_parallel=len(request.libraries),
        fail_fast=False,
    )
    pipeline_input = _build_pipeline_input_parallel(request)
    output = await pipeline.execute(pipeline_input)
    return _parallel_output_to_response(pipeline_id, request, output, state=pipeline.last_state)


def _resolve_graph_quality(
    output: Optional[Mapping[str, Any]],
    state: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve the structural graph-quality dict from state (preferred) or output.

    The real pipeline carries graph_quality on ``state`` (orchestrator
    ``_extract_graph_quality``); ``PipelineOutput`` does NOT copy it. The existing
    M-fo2 surface test injects it via ``output``. Prefer ``state`` when present
    (the real path) and fall back to ``output`` (synthetic/test path). Returns an
    empty dict (not None) so callers can ``.get(...)`` safely.
    """
    for source in (state, output):
        if source is None:
            continue
        gq = source.get("graph_quality")
        if isinstance(gq, dict):
            return gq
    return {}


def _robustness_from_state(
    state: Optional[Mapping[str, Any]],
) -> tuple[bool, Optional[str]]:
    """Gate ``robustness_validation_performed`` on the REAL refutation gate band.

    Mirrors the agent path's ``is_estimate_valid`` (PROCEED is usable-as-robust;
    REVIEW/BLOCK are not) and its caveat semantics (nodes/refutation.py):

    - refutation_results falsy / empty / ``skipped`` / no gate → ``(False,
      _ROBUSTNESS_UNVALIDATED_WARNING)`` (today's default-path behaviour; an
      honest skip is NOT a validation).
    - ``gate_decision == "proceed"`` → ``(True, None)`` (validated, no caveat).
    - ``gate_decision == "review"`` → ``(False, _ROBUSTNESS_REVIEW_WARNING)``.
    - ``gate_decision in {"block", "error"}`` or an ``error`` key → ``(False,
      _ROBUSTNESS_BLOCK_WARNING)`` (fail-closed: an errored/blocked refutation
      must NEVER flip robustness True).

    NOTE: the M-fo2 non-DAG structural override is applied by the response
    builders AFTER this helper (a cyclic graph forces False regardless of band).
    """
    rr = (state or {}).get("refutation_results")
    if not isinstance(rr, dict) or not rr or rr.get("skipped") is True:
        return False, _ROBUSTNESS_UNVALIDATED_WARNING
    if rr.get("error") is not None:
        return False, _ROBUSTNESS_BLOCK_WARNING
    gate = rr.get("gate_decision")
    if gate == "proceed":
        return True, None
    if gate == "review":
        return False, _ROBUSTNESS_REVIEW_WARNING
    if gate == "block":
        return False, _ROBUSTNESS_BLOCK_WARNING
    # Unknown / missing gate on a populated dict → fail-closed unvalidated.
    return False, _ROBUSTNESS_UNVALIDATED_WARNING


def _classify_structural_identification(
    graph_quality: Mapping[str, Any],
) -> tuple[Optional[str], bool]:
    """M-fo2 (precise): return ``(structural_identification, identification_blocked)``.

    Reads the precise fields the orchestrator stamps onto ``graph_quality``:

    - ``is_dag`` None/missing → the structural check did not run → ``(None, False)``.
    - ``is_dag`` True → ``("acyclic", False)``.
    - ``is_dag`` False → use ``structural_identification`` when present; otherwise
      derive from ``cycle_affects_identification`` and FAIL CLOSED (a non-DAG that
      lacks the precise flag is treated as ``undefined_cyclic``).
    """
    is_dag = graph_quality.get("is_dag")
    if is_dag is None:
        return None, False
    if is_dag is True:
        return "acyclic", False
    # is_dag is False — read the precise label, fail-closed on absence.
    label = graph_quality.get("structural_identification")
    if label in ("undefined_cyclic", "cycle_irrelevant"):
        return label, (label == "undefined_cyclic")
    affects = graph_quality.get("cycle_affects_identification")
    if affects is None:
        affects = True  # conservative: a non-DAG without the precise flag
    return ("undefined_cyclic" if affects else "cycle_irrelevant"), bool(affects)


class _StructuralGateOutcome(NamedTuple):
    """Result of the M-fo2 structural-identifiability gate applied by the builders."""

    robustness_performed: bool
    robustness_warning: Optional[str]
    warnings: List[str]
    requires_review: bool
    structural_identification: Optional[str]
    withhold_consensus: bool


def _apply_structural_identification_gate(
    *,
    graph_quality: Mapping[str, Any],
    robustness_performed: bool,
    robustness_warning: Optional[str],
    warnings: List[str],
) -> _StructuralGateOutcome:
    """M-fo2 (precise): quarantine ONLY when a cycle actually breaks identification.

    - ``undefined_cyclic`` (a directed cycle on the (T,Y) ancestral subgraph):
      backdoor adjustment is undefined → FORCE robustness False (overrides any
      PROCEED, a downgrade not a 503, per F1 Owner-decision 2), set
      ``requires_review=True``, WITHHOLD the consensus effect, and append an
      un-ignorable caveat to BOTH the warnings list and robustness_warning.
    - ``cycle_irrelevant`` (a cycle OFF the ancestral subgraph): the estimand is
      still identifiable → no robustness override, consensus preserved, only an
      informational warning.
    - ``acyclic`` / not-run: unchanged.
    """
    structural_identification, blocked = _classify_structural_identification(graph_quality)
    new_warnings = list(warnings)

    if blocked:
        if _NON_DAG_STRUCTURAL_WARNING not in new_warnings:
            new_warnings.append(_NON_DAG_STRUCTURAL_WARNING)
        if robustness_warning and _NON_DAG_STRUCTURAL_WARNING not in robustness_warning:
            combined: Optional[str] = f"{robustness_warning} {_NON_DAG_STRUCTURAL_WARNING}"
        else:
            combined = _NON_DAG_STRUCTURAL_WARNING
        return _StructuralGateOutcome(
            robustness_performed=False,
            robustness_warning=combined,
            warnings=new_warnings,
            requires_review=True,
            structural_identification=structural_identification,
            withhold_consensus=True,
        )

    if structural_identification == "cycle_irrelevant" and _CYCLE_IRRELEVANT_WARNING not in (
        new_warnings
    ):
        new_warnings.append(_CYCLE_IRRELEVANT_WARNING)

    return _StructuralGateOutcome(
        robustness_performed=robustness_performed,
        robustness_warning=robustness_warning,
        warnings=new_warnings,
        requires_review=False,
        structural_identification=structural_identification,
        withhold_consensus=False,
    )


def _sequential_output_to_response(
    pipeline_id: str,
    request: SequentialPipelineRequest,
    output: PipelineOutput,
    *,
    state: Optional[PipelineState] = None,
) -> SequentialPipelineResponse:
    """Adapt PipelineOutput → SequentialPipelineResponse.

    Builds one PipelineStageResult per requested stage, honoring the request's
    stage order. Fails closed with 503 when no library produced a *successful*
    result — see ``_run_real_sequential_pipeline`` for the contract.

    Note on "successful library" derivation:
        The engine appends to ``state["libraries_executed"]`` on every call
        (orchestrator.py:227), regardless of ``success``. The engine's failed
        executors also populate ``state["errors"]`` with one entry per failure.
        So a library is "successful" only if it appears in ``libraries_used``
        AND is NOT named in ``errors``.

    Note on per-library payload extraction:
        ``PipelineOutput.primary_result`` only carries the primary library's
        full payload. For non-primary stages we read from
        ``state["<lib>_result"]["result"]`` (via ``_extract_library_payload``)
        so we never silently drop a successful non-primary library's data.
    """
    libraries_used = list(output.get("libraries_used") or [])
    errors = output.get("errors") or []
    failed_libraries = {
        str(err.get("library")) for err in errors if isinstance(err, dict) and err.get("library")
    }
    successful_libraries = [lib for lib in libraries_used if lib not in failed_libraries]

    requested_libraries = [stage.library.value for stage in request.stages]
    _enforce_data_required_fail_close(requested_libraries, successful_libraries)

    error_by_library: Dict[str, str] = {
        str(err.get("library")): str(err.get("error") or "")
        for err in errors
        if isinstance(err, dict) and err.get("library")
    }

    stage_results: List[PipelineStageResult] = []
    for idx, stage_config in enumerate(request.stages, 1):
        lib_value = stage_config.library.value
        stage_results.append(
            _build_stage_result_from_output(
                stage_number=idx,
                stage_config_library=lib_value,
                stage_config_estimator=stage_config.estimator,
                output=output,
                state=state,
                successful_libraries=successful_libraries,
                error_by_library=error_by_library,
            )
        )

    stages_completed = sum(1 for r in stage_results if r.status == AnalysisStatus.COMPLETED)

    # M-fo2: read structural graph-quality from the (optional) mapping, guarding
    # the non-dict case so mypy narrows the type and a malformed value yields None.
    graph_quality = _resolve_graph_quality(output, state)

    # R6-F1: gate robustness on the REAL refutation suite (PROCEED → validated;
    # REVIEW/BLOCK/error/skipped/empty → False + a band-naming caveat). Only
    # append the caveat to ``warnings`` when one is set (a validated PROCEED has
    # no caveat — drop the "unvalidated" line entirely).
    robustness_performed, robustness_warning = _robustness_from_state(state)
    warnings = list(output.get("warnings") or [])
    if robustness_warning:
        warnings.append(robustness_warning)

    # M-fo2 (precise): quarantine ONLY when a cycle breaks identification of the
    # (T,Y) estimand (cycle on the ancestral subgraph). undefined_cyclic FORCES
    # robustness False + requires_review + WITHHOLDS the consensus; an off-subgraph
    # cycle (cycle_irrelevant) leaves the estimate untouched.
    gate = _apply_structural_identification_gate(
        graph_quality=graph_quality,
        robustness_performed=robustness_performed,
        robustness_warning=robustness_warning,
        warnings=warnings,
    )
    consensus_effect = None if gate.withhold_consensus else output.get("consensus_effect")

    return SequentialPipelineResponse(
        pipeline_id=pipeline_id,
        status=_derive_response_status(stages_completed, len(request.stages)),
        stages_completed=stages_completed,
        stages_total=len(request.stages),
        stage_results=stage_results,
        consensus_effect=consensus_effect,
        consensus_ci_lower=None,  # Not produced by the engine output today
        consensus_ci_upper=None,
        # #27: report the level the consensus CI WOULD use (CI itself None today).
        confidence_level=request.confidence_level,
        # H8: a REAL library-agreement metric (mean pairwise concordance), NOT
        # consensus_confidence (the mean of per-library confidences, which the API
        # previously mislabeled as agreement).
        library_agreement_score=(state.get("library_agreement_score") if state else None),
        effect_estimate_variance=None,
        total_latency_ms=int(output.get("total_latency_ms") or 0),
        created_at=datetime.now(timezone.utc),
        warnings=gate.warnings,
        robustness_validation_performed=gate.robustness_performed,
        robustness_warning=gate.robustness_warning,
        graph_is_dag=graph_quality.get("is_dag"),
        structural_quality=graph_quality.get("structural_quality"),
        requires_review=gate.requires_review,
        structural_identification=gate.structural_identification,
    )


def _parallel_output_to_response(
    pipeline_id: str,
    request: ParallelPipelineRequest,
    output: PipelineOutput,
    *,
    state: Optional[PipelineState] = None,
) -> ParallelPipelineResponse:
    """Adapt PipelineOutput → ParallelPipelineResponse.

    Fails closed with 503 when no library produced a successful result.
    Same "successful library" derivation and per-library payload reading
    as the sequential adapter — see its docstring for the rationale.
    """
    libraries_used = list(output.get("libraries_used") or [])
    errors = output.get("errors") or []
    error_by_library: Dict[str, str] = {
        str(err.get("library")): str(err.get("error") or "")
        for err in errors
        if isinstance(err, dict) and err.get("library")
    }
    successful_libraries = [lib for lib in libraries_used if lib not in error_by_library]

    requested_libraries = [lib.value for lib in request.libraries]
    _enforce_data_required_fail_close(requested_libraries, successful_libraries)

    library_results: Dict[str, Dict[str, Any]] = {}
    succeeded: List[str] = []
    failed: List[str] = []

    for lib in request.libraries:
        lib_value = lib.value
        if lib_value in successful_libraries:
            succeeded.append(lib_value)
            library_results[lib_value] = _extract_library_payload(lib_value, output, state=state)
        elif lib_value in error_by_library:
            failed.append(lib_value)
            library_results[lib_value] = {"error": error_by_library[lib_value]}
        else:
            # Library was requested but neither executed nor errored —
            # validate_input rejected it before run.
            failed.append(lib_value)
            library_results[lib_value] = {"error": "library skipped during execution"}

    # M-fo2: read structural graph-quality from state (real path) or output
    # (synthetic/test path); empty dict when absent so .get(...) is safe.
    graph_quality = _resolve_graph_quality(output, state)

    # R6-F1: gate robustness on the REAL refutation suite (see sequential builder).
    robustness_performed, robustness_warning = _robustness_from_state(state)
    warnings = list(output.get("warnings") or [])
    if robustness_warning:
        warnings.append(robustness_warning)

    # M-fo2 (precise): see the sequential builder. undefined_cyclic forces
    # robustness False + requires_review + withholds the consensus; cycle_irrelevant
    # leaves the estimate untouched.
    gate = _apply_structural_identification_gate(
        graph_quality=graph_quality,
        robustness_performed=robustness_performed,
        robustness_warning=robustness_warning,
        warnings=warnings,
    )
    consensus_effect = None if gate.withhold_consensus else output.get("consensus_effect")

    return ParallelPipelineResponse(
        pipeline_id=pipeline_id,
        status=(
            AnalysisStatus.COMPLETED
            if len(succeeded) == len(request.libraries)
            else AnalysisStatus.FAILED
        ),
        libraries_succeeded=succeeded,
        libraries_failed=failed,
        library_results=library_results,
        consensus_effect=consensus_effect,
        consensus_ci_lower=None,
        consensus_ci_upper=None,
        # #27: report the confidence level the consensus CI WOULD use. The real
        # engine does not emit a consensus CI today (lower/upper stay None), but
        # echoing the requested level keeps the field consistent with the demo
        # path and lets the UI label any future interval truthfully.
        confidence_level=request.confidence_level,
        # H8: real mean-pairwise-concordance agreement, not consensus_confidence.
        library_agreement_score=(state.get("library_agreement_score") if state else None),
        consensus_method=request.consensus_method,
        total_latency_ms=int(output.get("total_latency_ms") or 0),
        created_at=datetime.now(timezone.utc),
        warnings=gate.warnings,
        robustness_validation_performed=gate.robustness_performed,
        robustness_warning=gate.robustness_warning,
        graph_is_dag=graph_quality.get("is_dag"),
        structural_quality=graph_quality.get("structural_quality"),
        requires_review=gate.requires_review,
        structural_identification=gate.structural_identification,
    )


def _derive_response_status(stages_completed: int, stages_total: int) -> AnalysisStatus:
    """Derive API AnalysisStatus from completed/total stage counts."""
    if stages_completed == stages_total:
        return AnalysisStatus.COMPLETED
    return AnalysisStatus.FAILED


def _enforce_data_required_fail_close(
    requested_libraries: List[str],
    successful_libraries: List[str],
) -> None:
    """Fail-close with 503 when no library produced an answer to the question asked.

    Two fail-close conditions, both honest signals of "pipeline did not answer":

    1. **No library succeeded** — every executor returned success=False
       (typically because no DataFrame was resolvable from state).
    2. **Only symbolic-input libraries succeeded** when an effect question
       was asked — i.e. the request named at least one of
       ``_DATA_REQUIRED_LIBRARIES`` (dowhy/econml/causalml — the libraries
       that produce causal effect estimates) AND none of them succeeded.
       NetworkX alone cannot answer "what is the causal effect?" — it
       answers "what is the graph structure?". Returning 200 with only
       NetworkX in this case would be a labeling problem (succeeded =
       True; answered effect question = False).

    All-symbolic requested sets intentionally bypass this fail-close —
    a graph-only question is a valid use case and NetworkX is the
    canonical answer. (Note: today the request schemas enforce
    ``min_length=2`` on ``stages`` / ``libraries`` (see
    ``api/schemas/causal.py``), so a literal NetworkX-only API request
    would be rejected by Pydantic validation before reaching this
    helper. The bypass still matters for any future schema relaxation
    and for the symbolic-only path inside this helper.)

    Raises:
        HTTPException(503): with ``_NO_RESOLVABLE_DATA_DETAIL`` body.
    """
    if not successful_libraries:
        raise HTTPException(status_code=503, detail=_NO_RESOLVABLE_DATA_DETAIL)

    requested_data_required = {
        lib for lib in requested_libraries if lib in _DATA_REQUIRED_LIBRARIES
    }
    if requested_data_required:
        successful_data_required = {
            lib for lib in successful_libraries if lib in _DATA_REQUIRED_LIBRARIES
        }
        if not successful_data_required:
            # User asked for at least one effect-estimating library, none
            # succeeded. NetworkX's symbolic success doesn't answer the
            # effect question — fail-close.
            raise HTTPException(status_code=503, detail=_NO_RESOLVABLE_DATA_DETAIL)


def _build_stage_result_from_output(
    *,
    stage_number: int,
    stage_config_library: str,
    stage_config_estimator: Optional[str],
    output: PipelineOutput,
    state: Optional[PipelineState],
    successful_libraries: List[str],
    error_by_library: Dict[str, str],
) -> PipelineStageResult:
    """Build a PipelineStageResult for one stage.

    Reads real values from per-library result captured in ``state`` (so
    non-primary library payloads are not dropped). Marks the stage FAILED
    with the engine's descriptive error when the library did not succeed.
    """
    if stage_config_library not in successful_libraries:
        return PipelineStageResult(
            stage_number=stage_number,
            library=stage_config_library,
            estimator=stage_config_estimator,
            status=AnalysisStatus.FAILED,
            effect_estimate=None,
            ci_lower=None,
            ci_upper=None,
            p_value=None,
            additional_results={},
            latency_ms=0,
            error=error_by_library.get(
                stage_config_library,
                "library skipped or failed during pipeline execution",
            ),
        )

    payload = _extract_library_payload(stage_config_library, output, state=state)
    effect = payload.get("effect_estimate")
    ci_lower = payload.get("ci_lower")
    ci_upper = payload.get("ci_upper")
    p_value = payload.get("p_value")
    stage_latency = _get_stage_latency_ms(state, stage_config_library, output)

    return PipelineStageResult(
        stage_number=stage_number,
        library=stage_config_library,
        estimator=stage_config_estimator,
        status=AnalysisStatus.COMPLETED,
        effect_estimate=effect if isinstance(effect, (int, float)) else None,
        ci_lower=ci_lower if isinstance(ci_lower, (int, float)) else None,
        ci_upper=ci_upper if isinstance(ci_upper, (int, float)) else None,
        p_value=p_value if isinstance(p_value, (int, float)) else None,
        additional_results={
            k: v
            for k, v in payload.items()
            if k not in {"effect_estimate", "ci_lower", "ci_upper", "p_value"}
        },
        latency_ms=stage_latency,
        error=None,
    )


def _get_stage_latency_ms(
    state: Optional[PipelineState], library: str, output: PipelineOutput
) -> int:
    """Per-stage latency from state.stage_latencies, falling back to total."""
    if state is not None:
        stage_latencies = cast(Dict[str, Any], state).get("stage_latencies") or {}
        if isinstance(stage_latencies, dict):
            v = stage_latencies.get(library)
            if isinstance(v, (int, float)):
                return int(v)
    # Fallback: total latency (still real, just less granular)
    return int(output.get("total_latency_ms") or 0)


def _extract_library_payload(
    library: str,
    output: PipelineOutput,
    *,
    state: Optional[PipelineState] = None,
) -> Dict[str, Any]:
    """Extract a per-library payload for the API response.

    Resolution order:
        1. If ``state`` is provided AND ``state["<lib>_result"]`` is a
           success-flagged ``LibraryExecutionResult``, read its ``result``
           dict (the canonical per-library payload from the executor).
        2. Else if ``library`` is the primary library, fall back to
           ``output["primary_result"]``.
        3. Else return ``{"library": library}`` (the engine surfaced no
           per-library payload — this is a labeling honest minimum).

    Reading state first matters in parallel mode: when EconML (primary)
    fails and DoWhy (secondary) succeeds, ``output.primary_result`` is
    EconML's empty/error payload — reading it for DoWhy would drop the
    real DoWhy data. The per-library state fields preserve every executor's
    result.
    """
    result_payload = _read_library_result_from_state(library, state)
    if result_payload is None:
        # Fall back to primary_result when state is unavailable (defensive
        # path; the C-8 route always captures state).
        primary_lib = _output_primary_library(state, output)
        if primary_lib == library:
            result_payload = dict(output.get("primary_result") or {})
        else:
            result_payload = {}

    payload: Dict[str, Any] = {"library": library}

    if library == "dowhy":
        effect = result_payload.get("causal_effect")
        if isinstance(effect, (int, float)):
            payload["effect_estimate"] = float(effect)
        # #2014: DoWhy's uncertainty is its ``standard_error`` (HC1 of its OLS fit),
        # not ``ate_ci_*`` — it used to be dropped here, leaving the stage CI None.
        interval = _dowhy_interval(result_payload)
        if interval is not None:
            payload["ci_lower"], payload["ci_upper"] = interval
            payload["p_value"] = _te_pvalue_from_z(
                float(result_payload["causal_effect"]), result_payload["standard_error"]
            )
            payload["standard_error"] = float(result_payload["standard_error"])
            payload["standard_error_method"] = result_payload.get("standard_error_method")
        method = result_payload.get("dowhy_method")
        if isinstance(method, str):
            payload["method"] = method
        estimand = result_payload.get("identified_estimand")
        if isinstance(estimand, str):
            payload["identified_estimand"] = estimand
    elif library == "econml":
        ate = result_payload.get("ate") or result_payload.get("overall_ate")
        if isinstance(ate, (int, float)):
            payload["effect_estimate"] = float(ate)
        ci_lower = result_payload.get("ate_ci_lower") or result_payload.get("ci_lower")
        ci_upper = result_payload.get("ate_ci_upper") or result_payload.get("ci_upper")
        if isinstance(ci_lower, (int, float)):
            payload["ci_lower"] = float(ci_lower)
        if isinstance(ci_upper, (int, float)):
            payload["ci_upper"] = float(ci_upper)
        method = result_payload.get("econml_method") or result_payload.get("estimator")
        if isinstance(method, str):
            payload["method"] = method
    elif library == "causalml":
        ate = result_payload.get("ate")
        if isinstance(ate, (int, float)):
            payload["effect_estimate"] = float(ate)
        auuc = result_payload.get("auuc")
        if isinstance(auuc, (int, float)):
            payload["auuc"] = float(auuc)
        qini = result_payload.get("qini")
        if isinstance(qini, (int, float)):
            payload["qini"] = float(qini)
    elif library == "networkx":
        n_nodes = result_payload.get("n_nodes")
        if isinstance(n_nodes, (int, float)):
            payload["n_nodes"] = int(n_nodes)
        n_edges = result_payload.get("n_edges")
        if isinstance(n_edges, (int, float)):
            payload["n_edges"] = int(n_edges)
        is_dag = result_payload.get("is_dag")
        if isinstance(is_dag, bool):
            payload["is_dag"] = is_dag

    return payload


def _read_library_result_from_state(
    library: str, state: Optional[PipelineState]
) -> Optional[Dict[str, Any]]:
    """Read ``state["<lib>_result"]["result"]`` when present AND success=True.

    Returns ``None`` when state is absent OR the library has no successful
    result. Per-library state keys: ``dowhy_result``, ``econml_result``,
    ``causalml_result``, ``networkx_result``.
    """
    if state is None:
        return None
    state_dict = cast(Dict[str, Any], state)
    key = f"{library}_result"
    lib_result = state_dict.get(key)
    if not isinstance(lib_result, dict):
        return None
    if not lib_result.get("success"):
        return None
    result = lib_result.get("result")
    if isinstance(result, dict):
        return dict(result)
    return None


def _output_primary_library(
    state: Optional[PipelineState], output: PipelineOutput
) -> Optional[str]:
    """Best-effort primary library lookup from state or output."""
    if state is not None:
        config = cast(Dict[str, Any], state).get("config") or {}
        if isinstance(config, dict):
            primary = config.get("primary_library")
            if isinstance(primary, str):
                return primary
    libraries_used = output.get("libraries_used") or []
    if libraries_used:
        return libraries_used[0]
    return None


def _demo_stage_placeholder(
    *,
    stage_number: int,
    library: str,
    estimator: Optional[str],
    latency_ms: int,
) -> PipelineStageResult:
    """Pinned-zero placeholder used for explicit demo_mode=True flows.

    Never returns RNG values; the caller (with demo_mode=True) is responsible
    for labeling the surrounding envelope with ``is_demo=true``.
    """
    return PipelineStageResult(
        stage_number=stage_number,
        library=library,
        estimator=estimator,
        status=AnalysisStatus.COMPLETED,
        effect_estimate=0.0,
        ci_lower=0.0,
        ci_upper=0.0,
        p_value=1.0,
        additional_results={
            "n_samples": 0,
            "method": estimator or "default",
            "is_demo": True,
        },
        latency_ms=latency_ms,
        error=None,
    )


async def _execute_sequential_pipeline(
    pipeline_id: str,
    request: SequentialPipelineRequest,
    demo_mode: bool = False,
) -> SequentialPipelineResponse:
    """Execute sequential pipeline stages.

    Default path (``demo_mode=False``, #354 C-8): delegates to the real
    ``SequentialPipeline.execute()`` wired in C-1..C-6 (all 4 executors —
    DoWhy/EconML/CausalML/NetworkX — and 4-library aggregation). The
    caller MUST supply a DataFrame via ``request.filters['estimation_data_records']``
    (list of dicts). If no DataFrame is resolvable and every wired executor
    therefore returns ``success=False``, the response is an honest
    ``HTTPException(503)`` — there is still no production data backend that
    can resolve arbitrary ``data_source`` identifiers to real columns by name.

    With ``demo_mode=True``: returns pinned-zero placeholder stage results
    clearly labeled with ``is_demo=true``. This UI-demo branch is unchanged
    from the F-005 contract (see v4 §2.3); C-8 preserves it verbatim.

    Pre-C-8 behavior (now superseded): the default path raised 503
    unconditionally. The 503 stays as the honest no-data signal, but it
    now reflects the wired pipeline's actual outcome rather than a
    hardcoded short-circuit. See F-005 audit iter-1 HIGH-1 for the prior
    synthetic-data-fabrication trap that #354 was opened to fix.
    """
    if not demo_mode:
        # #354 C-8: invoke the wired pipeline. The helper raises
        # HTTPException(503) with _NO_RESOLVABLE_DATA_DETAIL when no library
        # produced a result (honest fail-close), or returns a real response
        # built from the engine's PipelineOutput when at least one library
        # succeeded. NO silent fallback to synthetic data, NO hardcoded values.
        #
        # The real 4-library pipeline (DoWhy/EconML/CausalML/NetworkX) is the
        # genuinely heavy in-process compute. Bound it to ONE per-worker
        # heavy-compute slot (OOM guard, P1b) so concurrent real pipelines
        # cannot stack and OOM-kill the cgroup. A saturated worker raises
        # HeavyComputeSaturated on enter (mapped to 503 + Retry-After) — nothing
        # is queued. The demo path below does NO heavy work and is intentionally
        # left unbounded. Both callers (sync + background task) route through
        # here, so bounding once covers both.
        async with heavy_compute_slot():
            return await _run_real_sequential_pipeline(pipeline_id, request)

    start_time = time.time()
    stage_results: List[PipelineStageResult] = []
    effect_estimates: List[float] = []
    warnings: List[str] = [
        "demo_mode=true: results are pinned-zero placeholders with is_demo=true; "
        "do NOT use for decisions.",
    ]

    for i, stage_config in enumerate(request.stages, 1):
        stage_start = time.time()
        stage_result = _demo_stage_placeholder(
            stage_number=i,
            library=stage_config.library.value,
            estimator=stage_config.estimator,
            latency_ms=int((time.time() - stage_start) * 1000),
        )
        effect_estimates.append(0.0)
        stage_results.append(stage_result)

    # Demo consensus is zero by construction (all stages return 0.0).
    consensus_effect = 0.0
    consensus_ci_lower = 0.0
    consensus_ci_upper = 0.0
    agreement_score = 1.0
    variance = 0.0

    total_latency_ms = int((time.time() - start_time) * 1000)
    stages_completed = len([r for r in stage_results if r.status == AnalysisStatus.COMPLETED])

    return SequentialPipelineResponse(
        pipeline_id=pipeline_id,
        status=AnalysisStatus.COMPLETED
        if stages_completed == len(request.stages)
        else AnalysisStatus.FAILED,
        stages_completed=stages_completed,
        stages_total=len(request.stages),
        stage_results=stage_results,
        consensus_effect=consensus_effect,
        consensus_ci_lower=consensus_ci_lower,
        consensus_ci_upper=consensus_ci_upper,
        confidence_level=request.confidence_level,
        library_agreement_score=agreement_score,
        effect_estimate_variance=variance,
        total_latency_ms=total_latency_ms,
        created_at=datetime.now(timezone.utc),
        warnings=warnings,
    )


@router.post(
    "/pipeline/parallel",
    response_model=ParallelPipelineResponse,
    summary="Run parallel multi-library analysis",
    operation_id="run_parallel_pipeline",
)
async def run_parallel_pipeline(
    request: ParallelPipelineRequest,
    demo_mode: bool = Query(
        default=False,
        description=(
            "If true, return pinned-zero placeholder results labeled with "
            "is_demo=true (for UI demonstrations only). Default is false: "
            "the endpoint runs real estimator selection or fails with 503."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> ParallelPipelineResponse:
    """
    Run parallel multi-library analysis.

    Executes multiple causal libraries simultaneously and computes
    consensus results weighted by confidence.

    Args:
        request: Parallel pipeline configuration
        demo_mode: If True, return clearly-labeled placeholder values

    Returns:
        ParallelPipelineResponse with library results and consensus
    """
    pipeline_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Parallel pipeline requested: {pipeline_id}",
        extra={
            "pipeline_id": pipeline_id,
            "libraries": [lib.value for lib in request.libraries],
            "demo_mode": demo_mode,
        },
    )

    if not demo_mode:
        # #354 C-8: invoke the wired ParallelPipeline. Helper raises
        # HTTPException(503) when no library produced a result (honest
        # fail-close), or returns a real response from the engine's
        # PipelineOutput. NO silent fallback, NO hardcoded values.
        try:
            # The real multi-library fan-out is the genuinely heavy in-process
            # compute. Bound it to ONE per-worker heavy-compute slot (OOM guard,
            # P1b) so concurrent real pipelines cannot stack and OOM-kill the
            # cgroup. A saturated worker raises HeavyComputeSaturated on enter
            # (mapped to 503 + Retry-After) — nothing is queued. The demo path
            # below does NO heavy work and is intentionally left unbounded.
            async with heavy_compute_slot():
                return await asyncio.wait_for(
                    _run_real_parallel_pipeline(pipeline_id, request),
                    timeout=request.timeout_seconds,
                )
        except HTTPException:
            raise
        except asyncio.TimeoutError as e:
            raise HTTPException(
                status_code=408,
                detail=f"Pipeline timed out after {request.timeout_seconds}s",
            ) from e
        except HeavyComputeSaturated:
            # Reject fast under load — surfaced as 503 + Retry-After by the app
            # exception handler. HeavyComputeSaturated is NOT an HTTPException,
            # so this must precede the broad last-resort handler below to avoid
            # being swallowed into a 500.
            raise
        except Exception as e:  # noqa: BLE001 - last-resort 500
            logger.error(f"Parallel pipeline failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from e

    try:
        # Run all libraries in parallel
        tasks = [
            _run_library_analysis(lib, request, demo_mode=demo_mode) for lib in request.libraries
        ]

        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=request.timeout_seconds,
        )

        # Process results
        library_results: Dict[str, Dict[str, Any]] = {}
        succeeded: List[str] = []
        failed: List[str] = []
        effect_estimates: List[float] = []

        for lib, result in zip(request.libraries, results, strict=False):
            if isinstance(result, HTTPException):
                # Real-path estimator unavailable for this library — surface
                # the upstream 503 to the client rather than fabricate.
                raise result
            if isinstance(result, Exception):
                library_results[lib.value] = {"error": str(result)}
                failed.append(lib.value)
            else:
                result_dict = cast(Dict[str, Any], result)
                library_results[lib.value] = result_dict
                succeeded.append(lib.value)
                if result_dict.get("effect_estimate") is not None:
                    effect_estimates.append(result_dict["effect_estimate"])

        # Compute consensus
        consensus_effect = None
        consensus_ci_lower = None
        consensus_ci_upper = None
        agreement_score = None

        if effect_estimates:
            import statistics

            # #27: derive the CI z-score from the requested confidence level
            # (default 0.95 => z~1.96, preserving the legacy half-width) instead
            # of a hardcoded magic number, and echo the level in the response so
            # the UI can label the interval truthfully. NB: this is the demo
            # path -- every library returns effect_estimate=0.0, so std==0.0 and
            # the interval is [consensus_effect, consensus_effect] at ANY z.
            z = z_score_for_confidence(request.confidence_level)
            consensus_effect = statistics.mean(effect_estimates)
            if len(effect_estimates) > 1:
                std = statistics.stdev(effect_estimates)
                consensus_ci_lower = consensus_effect - z * std
                consensus_ci_upper = consensus_effect + z * std
                cv = std / abs(consensus_effect) if consensus_effect != 0 else 1
                agreement_score = max(0, 1 - cv)
            else:
                consensus_ci_lower = consensus_effect
                consensus_ci_upper = consensus_effect
                agreement_score = 1.0

        total_latency_ms = int((time.time() - start_time) * 1000)

        warnings: List[str] = []
        if demo_mode:
            warnings.append(
                "demo_mode=true: results are pinned-zero placeholders with is_demo=true; "
                "do NOT use for decisions."
            )

        return ParallelPipelineResponse(
            pipeline_id=pipeline_id,
            status=AnalysisStatus.COMPLETED if succeeded else AnalysisStatus.FAILED,
            libraries_succeeded=succeeded,
            libraries_failed=failed,
            library_results=library_results,
            consensus_effect=consensus_effect,
            consensus_ci_lower=consensus_ci_lower,
            consensus_ci_upper=consensus_ci_upper,
            confidence_level=request.confidence_level,
            library_agreement_score=agreement_score,
            consensus_method=request.consensus_method,
            total_latency_ms=total_latency_ms,
            created_at=datetime.now(timezone.utc),
            warnings=warnings,
        )

    except HTTPException:
        raise
    except asyncio.TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail=f"Pipeline timed out after {request.timeout_seconds}s",
        ) from e
    except Exception as e:
        logger.error(f"Parallel pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from e


async def _run_library_analysis(
    library: CausalLibrary,
    request: ParallelPipelineRequest,
    demo_mode: bool = False,
) -> Dict[str, Any]:
    """Run analysis for a single library.

    Default path fails closed with HTTPException(503): there is no real data
    backend wired into the parallel pipeline. With ``demo_mode=True`` returns
    a pinned-zero placeholder labeled ``is_demo=true``. Never returns RNG
    values (F-005 fix).

    See F-005 audit iter-1 HIGH-1: synthetic-data + real-estimator in the
    default path is a labeling fabrication, not a functional fix.
    """
    if not demo_mode:
        raise HTTPException(status_code=503, detail=_NO_REAL_DATA_BACKEND_DETAIL)

    return {
        "library": library.value,
        "estimator": request.estimators.get(library.value) if request.estimators else None,
        "effect_estimate": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
        "p_value": 1.0,
        "n_samples": 0,
        "is_demo": True,
    }


@router.get(
    "/pipeline/{pipeline_id}", summary="Get pipeline status", operation_id="get_pipeline_status"
)
async def get_pipeline_status(
    pipeline_id: str,
) -> Dict[str, Any]:
    """
    Get status of a pipeline execution.

    Args:
        pipeline_id: Unique pipeline identifier

    Returns:
        Pipeline status and results
    """
    if pipeline_id not in _pipeline_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline {pipeline_id} not found",
        )

    return _pipeline_cache[pipeline_id]


# =============================================================================
# CROSS-VALIDATION ENDPOINT
# =============================================================================


@router.post(
    "/validate",
    response_model=CrossValidationResponse,
    summary="Run cross-library validation",
    operation_id="run_cross_validation",
)
async def run_cross_validation(
    request: CrossValidationRequest,
    demo_mode: bool = Query(
        default=False,
        description=(
            "If true, return pinned-zero placeholder results labeled with "
            "is_demo=true (for UI demonstrations only). Default is false: "
            "the endpoint runs real estimator selection or fails with 503."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> CrossValidationResponse:
    """
    Run cross-library validation (DoWhy ↔ CausalML).

    Compares effect estimates between libraries to validate results.

    Args:
        request: Cross-validation configuration
        demo_mode: If True, return clearly-labeled placeholder values

    Returns:
        CrossValidationResponse with agreement metrics
    """
    validation_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Cross-validation requested: {validation_id}",
        extra={
            "validation_id": validation_id,
            "primary_library": request.primary_library.value,
            "validation_library": request.validation_library.value,
            "demo_mode": demo_mode,
        },
    )

    if not demo_mode:
        # Default path has no real data backend; fail-closed (F-005 iter-1 HIGH-1).
        raise HTTPException(status_code=503, detail=_NO_REAL_DATA_BACKEND_DETAIL)

    # demo_mode=True: pinned-zero placeholder. Agreement is trivially perfect
    # because both libraries return the same zero, which we label explicitly.
    primary_effect = 0.0
    validation_effect = 0.0
    primary_ci = (0.0, 0.0)
    validation_ci = (0.0, 0.0)
    effect_difference = 0.0
    relative_difference = 0.0
    ci_overlap_ratio = 1.0
    agreement_score = 1.0
    validation_passed = agreement_score >= request.agreement_threshold

    latency_ms = int((time.time() - start_time) * 1000)

    # Surface the is_demo=true label as the FIRST recommendation so consumers
    # cannot miss it (CrossValidationResponse schema has no is_demo field, so
    # we encode the label in recommendations per F-005 iter-1 HIGH-3).
    recommendations: List[str] = [
        "is_demo=true: results are pinned-zero placeholders; do NOT use for decisions.",
    ]

    response = CrossValidationResponse(
        validation_id=validation_id,
        primary_library=request.primary_library.value,
        validation_library=request.validation_library.value,
        primary_effect=primary_effect,
        primary_ci=list(primary_ci),
        validation_effect=validation_effect,
        validation_ci=list(validation_ci),
        effect_difference=effect_difference,
        relative_difference=relative_difference,
        ci_overlap_ratio=ci_overlap_ratio,
        agreement_score=agreement_score,
        validation_passed=validation_passed,
        agreement_threshold=request.agreement_threshold,
        latency_ms=latency_ms,
        created_at=datetime.now(timezone.utc),
        recommendations=recommendations,
    )

    _validation_cache[validation_id] = response
    return response


# =============================================================================
# ESTIMATOR INFO ENDPOINT
# =============================================================================


# Single source of truth for the supported causal estimators. Both
# ``list_estimators`` (the /estimators endpoint) and ``causal_health_check``
# (``estimators_loaded``) read from this so the health count can never drift
# from the registry (previously ``estimators_loaded`` was a hardcoded ``12``).
_ESTIMATOR_REGISTRY: List[EstimatorInfo] = [
    # EconML
    EstimatorInfo(
        name="causal_forest",
        library=CausalLibrary.ECONML,
        estimator_type="CATE",
        description="Causal Forest for heterogeneous treatment effects",
        best_for=["Effect heterogeneity", "Feature importance"],
        parameters=["n_estimators", "min_samples_leaf", "max_depth"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="linear_dml",
        library=CausalLibrary.ECONML,
        estimator_type="CATE",
        description="Double Machine Learning with linear final stage",
        best_for=["High-dimensional confounders", "Linear effects"],
        parameters=["model_y", "model_t", "cv"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="ortho_forest",
        library=CausalLibrary.ECONML,
        estimator_type="CATE",
        description="Orthogonal Random Forest for CATE",
        best_for=["Non-linear effects", "SHAP integration"],
        parameters=["n_trees", "subsample_ratio", "max_depth"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="dr_learner",
        library=CausalLibrary.ECONML,
        estimator_type="CATE",
        description="Doubly Robust Learner",
        best_for=["Robustness to misspecification"],
        parameters=["model_propensity", "model_regression"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="x_learner",
        library=CausalLibrary.ECONML,
        estimator_type="Meta-Learner",
        description="X-Learner for heterogeneous effects",
        best_for=["Imbalanced treatment groups"],
        parameters=["models", "propensity_model"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="t_learner",
        library=CausalLibrary.ECONML,
        estimator_type="Meta-Learner",
        description="Two-Model approach",
        best_for=["Simple interpretation"],
        parameters=["models"],
        supports_confidence_intervals=False,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="s_learner",
        library=CausalLibrary.ECONML,
        estimator_type="Meta-Learner",
        description="Single-Model approach",
        best_for=["Limited data"],
        parameters=["overall_model"],
        supports_confidence_intervals=False,
        supports_heterogeneous_effects=True,
    ),
    # CausalML
    EstimatorInfo(
        name="uplift_random_forest",
        library=CausalLibrary.CAUSALML,
        estimator_type="Uplift",
        description="Uplift Random Forest for targeting",
        best_for=["Marketing optimization", "Customer targeting"],
        parameters=["n_estimators", "max_depth", "min_samples_treatment"],
        supports_confidence_intervals=False,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="uplift_gradient_boosting",
        library=CausalLibrary.CAUSALML,
        estimator_type="Uplift",
        description="Uplift Gradient Boosting",
        best_for=["High accuracy targeting"],
        parameters=["n_estimators", "learning_rate", "max_depth"],
        supports_confidence_intervals=False,
        supports_heterogeneous_effects=True,
    ),
    # DoWhy
    EstimatorInfo(
        name="propensity_score_matching",
        library=CausalLibrary.DOWHY,
        estimator_type="Identification",
        description="Propensity Score Matching",
        best_for=["Observational studies", "Selection bias"],
        parameters=["caliper", "n_neighbors"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=False,
    ),
    EstimatorInfo(
        name="inverse_propensity_weighting",
        library=CausalLibrary.DOWHY,
        estimator_type="Identification",
        description="Inverse Propensity Score Weighting",
        best_for=["Survey adjustments", "Treatment weighting"],
        parameters=["propensity_model", "stabilized"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=False,
    ),
    EstimatorInfo(
        name="instrumental_variable",
        library=CausalLibrary.DOWHY,
        estimator_type="Identification",
        description="Instrumental Variable (2SLS/LIML)",
        best_for=["Endogeneity", "Unmeasured confounders"],
        parameters=["instruments", "method"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=False,
    ),
]


@router.get(
    "/estimators",
    response_model=EstimatorListResponse,
    summary="List available estimators",
    operation_id="list_estimators",
)
async def list_estimators(
    library: Optional[CausalLibrary] = Query(None, description="Filter by library"),
) -> EstimatorListResponse:
    """
    List available causal estimators.

    Args:
        library: Optional filter by library

    Returns:
        EstimatorListResponse with estimator information
    """
    estimators = list(_ESTIMATOR_REGISTRY)

    # Filter by library if specified
    if library:
        estimators = [e for e in estimators if e.library == library]

    # Group by library
    by_library: Dict[str, List[str]] = {}
    for est in estimators:
        lib_name = est.library.value
        if lib_name not in by_library:
            by_library[lib_name] = []
        by_library[lib_name].append(est.name)

    return EstimatorListResponse(
        estimators=estimators,
        total=len(estimators),
        by_library=by_library,
    )


# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================


@router.get(
    "/health",
    response_model=CausalHealthResponse,
    summary="Causal engine health check",
    operation_id="causal_health_check",
)
async def causal_health_check() -> CausalHealthResponse:
    """
    Health check for causal inference engine.

    Returns:
        CausalHealthResponse with component status
    """
    libraries_available = {
        "dowhy": False,
        "econml": False,
        "causalml": False,
        "networkx": False,
    }

    # Check library availability
    try:
        import dowhy  # noqa: F401

        libraries_available["dowhy"] = True
    except ImportError:
        pass

    try:
        import econml  # noqa: F401

        libraries_available["econml"] = True
    except ImportError:
        pass

    try:
        import causalml  # noqa: F401

        libraries_available["causalml"] = True
    except ImportError:
        pass

    try:
        import networkx  # noqa: F401

        libraries_available["networkx"] = True
    except ImportError:
        pass

    # Check engine components
    hierarchical_ready = False
    pipeline_ready = False
    try:
        from src.causal_engine.hierarchical import HierarchicalAnalyzer  # noqa: F401

        hierarchical_ready = True
    except ImportError:
        pass

    try:
        from src.causal_engine.pipeline import PipelineOrchestrator  # noqa: F401

        pipeline_ready = True
    except ImportError:
        pass

    # Determine overall status
    all_libs = all(libraries_available.values())
    status = (
        "healthy" if all_libs else "degraded" if any(libraries_available.values()) else "unhealthy"
    )

    # #931: surface REAL recent causal-analysis activity from episodic_memories
    # (was a hardcoded 0/None stub from the original phase-B scaffold). The
    # count is the number of completed causal analyses in the last 24h; the
    # most-recent event's timestamp is ``last_analysis``. A read failure
    # degrades to an honest 0/None (never a fabricated value) so a transient
    # episodic-store issue can't take the whole health check down.
    analysis_count_24h, last_analysis = await _recent_causal_activity()

    return CausalHealthResponse(
        status=status,
        libraries_available=libraries_available,
        estimators_loaded=len(_ESTIMATOR_REGISTRY),  # real count from the registry
        pipeline_orchestrator_ready=pipeline_ready,
        hierarchical_analyzer_ready=hierarchical_ready,
        last_analysis=last_analysis,
        analysis_count_24h=analysis_count_24h,
        average_latency_ms=None,
        error=None if status == "healthy" else "Some libraries unavailable",
    )


async def _recent_causal_activity() -> tuple[int, Optional[datetime]]:
    """Return ``(count_last_24h, last_analysis_timestamp)`` for completed causal
    analyses, cached for a short window (review M1).

    The cache exists only to keep the public, frequently-polled health endpoint
    from amplifying into repeated DB reads; the cached value is the REAL reading
    (or the honest fallback), never a fabricated number.
    """
    now = time.monotonic()
    if now < _activity_cache["expires_at"]:
        return cast("tuple[int, Optional[datetime]]", _activity_cache["value"])

    value = await _read_causal_activity()
    _activity_cache["value"] = value
    _activity_cache["expires_at"] = now + _ACTIVITY_CACHE_TTL_SECONDS
    return value


async def _read_causal_activity() -> tuple[int, Optional[datetime]]:
    """Read ``(count_last_24h, last_analysis_timestamp)`` for completed causal
    analyses from episodic_memories.

    Both values are REAL (traced to ``causal_analysis_completed`` episodic rows)
    or an honest fallback (``0`` / ``None``) — never fabricated. On any read
    error we log and fall back so the health check stays available.
    """
    try:
        # ``days_back=1`` is the 24h window; provenance filter defaults to
        # excluding synthetic rows so the KPI reflects real activity.
        count = await count_memories_by_type(
            event_type=CAUSAL_COMPLETED_EVENT_TYPE,
            days_back=1,
        )
    except Exception:  # pragma: no cover - defensive
        logger.warning("causal health: 24h analysis count read failed", exc_info=True)
        count = 0

    last_analysis: Optional[datetime] = None
    try:
        recent = await get_recent_memories(
            limit=1,
            event_types=[CAUSAL_COMPLETED_EVENT_TYPE],
        )
        if recent:
            last_analysis = _parse_occurred_at(recent[0].get("occurred_at"))
    except Exception:  # pragma: no cover - defensive
        logger.warning("causal health: last-analysis read failed", exc_info=True)
        last_analysis = None

    return count, last_analysis


@router.get(
    "/history",
    response_model=CausalAnalysisHistoryResponse,
    summary="Recent completed causal analyses",
    operation_id="get_causal_analysis_history",
)
async def get_causal_analysis_history(
    limit: int = Query(20, ge=1, le=100, description="Maximum history items to return"),
    user: Dict[str, Any] = Depends(require_viewer),
) -> CausalAnalysisHistoryResponse:
    """Return recent completed causal analyses for the Analysis History tab.

    #931: feeds the previously-unwired History tab from REAL
    ``causal_analysis_completed`` episodic_memories rows (newest first). ATE,
    confidence and model are read from each row's ``raw_content`` when present;
    when a field is missing it stays ``None`` (never fabricated). An empty store
    yields an honest empty history rather than a synthesized series.
    """
    try:
        rows = await get_recent_memories(
            limit=limit,
            event_types=[CAUSAL_COMPLETED_EVENT_TYPE],
        )
    except Exception as exc:
        logger.error("Failed to read causal analysis history: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from exc

    items: List[CausalAnalysisHistoryItem] = []
    for row in rows:
        memory_id = row.get("memory_id")
        if not memory_id:
            # memory_id is the PK; a row without one can't be keyed honestly on
            # the client (empty keys collide). Skip rather than emit a blank id.
            logger.warning("causal history: skipping row with missing memory_id")
            continue
        occurred_at = _parse_occurred_at(row.get("occurred_at"))
        if occurred_at is None:
            # A row without a parseable timestamp can't be placed on the history
            # timeline honestly; skip it rather than invent a time.
            continue
        raw_content = row.get("raw_content")
        if not isinstance(raw_content, dict):
            raw_content = {}
        items.append(
            CausalAnalysisHistoryItem(
                memory_id=str(memory_id),
                event_type=str(row.get("event_type", CAUSAL_COMPLETED_EVENT_TYPE)),
                description=row.get("description"),
                occurred_at=occurred_at,
                agent_name=row.get("agent_name"),
                ate_estimate=_as_float(raw_content.get("ate_estimate")),
                confidence=_as_float(raw_content.get("confidence")),
                model_used=raw_content.get("model_used"),
            )
        )

    return CausalAnalysisHistoryResponse(items=items, total=len(items))


# =============================================================================
# CAUSAL VALUE CHAINS (dashboard "Primary Causal Value Chains" — REAL, dynamic)
# =============================================================================

# 'All'/'portfolio' selections from the Home dropdowns mean "no scope filter".
_ALL_BRAND_SENTINELS = {"all", "all brands", "portfolio", "all (combined portfolio)"}
_ALL_REGION_SENTINELS = {"all", "all us", "all regions", "all us regions"}


def _chain_node_sequence(row: Mapping[str, Any]) -> List[str]:
    """Ordered node names for a ``causal_paths`` row.

    Prefer the stored ``causal_chain.nodes`` ordering; fall back to
    ``start_node`` + ``intermediate_nodes`` + ``end_node``.
    """
    chain = row.get("causal_chain")
    if isinstance(chain, dict):
        nodes = chain.get("nodes")
        if isinstance(nodes, list) and len(nodes) >= 2 and all(isinstance(n, str) for n in nodes):
            return list(nodes)
    seq: List[str] = []
    start = row.get("start_node")
    if isinstance(start, str) and start:
        seq.append(start)
    inter = row.get("intermediate_nodes")
    if isinstance(inter, list):
        seq.extend(n for n in inter if isinstance(n, str) and n)
    end = row.get("end_node")
    if isinstance(end, str) and end:
        seq.append(end)
    return seq


def _chain_score(row: Mapping[str, Any]) -> float:
    """Rank key: |effect| x confidence (None-safe). Surfaces the strongest,
    best-supported chains first — never a fabricated magnitude."""
    eff = _as_optional_float(row.get("causal_effect_size"))
    conf = _as_optional_float(row.get("confidence_level"))
    return abs(eff or 0.0) * (conf or 0.0)


def _causal_path_to_graphpath(row: Mapping[str, Any]) -> GraphPath:
    """Map a ``causal_paths`` row to the ``GraphPath`` shape the dashboard renders.

    The chain-level effect (``causal_effect_size``) is placed on the TERMINAL
    edge as ``ate_estimate`` and the method on every edge — the dashboard reads
    the ate from the terminal edge and the method from the first, so the card
    surfaces the REAL effect/method. ``effect_size`` is intentionally NOT set as
    a number (it is a categorical label in this platform).
    """
    names = _chain_node_sequence(row)
    nodes = [
        GraphNode(
            id=f"var:{n}",
            type=EntityType.AGENT,
            name=n,
            properties={"original_type": "Variable"},
            created_at=None,
            updated_at=None,
        )
        for n in names
    ]
    conf_f = _as_optional_float(row.get("confidence_level"))
    eff_f = _as_optional_float(row.get("causal_effect_size"))
    method = row.get("method_used")

    rels: List[GraphRelationship] = []
    n_edges = len(nodes) - 1
    for i in range(n_edges):
        props: Dict[str, Any] = {}
        if method:
            props["method"] = method
        if i == n_edges - 1:  # terminal edge: chain-level effect + lifecycle/temporal
            if eff_f is not None:
                props["ate_estimate"] = eff_f
            # Real lifecycle/temporal signals for the dashboard's status badge —
            # NOT a confidence bucket. The frontend derives the tag from these.
            vstatus = row.get("validation_status")
            if vstatus:
                props["validation_status"] = vstatus
            cc = row.get("confirmation_count")
            if cc is not None:
                props["confirmation_count"] = cc
            ddate = row.get("discovery_date")
            if ddate:
                props["discovery_date"] = ddate
        rels.append(
            GraphRelationship(
                id="",
                type=RelationshipType.CAUSES,
                source_id=nodes[i].id,
                target_id=nodes[i + 1].id,
                properties=props,
                confidence=conf_f,
                created_at=None,
            )
        )

    plen = row.get("path_length")
    try:
        plen_i = int(plen) if plen is not None else n_edges
    except (TypeError, ValueError):
        plen_i = n_edges

    return GraphPath(
        nodes=nodes,
        relationships=rels,
        total_confidence=conf_f,
        path_length=plen_i,
    )


@router.get(
    "/value-chains",
    response_model=CausalChainResponse,
    summary="Top discovered causal value chains (brand/region scoped)",
    operation_id="get_causal_value_chains",
)
async def get_causal_value_chains(
    brand: Optional[str] = Query(
        None, description="Scope to a brand; omit or 'All' for the portfolio view"
    ),
    region: Optional[str] = Query(
        None, description="Scope to a region; omit or 'All US' for all regions"
    ),
    limit: int = Query(
        3, ge=1, le=20, description="Max distinct chains (top by |effect| x confidence)"
    ),
    user: Dict[str, Any] = Depends(require_viewer),
) -> CausalChainResponse:
    """Return the strongest REAL discovered causal value chains from ``causal_paths``.

    These are the live, dataset-derived chains the causal engine has *validated*
    (DoWhy backdoor estimation) — NOT a seeded graph fixture. Scoped by the Home
    dashboard's brand/region selectors, ranked by ``|effect| x confidence``, and
    de-duplicated by full pathway so the top-N are distinct value chains. Honors
    the synthetic-showcase provenance flag (``E2I_INCLUDE_SYNTHETIC``): on a
    synthetic-gold instance the synthetic chains ARE the substrate; on a strict
    real-data instance they are excluded verbatim.
    """
    start = time.time()
    try:
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Causal store unavailable")

        query = (
            client.table("causal_paths")
            .select(
                "path_id,start_node,end_node,intermediate_nodes,causal_chain,"
                "causal_effect_size,confidence_level,method_used,validation_status,"
                "confirmation_count,discovery_date,brand,region,path_length"
            )
            .eq("validation_status", "validated")
        )
        if brand and brand.strip().lower() not in _ALL_BRAND_SENTINELS:
            query = query.eq("brand", brand)
        if region and region.strip().lower() not in _ALL_REGION_SENTINELS:
            # causal_paths.region is stored lowercase (US-Census regions:
            # northeast/south/midwest/west). The dropdown sends title-case labels
            # ('Northeast'), so normalize to lowercase — an exact .eq against a
            # title-case value would silently return zero chains.
            query = query.eq("region", region.strip().lower())

        # Synthetic-showcase aware (SSOT). Showcase → include synthetic chains;
        # strict real-mode → excluded verbatim.
        query = apply_provenance_filter(query)

        # Pull a generous, effect-ordered slice; dedupe by pathway; rank by
        # |effect| x confidence so the top-N are DISTINCT, strongly-supported chains.
        result = await (
            query.order("causal_effect_size", desc=True).limit(max(limit * 12, 60)).execute()
        )
        rows: List[Dict[str, Any]] = result.data or []

        seen: set = set()
        distinct: List[Dict[str, Any]] = []
        for r in rows:
            seq = _chain_node_sequence(r)
            if len(seq) < 2:
                continue
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            distinct.append(r)

        distinct.sort(key=_chain_score, reverse=True)
        top = distinct[:limit]

        chains = [_causal_path_to_graphpath(r) for r in top]
        strongest = chains[0] if chains else None
        latency_ms = (time.time() - start) * 1000.0

        return CausalChainResponse(
            chains=chains,
            total_chains=len(chains),
            strongest_chain=strongest,
            # Heterogeneous pathways/scales — no honest scalar aggregate; the UI
            # hides the badge when this is None (never renders a fabricated 0.0%).
            aggregate_effect=None,
            query_latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Causal value-chains query failed")
        raise HTTPException(status_code=500, detail="Causal value-chains query failed") from exc


# =============================================================================
# TREATMENT EFFECTS (GET /causal/treatment-effects — cohort x brand ATE)
# =============================================================================

# The four cohorts the Treatment Effects surface supports. Each maps to the
# outcome column that becomes the binary label. The patient cohorts read
# patient_journeys; hcp_adoption reads hcp_brand_adoption JOIN hcp_profiles.
_TE_PATIENT_OUTCOME = {
    "initiation": "treatment_initiated",
    "persistence": "persistent_180d",
    "discontinuation": "discontinued_180d",
}
_TE_COHORTS = set(_TE_PATIENT_OUTCOME) | {"hcp_adoption"}
_TE_BRANDS = {"Remibrutinib", "Fabhalta", "Kisqali"}

# treatment column shared by all four cohorts (binary 0/1 arm).
_TE_TREATMENT_VAR = "treatment_arm"

# Numeric confounders per cohort family. geographic_region is DELIBERATELY
# EXCLUDED for the patient cohorts: it is a categorical string column that breaks
# DoWhy/EconML (they require numeric inputs). The HCP cohort joins hcp_profiles
# for the centrality confounders that drive treatment_arm by construction.
_TE_PATIENT_CONFOUNDERS = ["disease_severity", "academic_hcp"]
_TE_HCP_CONFOUNDERS = ["peer_influence_score", "influence_network_size"]


# Per-request compute budget (seconds) for the DoWhy+EconML fit under the
# heavy-compute slot. A single cohort fit is seconds; this bounds a degenerate
# run so a slow/contended box returns 408 rather than holding the slot forever.
_TE_TIMEOUT_SECONDS = 90.0


async def _resolve_treatment_effect_frame(
    cohort: str,
    brand: str,
) -> Optional["_TEFrameSpec"]:
    """Load a confounded estimation frame for (cohort, brand) from the DB.

    Returns a ``_TEFrameSpec`` (numeric-coerced + dropna'd DataFrame +
    treatment/outcome/confounder names) or ``None`` (caller fail-closes 503) when
    the cohort frame cannot be resolved or is empty after coercion. NEVER
    fabricates rows. The async Supabase client mirrors /value-chains.
    """
    import pandas as pd

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        return None

    if cohort in _TE_PATIENT_OUTCOME:
        outcome_var = _TE_PATIENT_OUTCOME[cohort]
        confounders = list(_TE_PATIENT_CONFOUNDERS)
        columns = ",".join([_TE_TREATMENT_VAR, outcome_var, *confounders])
        rows = await _te_paged_select(client, "patient_journeys", columns, brand)
        if not rows:
            return None
        df = pd.DataFrame(rows)
    else:
        # hcp_adoption: hcp_brand_adoption (treatment_arm, adopted) JOIN
        # hcp_profiles (peer_influence_score, influence_network_size) on hcp_id.
        # Two reads + a pandas merge (not the cohort_resolution hcp_profiles
        # branch, which uses a DIFFERENT continuous-treatment substrate).
        outcome_var = "adopted"
        confounders = list(_TE_HCP_CONFOUNDERS)
        adoption_rows = await _te_paged_select(
            client,
            "hcp_brand_adoption",
            "hcp_id,treatment_arm,adopted",
            brand,
        )
        if not adoption_rows:
            return None
        # hcp_profiles is NOT brand-partitioned; read its centrality covariates
        # paged across is_synthetic rows and merge by hcp_id (100% coverage
        # verified). brand filter does not apply here, so read without it.
        profile_rows: List[Dict[str, Any]] = []
        for page in range(_TE_MAX_PAGES):
            offset = page * _TE_PAGE_SIZE
            prof_q = (
                client.table("hcp_profiles")
                .select("hcp_id,peer_influence_score,influence_network_size")
                .eq("is_synthetic", True)
                .range(offset, offset + _TE_PAGE_SIZE - 1)
            )
            prof_res = await prof_q.execute()
            prof_batch: List[Dict[str, Any]] = prof_res.data or []
            profile_rows.extend(prof_batch)
            if len(prof_batch) < _TE_PAGE_SIZE:
                break
        if not profile_rows:
            return None
        adoption_df = pd.DataFrame(adoption_rows)
        profile_df = pd.DataFrame(profile_rows).drop_duplicates(subset="hcp_id")
        df = adoption_df.merge(profile_df, on="hcp_id", how="inner")
        if df.empty:
            return None

    # Build the numeric estimation frame: coerce treatment/outcome/confounders to
    # numeric, drop any row with a non-coercible/NA cell. n = surviving rows. An
    # empty frame after coercion -> None (honest 503), never a fabricated fit.
    use_cols = [_TE_TREATMENT_VAR, outcome_var, *confounders]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        logger.warning(
            "treatment-effects: cohort=%s brand=%s frame missing columns %s",
            cohort,
            brand,
            missing,
        )
        return None
    est = df[use_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if est.empty:
        return None
    return _TEFrameSpec(
        frame=est.reset_index(drop=True),
        treatment_var=_TE_TREATMENT_VAR,
        outcome_var=outcome_var,
        confounders=confounders,
    )


class _TEFrameSpec(NamedTuple):
    """Resolved estimation frame + runnable var-set for one (cohort, brand) cell."""

    frame: Any  # pd.DataFrame
    treatment_var: str
    outcome_var: str
    confounders: List[str]


async def _run_treatment_effect_estimate(
    cohort: str,
    brand: str,
    spec: "_TEFrameSpec",
) -> TreatmentEffectResponse:
    """Run the wired DoWhy+EconML sequential pipeline on the resolved frame.

    Prefers EconML's ate/ci/std (it carries the CI); falls back to DoWhy's
    causal_effect/standard_error (CI from that SE, #2014) when EconML fails. Raises
    HTTPException(503) when NEITHER executor produces a usable estimate. NEVER
    fabricates a number.
    """
    start = time.time()
    n = int(len(spec.frame))

    pipeline_input = PipelineInput(
        query=f"Treatment effect: cohort={cohort}, brand={brand}",
        treatment_var=spec.treatment_var,
        outcome_var=spec.outcome_var,
        confounders=list(spec.confounders),
        effect_modifiers=None,
        data_source=f"{cohort}/{brand}",
        filters={},
        estimation_data=spec.frame,
        mode="sequential",
        # Only DoWhy + EconML: NetworkX/CausalML are not needed for a single ATE
        # cell and would add latency. _get_execution_order filters SEQUENTIAL_ORDER
        # by this set, so DoWhy then EconML run in order.
        libraries_enabled=["dowhy", "econml"],
        cross_validate=None,
        run_refutation=False,
    )

    pipeline = _SurfaceCSequentialPipeline(fail_fast=False)
    await pipeline.execute(pipeline_input)
    state: Mapping[str, Any] = pipeline.last_state or {}

    # ---- Prefer EconML (carries CI) ----
    econml_result = state.get("econml_result")
    econml_payload = (
        econml_result.get("result")
        if isinstance(econml_result, dict) and isinstance(econml_result.get("result"), dict)
        else None
    )
    dowhy_result = state.get("dowhy_result")
    dowhy_payload = (
        dowhy_result.get("result")
        if isinstance(dowhy_result, dict) and isinstance(dowhy_result.get("result"), dict)
        else None
    )

    ate: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    std_error: Optional[float] = None
    estimator: Optional[str] = None

    if econml_payload is not None and econml_payload.get("ate") is not None:
        ate = _as_optional_float(econml_payload.get("ate"))
        ci_lower = _as_optional_float(econml_payload.get("ate_ci_lower"))
        ci_upper = _as_optional_float(econml_payload.get("ate_ci_upper"))
        std_error = _as_optional_float(econml_payload.get("ate_std"))
        est_name = econml_payload.get("estimator")
        estimator = str(est_name) if est_name is not None else None
    elif dowhy_payload is not None and dowhy_payload.get("causal_effect") is not None:
        # DoWhy fallback: the CI is the 95 % normal interval of its SE (#2014; it was
        # left None although the SE and its p-value were reported).
        ate = _as_optional_float(dowhy_payload.get("causal_effect"))
        std_error = _as_optional_float(dowhy_payload.get("standard_error"))
        estimator = dowhy_payload.get("dowhy_method")
        dowhy_interval = _dowhy_interval(dowhy_payload)
        if dowhy_interval is not None:
            ci_lower, ci_upper = dowhy_interval

    if ate is None:
        # Neither executor produced a usable estimate — honest fail-close.
        logger.warning(
            "treatment-effects: no usable estimate (cohort=%s brand=%s n=%d errors=%s)",
            cohort,
            brand,
            n,
            state.get("errors"),
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Causal pipeline produced no usable treatment-effect estimate for "
                f"cohort={cohort!r} brand={brand!r} (both DoWhy and EconML failed)."
            ),
        )

    p_value = _te_pvalue_from_z(ate, std_error)
    latency_ms = int((time.time() - start) * 1000)

    return TreatmentEffectResponse(
        cohort=cohort,
        brand=brand,
        treatment_var=spec.treatment_var,
        outcome_var=spec.outcome_var,
        confounders=list(spec.confounders),
        ate=ate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        std_error=std_error,
        n=n,
        estimator=estimator,
        method="dowhy+econml sequential",
        confidence_level=0.95,
        latency_ms=latency_ms,
        is_synthetic=True,
        warnings=[_ROBUSTNESS_UNVALIDATED_WARNING],
    )


@router.get(
    "/treatment-effects",
    response_model=TreatmentEffectResponse,
    summary="Estimate the treatment effect for a (cohort, brand) cell",
    operation_id="get_treatment_effect",
)
async def get_treatment_effect(
    cohort: str = Query(
        ...,
        description="Cohort: initiation | persistence | discontinuation | hcp_adoption",
    ),
    brand: str = Query(
        ...,
        description="Brand: Remibrutinib | Fabhalta | Kisqali",
    ),
    user: Dict[str, Any] = Depends(require_viewer),
) -> TreatmentEffectResponse:
    """Return a REAL average treatment effect for one (cohort, brand) cell.

    Loads a confounded cohort frame from the DB (patient_journeys for the patient
    cohorts; hcp_brand_adoption JOIN hcp_profiles for hcp_adoption), then runs the
    EXISTING DoWhy+EconML sequential pipeline to recover a de-confounded ATE + CI
    + p_value + n. Honors the synthetic-showcase substrate (is_synthetic=true).

    Fail-closed: 422 on an unknown cohort/brand; 503 when the cohort frame cannot
    be resolved (no rows) or the pipeline yields no usable estimate; 408 on
    timeout; 503 (Retry-After) when the heavy-compute slot is saturated. NEVER
    fabricates an effect.
    """
    cohort_key = cohort.strip().lower()
    if cohort_key not in _TE_COHORTS:
        raise HTTPException(
            status_code=422,
            detail=(f"Unknown cohort {cohort!r}. Expected one of: {sorted(_TE_COHORTS)}."),
        )
    if brand not in _TE_BRANDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown brand {brand!r}. Expected one of: {sorted(_TE_BRANDS)}.",
        )

    try:
        spec = await _resolve_treatment_effect_frame(cohort_key, brand)
        if spec is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No resolvable cohort data for cohort={cohort_key!r} brand={brand!r}. "
                    "The cohort frame was empty or unavailable; refusing to fabricate an effect."
                ),
            )
        # The DoWhy+EconML fit is the genuinely heavy in-process compute. Bound it
        # to ONE per-worker heavy-compute slot (OOM guard) + a wall-clock timeout
        # so a contended box returns 408 rather than holding the slot forever.
        async with heavy_compute_slot():
            return await asyncio.wait_for(
                _run_treatment_effect_estimate(cohort_key, brand, spec),
                timeout=_TE_TIMEOUT_SECONDS,
            )
    except HTTPException:
        raise
    except asyncio.TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail=f"Treatment-effect estimation timed out after {_TE_TIMEOUT_SECONDS}s",
        ) from e
    except HeavyComputeSaturated:
        # Reject fast under load — mapped to 503 + Retry-After by the app handler.
        # Must precede the broad handler so it is not swallowed into a 500.
        raise
    except Exception as e:  # noqa: BLE001 - last-resort 500
        logger.error(f"Treatment-effect estimation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from e
