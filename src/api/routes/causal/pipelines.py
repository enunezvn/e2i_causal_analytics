"""Pipeline routes for the causal package (#1991 debt 4).

POST /pipeline/sequential and /pipeline/parallel run the multi-library
orchestrators, GET /pipeline/{pipeline_id} polls a sequential run, and
POST /validate cross-validates. Also holds the Surface-C wiring: building a
PipelineInput from the request, the DoWhy-interval subclasses, the structural
identification gate, and the mapping from PipelineOutput onto the API response.

Import rule: may import ``_common``, ``datasets``, ``loaders`` and non-package
modules only; never another route module or the package root.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from src.api.dependencies.auth import require_analyst
from src.api.dependencies.compute import HeavyComputeSaturated, heavy_compute_slot
from src.api.schemas.causal import (
    AnalysisStatus,
    CausalLibrary,
    CrossValidationRequest,
    CrossValidationResponse,
    ParallelPipelineRequest,
    ParallelPipelineResponse,
    PipelineStageResult,
    SequentialPipelineRequest,
    SequentialPipelineResponse,
)
from src.causal.stats import z_score_for_confidence

# #354 C-8: real-pipeline wiring (replaces 503-default short-circuit in
# non-demo mode). Imported lazily-safely; the LibraryExecutor implementations
# inside ParallelPipeline / SequentialPipeline themselves guard their backend
# dependencies (dowhy/econml/causalml/networkx availability), so importing
# the orchestrator classes is cheap.
from src.causal_engine.pipeline.parallel import ParallelPipeline
from src.causal_engine.pipeline.sequential import SequentialPipeline
from src.causal_engine.pipeline.state import (
    PipelineInput,
    PipelineOutput,
    PipelineState,
)

from ._common import (
    _CYCLE_IRRELEVANT_WARNING,
    _DATA_REQUIRED_LIBRARIES,
    _GENERIC_500_DETAIL,
    _NO_REAL_DATA_BACKEND_DETAIL,
    _NO_RESOLVABLE_DATA_DETAIL,
    _NON_DAG_STRUCTURAL_WARNING,
    _ROBUSTNESS_BLOCK_WARNING,
    _ROBUSTNESS_REVIEW_WARNING,
    _ROBUSTNESS_UNVALIDATED_WARNING,
    _dowhy_interval,
    _resolve_pipeline_dataframe,
    _te_pvalue_from_z,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# In-memory storage (for demo — replace with a database in production).
_pipeline_cache: Dict[str, Dict[str, Any]] = {}
_validation_cache: Dict[str, CrossValidationResponse] = {}


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


# #2106: the two spellings of DoWhy's nonparametric-ATE identification label
# that reach `identified_estimand`. Literals, not `str(EstimandType...)`: dowhy
# is a heavy import and this route module is imported by the app at startup,
# so it must not pull dowhy in at import time; the 2067 test derives the first
# spelling from the library, so a DoWhy `__str__` change is caught there.
_DOWHY_NONPARAMETRIC_ATE_LABELS = frozenset(
    {
        "EstimandType.NONPARAMETRIC_ATE",  # str(EstimandType.NONPARAMETRIC_ATE): prod
        "nonparametric-ate",  # EstimandType.NONPARAMETRIC_ATE.value
    }
)


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

    Payload keys (#2106 — one vocabulary per key, never two):
        ``identified_estimand``: DoWhy's identification label, copied verbatim
            from its executor: ``_extract_estimand_label`` (dowhy.py) returns
            ``str(estimand_type)``, which for DoWhy's plain-Enum
            ``EstimandType`` is
            ``EstimandType.NONPARAMETRIC_ATE`` on the real executor — not the
            enum's value ``nonparametric-ate`` — else the estimand's class
            name. DoWhy ONLY — it is the estimand DoWhy identified from the
            graph, an identification step the other libraries never perform;
            the pipeline orchestrator reads it into ``identification_method``.
        ``estimand``: what the stage's reported ``effect_estimate`` ESTIMATES,
            in one vocabulary for every branch that can say: ``"ate"`` for an
            average treatment effect on the outcome as named,
            ``"risk_difference_on_indicator_y_gt_0"`` for CausalML's
            binarized case. DoWhy derives it only from the one identification
            whose mapping is certain (nonparametric ATE, in either spelling of
            ``_DOWHY_NONPARAMETRIC_ATE_LABELS`` → ``ate``); CausalML takes it
            from the executor's ``outcome_binarized`` discriminator. EconML
            carries no estimand claim today, and none is invented.
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
            # #2106: name what the effect ESTIMATES next to what was
            # IDENTIFIED. DoWhy's nonparametric ATE is the one identification
            # whose estimated quantity is certain (an ATE on the outcome as
            # named). It arrives in two spellings: the real executor emits
            # `str(estimand_type)` (`_extract_estimand_label`, dowhy.py), and
            # DoWhy's `EstimandType` is a plain Enum, so prod carries "EstimandType.NONPARAMETRIC_ATE"
            # (the deployed api's dowhy stage in the #2067 live cert); the
            # enum's value is "nonparametric-ate". Any other label keeps its
            # identification claim and gets no guessed `estimand`.
            if estimand in _DOWHY_NONPARAMETRIC_ATE_LABELS:
                payload["estimand"] = "ate"
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
        # This stage is NOT a member of the ATE consensus (#2027):
        # `effect_estimate` is the mean model-predicted uplift, served with
        # ci_lower / ci_upper / p_value = None (the only interval CausalML
        # has is a dispersion) and labelled by `data_provenance` +
        # `estimand` below. `consensus_effect` is DoWhy + EconML only.
        ate = result_payload.get("ate")
        if isinstance(ate, (int, float)):
            payload["effect_estimate"] = float(ate)
        auuc = result_payload.get("auuc")
        if isinstance(auuc, (int, float)):
            payload["auuc"] = float(auuc)
        qini = result_payload.get("qini")
        if isinstance(qini, (int, float)):
            payload["qini"] = float(qini)
        # #2067: name the estimand this `ate` belongs to, next to the number.
        # CausalML binarizes any outcome at zero before fitting, so on a
        # non-binary outcome `ate` is a risk difference on the derived
        # indicator (y > 0), not an ATE on the column the caller named. The
        # executor decides this before the fit; a payload that carries no
        # discriminator gets no estimand claim rather than a guessed one.
        # #2106: the key is `estimand` (what the number estimates), never
        # `identified_estimand` — CausalML performs no identification step,
        # and that key is DoWhy's identification label (see the docstring).
        binarized = result_payload.get("outcome_binarized")
        if isinstance(binarized, bool):
            payload["estimand"] = "risk_difference_on_indicator_y_gt_0" if binarized else "ate"
            distinct = result_payload.get("outcome_distinct_values")
            if isinstance(distinct, int):
                payload["outcome_distinct_values"] = distinct
        # The executor's honesty marker (these are mean model-predicted
        # uplift figures, not identification-validated) was dropped here.
        provenance = result_payload.get("data_provenance")
        if isinstance(provenance, str):
            payload["data_provenance"] = provenance
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
