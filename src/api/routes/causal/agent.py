"""Agent-analysis routes for the causal package (#1991 debt 4).

POST /agent-analyze submits an end-to-end causal_impact agent run and GET
/agent-analyze/{analysis_id} polls it. Also holds the cross-worker job store, the
background task that drives the graph under a wall-clock budget, the MLflow
recording, and the mapping from the finished agent state onto the API response.

Import rule: may import ``_common``, ``datasets``, ``loaders`` and non-package
modules only; never another route module or the package root.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.dependencies.auth import require_analyst
from src.api.dependencies.compute import heavy_compute_slot
from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.schemas.causal import (
    AGENT_FORCEABLE_ESTIMATORS,
    AgentCausalAnalysisRequest,
    AgentCausalAnalysisResponse,
    CausalDAGModel,
    EdgeProvenanceModel,
    EstimatorCandidate,
    EstimatorComparison,
    RefutationSummary,
    RefutationTestDetail,
)
from src.repositories.provenance import deployment_includes_synthetic

from ._common import (
    _AGENT_HARD_TIMEOUT_S,
    _CAUSAL_JOB_TTL_SECONDS,
    _REFUTATION_COMPUTE_BUDGET_S,
    _opt_float,
)
from .datasets import (
    _CAUSAL_DATASET_SPECS,
    _brand_scoped_covariates,
    _is_randomized_treatment,
    _negative_control_outcome,
)
from .loaders import _load_agent_estimation_frame, _resolve_requested_baselines

logger = logging.getLogger(__name__)

router = APIRouter()


_agent_analysis_store: DurableJobStore["AgentCausalAnalysisResponse"] = DurableJobStore(
    "causal:agent_analyze", AgentCausalAnalysisResponse, ttl_seconds=_CAUSAL_JOB_TTL_SECONDS
)


# =============================================================================
# AGENT ANALYSIS ENDPOINT (causal_impact agent, end-to-end)
# =============================================================================


def _normalised_panel(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The typed round-trip of the caller's panel: exactly what was validated at
    submit reaches the agent (unknown keys dropped, shapes normalised) — never
    the raw caller-authored dictionary (codex r3)."""
    from src.causal_engine.feature_role_panel import FeatureRolePanel

    return FeatureRolePanel.from_dict(payload).to_dict()


def _validate_feature_role_panel(
    request: AgentCausalAnalysisRequest,
    spec: Dict[str, Any],
    covariates: List[str],
) -> None:
    """Refuse (400) a panel that does not answer THIS question (codex r2)."""
    from src.causal_engine.feature_role_panel import FeatureRolePanel

    try:
        panel = FeatureRolePanel.from_dict(request.feature_role_panel or {})
    except Exception as exc:  # noqa: BLE001 — any malformed payload is a caller error
        raise HTTPException(
            status_code=400,
            detail=(
                "feature_role_panel does not parse as a FeatureRolePanel "
                f"(src.causal_engine.feature_role_panel.FeatureRolePanel.to_dict()): {exc}"
            ),
        ) from exc
    try:
        panel.validate_strict()
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"feature_role_panel violates an invariant: {exc}"
        ) from exc
    if (panel.treatment, panel.outcome) != (request.treatment_var, request.outcome_var):
        raise HTTPException(
            status_code=400,
            detail=(
                f"feature_role_panel was built for ({panel.treatment!r} -> {panel.outcome!r}) "
                f"but this analysis asks ({request.treatment_var!r} -> "
                f"{request.outcome_var!r}); build the panel for this question "
                "(scripts/measure_feature_role_panel.py) or omit it."
            ),
        )
    covered = [c for c in covariates if c in panel.records or c.split("=", 1)[0] in panel.records]
    if covariates and not covered:
        raise HTTPException(
            status_code=400,
            detail=(
                f"feature_role_panel covers none of this analysis' covariates "
                f"({covariates[:8]}{'...' if len(covariates) > 8 else ''}); it evaluated "
                f"{len(panel.records)} column(s) under manifest {panel.manifest_source!r}."
            ),
        )
    declared_manifest = spec.get("feature_manifest_source")
    if declared_manifest and declared_manifest != panel.manifest_source:
        raise HTTPException(
            status_code=400,
            detail=(
                f"feature_role_panel was built under manifest {panel.manifest_source!r} but "
                f"dataset {request.dataset!r} declares {declared_manifest!r}."
            ),
        )


@router.post(
    "/agent-analyze",
    response_model=AgentCausalAnalysisResponse,
    summary="Run the causal_impact agent end-to-end (DAG + effect + refutation)",
    operation_id="run_causal_agent_analysis",
)
async def run_causal_agent_analysis(
    request: AgentCausalAnalysisRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_analyst),
) -> AgentCausalAnalysisResponse:
    """Submit a causal_impact agent run (async) and return a pending handle.

    Leverages the agent: it builds the causal DAG, selects an estimator
    DATA-DRIVENLY via the energy-score router across the registry (or the forced
    one when ``estimator`` is set), estimates the treatment->outcome effect, and
    runs refutation + sensitivity. That work takes MINUTES, so it runs as a
    BackgroundTask and the client polls ``GET /causal/agent-analyze/{id}`` (same
    submit->poll shape as the hierarchical / pipeline endpoints).

    Data is validated + loaded SYNCHRONOUSLY here, so bad columns / no data
    fail-closed immediately with the right HTTP status (400/404/503); only the
    heavy agent run is deferred. Fail-closed throughout — never a fabricated ATE.
    """
    # Validate the optional estimator override BEFORE loading — the agent
    # restricts forced methods to _VALID_EXPLICIT_METHODS; surface an honest 400.
    if request.estimator and request.estimator not in AGENT_FORCEABLE_ESTIMATORS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Estimator '{request.estimator}' cannot be forced. Supported "
                f"overrides: {list(AGENT_FORCEABLE_ESTIMATORS)}. Omit `estimator` "
                "for Auto (the agent's data-driven routing across the registry)."
            ),
        )

    # Covariates default to the dataset's curated confounders (data-driven), and
    # can never include the treatment/outcome themselves.
    spec = _CAUSAL_DATASET_SPECS.get(request.dataset)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown causal dataset '{request.dataset}'. "
                f"Known datasets: {sorted(_CAUSAL_DATASET_SPECS)}"
            ),
        )
    # #1872: the spec-level covariate OFFER must not leak into a RANDOMIZED
    # treatment by default — randomization already closes every backdoor, and
    # the #1188 design keeps the RCT default-unadjusted (baselines stay a
    # separate opt-in efficiency role). An analyst's EXPLICIT picks are still
    # honored (pre-treatment adjustment of an RCT is legitimate when asked for).
    default_covariates = (
        []
        if _is_randomized_treatment(request.dataset, request.treatment_var)
        else spec["covariate"]
    )
    # Brand-aware adjustment set (Phase 2): drop indication-specific clinical
    # covariates that are NULL for this brand cohort so a gated off-brand column
    # never reaches EconML as NaN. Applies to BOTH the analyst's explicit picks and
    # the curated default. brand=None (all-brands) keeps universals only.
    covariates = [
        c
        for c in _brand_scoped_covariates(
            (request.covariates if request.covariates is not None else default_covariates),
            request.brand,
        )
        if c not in (request.treatment_var, request.outcome_var)
    ]

    # Lane E item 3(d): a feature-role panel is a TRUSTED causal input that
    # narrows the adjustment set, so its identity is established HERE, before
    # scheduling (codex r2): it must parse as a typed FeatureRolePanel, answer
    # THIS question (T, Y), cover at least one requested covariate (or the
    # source of a one-hot dummy), and — when the dataset spec declares its
    # manifest — come from that manifest.
    if request.feature_role_panel is not None:
        _validate_feature_role_panel(request, spec, covariates)

    # #1188: opt-in RCT baseline adjustment — resolve the flag to the curated
    # baseline list (400 on datasets without a baseline role).
    baseline_covariates = _resolve_requested_baselines(request.dataset, request.adjust_baselines)

    # Load synchronously -> fail-closed early (400 bad column / 404 dataset /
    # 503 no data) before scheduling the heavy run. ``brand`` (optional) scopes
    # the cohort to one brand (a row subset; brand stays out of the estimation
    # columns) so the analyst can analyze a single brand's patients.
    # #2007: the declared negative-control outcome (if any) rides along as a
    # PASSTHROUGH column — same rows, never a covariate, never in select_cols —
    # so the refutation node can re-fit the identical adjusted model on it.
    negative_control = _negative_control_outcome(
        request.dataset, request.treatment_var, request.outcome_var
    )
    df, select_cols = await _load_agent_estimation_frame(
        dataset=request.dataset,
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        covariates=covariates,
        limit=request.limit,
        brand=request.brand,
        baseline_covariates=baseline_covariates or None,
        passthrough_columns=[negative_control] if negative_control else None,
    )
    # The loader EXPANDS categorical columns (e.g. geographic_region) into
    # one-hot dummies; the agent needs the resolved frame columns (the dummy
    # names), not the raw categorical. Baselines (#1188) are split OUT of the
    # confounders — they are efficiency controls on a randomized design, never
    # part of the backdoor adjustment set. #1872: a column requested as a
    # BACKDOOR covariate keeps the confounder role even when adjust_baselines
    # also selects it (disease_severity holds both curated roles) — demoting a
    # backdoor variable to efficiency-only would silently unadjust the edge.
    _question_vars = (request.treatment_var, request.outcome_var)
    _covariate_roots = set(covariates)
    _baseline_roots = set(baseline_covariates) - _covariate_roots
    resolved_baselines = [
        c
        for c in select_cols
        if c not in _question_vars
        and (c in _baseline_roots or c.split("=", 1)[0] in _baseline_roots)
    ]
    resolved_covariates = [
        c for c in select_cols if c not in _question_vars and c not in resolved_baselines
    ]

    analysis_id = str(uuid.uuid4())
    data_source = "synthetic" if deployment_includes_synthetic() else "database"
    pending = AgentCausalAnalysisResponse(
        analysis_id=analysis_id,
        status="pending",
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        dataset=request.dataset,
        n_rows=int(df.shape[0]),
        data_source=data_source,
        dag=CausalDAGModel(),
        statistical_significance=False,
        refutation=RefutationSummary(),
        warnings=["Analysis submitted; poll GET /causal/agent-analyze/{id} for the result."],
        latency_ms=0,
    )
    await _agent_analysis_store.set(analysis_id, pending)
    background_tasks.add_task(
        _run_agent_analysis_task,
        analysis_id,
        request,
        df,
        resolved_covariates,
        data_source,
        resolved_baselines,
    )
    return pending


@router.get(
    "/agent-analyze/{analysis_id}",
    response_model=AgentCausalAnalysisResponse,
    summary="Poll a causal_impact agent run by id",
    operation_id="get_causal_agent_analysis",
)
async def get_causal_agent_analysis(analysis_id: str) -> AgentCausalAnalysisResponse:
    """Poll a submitted agent run. 404 until the submit registered it; then
    pending -> running -> completed / needs_review / failed."""
    job = await _agent_analysis_store.get(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return job


async def _run_agent_analysis_task(
    analysis_id: str,
    request: AgentCausalAnalysisRequest,
    df: "pd.DataFrame",  # type: ignore[name-defined] # noqa: F821
    covariates: List[str],
    data_source: str,
    baseline_covariates: Optional[List[str]] = None,
) -> None:
    """Background: run the agent on the pre-loaded frame; cache the result.

    The state is built DIRECTLY (not via CausalImpactAgent, whose wrapper would
    short-circuit a "synthetic" data_source to fast OLS) — so "Auto" runs the
    real energy-score selection across the registry. ``data_source`` here is only
    the response provenance label; the data always comes from data_cache. The
    refutation is bounded so the run completes in minutes (not the full
    ~610-re-estimation suite), with a generous wall-clock cap.
    """
    import time as _time

    prev = await _agent_analysis_store.get(analysis_id)
    if prev is not None:
        await _agent_analysis_store.set(analysis_id, prev.model_copy(update={"status": "running"}))

    parameters: Dict[str, Any] = {}
    if request.estimator:
        parameters["method"] = request.estimator
    parameters.setdefault(
        "refutation_config",
        {
            "bootstrap": {"num_bootstraps": 20},
            "placebo_treatment": {"num_simulations": 10},
            "data_subset": {"num_subsets": 5},
            "random_common_cause": {"num_simulations": 10},
        },
    )
    # #2007: the declared negative-control outcome (None when undeclared — the
    # runner then emits SKIPPED ``no_negative_control_declared``). The submit
    # endpoint fetched it as a passthrough column of ``df``; it is SPLIT OFF
    # here into its own data_cache entry, index-aligned with the estimation
    # frame, because two existing consumers treat every non-question column of
    # ``estimation_data`` as a covariate: guided discovery tiers all of them as
    # candidate confounders (graph_builder), and the estimator's no-backdoor
    # fallback adjusts on all of them (estimation). A control that entered
    # either would corrupt the very estimate it is meant to check.
    negative_control = _negative_control_outcome(
        request.dataset, request.treatment_var, request.outcome_var
    )
    data_cache: Dict[str, Any] = {"estimation_data": df}
    if negative_control and negative_control in df.columns:
        data_cache["negative_control_data"] = df[[negative_control]]
        data_cache["estimation_data"] = df.drop(columns=[negative_control])
    initial_state: Dict[str, Any] = {
        "query": (
            f"What is the causal effect of {request.treatment_var} on {request.outcome_var}?"
        ),
        "query_id": analysis_id,
        "treatment_var": request.treatment_var,
        "outcome_var": request.outcome_var,
        "confounders": covariates,
        # Fix 4 (two-channel confounder wiring): ``modeled_confounders`` is the
        # ADJUSTMENT-GUARANTEE channel — every covariate listed here is unioned
        # into the final adjustment set no matter what the discovered DAG shows,
        # so the estimate's conditioning set stays exactly the declared
        # covariates (unchanged vs the old wiring by construction).
        "modeled_confounders": covariates,
        # STRUCTURAL-PRIOR channel: deliberately EMPTY. The dataset spec's
        # covariate list is a role ALLOWLIST (an offer of adjustable columns),
        # not a per-question assertion that each is a genuine confounder — no
        # curated structural subset exists at this call site. Declaring them all
        # here (the old single-channel behavior) forced conf->treatment and
        # conf->outcome as REQUIRED edges for every covariate, making the
        # shipped DAG identical for real data and pure noise (measured:
        # F1 0.78, SHD 4 on the recovery benchmark). Empty anchors leave tiers +
        # the estimand edge as the only prior, so the DATA selects the
        # confounder edges (measured: F1 mean 0.93, SHD<=1 at n=2000) and the
        # corroboration gate scores real evidence (bootstrap stability) instead
        # of prior-determined renormalization.
        "anchored_confounders": [],
        # #1188: pre-treatment baselines for RCT efficiency adjustment —
        # deliberately NOT in confounders (they are not backdoor variables;
        # the estimation node routes them to the selector's
        # efficiency_controls channel).
        "baseline_covariates": list(baseline_covariates or []),
        "data_source": data_source,
        "data_cache": data_cache,
        # Learn the DAG from data via GUIDED discovery (graph_builder anchors the
        # treatment/outcome roles; the data selects the confounders). Falls back
        # to the domain DAG if discovery is skipped or not accepted by the gate.
        "auto_discover": request.auto_discover,
        "discovery_guided": True,
        # Lane E item 3(d): the feature-role panel, forwarded only when supplied
        # (graph_builder keys off presence; see the state docstring). The
        # declared covariates above stay as submitted — the panel narrows them in
        # graph_builder, with a named warning. ``approved_structure_roles`` is
        # NOT a request field: approval is resolved server-side (Lane B).
        **(
            {"feature_role_panel": _normalised_panel(request.feature_role_panel)}
            if request.feature_role_panel is not None
            else {}
        ),
        "parameters": parameters,
        "interpretation_depth": "standard",
        "brand": request.brand,
        # DESIGN declaration from the dataset spec (per-treatment): a genuinely
        # randomized assignment reports the E-value sensitivity as information
        # instead of an unmeasured-confounding BLOCK gate, and the narrative
        # stops calling the RCT "observational data". Fail-closed default False.
        "randomized_design": _is_randomized_treatment(request.dataset, request.treatment_var),
        # #2007: consumed by the refutation node (negative-control-outcome test);
        # the column itself is data_cache["negative_control_data"] (see above).
        "negative_control_outcome": negative_control,
        # Cooperative compute deadline so the refutation suite self-terminates
        # before the hard wait_for cap below (orphan-fix): timed-out runs return
        # cleanly instead of orphaning an uncancellable to_thread refutation.
        "compute_deadline": time.monotonic() + _REFUTATION_COMPUTE_BUDGET_S,
        "errors": [],
        "warnings": [],
        "fallback_used": False,
        "retry_count": 0,
    }

    start = _time.time()
    try:
        from src.agents.causal_impact.graph import create_causal_impact_graph

        graph = create_causal_impact_graph()
        # Bound concurrency to ONE per-worker heavy-compute slot (OOM guard),
        # mirroring the hierarchical / parallel endpoints.
        async with heavy_compute_slot():
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state), timeout=_AGENT_HARD_TIMEOUT_S
            )
        response = _agent_state_to_response(
            analysis_id=analysis_id,
            request=request,
            data_source=data_source,
            n_rows=int(df.shape[0]),
            final_state=final_state,
            latency_ms=int((_time.time() - start) * 1000),
        )
        # Store the result first so it is pollable immediately; the MLflow
        # trail below is best-effort observability (wave-51 Gap B) and runs
        # after, so tracking problems cannot affect the cached result.
        await _agent_analysis_store.set(analysis_id, response)
        await _record_agent_mlflow_run(
            request=request,
            analysis_id=analysis_id,
            response=response,
            final_state=final_state,
        )
    except Exception as e:  # noqa: BLE001 — cache a generic FAILED record
        logger.error(f"Background causal agent analysis failed: {e}", exc_info=True)
        await _agent_analysis_store.set(
            analysis_id,
            AgentCausalAnalysisResponse(
                analysis_id=analysis_id,
                status="failed",
                treatment_var=request.treatment_var,
                outcome_var=request.outcome_var,
                dataset=request.dataset,
                n_rows=int(df.shape[0]),
                data_source=data_source,
                dag=CausalDAGModel(),
                statistical_significance=False,
                refutation=RefutationSummary(),
                warnings=["Analysis failed due to an internal error."],
                latency_ms=int((_time.time() - start) * 1000),
            ),
        )


def _agent_mlflow_output(
    response: AgentCausalAnalysisResponse, final_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Adapt the API response onto the ``CausalImpactOutput`` field names the
    MLflow tracker reads.

    ``mlflow_tracker._extract_metrics`` / ``_log_params`` / ``_log_artifacts``
    consume the output with ``.get()`` only, so a plain dict carrying the same
    keys is a faithful payload; the detail metrics (latent diagnostic,
    refutation counts, sensitivity) all come from ``final_state``.
    """
    computation_latency_ms = (
        (final_state.get("graph_builder_latency_ms") or 0)
        + (final_state.get("estimation_latency_ms") or 0)
        + (final_state.get("refutation_latency_ms") or 0)
        + (final_state.get("sensitivity_latency_ms") or 0)
    )
    return {
        "query_id": response.analysis_id,
        "status": response.status,
        "ate_estimate": response.ate,
        "confidence_interval": (
            (response.ate_ci_lower, response.ate_ci_upper)
            if response.ate_ci_lower is not None and response.ate_ci_upper is not None
            else None
        ),
        "standard_error": response.standard_error,
        "p_value": response.p_value,
        "statistical_significance": response.statistical_significance,
        # None would break mlflow.log_metric — the tracker defaults 0.0 only
        # for an ABSENT key, not a present-but-None one.
        "confidence": response.confidence if response.confidence is not None else 0.0,
        "refutation_passed": bool(response.refutation.passed),
        "estimation_method": response.selected_estimator or "unknown",
        "model_used": response.selected_estimator or "unknown",
        "effect_type": "ate",
        "computation_latency_ms": float(computation_latency_ms),
        "interpretation_latency_ms": float(final_state.get("interpretation_latency_ms") or 0),
        "total_latency_ms": float(response.latency_ms),
    }


async def _record_agent_mlflow_run(
    *,
    request: AgentCausalAnalysisRequest,
    analysis_id: str,
    response: AgentCausalAnalysisResponse,
    final_state: Dict[str, Any],
) -> None:
    """Record a finished agent-analyze run to MLflow, best-effort (wave-51 Gap B).

    ``_run_agent_analysis_task`` invokes the causal_impact graph DIRECTLY (see
    its docstring for why), so it never traverses ``CausalImpactAgent.run()`` —
    the only seam ``CausalImpactMLflowTracker`` wrapped — and no
    ``e2i_causal/causal_impact`` experiment was ever recorded from the API.
    This records the same tracker payload after the fact.

    Two deliberate mechanics:

    * The run is opened AFTER the analysis completes, not around the (minutes
      long) ainvoke: mlflow's fluent active-run stack is per-THREAD
      (``ThreadLocalVariable``, mlflow 3.11 fluent.py) and ``start_run``
      raises on a non-empty stack — the per-minute health_score tracker holds
      fluent runs open in this same process, so a long-lived run here would
      collide both ways. Wall-clock lives in the ``total_latency_ms`` metric;
      the MLflow run duration itself is meaningless by design.
    * The whole recording executes in a WORKER thread via ``asyncio.to_thread``
      (fresh thread == empty fluent stack == no collision even at the instant a
      health_score run is open), which also keeps mlflow's blocking HTTP off
      the event loop.

    Never raises: the response is already stored, and observability must not
    fail the analysis (a raise here would hit the caller's generic-FAILED
    handler and clobber the good cached result).
    """
    try:
        from src.agents.causal_impact.mlflow_tracker import CausalImpactMLflowTracker

        tracker = CausalImpactMLflowTracker()
        output = _agent_mlflow_output(response, final_state)

        def _record_in_fresh_loop() -> None:
            async def _record() -> None:
                async with tracker.start_analysis_run(
                    experiment_name="default",
                    brand=request.brand,
                    treatment_var=request.treatment_var,
                    outcome_var=request.outcome_var,
                    query_id=analysis_id,
                ):
                    # Plain dicts are faithful payloads here: the tracker
                    # reads both TypedDicts with .get() only.
                    await tracker.log_analysis_result(output, final_state)  # type: ignore[arg-type]

            asyncio.run(_record())

        await asyncio.to_thread(_record_in_fresh_loop)
    except Exception as exc:  # noqa: BLE001 — observability is best-effort
        logger.warning(
            f"MLflow recording failed for agent analysis {analysis_id} (non-fatal): {exc}"
        )


def _refutation_tests_from_state(refutation: Dict[str, Any]) -> List[RefutationTestDetail]:
    """Map the agent's ``refutation_results['individual_tests']`` dict onto the
    per-test detail list the drill-down table renders.

    ``individual_tests`` is keyed by the CONTRACT test name (placebo_treatment,
    random_common_cause, data_subset, unobserved_common_cause, bootstrap); each
    value carries ``passed``/``original_effect``/``new_effect``/``p_value``/
    ``details``. Returns [] when refutation did not run (the FE then shows the
    honest 'refutation did not run' state rather than a misleading prompt).

    We surface the DICT KEY as ``test_name``, not the inner ``test_name`` field:
    to_legacy_format keys the sensitivity test under ``unobserved_common_cause``
    but sets its inner test_name to the raw enum ``sensitivity_e_value`` — using
    the inner value would make the FE fall back to the wrong label ("Random
    Common Cause") and duplicate that row. The key is the name the FE maps on.
    """
    individual = refutation.get("individual_tests")
    if not isinstance(individual, dict):
        return []
    tests: List[RefutationTestDetail] = []
    for key, t in individual.items():
        if not isinstance(t, dict):
            continue
        # Canonical (contract) name = the dict key; fall back to the inner field
        # only if the key is somehow empty.
        name = str(key) if key else str(t.get("test_name") or "")
        tests.append(
            RefutationTestDetail(
                test_name=name,
                passed=bool(t.get("passed", False)),
                status=(str(t["status"]) if t.get("status") else None),
                original_effect=_opt_float(t.get("original_effect")),
                new_effect=_opt_float(t.get("new_effect")),
                p_value=_opt_float(t.get("p_value")),
                details=(str(t["details"]) if t.get("details") else None),
            )
        )
    return tests


def _estimator_comparison_from_estimation(
    estimation: Dict[str, Any],
) -> Optional["EstimatorComparison"]:
    """Surface the energy-score selector's full evaluation (candidates + scores +
    rationale) so the UI can explain WHY the chosen estimator won.

    Returns None when only one estimator was evaluated (e.g. an explicitly-forced
    method) — a single-row "comparison" conveys nothing, so it is collapsed.
    """
    evaluated = estimation.get("all_estimators_evaluated") or []
    n_evaluated = int(estimation.get("n_estimators_evaluated") or len(evaluated))
    # Collapse only when a single estimator was CONSIDERED at all. An empty-backdoor
    # (randomized) run fits only OLS but carries the not-applicable rows for the
    # covariate-based estimators, so it still renders — and those rows explain WHY
    # only OLS applied (rather than the analyst wondering where they went).
    if len(evaluated) <= 1 or not evaluated:
        return None

    selected = estimation.get("selected_estimator") or estimation.get("method")
    candidates = [
        EstimatorCandidate(
            estimator=str(e.get("estimator")),
            success=bool(e.get("success")),
            skipped=bool(e.get("skipped", False)),
            energy_score=e.get("energy_score"),
            ate=e.get("ate"),
            error=e.get("error"),
            is_selected=(e.get("estimator") == selected),
        )
        for e in evaluated
    ]
    _energy_block = estimation.get("energy_score_data") or {}
    return EstimatorComparison(
        candidates=candidates,
        selection_reason=estimation.get("selection_reason"),
        energy_score_gap=estimation.get("energy_score_gap"),
        n_evaluated=n_evaluated,
        n_succeeded=int(
            estimation.get("n_estimators_succeeded")
            or sum(1 for e in evaluated if e.get("success"))
        ),
        quality_tier=_energy_block.get("quality_tier"),
        requires_review=bool(estimation.get("requires_review", False)),
    )


def _agent_state_to_response(
    *,
    analysis_id: str,
    request: AgentCausalAnalysisRequest,
    data_source: str,
    n_rows: int,
    final_state: Dict[str, Any],
    latency_ms: int,
) -> AgentCausalAnalysisResponse:
    """Map the causal_impact agent's final state onto the API response.

    Fail-closed status mirrors the agent's own gate (CausalImpactAgent
    ._build_output): a run is ``completed`` only with a real ATE, a non-blocked
    refutation gate, no sensitivity failure and no expert-review halt; ``review``
    band is surfaced as ``needs_review``; anything else is ``failed`` with the
    reason in warnings. An expert-review halt (#1971: a human rejected the DAG
    structure, or CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL is on and a REVIEW-band
    structure holds no approval) is ``failed`` with the halt message -- review
    id and how to resolve it -- in warnings, and the gate verdict in
    ``refutation.expert_review_decision`` / ``expert_review_id``. ``failed`` is
    reused rather than a fourth status because the frontend poll loop treats
    only completed / needs_review / failed as terminal. The agent's
    ``review_caveat`` (#1995: the band + HITL sentence naming the approval or
    rejection, reviewer and validity window) is surfaced as
    ``refutation.review_caveat`` and appended to warnings unless a warning
    already contains it -- the agent's halt line embeds it verbatim -- so the
    caveat text appears in warnings exactly once.
    """
    causal_graph = final_state.get("causal_graph") or {}
    estimation = final_state.get("estimation_result") or {}
    refutation = final_state.get("refutation_results") or {}
    sensitivity = final_state.get("sensitivity_analysis") or {}
    interpretation = final_state.get("interpretation") or {}

    dag = CausalDAGModel(
        nodes=list(causal_graph.get("nodes", []) or []),
        edges=[list(e) for e in (causal_graph.get("edges", []) or []) if len(e) == 2],
        treatment_nodes=list(causal_graph.get("treatment_nodes", []) or []),
        outcome_nodes=list(causal_graph.get("outcome_nodes", []) or []),
        adjustment_sets=[list(s) for s in (causal_graph.get("adjustment_sets", []) or [])],
        dag_dot=causal_graph.get("dag_dot"),
        # Fix 4: per-edge provenance (required_prior / discovered / curated),
        # computed by graph_builder. Malformed entries are dropped rather than
        # failing the response (legacy states simply have none).
        edge_provenance=[
            EdgeProvenanceModel(
                source=str(e["source"]),
                target=str(e["target"]),
                provenance=e["provenance"],
            )
            for e in (causal_graph.get("edge_provenance") or [])
            if isinstance(e, dict)
            and e.get("source")
            and e.get("target")
            and e.get("provenance") in ("required_prior", "discovered", "curated")
        ],
    )

    # How was the DAG built? Provenance must separate what the DATA contributed
    # from what the PRIORS asserted. Guided discovery seeds the estimand edge plus
    # BOTH common-cause edges for every modeled confounder as REQUIRED constraints,
    # so when those constraints alone account for every edge that ships, the data
    # cannot be credited with the structure.
    #
    # Not hypothetical: with the agent endpoints declaring every covariate a
    # confounder, the shipped DAG is IDENTICAL for a real frame and for a PURE
    # NOISE frame — yet the old classifier labelled the first 'discovered' and the
    # second 'domain_knowledge' (measured in tests/unit/test_causal_engine/
    # test_discovery/test_structural_recovery.py::TestProductionWiringIsPriorDetermined).
    #
    #   'discovered'       accepted, and the DAG carries edges BEYOND the priors
    #   'prior_asserted'   discovery's DAG was used (ACCEPT or AUGMENT) but every
    #                      shipped edge is prior-implied. The data may well agree;
    #                      a required-edge constraint makes agreement
    #                      indistinguishable from assertion, so no data
    #                      contribution is claimed either way.
    #   'augmented'        domain DAG + high-confidence discovered edges beyond
    #                      the priors
    #   'domain_knowledge' discovery skipped, rejected, or its DAG discarded
    #
    # ``discovery_dag_overridden`` guards an honesty corner case: the gate can
    # ACCEPT a discovered DAG that contradicts a curated confounder, in which case
    # graph_builder DISCARDS the discovered DAG for the manual domain one (so the
    # confounder stays adjusted). The gate decision stays "accept" (true — many
    # consumers rely on it), but the DAG that ships is manual, so provenance must
    # report 'domain_knowledge', NOT 'discovered'.
    discovery_ran = final_state.get("discovery_result") is not None
    _gate_dec = causal_graph.get("discovery_gate_decision")
    _dag_overridden = bool(causal_graph.get("discovery_dag_overridden"))

    # Confounders DECLARED to discovery as STRUCTURAL priors. Fix 4 split the
    # channels: when ``anchored_confounders`` is present it alone shapes the
    # required edges (graph_builder._resolve_anchored_confounders — the agent
    # endpoints pass [] so only the estimand edge is prior-implied); legacy
    # states without the key fall back to ``modeled_confounders`` /
    # ``confounders``, which the old single-channel wiring anchored wholesale.
    # This MUST track graph_builder's resolution exactly, or every prod run
    # would be mislabeled against the wrong prior shape.
    if "anchored_confounders" in final_state:
        _prior_confs = {str(c) for c in (final_state.get("anchored_confounders") or [])}
    else:
        _prior_confs = {
            str(c)
            for c in (
                final_state.get("modeled_confounders") or final_state.get("confounders") or []
            )
        }
    # Everything the caller declared through ANY channel — a declared covariate
    # echoed back is not a data finding, whichever channel carried it.
    _declared = _prior_confs | {
        str(c)
        for key in ("modeled_confounders", "confounders")
        for c in (final_state.get(key) or [])
    }
    # Exactly the edges the structural priors imply on their own. Treatment/outcome
    # are read from the DAG'S OWN node lists, not the request: the edges are
    # expressed in the graph's names, and a divergence would silently empty the
    # prior set and label every run 'discovered' — failing toward the OVERSTATING
    # label. Falling back to the request keeps the classifier working for states
    # that omit the node lists.
    _t = next(iter(dag.treatment_nodes or []), request.treatment_var)
    _o = next(iter(dag.outcome_nodes or []), request.outcome_var)
    _prior_edges = {(_t, _o)}
    for _conf in _prior_confs:
        _prior_edges.add((_conf, _t))
        _prior_edges.add((_conf, _o))
    # dag.edges was already filtered to [from, to] pairs at model construction.
    _beyond_priors = {(e[0], e[1]) for e in dag.edges} - _prior_edges

    if discovery_ran and _gate_dec == "accept" and not _dag_overridden:
        dag_source = "discovered" if _beyond_priors else "prior_asserted"
    elif discovery_ran and _gate_dec == "augment":
        # Same honesty rule as ACCEPT: an augment-gate DAG that carries nothing
        # beyond the prior-implied set has no data contribution to claim.
        dag_source = "augmented" if _beyond_priors else "prior_asserted"
    else:
        dag_source = "domain_knowledge"
    # Confounders the DATA identified: the backdoor adjustment set MINUS whatever
    # was declared up front. Echoing a declared covariate back as 'discovered' is
    # the same overstatement as the label above — under the all-covariates
    # declaration this correctly reports nothing rather than the caller's own list.
    _adj_sets = causal_graph.get("adjustment_sets", []) or []
    discovered_confounders = (
        sorted({str(c) for c in _adj_sets[0]} - _declared)
        if dag_source in ("discovered", "augmented") and _adj_sets
        else []
    )

    ate = estimation.get("ate")
    gate_decision = refutation.get("gate_decision") or final_state.get("gate_decision")
    refutation_error = final_state.get("refutation_error")
    sensitivity_failed = bool(final_state.get("sensitivity_error"))
    refutation_ran = bool(refutation) and not refutation_error
    gate_blocked = gate_decision == "block"
    needs_review = gate_decision == "review"
    # #1971: withheld on the expert-review gate -- may sit on a PROCEED gate.
    expert_review_halt = bool(final_state.get("expert_review_halt"))
    # #1995: the agent's band + HITL sentence (approval with reviewer and validity
    # window, rejection with reason, queued / blocked / unavailable). Built on
    # every REVIEW/BLOCK consult and on a PROCEED-band rejection; blank = not
    # consulted.
    review_caveat = str(final_state.get("review_caveat") or "").strip() or None

    if (
        ate is not None
        and refutation_ran
        and not gate_blocked
        and not sensitivity_failed
        and not expert_review_halt
    ):
        status = "needs_review" if needs_review else "completed"
    else:
        status = "failed"

    refutation_summary = RefutationSummary(
        gate_decision=gate_decision,
        passed=bool(refutation_ran and gate_decision == "proceed"),
        needs_review=needs_review,
        # Link to the expert-review queue row created for a REVIEW/BLOCK gate
        # (None when the run auto-proceeded), so the result references its review.
        expert_review_id=final_state.get("expert_review_id"),
        # #1971: the gate's verdict travels with its row id so consumers can
        # distinguish pending_review from an active structural approval.
        expert_review_decision=final_state.get("expert_review_decision"),
        # #1995: the caveat as a structured field, so a BLOCK-band run (which
        # never halts -- the statistical gate already withheld the estimate)
        # still records who approved / rejected the structure and why.
        review_caveat=review_caveat,
        tests_passed=refutation.get("tests_passed"),
        tests_total=refutation.get("total_tests"),
        sensitivity_e_value=sensitivity.get("e_value"),
        # Surface the per-test refutation results so the drill-down renders the
        # full table (placebo / random-common-cause / data-subset / bootstrap),
        # not just the pass/total count. The agent always computes these in the
        # leaderboard path; they were previously dropped here.
        tests=_refutation_tests_from_state(refutation),
    )

    # Surface honest warnings when the run did not yield a usable, validated effect.
    warnings: List[str] = list(final_state.get("warnings", []) or [])
    if ate is None:
        warnings.append("No treatment effect was estimated (the agent fail-closed on the data).")
    if not refutation_ran:
        warnings.append("Refutation did not run — the effect is unvalidated.")
    elif gate_blocked:
        warnings.append("Refutation gate BLOCKED — the estimate did not survive robustness checks.")
    if sensitivity_failed:
        warnings.append("Sensitivity analysis failed — robustness is unvalidated.")
    if expert_review_halt:
        warnings.append(
            str(final_state.get("error_message") or "")
            or "Estimate withheld on the expert-review gate (no reason recorded)."
        )
    if review_caveat and not any(review_caveat in w for w in warnings):
        # #1995: warnings is the drill-down's only prose channel. Containment
        # rule rather than "not halted": the agent's halt line embeds the caveat
        # verbatim (refutation.py _expert_review_halt_reason), in which case it
        # is already present and must not be repeated; a halt whose message does
        # NOT carry it (e.g. the fallback above) still gets it standalone. Either
        # way the caveat text appears exactly once.
        warnings.append(review_caveat)

    return AgentCausalAnalysisResponse(
        analysis_id=analysis_id,
        status=status,
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        dataset=request.dataset,
        n_rows=n_rows,
        data_source=data_source,
        dag=dag,
        dag_source=dag_source,
        discovered_confounders=discovered_confounders,
        # #1974: the persisted discovery row (None = not run / persist failed,
        # with the reason already in warnings via the state accumulator).
        discovered_dag_id=final_state.get("discovered_dag_id"),
        ate=ate,
        ate_ci_lower=estimation.get("ate_ci_lower"),
        ate_ci_upper=estimation.get("ate_ci_upper"),
        standard_error=estimation.get("standard_error"),
        p_value=estimation.get("p_value"),
        statistical_significance=bool(estimation.get("statistical_significance", False)),
        naive_ate=estimation.get("naive_ate"),
        naive_ate_ci_lower=estimation.get("naive_ate_ci_lower"),
        naive_ate_ci_upper=estimation.get("naive_ate_ci_upper"),
        confounding_bias_removed=estimation.get("confounding_bias_removed"),
        # #1188: honest adjustment framing — None for legacy states (unknown),
        # never a fabricated label.
        adjustment_type=estimation.get("adjustment_type"),
        baseline_covariates=list(estimation.get("baseline_covariates_adjusted") or []),
        selected_estimator=estimation.get("method") or estimation.get("selected_estimator"),
        estimator_comparison=_estimator_comparison_from_estimation(estimation),
        confidence=final_state.get("overall_confidence"),
        refutation=refutation_summary,
        narrative=interpretation.get("narrative"),
        executive_summary=interpretation.get("executive_summary"),
        recommendations=list(interpretation.get("recommendations", []) or []),
        key_insights=list(interpretation.get("key_findings", []) or []),
        warnings=warnings,
        latency_ms=latency_ms,
    )
