"""Discover-effects routes for the causal package (#1991 debt 4).

The validated-effects leaderboard: pick the candidate questions for a scope,
pre-rank them, then fan out over the agent task one question at a time, ranking
the VALIDATED effects as they land. Owns the cross-worker job store, the cancel
marker and the liveness heartbeat that repairs a job orphaned by a deploy.

Import rule: may import ``_common``, ``datasets``, ``loaders``, ``catalog`` and
``agent`` (the job fans out over the agent task) and non-package modules; never
the package root, and never a route module that imports this one.
"""

import asyncio
import contextlib
import logging
import uuid
from typing import Any, Dict, List, NamedTuple, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query

from src.api.dependencies.auth import require_analyst, require_viewer
from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.schemas.causal import (
    AgentCausalAnalysisRequest,
    AgentCausalAnalysisResponse,
    CausalDAGModel,
    ClinicalContext,
    DiscoveredEffect,
    DiscoverEffectsRequest,
    DiscoverEffectsResponse,
    DiscoverQuestion,
    DiscoverQuestionSelection,
    DiscoverQuestionsResponse,
    RefutationSummary,
)
from src.insights.robustness_phrase import gate_verdict_phrase
from src.repositories.provenance import deployment_includes_synthetic

# The task calls the agent task — and reads the agent JOB STORE — THROUGH the
# module namespace so a patch on ``agent._run_agent_analysis_task`` or on
# ``agent._agent_analysis_store`` reaches this reader (a ``from .agent import``
# would bind a copy here and the patch would miss it). The store especially:
# the agent task writes ``agent``'s global, so a second binding here would leave
# the agent side writing the real store whenever a test replaced only this one.
from . import agent as _agent
from ._common import _CAUSAL_JOB_TTL_SECONDS
from .catalog import _adjusted_partial_corr, _get_clinical_context_service
from .datasets import (
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_NUMERIC_COLUMNS,
    _DEFAULT_CAUSAL_DATASET,
    _DISCOVERY_ROW_CAP,
    _brand_scoped_covariates,
    _column_label,
    _default_auto_discover,
    _list_dataset_brands,
    _negative_control_outcome,
)
from .loaders import _get_causal_path_repo, _load_agent_estimation_frame

logger = logging.getLogger(__name__)

router = APIRouter()


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

# Cross-worker job store (Redis-backed; mirrors agent._agent_analysis_store). Each job
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
                    auto_discover=_default_auto_discover(dataset),
                    brand=q_brand,
                )
                await _agent._agent_analysis_store.set(
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
                await _agent._run_agent_analysis_task(aid, req, df, resolved_cov, data_source)
                resp = await _agent._agent_analysis_store.get(aid)
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
