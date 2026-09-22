"""
Proposed experiments from digital-twin simulations (#2206, owner item C)
=======================================================================

Owner directive on the fidelity chain's honest "skipped": **surface proposed
experiments**. Reasoned from the live data, not invented:

- There is no experiment-design table. An experiment is an ``ml_experiments``
  row; its ``status`` CHECK allows ``'draft'`` (migration 061) and nothing has
  ever written it. In real mode there are ZERO A/B experiments (every
  ``is_synthetic=false`` row is pipeline lineage; the 360 running rows are
  synthetic) — that is why the chain skips.
- What DOES exist: completed ``twin_simulations`` whose recommendation is
  ``deploy`` or ``refine``, each carrying a recommended sample size, duration,
  ATE and interval, none linked to an experiment. A twin run that ends in
  ``deploy`` IS the system proposing an experiment with concrete parameters.

So a **proposed experiment = a completed twin simulation with recommendation
deploy/refine and no ``experiment_design_id``**. This module lists them
(brand-grant-filtered, H11) and offers the ONE honest path to a real
experiment: a ``status='draft'`` ``ml_experiments`` row linked to the
simulation, which the owner promotes. Nothing auto-runs; every live
"active experiments" predicate selects ``status='running'``, so a draft never
counts anywhere.

Separate module (the ``/digital-twin`` route module is size-pinned by the
ratchet); included into that module's router, so it mounts under the same
``/api/digital-twin`` prefix without a new entry in ``src.api.main``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies.auth import (
    is_cross_brand_admin,
    require_operator,
    require_viewer,
    resolve_brand_for_read,
)
from src.api.routes.digital_twin_honesty import _stored_fidelity_fields
from src.api.schemas.digital_twin import (
    BrandEnum,
    DraftExperimentResponse,
    FidelityStatusEnum,
    ProposedExperimentItem,
    ProposedExperimentsResponse,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.data.per_hcp_cohort_columns import COHORT_OUTCOME_COLUMN

logger = logging.getLogger(__name__)

# No prefix/tags of its own: included into the /digital-twin router, whose
# prefix, tag and error responses apply to these routes.
router = APIRouter(
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

#: The lifecycle status a proposal becomes. Explicit on the INSERT: the
#: repository's ``create_experiment(status=None)`` inherits the DB default
#: ``'running'``, which would make a proposal count as an active experiment.
DRAFT_STATUS = "draft"
PROPOSAL_RECOMMENDATIONS = ("deploy", "refine")
NEXT_STEP = (
    "Still manual: promote this draft to 'running' and enroll units. The daily "
    "sweep, the final analysis and the fidelity roll-up then close the loop against "
    "the linked simulation."
)
OUTCOME_NOT_MEASURABLE = (
    " Note: {column} is recorded only on the synthetic-gold cohort rows today, so the "
    "real-mode final analysis of this draft will report insufficient_data until a real "
    "per-HCP endpoint is recorded (owner decision)."
)


def _proposal_sort_key(row: Dict[str, Any]) -> tuple:
    # deploy first, then the largest predicted effect.
    return (
        0 if row.get("recommendation") == "deploy" else 1,
        -float(row.get("simulated_ate") or 0),
    )


async def _twin_repo() -> Any:
    # Resolved through the route module so its test seam (patching
    # ``digital_twin._get_twin_repo``) covers these handlers too.
    from src.api.routes import digital_twin as _dt

    return await _dt._get_twin_repo()


async def _count_real_running_experiments(client: Any, brand: Optional[str]) -> int:
    """The real A/B portfolio: the Home tile's predicate, scoped to the caller's brand
    grant like the other counts (codex r1 #6). REAL means is_synthetic=false here
    regardless of deployment mode — apply_provenance_filter drops that predicate on a
    showcase instance (E2I_INCLUDE_SYNTHETIC), which would count the 360 synthetic
    running rows as real (codex r2 #1)."""
    query = (
        client.table("ml_experiments")
        .select("id", count="exact")
        .eq("status", "running")
        .not_.is_("intervention_channel", "null")
        .eq("is_synthetic", False)
    )
    if brand:
        query = query.eq("brand", brand)
    result = await query.execute()
    return int(result.count or 0)


async def _outcome_measurable_in_real_mode(client: Any) -> bool:
    """Whether any REAL per-HCP row records the twin's outcome column (codex r1 #1).

    Measured live 2026-09-22: every per_hcp_rollup row carrying
    cohort_conversion_outcome is is_synthetic=true (13,797 rows), and the real-mode
    outcome feed (ExperimentOutcomeRepository.load_arrays) excludes synthetic rows —
    so a real draft's final analysis cannot measure it today. Counted, never assumed.
    """
    # is_synthetic=false stated directly, not via apply_provenance_filter, which
    # skips the predicate on a showcase deployment (codex r2 #1).
    query = (
        client.table("business_metrics")
        .select("hcp_id", count="exact")
        .eq("metric_type", "per_hcp_rollup")
        .not_.is_(COHORT_OUTCOME_COLUMN, "null")
        .eq("is_synthetic", False)
        .limit(1)
    )
    result = await query.execute()
    return int(result.count or 0) > 0


def _caller_identity(user: Dict[str, Any]) -> Optional[str]:
    return user.get("email") or user.get("sub") or user.get("id")


@router.get(
    "/proposed-experiments",
    response_model=ProposedExperimentsResponse,
    summary="List proposed experiments",
    operation_id="list_proposed_experiments",
)
async def list_proposed_experiments(
    brand: Optional[BrandEnum] = Query(
        None, description="Filter by brand (omit for all you may see)"
    ),
    user: Dict[str, Any] = Depends(require_viewer),
) -> ProposedExperimentsResponse:
    """Completed twin simulations recommending deploy/refine, not yet linked to an
    experiment — ordered deploy first, then predicted effect — with the honest
    counts around them (linked simulations; real experiments running, 0 today)."""
    allowed, effective_brand = resolve_brand_for_read(user, brand.value if brand else None)
    if not allowed:
        raise HTTPException(status_code=403, detail="Brand not permitted for this user.")

    try:
        repo = await _twin_repo()
        rows = await repo.list_proposed_experiments(brand=effective_brand)
        linked = await repo.count_linked_simulations(brand=effective_brand)
        running = await _count_real_running_experiments(repo.client, effective_brand)
        measurable = await _outcome_measurable_in_real_mode(repo.client)

        # One STRICT model lookup per distinct model (codex r2 #2): a missing row
        # validates nothing; a failed lookup is a 500, never an 'unvalidated' label.
        model_rows: Dict[str, Optional[Dict[str, Any]]] = {}
        for row in rows:
            model_id = str(row.get("model_id") or "")
            if model_id and model_id not in model_rows:
                model_rows[model_id] = await repo.require_model(UUID(model_id))

        items: List[ProposedExperimentItem] = []
        for row in sorted(rows, key=_proposal_sort_key):
            fidelity = _stored_fidelity_fields(model_rows.get(str(row.get("model_id") or "")))
            items.append(
                ProposedExperimentItem(
                    simulation_id=str(row.get("simulation_id", "")),
                    model_id=str(row.get("model_id", "")),
                    brand=str(row.get("brand", "unknown")),
                    intervention_type=str(row.get("intervention_type", "unknown")),
                    intervention_config=row.get("intervention_config") or {},
                    simulated_ate=round(float(row.get("simulated_ate") or 0.0), 4),
                    simulated_ci_lower=_round4(row.get("simulated_ci_lower")),
                    simulated_ci_upper=_round4(row.get("simulated_ci_upper")),
                    recommendation=row.get("recommendation"),
                    recommendation_rationale=str(row.get("recommendation_rationale") or ""),
                    recommended_sample_size=row.get("recommended_sample_size"),
                    recommended_duration_weeks=row.get("recommended_duration_weeks"),
                    simulation_confidence=_round4(row.get("simulation_confidence")),
                    data_provenance=row.get("data_provenance"),
                    fidelity_status=FidelityStatusEnum(fidelity["fidelity_status"].value),
                    created_at=row.get("created_at"),
                    outcome_column=COHORT_OUTCOME_COLUMN,
                )
            )

        return ProposedExperimentsResponse(
            proposals=items,
            outcome_column=COHORT_OUTCOME_COLUMN,
            outcome_measurable_in_real_mode=measurable,
            total_proposed=len(items),
            total_linked=linked,
            real_experiments_running=running,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list proposed experiments: {e}")
        raise HTTPException(status_code=500, detail="Failed to list proposed experiments")


@router.post(
    "/proposed-experiments/{simulation_id}/draft",
    response_model=DraftExperimentResponse,
    status_code=201,
    summary="Create a draft experiment from a proposal",
    operation_id="create_draft_experiment_from_proposal",
    responses={
        404: {"model": ErrorResponse, "description": "Simulation not found"},
        409: {"model": ErrorResponse, "description": "Simulation already linked to an experiment"},
    },
)
async def create_draft_experiment(
    simulation_id: str,
    user: Dict[str, Any] = Depends(require_operator),
) -> DraftExperimentResponse:
    """Create ONE ``ml_experiments`` row with ``status='draft'`` from the twin's
    recommended parameters and link the simulation to it. The draft stays a draft
    until promoted; a failed link is a 500 naming both ids, never a 200 hiding an
    orphan (the ``/simulate`` rule)."""
    try:
        sim_uuid = UUID(str(simulation_id))
    except ValueError as bad:
        raise HTTPException(status_code=422, detail="simulation_id must be a UUID.") from bad

    repo = await _twin_repo()
    # Strict read (codex r2 #2): None only when there is no such row; a failure
    # is a 500, never a false 404.
    try:
        sim = await repo.require_simulation(sim_uuid)
    except Exception as e:
        logger.error(f"Simulation lookup failed for {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read the simulation")
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")

    # Fail-closed ownership (H11): 404, not 403, so existence is not leaked.
    sim_brand = sim.get("brand")
    if not is_cross_brand_admin(user) and (
        sim_brand is None or not resolve_brand_for_read(user, sim_brand)[0]
    ):
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")

    existing = sim.get("experiment_design_id")
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Simulation {simulation_id} is already linked to experiment {existing}.",
        )
    if sim.get("simulation_status") != "completed" or (
        sim.get("recommendation") not in PROPOSAL_RECOMMENDATIONS
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Only a completed simulation recommending deploy or refine proposes an "
                f"experiment (this one: status={sim.get('simulation_status')!r}, "
                f"recommendation={sim.get('recommendation')!r})."
            ),
        )

    intervention_type = str(sim.get("intervention_type") or "unknown")
    weeks = sim.get("recommended_duration_weeks")
    # Every diagnostic read happens BEFORE the mutation (codex r2 #5): a read that
    # fails after the committed draft would turn a success into a 500.
    try:
        model_row = (
            await repo.require_model(UUID(str(sim["model_id"]))) if sim.get("model_id") else None
        )
        measurable = await _outcome_measurable_in_real_mode(repo.client)
    except Exception as e:
        logger.error(f"Pre-draft reads failed for simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read the model / outcome state")
    fidelity = _stored_fidelity_fields(model_row)
    ate = float(sim.get("simulated_ate") or 0.0)
    ci_lo, ci_hi = sim.get("simulated_ci_lower"), sim.get("simulated_ci_upper")
    experiment_name = f"twin_proposal_{sim_brand}_{intervention_type}_{str(sim_uuid)[:8]}"
    description = (
        f"Proposed by digital-twin simulation {sim_uuid} ({sim.get('recommendation')}): "
        f"{sim.get('recommendation_rationale') or 'no rationale recorded'}. "
        f"Predicted ATE {ate:.4f}"
        + (
            f" [{float(ci_lo):.4f}, {float(ci_hi):.4f}]"
            if ci_lo is not None and ci_hi is not None
            else ""
        )
        + f" on {COHORT_OUTCOME_COLUMN}; model fidelity {fidelity['fidelity_status'].value}."
    )
    # Written directly (the ``prediction_synthesizer_deploy`` precedent):
    # ``MLExperimentRepository.create_experiment`` cannot carry the channel /
    # enrollment-plan columns and defaults status to the DB's 'running'.
    # ``mlflow_experiment_id`` (nullable, UNIQUE) is not invented; ``is_synthetic``
    # is not sent so the NOT NULL DEFAULT false makes the row real-mode visible.
    data: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "description": description,
        "prediction_target": COHORT_OUTCOME_COLUMN,
        "brand": sim_brand,
        "created_by": _caller_identity(user),
        "status": DRAFT_STATUS,
        "intervention_channel": intervention_type,
        "target_enrollment": sim.get("recommended_sample_size"),
        "planned_duration_days": int(weeks) * 7 if weeks is not None else None,
    }
    try:
        result = await repo.client.table("ml_experiments").insert(data).execute()
    except Exception as e:
        logger.error(f"Draft experiment insert failed for simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create the draft experiment")
    rows = getattr(result, "data", None) or []
    if not rows or not rows[0].get("id"):
        raise HTTPException(
            status_code=500,
            detail="The draft experiment insert returned no row; nothing was linked.",
        )
    experiment_id = UUID(str(rows[0]["id"]))

    # The link is a CLAIM (codex r1 #2): UPDATE … WHERE experiment_design_id IS NULL.
    # Two concurrent drafts both pass the pre-check and both insert; exactly one
    # claim succeeds. The loser removes the draft it just inserted and answers 409
    # naming the winner. A claim that RAISES leaves a draft the caller is told about.
    try:
        claimed = await repo.claim_experiment_link(sim_uuid, experiment_id)
    except Exception as e:
        logger.error(f"Claiming simulation {simulation_id} for draft {experiment_id} failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Draft experiment {experiment_id} was created but simulation {simulation_id} "
                "could not be linked to it; link it before relying on post-experiment "
                "fidelity tracking."
            ),
        )
    if not claimed:
        # The 409 must PROVE both halves (codex r2 #3): the orphan is gone AND the
        # winner is visible. Anything less is a 500 naming the unresolved draft.
        try:
            removed = await _delete_orphan_draft(repo.client, experiment_id)
            current = await repo.require_simulation(sim_uuid)
        except Exception as e:
            logger.error(f"Post-lost-claim cleanup failed for draft {experiment_id}: {e}")
            removed, current = False, None
        winner = (current or {}).get("experiment_design_id")
        if not removed or not winner:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Simulation {simulation_id} was claimed by a concurrent request, and the "
                    f"draft {experiment_id} created here "
                    + ("could not be removed" if not removed else "was removed")
                    + (
                        "; the winning experiment could not be read"
                        if not winner
                        else f"; it is linked to experiment {winner}"
                    )
                    + ". Resolve the draft by hand."
                ),
            )
        raise HTTPException(
            status_code=409,
            detail=(
                f"Simulation {simulation_id} was linked to experiment {winner} by a concurrent "
                f"request; the draft {experiment_id} created here was removed."
            ),
        )

    return DraftExperimentResponse(
        experiment_id=str(experiment_id),
        simulation_id=str(sim_uuid),
        experiment_name=experiment_name,
        brand=str(sim_brand),
        intervention_channel=intervention_type,
        prediction_target=COHORT_OUTCOME_COLUMN,
        target_enrollment=data["target_enrollment"],
        planned_duration_days=data["planned_duration_days"],
        created_by=data["created_by"],
        outcome_column=COHORT_OUTCOME_COLUMN,
        outcome_measurable_in_real_mode=measurable,
        linked=True,
        next_step=NEXT_STEP
        + ("" if measurable else OUTCOME_NOT_MEASURABLE.format(column=COHORT_OUTCOME_COLUMN)),
    )


async def _delete_orphan_draft(client: Any, experiment_id: UUID) -> bool:
    """Remove the draft a lost claim just inserted. True only when a row was deleted
    (PostgREST returns the deleted representation); a failure propagates to the
    caller, which reports the unresolved draft (codex r2 #3)."""
    result = await (
        client.table("ml_experiments")
        .delete()
        .eq("id", str(experiment_id))
        .eq("status", DRAFT_STATUS)
        .execute()
    )
    return bool(getattr(result, "data", None))


def _round4(value: Any) -> Optional[float]:
    return None if value is None else round(float(value), 4)
