"""
A/B → Digital Twin fidelity loop (#2206)
========================================

The post-experiment hops that feed ``digital_twin_models.fidelity_score``:

- ``_enqueue_final_analysis_on_stop`` / ``_reconcile_final_analysis`` — a persisted
  stopping decision produces ``compute_experiment_results(final)`` exactly once, and
  every daily sweep re-drives BOTH hops from their durable records (a FINAL row with
  no fidelity comparison enqueues ``fidelity_tracking_update`` directly).
- ``fidelity_tracking_update`` — compares the FINAL result with the experiment-linked
  twin simulation, persists the comparison, and rolls it up into the model's
  fidelity (``_roll_up_model_fidelity``). Routed to ``analytics`` (worker_medium).

Split out of ``ab_testing_tasks.py`` by the module-size ratchet; that module
re-exports these names. ``load_config`` / ``run_async`` are imported lazily from it
to avoid an import cycle.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, cast
from uuid import UUID

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# Decisions that end enrollment: the sequential test says the experiment is over.
_STOPPING_DECISIONS = frozenset({"stop_efficacy", "stop_futility", "stop_safety"})


async def _reconcile_final_analysis(experiment_id: UUID, previous_analyses: Any) -> Dict[str, bool]:
    """Re-drive BOTH post-stop hops from their durable records (#2206, codex r7/r8).

    A persisted stopping decision with no FINAL result row → enqueue
    compute_experiment_results(final). A FINAL row with no fidelity comparison
    for the experiment → enqueue fidelity_tracking_update directly (it skips
    until a simulation is linked; the next sweep tries again). Broker failures
    are logged and reported False; nothing here is one-shot.
    """
    out = {"final_analysis_enqueued": False, "fidelity_tracking_enqueued": False}
    decision = next(
        (
            getattr(a, "decision", None)
            for a in (previous_analyses or [])
            if getattr(getattr(a, "decision", None), "value", getattr(a, "decision", None))
            in _STOPPING_DECISIONS
        ),
        None,
    )
    if decision is None:
        return out
    from src.repositories.ab_results import ABResultsRepository

    repo = ABResultsRepository()
    if not await repo.get_results(experiment_id, analysis_type="final"):
        out["final_analysis_enqueued"] = await _enqueue_final_analysis_on_stop(
            experiment_id, decision
        )
        return out
    if await repo.get_fidelity_comparisons(experiment_id, limit=1):
        return out
    try:
        celery_app.send_task("src.tasks.fidelity_tracking_update", args=[str(experiment_id)])
        out["fidelity_tracking_enqueued"] = True
    except Exception as enqueue_err:
        logger.warning(
            "Could not enqueue fidelity_tracking_update for %s during reconciliation: %s",
            experiment_id,
            enqueue_err,
        )
    return out


async def _enqueue_final_analysis_on_stop(experiment_id: UUID, decision: Any) -> bool:
    """Producer for ``compute_experiment_results(final)`` (#2206).

    Fires only on a stopping decision and only when no FINAL result row exists yet
    (idempotent across the daily sweep). Best-effort: a broker failure must not
    fail the interim analysis that produced the decision.
    """
    value = getattr(decision, "value", decision)
    if value not in _STOPPING_DECISIONS:
        return False
    from src.repositories.ab_results import ABResultsRepository

    existing = await ABResultsRepository().get_results(experiment_id, analysis_type="final")
    if existing:
        return False
    try:
        celery_app.send_task(
            "src.tasks.compute_experiment_results",
            args=[str(experiment_id), "final"],
        )
    except Exception as enqueue_err:
        logger.warning(
            "Could not enqueue final analysis for %s after %s: %s",
            experiment_id,
            value,
            enqueue_err,
        )
        return False
    return True


async def _roll_up_model_fidelity(twin_simulation_id: UUID) -> Optional[Dict[str, Any]]:
    """Refresh the model behind ``twin_simulation_id`` from its A/B comparisons (#2206).

    Returns the written ``{model_id, fidelity_score, sample_count}``, or ``None`` when
    the simulation/model could not be resolved or no comparison carries a score. A
    failure here is logged, never raised: the comparison itself is already persisted.
    """
    try:
        from src.digital_twin.twin_repository import TwinRepository
        from src.memory.services.factories import get_async_supabase_client

        repo = TwinRepository(supabase_client=await get_async_supabase_client())
        sim = await repo.get_simulation(twin_simulation_id)
        model_id = (sim or {}).get("model_id")
        if not model_id:
            logger.warning(
                "No model behind twin simulation %s; fidelity not rolled up", twin_simulation_id
            )
            return None
        return await repo.refresh_model_fidelity_from_comparisons(UUID(str(model_id)))
    except Exception as exc:
        logger.error("Model fidelity roll-up failed for simulation %s: %s", twin_simulation_id, exc)
        return None


@celery_app.task(bind=True, name="src.tasks.fidelity_tracking_update")
def fidelity_tracking_update(
    self,
    experiment_id: str,
    twin_simulation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Update fidelity comparison with Digital Twin predictions.

    Compares actual experiment results with Digital Twin predictions
    to track simulation accuracy and identify calibration needs.

    Args:
        experiment_id: UUID of the experiment
        twin_simulation_id: Optional specific simulation to compare against

    Returns:
        Fidelity comparison results
    """
    logger.info(
        f"Updating fidelity tracking for experiment {experiment_id}: task {self.request.id}"
    )

    from src.tasks.ab_testing_tasks import load_config, run_async  # lazy: no import cycle

    config = load_config()
    fidelity_config = config.get("fidelity", {})
    start_time = time.time()

    async def execute_update():
        from src.services.results_analysis import ResultsAnalysisService

        try:
            results_service = ResultsAnalysisService()
            exp_uuid = UUID(experiment_id)

            # On-demand calibration: compare actual results with the twin's
            # predicted effect/CI. compare_experiment_to_twin (R1) does the fetch
            # from the REAL twin columns (simulated_ate / _ci_*) and raises
            # ValueError when results or a twin simulation are absent. The prior
            # inline query hit non-existent columns (id/predicted_effect/
            # confidence_interval), left predicted_effect/predicted_ci UNBOUND on
            # the explicit-id branch, and passed predicted_ci= (wrong arg name) —
            # all eliminated by routing through the convenience method (#705 H9).
            sim_uuid = UUID(twin_simulation_id) if twin_simulation_id else None
            try:
                comparison = await results_service.compare_experiment_to_twin(
                    experiment_id=exp_uuid,
                    twin_simulation_id=sim_uuid,  # None → resolves the linked sim
                    analysis_type="final",  # never an interim row (#2206)
                )
            except ValueError as ve:
                return {
                    "status": "skipped",
                    "experiment_id": experiment_id,
                    "reason": str(ve),
                }

            # Check if calibration is needed
            fidelity_config.get("acceptable_error", 0.2)
            calibration_trigger = fidelity_config.get("calibration_trigger_error", 0.3)

            calibration_needed = abs(comparison.prediction_error) > calibration_trigger

            if calibration_needed:
                logger.warning(
                    f"Digital Twin calibration needed for experiment {experiment_id}: "
                    f"prediction error = {comparison.prediction_error:.2%}"
                )

            # Close the loop (#2206): the comparison row alone never changed
            # digital_twin_models.fidelity_score (update_fidelity_score had no
            # caller), so the engine's gate read NULL forever. Refresh the model's
            # score from every comparison of that model's simulations.
            model_fidelity = await _roll_up_model_fidelity(comparison.twin_simulation_id)

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "completed",
                "experiment_id": experiment_id,
                "twin_simulation_id": str(comparison.twin_simulation_id),
                "predicted_effect": comparison.predicted_effect,
                "actual_effect": comparison.actual_effect,
                "prediction_error": comparison.prediction_error,
                # Real FidelityComparison field is ci_coverage (not the non-existent
                # confidence_interval_coverage, which AttributeError'd) (#705 H9).
                "ci_coverage": comparison.ci_coverage,
                "fidelity_score": comparison.fidelity_score,
                "calibration_needed": calibration_needed,
                "calibration_adjustment": comparison.calibration_adjustment,
                "model_fidelity": model_fidelity,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"Fidelity tracking update failed for {experiment_id}: {e}")
            return {
                "status": "failed",
                "experiment_id": experiment_id,
                "error": str(e),
            }

    return cast(Dict[str, Any], run_async(execute_update()))
