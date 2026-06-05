"""Celery tasks for the P2 heavy-compute offload (DARK by default).

These tasks move the two genuinely heavy in-process API compute paths off the
gunicorn workers onto ``worker_heavy``:

* ``compute_shap_values`` (queue ``shap``) — real-time SHAP for one instance.
* ``simulate_population`` (queue ``twins``) — digital-twin population simulation.

The task NAMES match the pre-existing ``task_routes`` entries in
``src/workers/celery_app.py`` (``src.tasks.compute_shap_values`` ->
``shap``; ``src.tasks.simulate_population`` -> ``twins``) so routing works
without touching the routing table.

Both take JSON-serializable inputs and return JSON-serializable scalar dicts
(the Celery result backend serializer is ``json``). The API routes enqueue these
via ``apply_async(queue=...)`` only when ``HEAVY_OFFLOAD_ENABLED`` is set;
otherwise they keep the P1 inline path. The heavy compute itself lives in the
shared runners (``src.mlops.shap_runner`` / ``src.digital_twin.simulation_runner``)
so the inline and offloaded paths run identical logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="src.tasks.compute_shap_values")
def compute_shap_values(self: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute SHAP values for a single prediction on ``worker_heavy``.

    Args:
        payload: JSON dict with ``features`` (numeric), ``model_type``,
            ``model_version_id`` and optional ``top_k``.

    Returns:
        JSON-safe dict: ``base_value``, ``shap_values``, ``explainer_type``,
        ``computation_time_ms``.
    """
    logger.info(
        "compute_shap_values task start model_type=%s version=%s",
        payload.get("model_type"),
        payload.get("model_version_id"),
    )
    # Heavy import stays inside the runner (worker process only).
    from src.mlops.shap_runner import run_single_shap

    result = run_single_shap(payload)
    logger.info("compute_shap_values task done")
    return result


@celery_app.task(bind=True, name="src.tasks.simulate_population")
def simulate_population(self: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a digital-twin population simulation on ``worker_heavy``.

    Args:
        payload: JSON dict mirroring
            ``src.digital_twin.simulation_runner.run_simulation_compute`` kwargs.

    Returns:
        JSON-safe ``SimulationResult`` dict the API route rebuilds via
        ``simulation_result_from_dict``.
    """
    logger.info(
        "simulate_population task start brand=%s twin_count=%s",
        payload.get("brand_value"),
        payload.get("twin_count"),
    )
    from src.digital_twin.simulation_runner import run_population_simulation

    result = run_population_simulation(payload)
    logger.info("simulate_population task done")
    return result


async def _train_twin_model_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a fail-closed repo and run the offline twin training job."""
    from src.digital_twin.models.twin_models import Brand, TwinType
    from src.digital_twin.training_job import train_and_persist_twin
    from src.digital_twin.twin_repository import TwinRepository
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    repo = TwinRepository(supabase_client=client)
    return await train_and_persist_twin(
        twin_type=TwinType(payload["twin_type"]),
        brand=Brand(payload["brand"]),
        repo=repo,
        data_source=payload.get("data_source"),
        target_column=payload.get("target_column", "outcome"),
        algorithm=payload.get("algorithm", "random_forest"),
        synthetic=bool(payload.get("synthetic", False)),
        n_rows=int(payload.get("n_rows", 2000)),
        seed=int(payload.get("seed", 0)),
    )


@celery_app.task(bind=True, name="src.tasks.train_twin_model")
def train_twin_model(self: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Train + persist a digital-twin model on ``worker_heavy`` (queue ``ml``).

    Fills the pre-existing ``src.tasks.train_twin_model`` -> ``ml`` route. The
    worker ships dark on the 16 GB box, so the same training also runs inline via
    ``train_and_persist_twin`` (e.g. the admin ``/digital-twin/train`` endpoint).

    Args:
        payload: JSON dict with ``twin_type``, ``brand`` and optionally
            ``synthetic`` / ``data_source`` / ``target_column`` / ``algorithm`` /
            ``n_rows`` / ``seed``.

    Returns:
        JSON-safe training result (``model_id``, ``model_uri``, ``r2_score``,
        ``data_provenance`` …).
    """
    # Reuse the codebase's thread-local-loop helper (NOT bare asyncio.run): the
    # async Supabase client is process-cached and bound to the loop it was created
    # on, so asyncio.run's per-call create-then-close loop would raise "Event loop
    # is closed" on the 2nd+ task in a long-lived prefork worker. run_async reuses
    # a live thread-local loop, matching execute_twin_retraining. Function-local
    # import keeps the heavy ab_testing_tasks module off the API process.
    from src.tasks.ab_testing_tasks import run_async

    logger.info(
        "train_twin_model task start brand=%s twin_type=%s",
        payload.get("brand"),
        payload.get("twin_type"),
    )
    result = run_async(_train_twin_model_async(payload))
    logger.info("train_twin_model task done model=%s", result.get("model_id"))
    return result


__all__ = ["compute_shap_values", "simulate_population", "train_twin_model"]
