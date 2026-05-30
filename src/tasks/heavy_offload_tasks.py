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


__all__ = ["compute_shap_values", "simulate_population"]
