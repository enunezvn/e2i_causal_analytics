"""Shared heavy-compute helper for real-time SHAP explanation (P2 offload).

Single source of truth for the SHAP computation so the synchronous API path
(``src/api/routes/explain.py`` via ``RealTimeSHAPService.compute_shap``) and the
Celery ``compute_shap_values`` task run the *same* explainer call and return the
*same* JSON-safe scalar dict.

Offload boundary
----------------
Only the genuinely heavy/memory-bound SHAP compute moves to the worker:
``RealTimeSHAPExplainer.compute_shap_values(features, model_type,
model_version_id, top_k)`` -> a ``SHAPResult``. Feature retrieval (Feast),
prediction (BentoML), PII masking, narrative, and the audit-write
``background_task`` all stay on the API process — they are light and carry
request-scoped state (masked IDs, background tasks) that must not cross the
Celery boundary.

The returned dict matches the subset of ``SHAPResult`` the route consumes when
building ``FeatureContribution``s, so the route's response shape is unchanged on
both paths.

Heavy imports stay function-local so importing this module from the API process
(only to enqueue ``.apply_async()``) does not pull SHAP/model libs into the API
worker's memory budget.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def shap_result_to_dict(result: "Any") -> Dict[str, Any]:
    """Serialize the consumed subset of a ``SHAPResult`` to a JSON-safe dict.

    Matches what ``RealTimeSHAPService.compute_shap`` reads off the result:
    ``base_value``, ``shap_values`` (already top-k filtered by the explainer),
    ``explainer_type`` (enum -> its ``.value`` string), and
    ``computation_time_ms``.
    """
    return {
        "base_value": float(result.base_value),
        "shap_values": {k: float(v) for k, v in result.shap_values.items()},
        "explainer_type": result.explainer_type.value,
        "computation_time_ms": float(result.computation_time_ms),
    }


def run_shap_compute(
    *,
    features: Dict[str, Any],
    model_type: str,
    model_version_id: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the SHAP explainer for one instance, returning a JSON-safe dict.

    The explainer's ``compute_shap_values`` is async (it offloads the CPU-bound
    SHAP math to its own thread pool internally), so we drive it to completion on
    a private event loop. We run it on a dedicated worker thread rather than via
    ``asyncio.run`` so it is correct whether or not the *calling* thread already
    has a running loop (``asyncio.run`` raises "cannot be called from a running
    event loop"). The Celery ``worker_heavy`` process has no running loop, but a
    thread-isolated loop keeps this entrypoint safe under any caller (tests,
    eager mode, a future async caller).

    Args:
        features: Numeric feature dict for the instance (already prepared by the
            caller, mirroring ``_prepare_numeric_features``).
        model_type: ``ModelType`` value string.
        model_version_id: Resolved model version id.
        top_k: Optional top-k filter passed straight through to the explainer.

    Returns:
        JSON-safe dict (see :func:`shap_result_to_dict`).
    """
    from src.mlops.shap_explainer_realtime import RealTimeSHAPExplainer

    explainer = RealTimeSHAPExplainer()
    result = _run_coro_blocking(
        explainer.compute_shap_values(
            features=features,
            model_type=model_type,
            model_version_id=model_version_id,
            top_k=top_k,
        )
    )
    return shap_result_to_dict(result)


def _run_coro_blocking(coro: "Any") -> "Any":
    """Run an awaitable to completion on a private loop in a dedicated thread.

    Safe regardless of whether the calling thread already has a running event
    loop (``asyncio.run`` is not). The new thread has no loop, so a fresh one is
    created, used, and closed there.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _runner() -> "Any":
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()


def run_single_shap(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Celery-task entrypoint: single-instance SHAP from a JSON payload."""
    return run_shap_compute(
        features=dict(payload["features"]),
        model_type=str(payload["model_type"]),
        model_version_id=str(payload["model_version_id"]),
        top_k=payload.get("top_k"),
    )


__all__ = [
    "run_shap_compute",
    "run_single_shap",
    "shap_result_to_dict",
]
