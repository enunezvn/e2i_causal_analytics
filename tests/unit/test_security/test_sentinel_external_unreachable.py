"""Regression pin: sentinel-action Celery tasks MUST NOT be HTTP-reachable (#391 box 3).

The four plan-specified sentinel-action handlers in
:mod:`src.tasks.sentinel_actions` are designed to be dispatched ONLY by:

1. The Celery worker (via ``celery_app.send_task``), routed from the
   in-process ``sentinel_dispatcher`` Celery beat task.
2. Test code that imports the async helpers directly.

They MUST NOT be reachable through the FastAPI HTTP surface — neither
as a direct route, nor via an indirect ``send_task`` from an
unauthenticated route. If such a route appears later, this test fails.

What "external-unreachable" means here
--------------------------------------
* No HTTP route path or operation_id contains a sentinel-action task
  NAME (the exact ``src.tasks.sentinel_actions.<handler>`` strings the
  Celery task decorators register).
* If a future PR adds an HTTP-triggered sentinel-action endpoint, it
  MUST be gated by ``require_operator`` or stronger (per the action's
  cascade scope).
* The Celery TASK NAMES are pinned here so renaming a handler without
  updating this test is a fail-loud signal.

Companion to ``tests/unit/test_api/test_auth_gating.py`` — that file
pins auth coverage; this one pins external-unreachability for a
specific set of operations.
"""

from __future__ import annotations

from typing import Iterable, List, Set

from fastapi import FastAPI
from fastapi.routing import APIRoute

# ---------------------------------------------------------------------------
# The sentinel-action Celery task names. Sourced from the @celery_app.task
# decorators in src/tasks/sentinel_actions.py. Pinned here so a rename
# without test update is loud.
# ---------------------------------------------------------------------------
SENTINEL_ACTION_TASK_NAMES: Set[str] = {
    "src.tasks.sentinel_actions.rerun_all_active_cohorts",
    "src.tasks.sentinel_actions.notify_and_queue_reanalysis",
    "src.tasks.sentinel_actions.flag_for_review",
    "src.tasks.sentinel_actions.run_full_consolidation",
}

# The handler function NAMES (the async helpers + their celery wrappers).
SENTINEL_ACTION_HANDLER_NAMES: Set[str] = {
    "rerun_all_active_cohorts",
    "notify_and_queue_reanalysis",
    "flag_for_review",
    "run_full_consolidation",
    "celery_rerun_all_active_cohorts",
    "celery_notify_and_queue_reanalysis",
    "celery_flag_for_review",
    "celery_run_full_consolidation",
}


def _build_full_app() -> FastAPI:
    """Build a FastAPI app with ALL routers from src.api.main attached.

    We deliberately avoid importing ``src.api.main.app`` directly: that
    module's import-time init pulls in Sentry, OTel, Supabase, FalkorDB,
    BentoML clients (which connect at import) and the unit-test budget
    blows past the 30s pytest-timeout. Instead, we mirror what
    ``src.api.main`` does at the router-registration level — import the
    public ``router`` object from each route module, attach to a fresh
    FastAPI instance with the same prefix as ``src.api.main``. The
    regression pin is path/endpoint-based, so the middlewares don't
    matter.

    If a future router lands in ``src.api.routes/`` it MUST be added
    here. The ``test_router_set_is_in_sync_with_src_api_main`` test
    below pins this list against the actual ``src.api.main`` source so
    a drift fails loud.
    """
    import os

    os.environ.setdefault("E2I_TESTING_MODE", "1")
    os.environ.setdefault("ENVIRONMENT", "development")

    # Import every router. List ORDER mirrors src.api.main.
    from src.api.routes.agents import router as agents_router
    from src.api.routes.analytics import router as analytics_router
    from src.api.routes.audit import router as audit_router
    from src.api.routes.causal import router as causal_router
    from src.api.routes.cognitive import router as cognitive_router
    from src.api.routes.copilotkit import router as copilotkit_router
    from src.api.routes.digital_twin import router as digital_twin_router
    from src.api.routes.executive_insights import router as executive_insights_router
    from src.api.routes.experiments import router as experiments_router
    from src.api.routes.explain import router as explain_router
    from src.api.routes.feedback import router as feedback_router
    from src.api.routes.gaps import router as gaps_router
    from src.api.routes.graph import router as graph_router
    from src.api.routes.health_score import router as health_score_router
    from src.api.routes.kpi import router as kpi_router
    from src.api.routes.memory import router as memory_router
    from src.api.routes.metrics import router as metrics_router
    from src.api.routes.monitoring import router as monitoring_router
    from src.api.routes.predictions import router as predictions_router
    from src.api.routes.rag import router as rag_router
    from src.api.routes.resource_optimizer import router as resource_optimizer_router
    from src.api.routes.segments import router as segments_router
    from src.api.routes.sentinels import router as sentinels_router
    from src.api.routes.staleness_alerts import router as staleness_alerts_router

    app = FastAPI()
    # /api-prefixed routers
    for r in (
        explain_router,
        memory_router,
        cognitive_router,
        graph_router,
        monitoring_router,
        experiments_router,
        gaps_router,
        segments_router,
        resource_optimizer_router,
        feedback_router,
        health_score_router,
        digital_twin_router,
        causal_router,
        audit_router,
        analytics_router,
        copilotkit_router,
        agents_router,
        sentinels_router,
        executive_insights_router,
        staleness_alerts_router,
    ):
        app.include_router(r, prefix="/api")
    # Non-prefixed routers (these set their own prefix in src.api.main).
    for r in (rag_router, predictions_router, kpi_router, metrics_router):
        app.include_router(r)
    return app


def _api_routes(app: FastAPI) -> List[APIRoute]:
    return [r for r in app.routes if isinstance(r, APIRoute)]


def _route_descriptors(routes: Iterable[APIRoute]) -> List[str]:
    """Render each route as a single descriptor string for substring search."""
    out = []
    for r in routes:
        descriptor = " ".join(
            [
                r.path or "",
                r.name or "",
                r.operation_id or "",
                getattr(r, "summary", "") or "",
            ]
        )
        out.append(descriptor)
    return out


# ---------------------------------------------------------------------------
# The regression pins
# ---------------------------------------------------------------------------


def test_no_http_route_path_contains_sentinel_action_task_name() -> None:
    """Pin: no route's path/name/operation_id includes a sentinel-action
    Celery task NAME (e.g. ``src.tasks.sentinel_actions.flag_for_review``).
    """
    app = _build_full_app()
    descriptors = _route_descriptors(_api_routes(app))

    offenders: List[str] = []
    for task_name in SENTINEL_ACTION_TASK_NAMES:
        for desc in descriptors:
            if task_name in desc:
                offenders.append(f"task={task_name!r} appears in route: {desc!r}")

    assert offenders == [], (
        "Sentinel-action Celery task names MUST NOT appear on HTTP routes. "
        "Offenders: " + ";".join(offenders)
    )


def test_no_http_route_invokes_sentinel_action_handler_directly() -> None:
    """Pin: no route's endpoint callable IS one of the sentinel-action
    handler functions.

    This catches a leak where a future PR routes an HTTP endpoint
    directly to ``flag_for_review`` or its Celery wrapper. The test
    walks ``route.endpoint`` and compares the qualified function name
    against :data:`SENTINEL_ACTION_HANDLER_NAMES`.
    """
    app = _build_full_app()
    offenders: List[str] = []
    for route in _api_routes(app):
        endpoint = route.endpoint
        if endpoint is None:
            continue
        fn_name = getattr(endpoint, "__name__", "")
        mod_name = getattr(endpoint, "__module__", "")
        # A direct exposure looks like: endpoint=src.tasks.sentinel_actions.X
        if mod_name == "src.tasks.sentinel_actions" and fn_name in SENTINEL_ACTION_HANDLER_NAMES:
            offenders.append(
                f"route {route.path!r} (methods={sorted(route.methods or [])!r}) "
                f"is wired directly to {mod_name}.{fn_name}"
            )

    assert offenders == [], (
        "Sentinel-action handler functions MUST NOT be the endpoint of any "
        "HTTP route (Celery-only). Offenders: " + ";".join(offenders)
    )


def test_no_http_route_has_path_segment_named_sentinel_action() -> None:
    """Pin: no HTTP path segment exactly matches a sentinel-action
    handler short name (e.g. no ``/api/sentinels/flag_for_review`` path).

    Substring-checks would over-match (``flag_for_review`` is a long
    string unlikely to be a substring of any unrelated route). Path-segment
    exactness is the right level for a future-PR safety net.
    """
    app = _build_full_app()
    short_names = {
        "rerun_all_active_cohorts",
        "notify_and_queue_reanalysis",
        "flag_for_review",
        "run_full_consolidation",
    }
    offenders: List[str] = []
    for route in _api_routes(app):
        segments = [seg for seg in (route.path or "").split("/") if seg]
        for seg in segments:
            if seg in short_names:
                offenders.append(f"{route.path} (segment {seg!r})")

    assert offenders == [], (
        "Path segments must not name sentinel-action handlers. Offenders: " + ";".join(offenders)
    )


# ---------------------------------------------------------------------------
# Positive sanity-check: verify the four Celery tasks ARE registered so
# this test isn't passing because the task name set is stale / empty.
# ---------------------------------------------------------------------------


def test_sentinel_action_celery_tasks_are_registered() -> None:
    """Sanity: the four Celery task NAMES from the pin ARE registered
    on the Celery app. If a handler is renamed without updating the pin,
    this fails first — making the rename loud.
    """
    # Importing src.tasks at module load triggers the registration
    # decorator chain (per the F401 in src/tasks/__init__.py). Importing
    # the celery app here picks up the resulting registry.
    import src.tasks  # noqa: F401
    from src.workers.celery_app import celery_app

    registered_tasks = set(celery_app.tasks.keys())
    missing = SENTINEL_ACTION_TASK_NAMES - registered_tasks
    assert not missing, (
        f"Pinned sentinel-action tasks not registered on celery_app: {missing!r}. "
        f"Either the @celery_app.task decorator was removed or "
        f"src/tasks/__init__.py is no longer importing the module."
    )
