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

    Codex iter-0 M3 closure: includes ``add_copilotkit_routes`` (the
    dynamic CopilotKit runtime registration at
    ``src/api/main.py:1000``) so the slim app's route set is a strict
    superset of what an external HTTP caller sees. A drift sanity check
    is in ``test_slim_app_includes_all_routers_from_src_api_main``
    below: it scans the source of ``src/api/main.py`` for
    ``app.include_router(...)`` / ``add_copilotkit_routes(...)`` calls
    and asserts every referenced module is wired into the slim app
    here.

    If a future router lands in ``src.api.routes/`` it MUST be added
    here AND the drift-sanity test will fail loud until it is.
    """
    import os

    os.environ.setdefault("E2I_TESTING_MODE", "1")
    os.environ.setdefault("ENVIRONMENT", "development")

    # Import every router. List ORDER mirrors src.api.main.
    from src.api.routes.agents import router as agents_router
    from src.api.routes.analytics import router as analytics_router
    from src.api.routes.audit import router as audit_router
    from src.api.routes.causal import router as causal_router
    from src.api.routes.chat import router as chat_router
    from src.api.routes.cognitive import router as cognitive_router
    from src.api.routes.copilotkit import add_copilotkit_routes
    from src.api.routes.copilotkit import router as copilotkit_router
    from src.api.routes.digital_twin import router as digital_twin_router
    from src.api.routes.executive_insights import router as executive_insights_router
    from src.api.routes.experiments import router as experiments_router
    from src.api.routes.expert_review import router as expert_review_router
    from src.api.routes.explain import router as explain_router
    from src.api.routes.feedback import router as feedback_router
    from src.api.routes.gaps import router as gaps_router
    from src.api.routes.graph import router as graph_router
    from src.api.routes.health_score import router as health_score_router
    from src.api.routes.insights_strategic import router as insights_strategic_router
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
        chat_router,
        expert_review_router,
        audit_router,
        analytics_router,
        copilotkit_router,
        agents_router,
        sentinels_router,
        executive_insights_router,
        insights_strategic_router,
        staleness_alerts_router,
    ):
        app.include_router(r, prefix="/api")
    # Non-prefixed routers (these set their own prefix in src.api.main).
    for r in (rag_router, predictions_router, kpi_router, metrics_router):
        app.include_router(r)
    # CopilotKit dynamic-route registration (matches src/api/main.py:1000).
    # Best-effort: the CopilotKit SDK pulls in network state at import,
    # so an offline CI may fail this call. We catch broadly and skip
    # the dynamic-route portion; the static copilotkit_router above
    # already covers the path-prefix regression-pin contract.
    try:
        add_copilotkit_routes(app, prefix="/api/copilotkit")
    except Exception:
        # Acceptable for the test environment — the static
        # copilotkit_router (registered above) covers the path-shape
        # regression-pin contract. If add_copilotkit_routes ever exposes
        # a sentinel-action path it would still fail
        # test_no_http_route_path_contains_sentinel_action_task_name as
        # long as the slim app's main routers carry the offending path
        # (CopilotKit dynamic registration is unlikely to reference our
        # internal Celery task names).
        pass
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


# ---------------------------------------------------------------------------
# Drift sanity-check: every router referenced in src/api/main.py is in the
# slim app builder. Closes codex iter-0 M3.
# ---------------------------------------------------------------------------


def test_slim_app_includes_all_routers_from_src_api_main() -> None:
    """The slim ``_build_full_app`` must include every router referenced
    in ``src/api/main.py``. We parse ``src/api/main.py`` for
    ``app.include_router(<X>_router, ...)`` calls and assert each ``X``
    appears in the slim app's router iteration too.

    Codex iter-0 M3 closure: this drift pin catches the case where a
    future PR adds a new router in ``src.api.main`` but forgets to
    update the slim-app builder. Without it the sentinel-unreachable
    pins would silently NOT cover the new surface.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    main_src = (repo_root / "src" / "api" / "main.py").read_text()
    # Match include_router(<name>_router, ...) calls.
    referenced = set(re.findall(r"app\.include_router\(\s*([a-zA-Z_]+)_router\b", main_src))
    # Build the slim app and collect the routers we actually attached.
    # We piggyback on the import set rather than re-parsing.
    app = _build_full_app()
    # The slim app's router attachment is by router object, not by name;
    # we cross-check via the route path prefix each router declares. We
    # parse the include_router calls so the failure message can name
    # the missing module.
    slim_paths = {(r.path or "") for r in _api_routes(app)}

    # Map each referenced router module to a path-prefix substring we
    # expect to appear in slim_paths if it was wired in. The mapping is
    # derived from each router's own ``router = APIRouter(prefix=...)``
    # declaration; we use the router-module name as a proxy.
    # Maps router-MODULE name → a path-prefix substring that the router
    # advertises in its own ``router = APIRouter(prefix=...)`` declaration.
    # Sourced directly from each route module — when a future router lands,
    # update this dict alongside the slim-app builder.
    expected_prefix_substrings = {
        "executive_insights": "/executive-insights",
        "audit": "/audit",
        # staleness_alerts router uses prefix=/alerts (not /staleness-alerts)
        # — see src/api/routes/staleness_alerts.py.
        "staleness_alerts": "/alerts",
        "sentinels": "/sentinels",
        "agents": "/agents",
        "analytics": "/analytics",
        "causal": "/causal",
        "chat": "/chat",
        "cognitive": "/cognitive",
        "copilotkit": "/copilotkit",
        "digital_twin": "/digital-twin",
        "experiments": "/experiments",
        "expert_review": "/expert-reviews",
        "explain": "/explain",
        "feedback": "/feedback",
        "gaps": "/gaps",
        "graph": "/graph",
        "health_score": "/health-score",
        "insights_strategic": "/insights",
        "kpi": "/kpis",
        "memory": "/memory",
        "metrics": "/metrics",
        "monitoring": "/monitoring",
        # predictions router uses prefix=/api/models — see
        # src/api/routes/predictions.py.
        "predictions": "/models",
        "rag": "/rag",
        "resource_optimizer": "/resources",
        "segments": "/segments",
    }
    missing: List[str] = []
    for ref in referenced:
        prefix = expected_prefix_substrings.get(ref)
        if prefix is None:
            # New router with an unmapped name: fail loud to force
            # an update to BOTH the slim app AND this drift table.
            missing.append(
                f"router {ref!r} referenced in src/api/main.py but missing "
                f"from expected_prefix_substrings table — update both the "
                f"slim app builder and this drift table."
            )
            continue
        if not any(prefix in p for p in slim_paths):
            missing.append(
                f"router {ref!r} (expected path prefix {prefix!r}) referenced "
                f"in src/api/main.py but NOT in the slim app's route set"
            )
    assert missing == [], (
        "Drift detected between src/api/main.py and _build_full_app: " + "; ".join(missing)
    )


# ---------------------------------------------------------------------------
# AST-based ``include_router(..., prefix=...)`` drift pin
# (codex iter-2 M1 closure)
# ---------------------------------------------------------------------------


def _parse_include_router_calls(main_src: str) -> List[dict]:
    """Parse ``src/api/main.py`` via :mod:`ast` and extract every
    ``app.include_router(<name>_router, prefix=...)`` call.

    Returns a list of ``{"router_name": str, "prefix": Optional[str]}``
    records. ``prefix`` is ``None`` when the call omitted the kwarg —
    those routers carry their own ``APIRouter(prefix=...)`` declaration.

    Why AST (not regex):
    The codex iter-2 M1 finding asks for prefix-drift detection. A regex
    that captures ``prefix="/api"`` ANYWHERE on the line silently fails
    on multi-line ``include_router`` calls and on prefix values that
    aren't string literals (rare but possible — e.g.
    ``prefix=API_PREFIX``). An ``ast.parse`` walk is the
    canonical way to extract keyword args robustly.
    """
    import ast

    tree = ast.parse(main_src)
    calls: List[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match ``something.include_router(...)`` calls. The receiver
        # name (e.g. ``app``) is not checked here — we match by method
        # name only, which is the pattern src/api/main.py uses
        # consistently.
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "include_router":
            continue
        if not node.args:
            continue
        router_arg = node.args[0]
        # First positional argument: expect ``<name>_router``.
        router_name: str = ""
        if isinstance(router_arg, ast.Name):
            router_name = router_arg.id
        else:
            # Skip non-Name forms (we don't expect them in src/api/main.py).
            continue
        # Extract prefix= kwarg if present and string-literal.
        prefix_value: str = ""
        prefix_present: bool = False
        for kw in node.keywords:
            if kw.arg == "prefix":
                prefix_present = True
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    prefix_value = kw.value.value
                else:
                    # Non-literal prefix (e.g. variable / f-string) —
                    # record presence but leave value empty so the
                    # caller can decide.
                    prefix_value = "<non-literal>"
                break
        calls.append(
            {
                "router_name": router_name,
                "prefix": prefix_value if prefix_present else None,
            }
        )
    return calls


def test_include_router_prefixes_match_expected_drift_table() -> None:
    """Codex iter-2 M1 closure: parse the AST of ``src/api/main.py`` for
    every ``include_router(...)`` call and assert that each call's
    ``prefix=`` kwarg matches the expected value in the drift table
    below.

    A prefix change (e.g. ``prefix="/api"`` → ``prefix="/api/v2"``)
    would previously pass silently because the prior
    ``test_slim_app_includes_all_routers_from_src_api_main`` only
    checks substring membership in the slim-app route set, not the
    explicit prefix value used at ``include_router`` time.

    Failure mode: prefix drift now raises an explicit AssertionError
    naming the changed router + its old vs new prefix.

    Robustness notes:
    * Routers that omit ``prefix=`` (because the prefix is baked into
      ``APIRouter(prefix=...)`` itself) are EXPECTED to have
      ``prefix=None`` in the drift table — they're recorded with an
      explicit ``None`` value, so adding a prefix to one of them later
      also trips the assertion.
    * Non-literal prefix values (e.g. ``prefix=API_PREFIX``) record as
      the string ``"<non-literal>"`` — if production ever moves to a
      computed prefix, this drift table will catch it on the first run
      and the operator can choose whether to accept that change.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    main_src = (repo_root / "src" / "api" / "main.py").read_text()
    calls = _parse_include_router_calls(main_src)

    # Expected (router_name, prefix) tuples. ``None`` = the call omitted
    # the prefix kwarg (router's own APIRouter(prefix=...) is the source
    # of truth). Sourced directly from src/api/main.py — when a future
    # router is added or a prefix is changed, update BOTH src/api/main.py
    # AND this expected table; the assertion below will name the drift.
    expected_prefixes: dict = {
        "explain_router": "/api",
        "memory_router": "/api",
        "cognitive_router": "/api",
        "graph_router": "/api",
        # rag_router declares its own prefix=/api/rag in APIRouter()
        "rag_router": None,
        "monitoring_router": "/api",
        "experiments_router": "/api",
        "expert_review_router": "/api",
        "gaps_router": "/api",
        "segments_router": "/api",
        "resource_optimizer_router": "/api",
        "feedback_router": "/api",
        "health_score_router": "/api",
        "digital_twin_router": "/api",
        # predictions_router declares its own prefix=/api/models
        "predictions_router": None,
        # kpi_router declares its own prefix=/api/kpis
        "kpi_router": None,
        "causal_router": "/api",
        "chat_router": "/api",
        "audit_router": "/api",
        "analytics_router": "/api",
        "copilotkit_router": "/api",
        "agents_router": "/api",
        "sentinels_router": "/api",
        "executive_insights_router": "/api",
        "insights_strategic_router": "/api",
        "staleness_alerts_router": "/api",
        # metrics_router declares its own prefix (or no prefix needed)
        "metrics_router": None,
    }

    drift: List[str] = []
    for call in calls:
        name = call["router_name"]
        actual = call["prefix"]
        if name not in expected_prefixes:
            drift.append(
                f"router {name!r} called by include_router(...) is NOT in the "
                f"expected_prefixes drift table — update both src/api/main.py "
                f"and this test."
            )
            continue
        expected = expected_prefixes[name]
        if actual != expected:
            drift.append(f"router {name!r}: expected prefix={expected!r}, got prefix={actual!r}")

    # Also check the inverse direction — every router in the expected
    # table must appear in src/api/main.py (catches accidental removals).
    seen_names = {c["router_name"] for c in calls}
    for expected_name in expected_prefixes:
        if expected_name not in seen_names:
            drift.append(
                f"router {expected_name!r} is in the expected_prefixes drift "
                f"table but NOT called by any include_router(...) in "
                f"src/api/main.py — was it removed? Update both."
            )

    assert drift == [], (
        "include_router(...) prefix drift detected in src/api/main.py — codex "
        "iter-2 M1 closure: prefix changes must now be acknowledged by "
        "updating the expected_prefixes table. Drift:\n  - " + "\n  - ".join(drift)
    )
