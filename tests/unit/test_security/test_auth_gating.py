"""Auth-gating regression pin for crystal/provenance API routers (#391 security box 2).

This test enumerates every route in the ``executive_insights``, ``audit``,
and ``staleness_alerts`` routers and asserts that EACH one carries a
``require_auth``-or-stronger dependency. A new route added later that
forgets ``Depends(require_auth)`` will FAIL this test — that's the
intended regression pin.

Why this shape (route-level dependency introspection)
-----------------------------------------------------
Inspecting routes through ``app.routes`` and walking ``route.dependant.
dependencies`` is the canonical FastAPI pattern for testing auth coverage.
It exercises the REAL ``Depends(...)`` graph, not a monkey-patched stub
(per memory feedback-test-must-exercise-real-catch-not-mock). The test
fails if a future PR adds a route without auth, even if its tests
individually pass.

For the HTTP-shape tests (401 without auth header, 200 with faked auth)
we use FastAPI's ``TestClient`` against an app with NO middleware-level
TESTING_MODE bypass — the route-level ``Depends(require_auth)`` is
what gates access. We DO temporarily disable the env-level testing-mode
shim by patching ``TESTING_MODE = False`` on the auth module so the
real dependency runs.

NOTE: ``tests/unit/test_api/conftest.py`` sets ``E2I_TESTING_MODE=1`` at
import time, so the auth dependency's TESTING_MODE branch is on by
default. The route-level introspection part of this test does NOT
depend on that flag (it just inspects the dependency graph). The
HTTP-shape part uses TestClient with auth NOT bypassed via a separate
fixture below.
"""

from __future__ import annotations

import os
from typing import List, Set

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app() -> FastAPI:
    """Build a FastAPI app with the three protected routers attached.

    Does NOT install ``JWTAuthMiddleware`` — the test exercises the
    route-level ``Depends(require_auth)`` directly (route-level deps
    are the authoritative gate; middleware is the cross-cutting backup).
    """
    from src.api.routes.audit import router as audit_router
    from src.api.routes.executive_insights import router as executive_insights_router
    from src.api.routes.staleness_alerts import router as staleness_alerts_router

    app = FastAPI()
    app.include_router(executive_insights_router, prefix="/api")
    app.include_router(audit_router, prefix="/api")
    app.include_router(staleness_alerts_router, prefix="/api")
    return app


def _route_has_auth_dependency(route: APIRoute) -> bool:
    """Return True iff this route's dependency tree includes one of the
    auth dependencies (``require_auth`` or any role-stronger gate).

    Walks the FastAPI ``dependant`` tree (deps-of-deps) and matches
    against the qualified function name. Stronger roles (operator/admin)
    transitively depend on ``require_auth`` so checking just the leaf
    names is enough, but we walk the whole tree to be defensive.
    """
    # Names of the four functions in src.api.dependencies.auth that
    # satisfy "this route requires authentication".
    auth_func_names = {
        "require_auth",
        "require_viewer",
        "require_analyst",
        "require_operator",
        "require_admin",
    }

    seen_funcs: Set[str] = set()
    queue = list(route.dependant.dependencies)
    while queue:
        dep = queue.pop()
        fn = getattr(dep, "call", None)
        if fn is not None:
            fn_name = getattr(fn, "__name__", "")
            seen_funcs.add(fn_name)
            if fn_name in auth_func_names:
                return True
        queue.extend(dep.dependencies)
    return False


def _api_routes(app: FastAPI) -> List[APIRoute]:
    """Return all APIRoute instances on the app (excludes Mount, etc.)."""
    return [r for r in app.routes if isinstance(r, APIRoute)]


# ---------------------------------------------------------------------------
# Route-level dependency-graph tests (the regression pin)
# ---------------------------------------------------------------------------


def test_all_executive_insights_routes_are_auth_gated() -> None:
    """Every route under ``/api/executive-insights*`` has a require_auth dep."""
    app = _build_app()
    unprotected: List[str] = []
    for route in _api_routes(app):
        if not route.path.startswith("/api/executive-insights"):
            continue
        if not _route_has_auth_dependency(route):
            unprotected.append(f"{','.join(sorted(route.methods or []))} {route.path}")
    assert unprotected == [], (
        "Found executive-insights routes without auth-gating: "
        f"{unprotected!r}. Every route in this router MUST declare "
        "Depends(require_auth) or stronger."
    )


def test_all_audit_routes_are_auth_gated() -> None:
    """Every route under ``/api/audit*`` has a require_auth dep."""
    app = _build_app()
    unprotected: List[str] = []
    for route in _api_routes(app):
        if not route.path.startswith("/api/audit"):
            continue
        if not _route_has_auth_dependency(route):
            unprotected.append(f"{','.join(sorted(route.methods or []))} {route.path}")
    assert unprotected == [], (
        "Found audit routes without auth-gating: "
        f"{unprotected!r}. Every route in this router MUST declare "
        "Depends(require_auth) or stronger."
    )


def test_all_staleness_alerts_routes_are_auth_gated() -> None:
    """Every route under ``/api/staleness-alerts*`` has a require_auth dep."""
    app = _build_app()
    unprotected: List[str] = []
    for route in _api_routes(app):
        if not route.path.startswith("/api/staleness-alerts"):
            continue
        if not _route_has_auth_dependency(route):
            unprotected.append(f"{','.join(sorted(route.methods or []))} {route.path}")
    assert unprotected == [], (
        "Found staleness-alerts routes without auth-gating: "
        f"{unprotected!r}. Every route in this router MUST declare "
        "Depends(require_auth) or stronger."
    )


def test_executive_insights_crystallize_requires_operator_not_just_auth() -> None:
    """``POST /api/executive-insights/crystallize`` MUST be operator-gated.

    Crystallization is a state-mutating write that triggers downstream
    cascades. Even though every authenticated user might be reading
    these endpoints, only operators should be able to trigger fresh
    crystallization runs. This pin keeps the gate from being weakened.
    """
    app = _build_app()
    target_routes = [
        r
        for r in _api_routes(app)
        if r.path == "/api/executive-insights/crystallize" and "POST" in (r.methods or set())
    ]
    assert target_routes, "POST /api/executive-insights/crystallize not found"
    for route in target_routes:
        # Walk dependants to find require_operator (or stronger: admin).
        operator_func_names = {"require_operator", "require_admin"}
        found = False
        queue = list(route.dependant.dependencies)
        while queue:
            dep = queue.pop()
            fn = getattr(dep, "call", None)
            if fn is not None and getattr(fn, "__name__", "") in operator_func_names:
                found = True
                break
            queue.extend(dep.dependencies)
        assert found, (
            "POST /api/executive-insights/crystallize MUST be gated by "
            "require_operator (or require_admin); the plain require_auth "
            "is too weak for a state-mutating write."
        )


# ---------------------------------------------------------------------------
# Enumeration sanity-check — fail-loud if a router goes silent
# ---------------------------------------------------------------------------


def test_executive_insights_router_exposes_expected_minimum_routes() -> None:
    """At least 4 routes are expected on the executive_insights router
    (list, portfolio-summary, get-one, crystallize). The exact set is
    pinned so a refactor that drops one of these would fail loud."""
    app = _build_app()
    paths = {r.path for r in _api_routes(app) if r.path.startswith("/api/executive-insights")}
    # NOTE: portfolio-summary MUST be declared BEFORE /{insight_id} in
    # the router. The pinned set below also ensures the route exists at
    # all (not just that the order is right).
    required = {
        "/api/executive-insights",
        "/api/executive-insights/portfolio-summary",
        "/api/executive-insights/{insight_id}",
        "/api/executive-insights/crystallize",
    }
    missing = required - paths
    assert not missing, (
        f"executive-insights router is missing expected routes: {missing!r}. "
        f"Found paths: {sorted(paths)!r}"
    )


def test_audit_router_exposes_expected_minimum_routes() -> None:
    """The audit router has at least 4 routes (workflow entries/verify/summary
    + recent)."""
    app = _build_app()
    paths = {r.path for r in _api_routes(app) if r.path.startswith("/api/audit")}
    required = {
        "/api/audit/workflow/{workflow_id}",
        "/api/audit/workflow/{workflow_id}/verify",
        "/api/audit/workflow/{workflow_id}/summary",
        "/api/audit/recent",
    }
    missing = required - paths
    assert not missing, (
        f"audit router is missing expected routes: {missing!r}. Found paths: {sorted(paths)!r}"
    )


# ---------------------------------------------------------------------------
# HTTP-shape tests — assert 401 when middleware sees no Authorization
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_real_auth_middleware(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Build an app with ``JWTAuthMiddleware`` AND testing-mode OFF.

    This forces the middleware to enforce JWT validation. Without an
    Authorization header, the middleware returns 401 BEFORE any route
    handler runs. We can then assert protected paths really are
    enforced and not just stamped with a dependency that the testing-
    mode shim short-circuits.

    Cleanup: ``monkeypatch.setenv`` + reload restore the prior state at
    test teardown.
    """
    # Disable testing-mode bypass for this fixture only.
    monkeypatch.delenv("E2I_TESTING_MODE", raising=False)

    # Re-import the auth modules so TESTING_MODE is re-evaluated.
    import importlib

    import src.api.dependencies.auth as auth_dep
    import src.api.middleware.auth_middleware as auth_mw

    # #1438: ``importlib.reload`` rebinds *every* symbol in the module's
    # __dict__ — including the ``AuthError`` class object. Any module that
    # imported ``AuthError`` by name BEFORE this reload (e.g.
    # ``src.api.routes.copilotkit``, pulled in earlier by the copilotkit
    # test files) keeps a frozen reference to the *original* class, while a
    # fresh ``from src.api.dependencies.auth import AuthError`` after the
    # reload returns the *new* one. ``pytest.raises(AuthError)`` in a later
    # test then cannot match an ``AuthError`` raised from copilotkit — a
    # cross-file leak that CI's loadscope sharding (one file per worker)
    # masks but a serial run exposes. Snapshot the pre-reload module dicts
    # so the teardown can restore them and the rebinding never escapes.
    _auth_dep_dict_snapshot = auth_dep.__dict__.copy()
    _auth_mw_dict_snapshot = auth_mw.__dict__.copy()

    importlib.reload(auth_dep)
    importlib.reload(auth_mw)

    from src.api.middleware.auth_middleware import JWTAuthMiddleware
    from src.api.routes.audit import router as audit_router
    from src.api.routes.executive_insights import router as executive_insights_router
    from src.api.routes.staleness_alerts import router as staleness_alerts_router

    app = FastAPI()
    app.add_middleware(JWTAuthMiddleware)
    app.include_router(executive_insights_router, prefix="/api")
    app.include_router(audit_router, prefix="/api")
    app.include_router(staleness_alerts_router, prefix="/api")

    yield app

    # Restore the exact pre-reload module state. This undoes the reload's
    # rebinding of ``AuthError`` et al. (so no other test sees a diverged
    # class identity) AND restores testing-mode for the tests that follow
    # — the snapshot was taken while ``TESTING_MODE`` was still on, before
    # the reload flipped it off. ``clear()`` + ``update()`` makes each
    # module's __dict__ byte-for-byte identical to its pre-fixture state.
    auth_dep.__dict__.clear()
    auth_dep.__dict__.update(_auth_dep_dict_snapshot)
    auth_mw.__dict__.clear()
    auth_mw.__dict__.update(_auth_mw_dict_snapshot)


def test_unauthenticated_request_to_executive_insights_returns_401(
    app_with_real_auth_middleware: FastAPI,
) -> None:
    """GET /api/executive-insights without auth header → 401."""
    from fastapi.testclient import TestClient

    with TestClient(app_with_real_auth_middleware) as client:
        response = client.get("/api/executive-insights")
        assert response.status_code == 401, (
            f"Expected 401 without auth header, got {response.status_code}: {response.text}"
        )


def test_unauthenticated_request_to_portfolio_summary_returns_401(
    app_with_real_auth_middleware: FastAPI,
) -> None:
    """GET /api/executive-insights/portfolio-summary without auth → 401."""
    from fastapi.testclient import TestClient

    with TestClient(app_with_real_auth_middleware) as client:
        response = client.get("/api/executive-insights/portfolio-summary")
        assert response.status_code == 401


def test_unauthenticated_request_to_audit_recent_returns_401(
    app_with_real_auth_middleware: FastAPI,
) -> None:
    """GET /api/audit/recent without auth → 401."""
    from fastapi.testclient import TestClient

    with TestClient(app_with_real_auth_middleware) as client:
        response = client.get("/api/audit/recent")
        assert response.status_code == 401


def test_unauthenticated_post_crystallize_returns_401(
    app_with_real_auth_middleware: FastAPI,
) -> None:
    """POST /api/executive-insights/crystallize without auth → 401."""
    from fastapi.testclient import TestClient

    with TestClient(app_with_real_auth_middleware) as client:
        response = client.post(
            "/api/executive-insights/crystallize",
            json={"brand": "kisqali"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# CopilotKit allowlist regression pin (#399 closure — was: auth-gap pin)
# ---------------------------------------------------------------------------


def test_copilotkit_probe_paths_pinned_in_public_paths() -> None:
    """REGRESSION PIN — CopilotKit SDK-probe surface MUST remain in
    ``PUBLIC_PATHS`` so the React provider can bootstrap without an
    authenticated session.

    This test replaces the prior ``test_copilotkit_auth_gap_pinned_for_issue_399``
    pin (which asserted the CURRENT-broken state pre-fix). #399 closed
    by removing the ``^/api/copilotkit(/.*)?$`` catch-all from
    ``PUBLIC_PATH_PATTERNS``; the three explicit allowlist entries
    survived because they're load-bearing for SDK initialization:

    * ``/api/copilotkit`` — base handler (root of the SDK runtime;
      registered at ``src/api/routes/copilotkit.py:2822`` via
      ``add_copilotkit_routes`` as ``copilotkit_handler_base``).
    * ``/api/copilotkit/status`` — health endpoint served by the
      static router (``@router.get("/status")`` at
      ``src/api/routes/copilotkit.py:2860``; no auth in the function
      signature, so its public-ness lives entirely in the middleware
      allowlist).
    * ``/api/copilotkit/info`` — discovery endpoint that returns
      available agents + actions; called by the CopilotKit React SDK
      at provider mount time before an access token may be available.

    If a future PR REMOVES one of these three entries, SDK
    initialization will silently start requiring auth and the chat UI
    will fail to bootstrap for unauthenticated users (e.g. on the
    public landing page if it exposes CopilotKit). This test catches
    that regression.

    Counterpart to ``test_copilotkit_dynamic_routes_require_auth``
    below, which asserts the inverse: every NON-probe sub-path
    requires auth.
    """
    from src.api.middleware.auth_middleware import PUBLIC_PATHS

    # Pin the EXACT three SDK-probe entries (mirror of the pre-#399
    # pin's expected_exact, but now asserting these are the ONLY
    # CopilotKit entries in PUBLIC_PATHS — i.e. the allowlist does not
    # creep wider over time).
    copilotkit_exact_entries = {
        (method, path)
        for method, path in PUBLIC_PATHS
        if path == "/api/copilotkit" or path.startswith("/api/copilotkit/")
    }
    expected_exact = {
        ("*", "/api/copilotkit"),
        # #399 iter-2: /status is GET-only public (the static router
        # has GET-only handler; POST falls through to the dynamic
        # catch-all which now requires auth at the middleware).
        ("GET", "/api/copilotkit/status"),
        ("*", "/api/copilotkit/info"),
    }
    missing_exact = expected_exact - copilotkit_exact_entries
    assert not missing_exact, (
        f"Expected the three pinned /api/copilotkit SDK-probe entries in "
        f"PUBLIC_PATHS (see docstring); missing: {missing_exact!r}. Current "
        f"copilotkit entries: {copilotkit_exact_entries!r}. If a probe surface "
        f"was removed intentionally, update this pin alongside the change."
    )
    # Stricter: ALSO assert no UNEXPECTED CopilotKit entries crept in.
    # Allowlist creep is the inverse failure mode of the prior #399 gap
    # and equally dangerous: a future PR that adds
    # ``("*", "/api/copilotkit/agent")`` would silently re-open the
    # dynamic-route bypass.
    extra_exact = copilotkit_exact_entries - expected_exact
    assert not extra_exact, (
        f"Unexpected CopilotKit entries in PUBLIC_PATHS: {extra_exact!r}. "
        f"The #399 closure narrowed the public surface to exactly the three "
        f"SDK-probe entries listed above. If a new public probe is genuinely "
        f"needed, update both this test and the docstring rationale."
    )


def test_copilotkit_static_routes_present_in_auth_test_app() -> None:
    """Codex iter-2 M2 closure (iter-3 LOW-2 strengthened):
    pin the minimum expected set of static CopilotKit routes so partial
    removals (e.g. dropping ``/chat/stream`` while leaving ``/status``)
    fail loud — the prior shape asserted only non-empty, which silently
    allowed partial regressions.

    The ``_build_app`` helper above includes the three protected
    routers (``executive_insights``, ``audit``, ``staleness_alerts``).
    Production also mounts the static-CopilotKit router via
    ``app.include_router(copilotkit_router, prefix="/api")`` and
    ``add_copilotkit_routes(app, prefix="/api/copilotkit")`` at
    ``src/api/main.py:997-1000``.

    The static portion (``copilotkit_router``) is the discovery / status
    / chat / feedback / analytics set declared in
    ``src/api/routes/copilotkit.py``. Per the codex iter-2 M2 fix shape
    (strengthened by iter-3 LOW-2) we enumerate the EXPECTED MINIMUM
    set here so a removal or rename trips the assertion. New static
    routes added later are NOT flagged (we use a subset check, not
    equality) — that's intentional: the regression-pin role is to
    catch removals, not lock the surface.

    The DYNAMIC catch-all routes registered by
    ``add_copilotkit_routes`` are NOT covered here — they are pinned
    by ``test_copilotkit_dynamic_routes_require_auth`` below. #399 was
    closed by removing the catch-all regex from
    ``PUBLIC_PATH_PATTERNS`` so any unknown sub-path now requires JWT.
    """
    from src.api.routes.copilotkit import router as copilotkit_router

    # Build an app that includes the static copilotkit router exactly as
    # production does. We don't include dynamic add_copilotkit_routes
    # here because that path is the subject of #399's tracking pin (not
    # the regression-pin contract this test is enforcing).
    app = FastAPI()
    app.include_router(copilotkit_router, prefix="/api")

    paths = {r.path for r in _api_routes(app) if r.path.startswith("/api/copilotkit")}
    # Pin the EXPECTED minimum set of static routes (codex iter-3 LOW-2
    # closure: the prior shape only asserted ``paths`` was non-empty,
    # which silently allowed partial removals — e.g. removing
    # ``/chat/stream`` while keeping ``/status`` would still pass.
    # The pinned superset below ensures all currently-known static
    # routes survive a refactor; new additions are flagged by the
    # equality check at the end if the operator wants to add them).
    expected_minimum_static = {
        "/api/copilotkit/status",
        "/api/copilotkit/chat/stream",
        "/api/copilotkit/chat",
        "/api/copilotkit/feedback",
        "/api/copilotkit/feedback/stats",
        "/api/copilotkit/analytics/usage",
        "/api/copilotkit/analytics/agents",
        "/api/copilotkit/analytics/errors",
    }
    missing = expected_minimum_static - paths
    assert not missing, (
        f"Expected static CopilotKit routes are missing from the test app: "
        f"{missing!r}. Found paths: {sorted(paths)!r}. If a route was "
        f"removed intentionally, update this pin alongside the change."
    )
    # Document that we are explicitly NOT covering dynamic-route auth
    # here — that surface is pinned by
    # ``test_copilotkit_dynamic_routes_require_auth`` below (#399 closure).
    # The dynamic routes registered by
    # ``add_copilotkit_routes(app, prefix="/api/copilotkit")`` at
    # ``src/api/main.py:1000`` are NOT present in this app's route set
    # (we did not call ``add_copilotkit_routes`` here) — confirm absence:
    dynamic_route_paths = {p for p in paths if "{path:path}" in p}
    assert dynamic_route_paths == set(), (
        "This test app intentionally excludes dynamic CopilotKit catch-all "
        "routes (registered via add_copilotkit_routes). Their auth state is "
        f"pinned separately in test_copilotkit_dynamic_routes_require_auth "
        f"and tracked in #399. Found unexpected dynamic routes: "
        f"{dynamic_route_paths!r}."
    )


# ---------------------------------------------------------------------------
# CopilotKit allowlist positive controls (#399 closure)
# ---------------------------------------------------------------------------


def test_copilotkit_probe_paths_remain_public() -> None:
    """ALLOWLIST: the three SDK-discovery surfaces stay public so the
    CopilotKit React provider can bootstrap without an authenticated
    session.

    These three exact paths are SDK protocol probes:
      * ``/api/copilotkit`` — base handler (root of the SDK runtime)
      * ``/api/copilotkit/status`` — health endpoint served by the
        static router (operation_id=get_copilotkit_status at
        ``src/api/routes/copilotkit.py:2860``); no auth in the function
        signature, so its public-ness lives entirely in the middleware
        allowlist.
      * ``/api/copilotkit/info`` — discovery endpoint that returns
        available agents + actions; the CopilotKit React SDK calls this
        at provider mount time.

    The fix for #399 narrows the public surface to JUST these three,
    dropping the catch-all regex that previously allowed every sub-path.
    """
    from src.api.middleware.auth_middleware import _is_public_path

    assert _is_public_path("GET", "/api/copilotkit") is True
    assert _is_public_path("POST", "/api/copilotkit") is True
    assert _is_public_path("GET", "/api/copilotkit/status") is True
    assert _is_public_path("GET", "/api/copilotkit/info") is True
    # #399 iter-2: POST /status is NOT public anymore (it would
    # otherwise fall through to the dynamic catch-all → SDK fallback).
    # /status's static GET-only handler does NOT serve POST, so POSTing
    # there is an attempt to reach SDK code that should require auth.
    assert _is_public_path("POST", "/api/copilotkit/status") is False


def test_copilotkit_dynamic_routes_require_auth() -> None:
    """ALLOWLIST: SDK execution endpoints require JWT auth.

    Closes #399. The CopilotKit SDK runtime exposes several
    execution-side paths via the dynamic catch-all registered at
    ``src/api/routes/copilotkit.py:2832``:

      * ``/api/copilotkit/agent/{name}`` — execute named agent
      * ``/api/copilotkit/agent/{name}/state`` — get agent state
      * ``/api/copilotkit/action/{name}`` — execute named action
      * ``/api/copilotkit/agents/execute`` — v1 batch agent execute
      * ``/api/copilotkit/actions/execute`` — v1 batch action execute

    These paths handle ACTUAL DATA FLOW (LLM invocation, action
    side-effects, agent state mutations) — they MUST require an
    authenticated caller. The fix for #399 removes the catch-all regex
    ``^/api/copilotkit(/.*)?$`` from ``PUBLIC_PATH_PATTERNS`` so these
    paths fall through to the default JWT-required branch.

    The frontend already sends ``Authorization: Bearer ${accessToken}``
    on every CopilotKit SDK call via the ``<CopilotKit headers={...}>``
    pattern at ``frontend/src/providers/E2ICopilotProvider.tsx:456-468``,
    so authenticated users see no change.
    """
    from src.api.middleware.auth_middleware import _is_public_path

    # Execution endpoints — all must require auth (NOT public).
    assert _is_public_path("POST", "/api/copilotkit/agents/execute") is False
    assert _is_public_path("POST", "/api/copilotkit/actions/execute") is False
    assert _is_public_path("POST", "/api/copilotkit/agent/foo") is False
    assert _is_public_path("GET", "/api/copilotkit/agent/foo/state") is False
    assert _is_public_path("POST", "/api/copilotkit/action/anything") is False

    # Catch-all subpath patterns (smoke check — anything not in the
    # explicit allowlist must require auth):
    assert _is_public_path("GET", "/api/copilotkit/some/unknown/path") is False
    assert _is_public_path("POST", "/api/copilotkit/v2/agent/bar") is False

    # #399 iter-2 H1 closure: POST /api/copilotkit/status. The static
    # router serves GET-only at /status (no POST handler), so a POST
    # would fall through to the dynamic catch-all and reach SDK code
    # that should require auth. Codex iter-1 H1 found this gap;
    # method-restricting the allowlist entry to GET catches it at the
    # middleware before any handler dispatch.
    assert _is_public_path("POST", "/api/copilotkit/status") is False
    assert _is_public_path("PUT", "/api/copilotkit/status") is False
    assert _is_public_path("DELETE", "/api/copilotkit/status") is False


def _build_mock_request_with_headers(headers: dict[str, str]) -> object:
    """Build a Mock request stub for unit testing the auth helper.

    Hand-crafted Starlette Request scopes are fragile (workers crash on
    incomplete scope dicts under xdist parallelism). A Mock with the
    fields the helper actually reads — ``headers`` (dict-like .get) +
    ``state`` (attribute-settable namespace) — is sufficient for the
    helper's contract and stable across pytest-xdist workers.
    """
    from unittest.mock import MagicMock

    request = MagicMock()
    request.headers = headers
    # Allow ``request.state.user = ...`` attribute assignment.
    request.state = MagicMock()
    return request


def test_copilotkit_execution_post_to_public_path_requires_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ALLOWLIST defense-in-depth: even on the middleware-public
    ``/api/copilotkit`` + ``/info`` paths, POST bodies that route to
    execution (``agent/run``, ``action/run``, ``agent/connect``, SDK
    fallback) must require a JWT.

    Codex iter-0 H1+H2 closure: the CopilotKit JSON-RPC protocol mixes
    discovery and execution under the same paths via the request body's
    ``method`` field. Path-based allowlist alone CANNOT distinguish the
    two — only the body can. The ``copilotkit_custom_handler``
    inspects the body at
    ``src/api/routes/copilotkit.py:2590-2624`` to detect discovery
    requests; this test pins that execution-shaped bodies hit the new
    in-handler auth check
    (``_require_auth_for_copilotkit_execution``) and produce 401 when
    no Bearer token is present.

    Production note: TESTING_MODE bypasses this check (mirrors
    ``require_auth``). The test forces TESTING_MODE off so the real
    JWT path runs.
    """
    from src.api.routes import copilotkit as copilotkit_module

    # Force TESTING_MODE off so the real JWT branch runs.
    monkeypatch.setattr(copilotkit_module, "TESTING_MODE", False)

    request = _build_mock_request_with_headers({})

    import asyncio

    from src.api.dependencies.auth import AuthError

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(AuthError):
            loop.run_until_complete(
                copilotkit_module._require_auth_for_copilotkit_execution(request)  # type: ignore[arg-type]
            )
    finally:
        loop.close()


def test_copilotkit_execution_helper_rejects_malformed_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed Authorization header (missing 'Bearer' prefix, wrong
    parts count, etc.) is rejected by the in-handler auth check."""
    from src.api.routes import copilotkit as copilotkit_module

    monkeypatch.setattr(copilotkit_module, "TESTING_MODE", False)

    from src.api.dependencies.auth import AuthError

    # Wrong scheme.
    request_wrong_scheme = _build_mock_request_with_headers({"Authorization": "Basic xyz"})
    # Single-part (no token).
    request_single_part = _build_mock_request_with_headers({"Authorization": "Bearer"})
    # Three-part (extra junk).
    request_three_part = _build_mock_request_with_headers({"Authorization": "Bearer abc extra"})

    import asyncio

    for req in (request_wrong_scheme, request_single_part, request_three_part):
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(AuthError):
                loop.run_until_complete(
                    copilotkit_module._require_auth_for_copilotkit_execution(req)  # type: ignore[arg-type]
                )
        finally:
            loop.close()


def test_copilotkit_execution_helper_passes_in_testing_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ALLOWLIST: when ``E2I_TESTING_MODE=1``, the in-handler auth
    helper short-circuits to the TEST_USER mock (mirrors
    ``require_auth``'s testing-mode branch at
    ``src/api/dependencies/auth.py:304-306``). This keeps integration
    tests + e2e flows working without real JWT issuance.
    """
    from src.api.routes import copilotkit as copilotkit_module

    monkeypatch.setattr(copilotkit_module, "TESTING_MODE", True)

    request = _build_mock_request_with_headers({})

    import asyncio

    loop = asyncio.new_event_loop()
    try:
        user = loop.run_until_complete(
            copilotkit_module._require_auth_for_copilotkit_execution(request)  # type: ignore[arg-type]
        )
    finally:
        loop.close()

    # TEST_USER is the fixture-shape mock returned by require_auth in
    # testing mode.
    assert isinstance(user, dict)
    assert user.get("id") or user.get("user_id") or user.get("sub")


def test_copilotkit_public_path_patterns_does_not_contain_catchall() -> None:
    """ALLOWLIST: the catch-all regex ``^/api/copilotkit(/.*)?$`` MUST
    NOT appear in ``PUBLIC_PATH_PATTERNS``.

    This is the literal counterpart to the pre-#399 pin
    ``test_copilotkit_auth_gap_pinned_for_issue_399``: the gap was that
    the catch-all regex existed; the fix is its removal. A future PR
    that re-adds the catch-all (broad regex starting with
    ``^/api/copilotkit``) will trip this assertion.

    Narrowed regexes scoped to specific SDK probes (e.g. matching only
    ``/api/copilotkit/status``) are acceptable — we assert against the
    SPECIFIC broad shape, not against any regex mentioning copilotkit.
    """
    from src.api.middleware.auth_middleware import PUBLIC_PATH_PATTERNS

    FORBIDDEN_CATCHALL_REGEX = r"^/api/copilotkit(/.*)?$"
    matching = [
        (method, pattern)
        for method, pattern in PUBLIC_PATH_PATTERNS
        if pattern == FORBIDDEN_CATCHALL_REGEX
    ]
    assert not matching, (
        f"#399 closure broken: the pre-fix catch-all pattern "
        f"{FORBIDDEN_CATCHALL_REGEX!r} reappeared in PUBLIC_PATH_PATTERNS. "
        f"This is the dynamic-route auth bypass that #399 fixed. Current "
        f"PUBLIC_PATH_PATTERNS: {PUBLIC_PATH_PATTERNS!r}. Either narrow the "
        f"regex (e.g. to only /api/copilotkit/(status|info|health)) or "
        f"remove it entirely and rely on the explicit PUBLIC_PATHS entries."
    )


# ---------------------------------------------------------------------------
# Graph endpoint allowlist regression pin (PHI hardening)
#
# The knowledge-graph DATA endpoints were public "for demo visualization"
# but (1) return Patient/HCP nodes + relationships (PHI/PII) cross-tenant
# to any anonymous caller, and (2) forward user-supplied ``entity_types`` /
# ``relationship_types`` into a Cypher string-interpolation in
# ``src/memory/semantic_memory.py`` — an UNAUTHENTICATED Cypher-injection
# surface. Only ``/api/graph/health`` stays public.
#
# The frontend graph client sends ``Authorization: Bearer`` via the shared
# apiClient (``frontend/src/lib/api-client.ts``) and the dashboard sits
# behind a ``ProtectedRoute`` login wall, so authenticated users observe no
# change — only anonymous access is removed.
# ---------------------------------------------------------------------------

_GRAPH_DATA_ENTRIES = {
    ("GET", "/api/graph/nodes"),
    ("GET", "/api/graph/relationships"),
    ("GET", "/api/graph/stats"),
    ("POST", "/api/graph/causal-chains"),
}


def test_graph_data_endpoints_not_in_public_paths() -> None:
    """REGRESSION PIN — graph DATA endpoints MUST NOT be in ``PUBLIC_PATHS``.

    They return PHI/PII (Patient/HCP nodes + edges) and expose an
    unauthenticated Cypher-injection surface. Matches on the bare path
    (any method) so a future ``("*", ...)`` re-add is also caught.
    """
    from src.api.middleware.auth_middleware import PUBLIC_PATHS

    graph_data_paths = {path for _method, path in _GRAPH_DATA_ENTRIES}
    leaked = [(method, path) for method, path in PUBLIC_PATHS if path in graph_data_paths]
    assert not leaked, (
        f"Graph DATA endpoints re-appeared in PUBLIC_PATHS: {leaked!r}. "
        f"These expose Patient/HCP PHI/PII and an unauthenticated Cypher "
        f"injection surface (entity_types/relationship_types are string-"
        f"interpolated into Cypher in src/memory/semantic_memory.py). They "
        f"MUST require a JWT. Only /api/graph/health may be public."
    )


def test_graph_data_endpoints_require_auth_via_is_public_path() -> None:
    """The public-path resolver treats graph DATA endpoints as protected."""
    from src.api.middleware.auth_middleware import _is_public_path

    for method, path in _GRAPH_DATA_ENTRIES:
        assert _is_public_path(method, path) is False, (
            f"{method} {path} must require auth (PHI/PII + Cypher injection)."
        )


def test_graph_health_remains_public() -> None:
    """The graph health probe stays public (no PHI; dashboard liveness)."""
    from src.api.middleware.auth_middleware import _is_public_path

    assert _is_public_path("GET", "/api/graph/health") is True


# ---------------------------------------------------------------------------
# Handler-dispatch end-to-end tests (#399 codex iter-1 M-finding closure)
#
# What's heavy here: building a FastAPI app via ``add_copilotkit_routes``
# and routing requests through it via TestClient. That triggers SDK
# initialization, request marshalling, ASGI middleware traversal, and
# the handler's full body-read + body-parse + branch-dispatch path
# end-to-end. Combined with the (already-loaded-at-collection-time)
# CopilotKit + LangChain + Anthropic + LangGraph module graph, the
# cumulative per-worker memory footprint can push xdist workers over
# the OOM threshold when running alongside other test files.
#
# Same shape as the SSE e2e in PR #394 (see memory file
# ``feat_393_394_388_390_parallel_close_20260520``).
#
# Note on import cost: the unit-level helper tests above
# (``test_copilotkit_execution_post_to_public_path_*``) DO import
# ``src.api.routes.copilotkit`` to reach ``_require_auth_for_copilotkit_execution``,
# so they ALSO incur the module-import cost at collection time. That
# import cost alone is NOT what crashes workers — every test in this
# file shares it. The marginal cost that pushes workers over the edge
# is the TestClient EXECUTION (full ASGI request lifecycle on top of
# the already-loaded module graph). Skipping just the 3 TestClient
# tests on CI is sufficient; the helper tests stay always-on because
# they don't add the execution-time pressure.
#
# Solution: ``@_SKIP_COPILOTKIT_E2E_ON_CI`` gates ONLY the 3 TestClient
# tests on ``CI=true and not E2I_RUN_COPILOTKIT_E2E``. Local
# pre-release verification can opt in via the env override.
# ---------------------------------------------------------------------------

_SKIP_COPILOTKIT_E2E_ON_CI = pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.environ.get("E2I_RUN_COPILOTKIT_E2E"),
    reason=(
        "CopilotKit TestClient handler-dispatch adds full ASGI request "
        "lifecycle on top of the already-loaded CopilotKit/LangChain/"
        "Anthropic/LangGraph module graph; cumulative xdist worker "
        "memory has triggered OOM crashes alongside other test files. "
        "Helper tests above (which only invoke the auth helper directly) "
        "stay always-on because they don't add the execution-time "
        "pressure — the module imports they share are already paid at "
        "collection time. Set E2I_RUN_COPILOTKIT_E2E=1 to opt into the "
        "TestClient tests locally for pre-release verification."
    ),
)


def _build_copilotkit_app(monkeypatch: pytest.MonkeyPatch) -> "FastAPI":  # type: ignore[name-defined]  # noqa: F821
    """Build a FastAPI app with the dynamic catch-all CopilotKit routes
    attached but the SDK creator stubbed.

    ``create_copilotkit_sdk()`` (in ``src/api/routes/copilotkit.py``)
    instantiates a real SDK with network/secret dependencies; we mock
    it so the handler dispatch path is exercised without those.
    """
    from unittest.mock import MagicMock

    from fastapi import FastAPI

    from src.api.routes import copilotkit as copilotkit_module

    # Mock SDK + agents list. ``copilotkit_custom_handler`` calls
    # ``sdk.agents(sdk_context) if callable(sdk.agents) else sdk.agents``
    # and iterates results when matching agent_name. For the auth-gating
    # tests we never reach the agent lookup (auth check fires first), so
    # an empty list suffices.
    mock_sdk = MagicMock()
    mock_sdk.agents = []
    mock_sdk.actions = []
    monkeypatch.setattr(copilotkit_module, "create_copilotkit_sdk", lambda: mock_sdk)
    # ``transform_info_response`` is called for discovery requests; we
    # mock to return a deterministic, JSON-serializable response so
    # discovery tests can assert against the shape.
    monkeypatch.setattr(
        copilotkit_module,
        "transform_info_response",
        lambda sdk: {"agents": {}, "actions": [], "version": "test"},
    )

    app = FastAPI()
    copilotkit_module.add_copilotkit_routes(app, prefix="/api/copilotkit")
    return app


@_SKIP_COPILOTKIT_E2E_ON_CI
def test_handler_post_with_execution_body_returns_401_without_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: POST /api/copilotkit (or /info) with execution body
    returns 401 when no Authorization header is present.

    Closes codex iter-1 M-finding: the unit tests on
    ``_require_auth_for_copilotkit_execution`` exercise the helper but
    not the handler dispatch. This test routes through the real
    ``copilotkit_custom_handler`` via FastAPI TestClient and asserts
    the auth response shape.
    """
    from fastapi.testclient import TestClient

    from src.api.routes import copilotkit as copilotkit_module

    # Force TESTING_MODE off so the real JWT branch runs.
    monkeypatch.setattr(copilotkit_module, "TESTING_MODE", False)

    app = _build_copilotkit_app(monkeypatch)
    client = TestClient(app)

    # Execution body on base path → 401.
    resp = client.post(
        "/api/copilotkit",
        json={"method": "agent/run", "params": {"agentId": "test"}, "messages": []},
    )
    assert resp.status_code == 401, (
        f"POST /api/copilotkit with execution body must require auth "
        f"(codex iter-0 H1). Got status={resp.status_code}, body={resp.text!r}"
    )
    body = resp.json()
    assert "Authentication required" in body.get("error", ""), (
        f"401 response must explicitly say auth required. Got: {body!r}"
    )

    # Same shape on /info path.
    resp_info = client.post(
        "/api/copilotkit/info",
        json={"method": "agent/run", "params": {"agentId": "test"}, "messages": []},
    )
    assert resp_info.status_code == 401, (
        f"POST /api/copilotkit/info with execution body must require auth "
        f"(codex iter-0 H2). Got status={resp_info.status_code}, body={resp_info.text!r}"
    )


@_SKIP_COPILOTKIT_E2E_ON_CI
def test_handler_post_with_discovery_body_returns_200_without_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: POST /api/copilotkit (or /info) with discovery body
    (empty / ``{}`` / ``{"method":"info"}``) returns 200 WITHOUT auth.

    The middleware-public allowlist preserves SDK bootstrap behavior:
    the CopilotKit React provider can mount and discover available
    agents before an access token is available.
    """
    from fastapi.testclient import TestClient

    from src.api.routes import copilotkit as copilotkit_module

    # TESTING_MODE off so we exercise the real branch logic — discovery
    # short-circuits BEFORE the auth check, so we still get 200.
    monkeypatch.setattr(copilotkit_module, "TESTING_MODE", False)

    app = _build_copilotkit_app(monkeypatch)
    client = TestClient(app)

    # Empty {} body → discovery.
    resp_empty = client.post("/api/copilotkit", json={})
    assert resp_empty.status_code == 200, (
        f"POST /api/copilotkit with empty body must serve discovery without auth. "
        f"Got status={resp_empty.status_code}, body={resp_empty.text!r}"
    )

    # {"method": "info"} body → discovery.
    resp_method_info = client.post("/api/copilotkit/info", json={"method": "info"})
    assert resp_method_info.status_code == 200, (
        f"POST /api/copilotkit/info with method=info body must serve discovery. "
        f"Got status={resp_method_info.status_code}, body={resp_method_info.text!r}"
    )

    # {"action": "getInfo"} body → discovery.
    resp_action_getinfo = client.post("/api/copilotkit", json={"action": "getInfo"})
    assert resp_action_getinfo.status_code == 200, (
        f"POST /api/copilotkit with action=getInfo body must serve discovery. "
        f"Got status={resp_action_getinfo.status_code}, body={resp_action_getinfo.text!r}"
    )


@_SKIP_COPILOTKIT_E2E_ON_CI
def test_handler_get_info_returns_200_without_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: GET /api/copilotkit/info returns 200 without auth
    (the primary SDK discovery handshake).
    """
    from fastapi.testclient import TestClient

    from src.api.routes import copilotkit as copilotkit_module

    monkeypatch.setattr(copilotkit_module, "TESTING_MODE", False)

    app = _build_copilotkit_app(monkeypatch)
    client = TestClient(app)

    resp = client.get("/api/copilotkit/info")
    assert resp.status_code == 200, (
        f"GET /api/copilotkit/info must serve discovery without auth. "
        f"Got status={resp.status_code}, body={resp.text!r}"
    )
    body = resp.json()
    assert "version" in body, f"Discovery response must include version. Got: {body!r}"
