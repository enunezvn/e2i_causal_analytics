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

    # Restore testing-mode (other tests in this run depend on it).
    monkeypatch.setenv("E2I_TESTING_MODE", "1")
    importlib.reload(auth_dep)
    importlib.reload(auth_mw)


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
        ("*", "/api/copilotkit/status"),
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
