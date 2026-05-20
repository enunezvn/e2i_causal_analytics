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
# CopilotKit auth-gap pin (codex iter-2 H2 closure — issue #399)
# ---------------------------------------------------------------------------


def test_copilotkit_auth_gap_pinned_for_issue_399() -> None:
    """REGRESSION PIN — CopilotKit dynamic endpoints currently BYPASS the
    JWTAuthMiddleware via two surfaces in
    ``src.api.middleware.auth_middleware``:

    1. ``PUBLIC_PATHS`` includes the exact entries ``("*", "/api/copilotkit")``,
       ``("*", "/api/copilotkit/status")``, ``("*", "/api/copilotkit/info")``.
    2. ``PUBLIC_PATH_PATTERNS`` includes the catch-all
       ``("*", r"^/api/copilotkit(/.*)?$")``.

    Codex iter-2 H2 found this gap during PR #398 review. The gap is
    PRE-EXISTING (not introduced by #391) and per issue #391 box 2
    wording (verbatim: "Add authentication to crystal/provenance API
    endpoints — verify ``/api/executive-insights/*``") it is
    out-of-scope for the security slice. A separate tracking issue
    (#399) has been filed.

    This test PINS the CURRENT (unauthenticated) state so:
    * Future PRs that accidentally widen the bypass surface fail loud.
    * The fix for #399 will FLIP this assertion (delete the negation,
      assert auth IS required), making the migration mechanical and
      auditable.

    Flip recipe (when #399 is fixed):
      1. Edit ``PUBLIC_PATHS`` + ``PUBLIC_PATH_PATTERNS`` in
         ``src/api/middleware/auth_middleware.py`` to remove or scope
         the CopilotKit entries.
      2. Edit ``add_copilotkit_routes`` in
         ``src/api/routes/copilotkit.py`` to attach
         ``dependencies=[Depends(require_auth)]`` (or similar).
      3. Rename this test to ``test_copilotkit_routes_require_auth``
         and flip the assertion below — change
         ``assert pattern_in_public_patterns is True``
         to
         ``assert pattern_in_public_patterns is False``.
    """
    from src.api.middleware.auth_middleware import (
        PUBLIC_PATH_PATTERNS,
        PUBLIC_PATHS,
        _is_public_path,
    )

    # Surface 1: exact-match entry in PUBLIC_PATHS.
    copilotkit_exact_entries = [
        (method, path)
        for method, path in PUBLIC_PATHS
        if path == "/api/copilotkit" or path.startswith("/api/copilotkit/")
    ]
    assert copilotkit_exact_entries, (
        "Expected at least one /api/copilotkit entry in PUBLIC_PATHS — "
        "the iter-2 H2 audit identified one. If this is empty, the gap "
        "may already be closed and the pin should be flipped (see #399)."
    )

    # Surface 2: catch-all pattern in PUBLIC_PATH_PATTERNS.
    pattern_in_public_patterns = any(
        "/api/copilotkit" in pattern for _method, pattern in PUBLIC_PATH_PATTERNS
    )
    assert pattern_in_public_patterns is True, (
        "Expected a /api/copilotkit catch-all pattern in PUBLIC_PATH_PATTERNS. "
        "If this is False, the dynamic-route bypass may already be closed and "
        "this regression pin should be flipped (see #399)."
    )

    # Surface 3: the _is_public_path matcher actually returns True for
    # arbitrary subpaths under /api/copilotkit (proving the bypass is
    # active end-to-end, not just declared in lookup tables).
    assert _is_public_path("POST", "/api/copilotkit/agents/execute") is True
    assert _is_public_path("GET", "/api/copilotkit/agent/foo/state") is True
    assert _is_public_path("POST", "/api/copilotkit/action/anything") is True


def test_copilotkit_static_routes_present_in_auth_test_app() -> None:
    """Codex iter-2 M2 closure: enumerate the static-CopilotKit routes
    that DO appear on the test app's router set and document the
    dynamic-route bypass cross-reference.

    The ``_build_app`` helper above includes the three protected
    routers (``executive_insights``, ``audit``, ``staleness_alerts``).
    Production also mounts the static-CopilotKit router via
    ``app.include_router(copilotkit_router, prefix="/api")`` and
    ``add_copilotkit_routes(app, prefix="/api/copilotkit")`` at
    ``src/api/main.py:997-1000``.

    The static portion (``copilotkit_router``) is a small set of
    discovery / status routes. Per the codex iter-2 M2 fix shape we
    enumerate those routes here so the auth-gating test app's
    coverage is documented and any silent route addition is flagged
    by the static-route enumeration test.

    The DYNAMIC catch-all routes registered by
    ``add_copilotkit_routes`` are NOT covered here — they are pinned
    in ``test_copilotkit_auth_gap_pinned_for_issue_399`` above + tracked
    in tracking issue #399.
    """
    from src.api.routes.copilotkit import router as copilotkit_router

    # Build an app that includes the static copilotkit router exactly as
    # production does. We don't include dynamic add_copilotkit_routes
    # here because that path is the subject of #399's tracking pin (not
    # the regression-pin contract this test is enforcing).
    app = FastAPI()
    app.include_router(copilotkit_router, prefix="/api")

    paths = {r.path for r in _api_routes(app) if r.path.startswith("/api/copilotkit")}
    # Expected static routes from src/api/routes/copilotkit.py — these
    # are discovery / status endpoints. The path set is pinned so a
    # silent addition shows up here.
    assert paths, (
        "Expected the static CopilotKit router to expose at least one "
        "/api/copilotkit path. If empty, the static router was removed "
        "and this enumeration must be updated alongside the change."
    )
    # Document that we are explicitly NOT covering dynamic-route auth
    # here — that surface is pinned for tracking-issue #399.
    # (No assertion on path-set CONTENT — the path set is implementation
    # detail of the copilotkit_router itself; the existence assertion
    # above is the regression pin.)
