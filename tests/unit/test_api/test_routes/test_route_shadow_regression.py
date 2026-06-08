"""Regression tests for FastAPI route-ordering shadows (audit C1 + C2).

Root cause: a parameterized ``GET /{id}`` registered BEFORE a static sibling
``GET /static`` captures the static path (first-match-wins on registration
order). The global JWT middleware runs before routing, so an unauthenticated
probe returns 401 for every path and CANNOT detect the shadow. The only
faithful, auth-independent check is the live Starlette matcher on the real
``app.router`` object — exactly what these tests assert.

See docs/reports/frontend-backend-api-connectivity-audit-20260608.md §3.
"""

import pytest
from starlette.routing import Match

from src.api.main import app


def _resolve_get(path: str) -> str | None:
    """Return the name of the handler the live app resolves a GET ``path`` to.

    Mirrors the faithful matcher used in the audit. The ``getattr(..., None)``
    guard is required: ``app.router.routes`` includes ``APIWebSocketRoute``
    objects that have no ``.methods`` attribute.
    """
    scope = {"type": "http", "method": "GET", "path": path}
    for route in app.router.routes:
        methods = getattr(route, "methods", None)
        if not methods or "GET" not in methods:
            continue
        if route.matches(scope)[0] == Match.FULL:
            return route.name
    return None


@pytest.mark.parametrize(
    ("path", "expected_handler"),
    [
        # C1 — gaps router
        ("/api/gaps/opportunities", "list_opportunities"),
        ("/api/gaps/health", "get_gap_health"),
        # C2 — feedback router
        ("/api/feedback/patterns", "list_patterns"),
        ("/api/feedback/updates", "list_updates"),
        ("/api/feedback/health", "get_feedback_health"),
    ],
)
def test_static_get_routes_are_not_shadowed(path: str, expected_handler: str) -> None:
    """Each static GET path must resolve to its own handler, not the catch-all."""
    resolved = _resolve_get(path)
    assert resolved == expected_handler, (
        f"{path} resolved to {resolved!r}; expected {expected_handler!r}. "
        "A parameterized /{id} route is shadowing it (registration order)."
    )


def test_parameterized_get_routes_still_resolve() -> None:
    """The moved /{id} routes must still catch genuine id-shaped paths."""
    assert _resolve_get("/api/gaps/gap_abc123") == "get_gap_analysis"
    assert _resolve_get("/api/feedback/fb_abc123") == "get_learning_results"
