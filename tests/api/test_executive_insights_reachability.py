"""M5 triage guard: the orphan endpoints classified KEEP/REWIRE must stay
reachable. Uses the live Starlette route-matcher on the real app object —
the only faithful, auth-independent route-resolution check (the global JWT
middleware returns 401 for any path before routing, so an HTTP probe cannot
detect route resolution; see audit §6 INFO measurement caveat)."""

from __future__ import annotations

import pytest
from starlette.routing import Match

from src.api.main import app


def _resolve(method: str, path: str) -> str | None:
    """Return the name of the handler that FULL-matches (method, path), or None."""
    scope = {"type": "http", "method": method, "path": path}
    for route in app.router.routes:
        methods = getattr(route, "methods", None)
        if not methods or method not in methods:
            continue
        if route.matches(scope)[0] == Match.FULL:
            return route.name
    return None


@pytest.mark.parametrize(
    ("method", "path", "expected_handler"),
    [
        # executive-insights (REWIRE — UI now consumes these via Task 2/3)
        ("GET", "/api/executive-insights", "list_executive_insights"),
        ("GET", "/api/executive-insights/portfolio-summary", "get_portfolio_summary"),
        # feedback GEPA optimizer surface (KEEP — feeds DSPy loop PR #792)
        ("GET", "/api/feedback/agent/tool_composer/signals", "get_optimization_signals"),
        ("GET", "/api/feedback/agent/tool_composer/gepa-batch", "get_gepa_training_batch"),
        # causal async status poll (KEEP)
        ("GET", "/api/causal/pipeline/some-id", "get_pipeline_status"),
        # staleness SSE feed (KEEP)
        ("GET", "/api/alerts/stream", "alerts_stream"),
    ],
)
def test_m5_kept_routes_resolve_to_real_handlers(
    method: str, path: str, expected_handler: str
) -> None:
    assert _resolve(method, path) == expected_handler
