"""GET /admin/observability/tool-composer: thin route (spec §8).

Admin-gated like every admin.py sibling, bounded days, and the aggregation runs off-thread. The
route owns one piece of logic: it awaits the SAME cached reliability reader the planner uses and
hands those verdicts to the sync service, so the verdict word on the page and the verdict word in
the planning prompt can never be computed two different ways.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import pytest

from src.agents.tool_composer.reliability import ToolReliability
from src.api.dependencies.auth import require_admin
from src.api.routes.admin import (
    get_tool_composer_observability_service,
    tool_composer_overview,
)


def _verdict_row(name: str = "gap_calculator") -> ToolReliability:
    return ToolReliability.from_row(
        {
            "tool_name": name,
            "category": "GAP",
            "source_agent": "gap_analyzer",
            "version": "1.0.0",
            "declared_latency_ms": 4000.0,
            "n_invoked": 40,
            "n_succeeded": 28,
            "n_refused": 0,
            "n_health_failures": 12,
            "n_health": 40,
            "n_retried": 0,
            "n_synthetic": 0,
            "p50_latency_ms": 900.0,
            "p95_latency_ms": 2500.0,
            "last_executed_at": "2026-09-11T00:00:00+00:00",
            "most_common_health_error": "timeout",
        }
    )


class _FakeReader:
    def __init__(self) -> None:
        self.days: List[int] = []

    async def get(self, days: int = 30) -> Dict[str, ToolReliability]:
        self.days.append(days)
        return {"gap_calculator": _verdict_row()}


class _FakeService:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    def overview(self, days: int, verdicts: Optional[Dict[str, ToolReliability]]) -> Dict[str, Any]:
        self.calls.append((days, verdicts))
        return {"compositions": {"total": 0}, "tools": [], "recent_failures": []}


async def test_route_awaits_the_reader_and_hands_its_verdicts_to_the_service():
    reader = _FakeReader()
    service = _FakeService()

    result = await tool_composer_overview(
        days=7,
        admin={"id": "admin-1"},
        service=service,
        reader=reader,
    )

    assert reader.days == [7]
    (days, verdicts) = service.calls[0]
    assert days == 7
    assert verdicts["gap_calculator"].verdict == "caveat"
    assert set(result) == {"compositions", "tools", "recent_failures"}


async def test_a_failed_reliability_read_still_returns_the_page():
    """The reader fails open to {}; the composition aggregates do not depend on it."""

    class _EmptyReader:
        async def get(self, days: int = 30) -> Dict[str, ToolReliability]:
            return {}

    service = _FakeService()

    result = await tool_composer_overview(
        days=30, admin={"id": "a"}, service=service, reader=_EmptyReader()
    )

    assert service.calls[0][1] == {}
    assert "compositions" in result


def test_route_is_admin_gated_with_bounded_days():
    parameters = inspect.signature(tool_composer_overview).parameters

    admin_default = parameters["admin"].default
    assert getattr(admin_default, "dependency", None) is require_admin

    days_default = parameters["days"].default
    assert days_default.default == 30
    assert days_default.metadata[0].ge == 1
    assert days_default.metadata[1].le == 365


def test_the_aggregation_runs_off_thread():
    """The service is synchronous by design, like its admin.py siblings."""
    assert "asyncio.to_thread" in inspect.getsource(tool_composer_overview)
    assert not inspect.iscoroutinefunction(_FakeService.overview)


def test_the_service_provider_is_a_lazy_singleton(monkeypatch):
    import src.api.routes.admin as admin_module

    built: List[int] = []

    class _Service:
        def __init__(self) -> None:
            built.append(1)

    monkeypatch.setattr(admin_module, "_tool_composer_obs_service", None)
    monkeypatch.setattr(admin_module, "ToolComposerObservabilityService", _Service)

    first = get_tool_composer_observability_service()
    second = get_tool_composer_observability_service()

    assert first is second and built == [1]


def test_the_route_and_the_planner_share_one_reader():
    """Two readers would mean two caches, and two different verdict words for up to a TTL."""
    from src.agents.tool_composer.planner import ToolPlanner
    from src.agents.tool_composer.reliability import default_reliability_reader
    from src.api.routes.admin import get_tool_composer_reliability_reader

    shared = default_reliability_reader()

    assert get_tool_composer_reliability_reader() is shared

    planner = object.__new__(ToolPlanner)
    planner._reliability_reader = None
    assert planner._resolve_reliability_reader() is shared


@pytest.mark.parametrize("days", [1, 365])
def test_the_bounds_are_inclusive(days):
    parameters = inspect.signature(tool_composer_overview).parameters
    metadata = parameters["days"].default.metadata
    assert metadata[0].ge <= days <= metadata[1].le
