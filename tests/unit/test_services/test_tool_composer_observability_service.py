"""The tool-composer observability aggregates (spec §8).

The page and the planning prompt must describe the SAME population: if the tool rows exclude
synthetic-substrate runs, the composition counts beside them cannot include those runs, and a
percentile shown here cannot be computed differently from the one the database computes for
tools. These tests pin the parts a real database would otherwise only reveal later.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.services.tool_composer_observability_service import (
    ToolComposerObservabilityService,
    _percentile,
)


class FakeQuery:
    def __init__(self, client: "FakeClient", table: str) -> None:
        self.client = client
        self.table = table
        self.filters: List[tuple] = []
        self.orders: List[tuple] = []
        # Not ``self.range``: that would shadow the range() method this double must expose.
        self.window: Optional[tuple] = None

    def select(self, *_columns: str) -> "FakeQuery":
        return self

    def gte(self, column: str, value: Any) -> "FakeQuery":
        self.filters.append(("gte", column, value))
        return self

    def eq(self, column: str, value: Any) -> "FakeQuery":
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column: str, values: List[Any]) -> "FakeQuery":
        self.filters.append(("in", column, list(values)))
        return self

    def order(self, column: str, desc: bool = False) -> "FakeQuery":
        self.orders.append((column, desc))
        return self

    def range(self, start: int, end: int) -> "FakeQuery":  # noqa: A003 - supabase-py's name
        self.window = (start, end)
        return self

    def execute(self) -> Any:
        self.client.queries.append(self)
        rows = self.client.rows.get(self.table, [])
        if self.window is not None:
            start, end = self.window
            rows = rows[start : end + 1]
        return type("Result", (), {"data": rows})()


class FakeClient:
    def __init__(self, rows: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
        self.rows = rows or {}
        self.queries: List[FakeQuery] = []

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)

    def queries_for(self, table: str) -> List[FakeQuery]:
        return [q for q in self.queries if q.table == table]


def _episode(**over: Any) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    episode = {
        "episode_id": "11111111-1111-1111-1111-111111111111",
        "composition_id": "comp_a",
        "query_text": "q",
        "status": "COMPLETED",
        "outcome": "success",
        "failed_phase": None,
        "error_type": None,
        "plan_source": "llm",
        "entry_point": "chat_tool",
        "total_latency_ms": 100.0,
        "last_activity_at": now.isoformat(),
        "tools_executed": 2,
        "tools_succeeded": 2,
        "is_synthetic": False,
    }
    episode.update(over)
    return episode


def _service(rows: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Any:
    return ToolComposerObservabilityService(client=FakeClient(rows or {}))


# ---------------------------------------------------------------------------
# Percentiles agree with the database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, fraction, expected",
    [
        ([100.0, 1000.0], 0.50, 550.0),  # percentile_cont interpolates
        ([100.0, 1000.0], 0.95, 955.0),
        ([5.0], 0.95, 5.0),
        ([], 0.50, None),
        ([1.0, 2.0, 3.0], 0.50, 2.0),
    ],
)
def test_percentiles_interpolate_like_percentile_cont(values, fraction, expected):
    assert _percentile(values, fraction) == expected


# ---------------------------------------------------------------------------
# Terminal statuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "TIMEOUT"])
def test_a_terminal_episode_is_never_unfinished_or_abandoned(status):
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    rows = {"composer_episodes": [_episode(status=status, last_activity_at=stale)]}

    compositions = _service(rows).overview(30)["compositions"]

    assert compositions["unfinished"] == 0 and compositions["abandoned"] == 0


def test_an_in_flight_episode_gone_quiet_is_abandoned():
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    rows = {
        "composer_episodes": [_episode(status="EXECUTING", outcome=None, last_activity_at=stale)]
    }

    compositions = _service(rows).overview(30)["compositions"]

    assert compositions["unfinished"] == 1 and compositions["abandoned"] == 1


# ---------------------------------------------------------------------------
# One population: the window, and the provenance flag
# ---------------------------------------------------------------------------


def test_window_membership_is_when_the_composition_started():
    service = _service({"composer_episodes": []})

    service.overview(30)

    episodes = service.client.queries_for("composer_episodes")[0]
    assert [f for f in episodes.filters if f[0] == "gte"][0][1] == "created_at"
    assert all(f[1] != "last_activity_at" for f in episodes.filters)


def test_synthetic_runs_are_filtered_out_unless_the_deployment_includes_them(monkeypatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    service = _service({"composer_episodes": []})
    service.overview(30)
    filters = service.client.queries_for("composer_episodes")[0].filters
    assert ("eq", "is_synthetic", False) in filters

    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "1")
    included = _service({"composer_episodes": []})
    included.overview(30)
    assert all(
        f[1] != "is_synthetic" for f in included.client.queries_for("composer_episodes")[0].filters
    )


# ---------------------------------------------------------------------------
# Paging cannot drop or double-count a row
# ---------------------------------------------------------------------------


def test_episode_paging_orders_by_a_unique_key_as_well():
    """last_activity_at moves under heartbeats and repeats; a page boundary needs a tiebreaker."""
    service = _service({"composer_episodes": []})

    service.overview(30)

    orders = service.client.queries_for("composer_episodes")[0].orders
    assert any(column == "episode_id" for column, _ in orders), orders


def test_steps_are_fetched_for_every_failed_episode_in_pages():
    failures = [
        _episode(
            episode_id=f"1111111{n:04d}-1111-1111-1111-111111111111",
            composition_id=f"comp_{n}",
            status="FAILED",
            outcome="failed",
        )
        for n in range(12)
    ]
    rows = {"composer_episodes": failures, "composition_steps": []}
    service = _service(rows)

    result = service.overview(30)

    # At most ten recent failures are shown, and every one of them had its steps looked up.
    assert len(result["recent_failures"]) == 10
    asked = [f for q in service.client.queries_for("composition_steps") for f in q.filters]
    looked_up = {cid for kind, _, values in asked if kind == "in" for cid in values}
    assert len(looked_up) == 10
