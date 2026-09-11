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
    """A query builder that actually filters, orders and pages.

    A double that ignored its own filters would let a pagination bug pass: the tests below
    depend on it returning what the query asked for, not everything it holds.
    """

    def __init__(self, client: "FakeClient", table: str) -> None:
        self.client = client
        self.table = table
        self.filters: List[tuple] = []
        self.orders: List[tuple] = []
        self.columns: Optional[set] = None
        # Not ``self.range``: that would shadow the range() method this double must expose.
        self.window: Optional[tuple] = None

    def select(self, *columns: str) -> "FakeQuery":
        # Honour the projection: a double that returned columns the query never asked for would
        # hide a missing one, which is exactly how the created_at ordering defect escaped.
        self.columns = {c.strip() for spec in columns for c in spec.split(",") if c.strip()}
        return self

    def gte(self, column: str, value: Any) -> "FakeQuery":
        self.filters.append(("gte", column, value))
        return self

    def gt(self, column: str, value: Any) -> "FakeQuery":
        self.filters.append(("gt", column, value))
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

    def limit(self, count: int) -> "FakeQuery":
        self.window = (0, count - 1)
        return self

    def _matches(self, row: Dict[str, Any]) -> bool:
        for kind, column, value in self.filters:
            cell = row.get(column)
            if kind == "eq" and cell != value:
                return False
            if kind == "gte" and not (cell is not None and str(cell) >= str(value)):
                return False
            if kind == "gt" and not (cell is not None and str(cell) > str(value)):
                return False
            if kind == "in" and str(cell) not in {str(v) for v in value}:
                return False
        return True

    def execute(self) -> Any:
        self.client.queries.append(self)
        rows = [row for row in self.client.rows.get(self.table, []) if self._matches(row)]
        for column, desc in reversed(self.orders):
            rows.sort(key=lambda r: str(r.get(column) or ""), reverse=desc)
        if self.columns is not None:
            rows = [{k: v for k, v in row.items() if k in self.columns} for row in rows]
        if self.window is not None:
            start, end = self.window
            rows = rows[start : end + 1]
        self.client.after_page(self.table)
        return type("Result", (), {"data": rows})()


class FakeClient:
    def __init__(
        self,
        rows: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        after_page: Optional[Any] = None,
    ) -> None:
        self.rows = rows or {}
        self.queries: List[FakeQuery] = []
        self._after_page = after_page

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)

    def queries_for(self, table: str) -> List[FakeQuery]:
        return [q for q in self.queries if q.table == table]

    def after_page(self, table: str) -> None:
        """Hook: lets a test insert a row between page reads, as a live database would."""
        if self._after_page is not None:
            self._after_page(self, table)


def _episode(**over: Any) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    episode = {
        "episode_id": "11111111-1111-1111-1111-111111111111",
        "composition_id": "comp_a",
        # Window membership is created_at; the double filters on it, so a row without one
        # would silently drop out of every query.
        "created_at": now.isoformat(),
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


def _many_episodes(count: int, *, prefix: str = "a") -> List[Dict[str, Any]]:
    return [
        _episode(
            episode_id=f"{prefix}{n:07d}-1111-1111-1111-111111111111",
            composition_id=f"comp_{prefix}{n}",
        )
        for n in range(count)
    ]


def test_a_composition_started_mid_read_is_neither_counted_twice_nor_lost(monkeypatch):
    """Offset paging double-counts under concurrent inserts: row N-1 slides to offset N.

    Compositions are being recorded while an admin reads the page, so this is the ordinary
    case, not a rare one.
    """
    monkeypatch.setattr("src.services.tool_composer_observability_service._PAGE", 10, raising=False)
    rows = {"composer_episodes": _many_episodes(25)}
    inserted: List[int] = []

    def insert_one(client: FakeClient, table: str) -> None:
        # One new episode arrives between pages, as a live recorder would write it. It is NEWER,
        # so under `created_at DESC` it takes first place and pushes every read row down by one.
        if table == "composer_episodes" and not inserted:
            inserted.append(1)
            client.rows["composer_episodes"].append(
                _episode(
                    episode_id="00000000-1111-1111-1111-111111111111",
                    composition_id="comp_arrived",
                    created_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
                )
            )

    service = ToolComposerObservabilityService(client=FakeClient(rows, after_page=insert_one))
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    fetched = service._fetch_episodes(since, True)

    ids = [row["episode_id"] for row in fetched]
    assert len(ids) == len(set(ids)), "an episode was read twice across a page boundary"
    assert len(ids) in (25, 26), f"an episode was lost across a page boundary: {len(ids)}"


def test_the_recent_failures_are_the_newest_ones_not_the_first_by_id():
    """The ten shown are the ten most recent, so the id order must not decide the ordering."""
    now = datetime.now(timezone.utc)
    failures = [
        _episode(
            # UUID order runs opposite to time order: oldest episode sorts first by id.
            episode_id=f"{n:08d}-1111-1111-1111-111111111111",
            composition_id=f"comp_{n}",
            created_at=(now - timedelta(minutes=n)).isoformat(),
            status="FAILED",
            outcome="failed",
        )
        for n in range(14)
    ]
    service = _service({"composer_episodes": failures, "composition_steps": []})

    shown = [row["composition_id"] for row in service.overview(30)["recent_failures"]]

    assert shown == [f"comp_{n}" for n in range(10)]


def test_steps_are_fetched_across_chunks_and_pages_exactly_once(monkeypatch):
    monkeypatch.setattr("src.services.tool_composer_observability_service._PAGE", 3, raising=False)
    monkeypatch.setattr(
        "src.services.tool_composer_observability_service._IN_CHUNK", 2, raising=False
    )
    failures = [
        _episode(
            episode_id=f"{n:08d}-1111-1111-1111-111111111111",
            composition_id=f"comp_{n}",
            status="FAILED",
            outcome="failed",
        )
        for n in range(5)
    ]
    steps = [
        {
            "episode_id": episode["episode_id"],
            "step_id": f"{episode['episode_id']}-{n}",
            "step_number": n,
            "tool_name": "gap_calculator",
            "outcome_class": "error",
        }
        for episode in failures
        for n in range(4)  # more than one page of steps per episode
    ]
    service = _service({"composer_episodes": failures, "composition_steps": steps})

    failures_out = service.overview(30)["recent_failures"]

    by_composition = {row["composition_id"]: row["step_classes"] for row in failures_out}
    assert len(by_composition) == 5
    for composition_id, classes in by_composition.items():
        numbers = [s["step_number"] for s in classes]
        assert numbers == [0, 1, 2, 3], f"{composition_id} got {numbers}"


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
