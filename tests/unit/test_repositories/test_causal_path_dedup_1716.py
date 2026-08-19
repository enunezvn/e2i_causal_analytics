"""#1716 — dedup BEFORE the cap in ``search_paths_for_outcome`` (both twins).

Incident (2026-08-19 full eval, turn 4.7): repeated synthetic loads insert a
fresh random ``path_id`` per run for the SAME causal question, so the registry
holds ~14 raw copies per distinct (cause, outcome, brand) identity (measured:
2,729 rows over 193 identities). The old read applied ``limit`` to RAW rows,
so 15 slots were consumed by 10 copies of one high-confidence
``treatment_arm -> treatment_initiated`` path and the distinct, directly
relevant ``trigger_accepted -> treatment_initiated`` path (0.892, clearing the
0.7 floor) never surfaced.

These tests drive the repository through a PostgREST-faithful fake builder
(``.limit``/``.range`` actually slice the confidence-ordered fixture at
``execute()``), so the pre-fix code REALLY returns 15 raw duplicates — the
red-first pair is genuine, not a mock artifact.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.repositories.causal_path import (
    CausalPathRepository,
    causal_path_identity,
    search_paths_for_outcome_sync,
)

# --------------------------------------------------------------------------
# PostgREST-faithful fake query builder
# --------------------------------------------------------------------------


class _FakeQuery:
    """Supabase query-builder stand-in that HONORS limit/range slicing.

    The fixture rows are pre-sorted confidence-desc (what ``.order`` would
    produce); ``execute()`` returns ``rows[offset : offset + limit]`` exactly
    like PostgREST. A mock that returned every row regardless of ``.limit``
    would let the OLD (cap-before-dedup) code pass these tests vacuously.
    """

    def __init__(self, rows: List[Dict[str, Any]], calls: List[Any]):
        self._rows = rows
        self._calls = calls
        self._offset = 0
        self._limit: Any = None

    # -- filter/order chain (recorded, not applied: fixture is pre-filtered) --
    def select(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        self._calls.append(("select", args))
        return self

    def or_(self, expr: str) -> "_FakeQuery":
        self._calls.append(("or_", expr))
        return self

    def ilike(self, col: str, value: str) -> "_FakeQuery":
        self._calls.append(("ilike", col, value))
        return self

    def gte(self, col: str, value: float) -> "_FakeQuery":
        self._calls.append(("gte", col, value))
        return self

    def eq(self, col: str, value: Any) -> "_FakeQuery":
        self._calls.append(("eq", col, value))
        return self

    def order(self, col: str, desc: bool = False) -> "_FakeQuery":
        self._calls.append(("order", col, desc))
        return self

    # -- slicing (applied) --
    def limit(self, n: int) -> "_FakeQuery":
        self._calls.append(("limit", n))
        self._limit = n
        return self

    def range(self, start: int, end: int) -> "_FakeQuery":
        # Mirrors postgrest BaseSelectRequestBuilder.range: offset=start,
        # limit=end-start+1.
        self._calls.append(("range", start, end))
        self._offset = start
        self._limit = end - start + 1
        return self

    def _sliced(self) -> List[Dict[str, Any]]:
        stop = None if self._limit is None else self._offset + self._limit
        return self._rows[self._offset : stop]


class _FakeAsyncQuery(_FakeQuery):
    async def execute(self) -> Any:
        result = MagicMock()
        result.data = self._sliced()
        return result


class _FakeSyncQuery(_FakeQuery):
    def execute(self) -> Any:
        result = MagicMock()
        result.data = self._sliced()
        return result


class _FakeClient:
    """Each ``table()`` call yields a FRESH builder over the same fixture —
    the paginated read rebuilds its query per page (the real builder's
    ``.range`` uses ``params.add``, so builders are not reusable)."""

    def __init__(self, rows: List[Dict[str, Any]], query_cls: type):
        self._rows = rows
        self._query_cls = query_cls
        self.calls: List[Any] = []

    def table(self, name: str) -> _FakeQuery:
        self.calls.append(("table", name))
        return self._query_cls(self._rows, self.calls)


# --------------------------------------------------------------------------
# Fixture modeled on the measured incident payload
# --------------------------------------------------------------------------


def _row(
    path_id: str,
    start: str,
    end: str,
    brand: str,
    confidence: float,
    via: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "path_id": path_id,
        "start_node": start,
        "end_node": end,
        "brand": brand,
        "confidence_level": confidence,
        "causal_effect_size": 0.4,
        "intermediate_nodes": via or ["disease_severity"],
        "method_used": "backdoor.linear_regression",
        "is_synthetic": True,
    }


def _incident_rows() -> List[Dict[str, Any]]:
    """>15 raw rows of high-confidence duplicates + ONE distinct path above
    the floor — the exact shape that crowded out turn 4.7's answer.
    Confidence-desc order, as the query's ``.order`` returns."""
    rows: List[Dict[str, Any]] = []
    # 11 copies (distinct path_ids, repeated loads) of one identity @ 0.945
    for i in range(11):
        rows.append(
            _row(f"scp_dupA{i:02d}", "treatment_arm", "treatment_initiated", "Remibrutinib", 0.945)
        )
    # 5 copies of a second identity @ 0.944
    for i in range(5):
        rows.append(
            _row(f"scp_dupB{i:02d}", "treatment_arm", "persistent_180d", "Remibrutinib", 0.944)
        )
    # 4 copies of a third identity @ 0.944
    for i in range(4):
        rows.append(
            _row(f"scp_dupC{i:02d}", "treatment_arm", "treatment_initiated", "Kisqali", 0.944)
        )
    # THE distinct, directly relevant path — clears the 0.7 floor, ranked last
    rows.append(
        _row(
            "scp_trigger01",
            "trigger_accepted",
            "treatment_initiated",
            "Kisqali",
            0.892,
            via=["hcp_follow_up"],
        )
    )
    return rows


TRIGGER_ID = "scp_trigger01"


def _assert_deduped_and_trigger_survives(paths: List[Dict[str, Any]]) -> None:
    returned_ids = [p["path_id"] for p in paths]
    # THE incident assertion: the distinct path above the floor must survive
    # into the capped result — pre-fix, 15 raw duplicate rows fill the cap
    # and this fails.
    assert TRIGGER_ID in returned_ids, (
        f"distinct trigger_accepted path crowded out by duplicates; got {returned_ids}"
    )
    # No identity may appear twice post-dedup.
    identities = [causal_path_identity(p) for p in paths]
    assert len(identities) == len(set(identities)), f"duplicate identities returned: {identities}"
    # 3 duplicate groups + 1 distinct path = 4 distinct identities.
    assert len(paths) == 4
    # Representatives keep confidence-desc order; first-seen = max-confidence.
    confs = [p["confidence_level"] for p in paths]
    assert confs == sorted(confs, reverse=True)
    assert paths[0]["confidence_level"] == 0.945


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_distinct_path_survives_the_cap() -> None:
    client = _FakeClient(_incident_rows(), _FakeAsyncQuery)
    repo = CausalPathRepository(supabase_client=client)

    paths = await repo.search_paths_for_outcome(
        "treatment_initiated",
        min_confidence=0.7,
        limit=15,
        include_synthetic=True,
    )

    _assert_deduped_and_trigger_survives(paths)


@pytest.mark.unit
def test_sync_twin_distinct_path_survives_the_cap() -> None:
    """The sync twin (#1475) shares the defect — same cap-before-dedup read."""
    client = _FakeClient(_incident_rows(), _FakeSyncQuery)

    paths = search_paths_for_outcome_sync(
        "treatment_initiated",
        client=client,
        min_confidence=0.7,
        limit=15,
        include_synthetic=True,
    )

    _assert_deduped_and_trigger_survives(paths)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_limit_counts_distinct_paths_not_raw_rows() -> None:
    """With more distinct identities than the cap, exactly ``limit`` distinct
    paths return — the highest-confidence ones — even when every identity
    carries duplicate rows."""
    rows: List[Dict[str, Any]] = []
    for i in range(20):  # 20 identities, 2 raw copies each, conf desc
        conf = round(0.95 - i * 0.01, 3)
        for copy in range(2):
            rows.append(_row(f"scp_i{i:02d}c{copy}", f"driver_{i:02d}", "outcome", "Kisqali", conf))

    client = _FakeClient(rows, _FakeAsyncQuery)
    repo = CausalPathRepository(supabase_client=client)

    paths = await repo.search_paths_for_outcome(
        "outcome", min_confidence=0.7, limit=15, include_synthetic=True
    )

    assert len(paths) == 15
    starts = [p["start_node"] for p in paths]
    assert starts == [f"driver_{i:02d}" for i in range(15)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pagination_reaches_past_the_first_page() -> None:
    """A distinct path buried past the first DB page (duplicates measured up
    to raw rank ~866 live) is still found: the read pages forward until the
    cap is met in DISTINCT paths or rows run out."""
    rows: List[Dict[str, Any]] = []
    for i in range(505):  # one identity floods page 1 (page size 500)
        rows.append(_row(f"scp_flood{i:03d}", "treatment_arm", "treatment_initiated", "K", 0.945))
    rows.append(_row("scp_deep01", "trigger_accepted", "treatment_initiated", "K", 0.892))

    client = _FakeClient(rows, _FakeAsyncQuery)
    repo = CausalPathRepository(supabase_client=client)

    paths = await repo.search_paths_for_outcome(
        "treatment_initiated", min_confidence=0.7, limit=15, include_synthetic=True
    )

    ids = [p["path_id"] for p in paths]
    assert ids == ["scp_flood000", "scp_deep01"]
