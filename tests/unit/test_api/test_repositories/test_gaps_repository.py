import pytest

from src.api.routes.gaps import AnalysisStatus, GapAnalysisResponse


class _FakeExec:
    def __init__(self, rows):
        self._rows = rows

    def execute(self):  # supabase-py .execute() is SYNC; repo offloads via to_thread
        class _R:
            data = self._rows

        return _R()


class _FakeQuery:
    """Records upserts into a shared dict keyed by primary key; supports
    select().eq().order().limit() chaining used by the repo."""

    def __init__(self, store, rows=None):
        self._store = store
        self._rows = rows if rows is not None else list(store.values())

    def upsert(self, row, on_conflict=None):
        self._store[row["analysis_id"]] = row
        return _FakeExec([row])

    def select(self, *_a, **_k):
        return _FakeQuery(self._store, list(self._store.values()))

    def eq(self, col, val):
        return _FakeQuery(self._store, [r for r in self._rows if r.get(col) == val])

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        class _R:
            data = self._rows

        return _R()


class _FakeClient:
    def __init__(self):
        self.gap_analyses: dict = {}

    def table(self, name):
        assert name == "gap_analyses"
        return _FakeQuery(self.gap_analyses)


@pytest.mark.asyncio
async def test_upsert_then_get_roundtrips_across_boundary():
    from src.api.repositories.gaps_repository import GapsRepository

    repo = GapsRepository(client=_FakeClient())
    resp = GapAnalysisResponse(
        analysis_id="gap_deadbeef0001",
        status=AnalysisStatus.COMPLETED,
        brand="kisqali",
        metrics_analyzed=["trx"],
        segments_analyzed=3,
    )
    await repo.upsert(resp)
    got = await repo.get("gap_deadbeef0001")
    assert got is not None
    assert got.analysis_id == "gap_deadbeef0001"
    assert got.status == AnalysisStatus.COMPLETED
    assert got.brand == "kisqali"


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    from src.api.repositories.gaps_repository import GapsRepository

    repo = GapsRepository(client=_FakeClient())
    assert await repo.get("gap_does_not_exist") is None


@pytest.mark.asyncio
async def test_list_completed_filters_by_brand():
    from src.api.repositories.gaps_repository import GapsRepository

    client = _FakeClient()
    repo = GapsRepository(client=client)
    await repo.upsert(
        GapAnalysisResponse(
            analysis_id="gap_a",
            status=AnalysisStatus.COMPLETED,
            brand="kisqali",
            metrics_analyzed=["trx"],
            segments_analyzed=1,
        )
    )
    await repo.upsert(
        GapAnalysisResponse(
            analysis_id="gap_b",
            status=AnalysisStatus.PENDING,
            brand="kisqali",
            metrics_analyzed=["trx"],
            segments_analyzed=1,
        )
    )
    await repo.upsert(
        GapAnalysisResponse(
            analysis_id="gap_c",
            status=AnalysisStatus.COMPLETED,
            brand="cosentyx",
            metrics_analyzed=["trx"],
            segments_analyzed=1,
        )
    )

    rows = await repo.list_completed(brand="kisqali")
    ids = {r.analysis_id for r in rows}
    assert ids == {"gap_a"}  # only COMPLETED + brand match
