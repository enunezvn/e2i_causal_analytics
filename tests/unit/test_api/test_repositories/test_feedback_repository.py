import pytest

from src.api.routes.feedback import (
    DetectedPattern,
    KnowledgeUpdate,
    LearningResponse,
    LearningStatus,
    PatternSeverity,
    PatternType,
    UpdateStatus,
    UpdateType,
)


class _FakeQuery:
    def __init__(self, store, pk, rows=None):
        self._store = store
        self._pk = pk
        self._rows = rows if rows is not None else list(store.values())

    def upsert(self, row, on_conflict=None):
        self._store[row[self._pk]] = row
        return _FakeQuery(self._store, self._pk, [row])

    def insert(self, row):
        self._store[row[self._pk]] = row
        return _FakeQuery(self._store, self._pk, [row])

    def select(self, *_a, **_k):
        return _FakeQuery(self._store, self._pk, list(self._store.values()))

    def eq(self, col, val):
        return _FakeQuery(self._store, self._pk, [r for r in self._rows if r.get(col) == val])

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        class _R:
            data = self._rows

        return _R()


_PK = {
    "feedback_learning_batches": "batch_id",
    "feedback_patterns": "pattern_id",
    "feedback_knowledge_updates": "update_id",
    "feedback_items": "feedback_id",
}


class _FakeClient:
    def __init__(self):
        self.stores = {name: {} for name in _PK}

    def table(self, name):
        return _FakeQuery(self.stores[name], _PK[name])


@pytest.mark.asyncio
async def test_learning_batch_roundtrip():
    from src.api.repositories.feedback_repository import FeedbackRepository

    repo = FeedbackRepository(client=_FakeClient())
    resp = LearningResponse(batch_id="fb_0001", status=LearningStatus.COMPLETED)
    await repo.upsert_batch(resp)
    got = await repo.get_batch("fb_0001")
    assert got is not None and got.batch_id == "fb_0001"
    assert got.status == LearningStatus.COMPLETED
    assert await repo.get_batch("fb_missing") is None


@pytest.mark.asyncio
async def test_pattern_and_update_listing():
    from src.api.repositories.feedback_repository import FeedbackRepository

    repo = FeedbackRepository(client=_FakeClient())
    await repo.upsert_pattern(
        DetectedPattern(
            pattern_id="p1",
            pattern_type=PatternType.ACCURACY_ISSUE,
            description="x",
            frequency=4,
            severity=PatternSeverity.HIGH,
            affected_agents=["gap_analyzer"],
            example_feedback_ids=["f1"],
            root_cause_hypothesis="y",
            confidence=0.8,
        )
    )
    patterns = await repo.list_patterns()
    assert [p.pattern_id for p in patterns] == ["p1"]

    upd = KnowledgeUpdate(
        update_id="u1",
        update_type=UpdateType.PROMPT_REFINEMENT,
        status=UpdateStatus.PROPOSED,
        target_agent="gap_analyzer",
        target_component="prompt",
        proposed_value="v",
        rationale="r",
        expected_improvement="i",
    )
    await repo.upsert_update(upd)
    assert (await repo.get_update("u1")).status == UpdateStatus.PROPOSED

    upd.status = UpdateStatus.APPLIED
    await repo.upsert_update(upd)
    assert (await repo.get_update("u1")).status == UpdateStatus.APPLIED
