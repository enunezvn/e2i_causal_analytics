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
    def __init__(self, store, pk, rows=None, projection=None):
        self._store = store
        self._pk = pk
        self._rows = rows if rows is not None else list(store.values())
        self._projection = projection

    def upsert(self, row, on_conflict=None):
        self._store[row[self._pk]] = row
        return _FakeQuery(self._store, self._pk, [row])

    def insert(self, row):
        self._store[row[self._pk]] = row
        return _FakeQuery(self._store, self._pk, [row])

    def select(self, *cols, **_k):
        # #1262: honor the projection like PostgREST does — a fake that
        # returns every column regardless passes tests for code whose real
        # query never fetched the column it reads (proved by mutation:
        # reverting list_patterns to select("payload") kept both #1244
        # backfill tests green while killing the backfill in production).
        # Filters still apply to full rows (PostgREST filters the table,
        # not the projection) — columns are stripped at execute().
        projection = None
        if cols and "*" not in cols:
            projection = {c.strip() for c in ",".join(cols).split(",")}
        return _FakeQuery(self._store, self._pk, list(self._store.values()), projection)

    def eq(self, col, val):
        return _FakeQuery(
            self._store,
            self._pk,
            [r for r in self._rows if r.get(col) == val],
            self._projection,
        )

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        rows = self._rows
        if self._projection is not None:
            rows = [{k: v for k, v in r.items() if k in self._projection} for r in rows]

        class _R:
            data = rows

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


@pytest.mark.asyncio
async def test_list_patterns_injects_detected_at_from_row_created_at():
    """#1244: DetectedPattern.detected_at is plumbed from the persistence
    row's created_at (DB default now()) when the payload doesn't carry one —
    the frontend's Recent Activity timestamp has no other source (patterns
    written before this field existed have created_at but no payload field).
    """
    from src.api.repositories.feedback_repository import FeedbackRepository

    client = _FakeClient()
    repo = FeedbackRepository(client=client)
    await repo.upsert_pattern(
        DetectedPattern(
            pattern_id="p_ts",
            pattern_type=PatternType.ACCURACY_ISSUE,
            severity=PatternSeverity.HIGH,
            description="x",
            frequency=2,
            affected_agents=["cognitive_investigator"],
            example_feedback_ids=["f1"],
            root_cause_hypothesis="y",
            confidence=0.8,
        )
    )
    # Simulate the DB default: the stored row gains created_at server-side.
    client.stores["feedback_patterns"]["p_ts"]["created_at"] = "2026-07-15T22:13:29+00:00"

    patterns = await repo.list_patterns()
    assert len(patterns) == 1
    got = patterns[0]
    assert got.detected_at is not None
    assert got.detected_at.isoformat().startswith("2026-07-15T22:13:29")


@pytest.mark.asyncio
async def test_list_patterns_payload_detected_at_wins_over_row_created_at():
    """A payload that already carries detected_at (written by post-#1244 code)
    must keep it — the row's created_at only backfills legacy payloads."""
    from datetime import datetime, timezone

    from src.api.repositories.feedback_repository import FeedbackRepository

    client = _FakeClient()
    repo = FeedbackRepository(client=client)
    stamped = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    await repo.upsert_pattern(
        DetectedPattern(
            pattern_id="p_keep",
            pattern_type=PatternType.ACCURACY_ISSUE,
            severity=PatternSeverity.MEDIUM,
            description="x",
            frequency=1,
            affected_agents=["gap_analyzer"],
            example_feedback_ids=[],
            root_cause_hypothesis="z",
            confidence=0.7,
            detected_at=stamped,
        )
    )
    client.stores["feedback_patterns"]["p_keep"]["created_at"] = "2026-07-15T00:00:00+00:00"

    got = (await repo.list_patterns())[0]
    assert got.detected_at == stamped
