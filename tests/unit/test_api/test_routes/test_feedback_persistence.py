import pytest

import src.api.routes.feedback as fb
from src.api.routes.feedback import (
    DetectedPattern,
    LearningResponse,
    LearningStatus,
    PatternSeverity,
    PatternType,
    get_learning_results,
    list_patterns,
)


class _Repo:
    def __init__(self):
        self.batches, self.patterns, self.updates = {}, {}, {}

    async def upsert_batch(self, r):
        self.batches[r.batch_id] = r

    async def get_batch(self, b):
        return self.batches.get(b)

    async def upsert_pattern(self, p):
        self.patterns[p.pattern_id] = p

    async def list_patterns(self):
        return list(self.patterns.values())

    async def upsert_update(self, u):
        self.updates[u.update_id] = u

    async def get_update(self, u):
        return self.updates.get(u)

    async def list_updates(self):
        return list(self.updates.values())

    async def count_recent_and_last(self):
        return list(self.batches.values())

    async def append_item(self, i):
        pass


@pytest.fixture
def repo(monkeypatch):
    r = _Repo()
    monkeypatch.setattr(fb, "_get_repo", lambda: r)
    monkeypatch.setattr(fb, "_use_inmemory_fallback", lambda: False)
    fb._learning_store.clear()
    fb._patterns_store.clear()
    fb._updates_store.clear()
    return r


@pytest.mark.asyncio
async def test_get_batch_reads_from_repo(repo):
    await repo.upsert_batch(LearningResponse(batch_id="fb_x", status=LearningStatus.COMPLETED))
    got = await get_learning_results("fb_x")
    assert got.batch_id == "fb_x"


@pytest.mark.asyncio
async def test_list_patterns_reads_from_repo(repo):
    await repo.upsert_pattern(
        DetectedPattern(
            pattern_id="p9",
            pattern_type=PatternType.ACCURACY_ISSUE,
            description="d",
            frequency=3,
            severity=PatternSeverity.CRITICAL,
            affected_agents=["a"],
            example_feedback_ids=["f"],
            root_cause_hypothesis="rc",
            confidence=0.9,
        )
    )
    resp = await list_patterns(severity=None, pattern_type=None, agent=None, limit=50)
    assert [p.pattern_id for p in resp.patterns] == ["p9"]
    assert resp.critical_count == 1
