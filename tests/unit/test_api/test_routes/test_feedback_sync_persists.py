"""Regression: the SYNC learning path (``async_mode=False``) must persist the
patterns and updates it detects, not just the batch.

Root cause this guards against
------------------------------
``run_learning_cycle(async_mode=False)`` executed the cycle and persisted the
**batch** (``_persist_batch``) but never looped over the result's
``detected_patterns`` / ``proposed_updates`` to call ``_persist_pattern`` /
``_persist_update``. Only the ASYNC background task (``_run_learning_task``)
did. The FeedbackLearning page drives the sync path
(``useQuickLearningCycle`` -> ``async_mode=false``), so detected patterns and
proposed updates were computed and thrown away -> the Patterns / Updates tabs
stayed empty even though a real cycle ran.

These tests assert that after a sync cycle the patterns/updates are listable
from the repo (i.e. what the GET /feedback/patterns and /feedback/updates
endpoints read), exactly as the async path already guarantees.
"""

import pytest

import src.api.routes.feedback as fb
from src.api.routes.feedback import (
    DetectedPattern,
    KnowledgeUpdate,
    LearningResponse,
    LearningStatus,
    PatternSeverity,
    PatternType,
    RunLearningRequest,
    UpdateStatus,
    UpdateType,
    list_patterns,
    list_updates,
    run_learning_cycle,
)


class _Repo:
    """Minimal in-memory stand-in for FeedbackRepository."""

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


def _cycle_result_with_one_pattern_and_update() -> LearningResponse:
    """A completed cycle carrying one detected pattern and one proposed update.

    Mirrors the real shape ``_execute_learning_cycle`` returns; we stub the
    cycle itself so the test is deterministic and offline (no agent graph, no
    DB read of learning_signals).
    """
    return LearningResponse(
        batch_id="",  # set by the endpoint
        status=LearningStatus.COMPLETED,
        detected_patterns=[
            DetectedPattern(
                pattern_id="pat_sync1",
                pattern_type=PatternType.ACCURACY_ISSUE,
                description="Multiple low ratings for causal_impact responses",
                frequency=4,
                severity=PatternSeverity.HIGH,
                affected_agents=["causal_impact"],
                example_feedback_ids=["fbi_1", "fbi_2"],
                root_cause_hypothesis="Insufficient grounding in RWD",
                confidence=0.82,
            )
        ],
        proposed_updates=[
            KnowledgeUpdate(
                update_id="upd_sync1",
                update_type=UpdateType.PROMPT_REFINEMENT,
                status=UpdateStatus.PROPOSED,
                target_agent="causal_impact",
                target_component="system_prompt",
                proposed_value="Add explicit RWD grounding instruction",
                rationale="Addresses pat_sync1",
                expected_improvement="Higher answer ratings",
            )
        ],
        patterns_detected=1,
        updates_proposed=1,
    )


@pytest.mark.asyncio
async def test_sync_cycle_persists_detected_patterns(repo, monkeypatch):
    """async_mode=False must persist detected patterns so the tab is non-empty."""

    async def _fake_execute(_request):
        return _cycle_result_with_one_pattern_and_update()

    monkeypatch.setattr(fb, "_execute_learning_cycle", _fake_execute)

    # FastAPI normally injects these; call the handler directly.
    from fastapi import BackgroundTasks

    await run_learning_cycle(
        request=RunLearningRequest(min_feedback_count=1),
        background_tasks=BackgroundTasks(),
        async_mode=False,
        user={"sub": "tester", "role": "operator"},
    )

    # Read back via the same path the GET /feedback/patterns endpoint uses.
    resp = await list_patterns(severity=None, pattern_type=None, agent=None, limit=50)
    assert [p.pattern_id for p in resp.patterns] == ["pat_sync1"]


@pytest.mark.asyncio
async def test_sync_cycle_persists_proposed_updates(repo, monkeypatch):
    """async_mode=False must persist proposed updates so the tab is non-empty."""

    async def _fake_execute(_request):
        return _cycle_result_with_one_pattern_and_update()

    monkeypatch.setattr(fb, "_execute_learning_cycle", _fake_execute)

    from fastapi import BackgroundTasks

    await run_learning_cycle(
        request=RunLearningRequest(min_feedback_count=1),
        background_tasks=BackgroundTasks(),
        async_mode=False,
        user={"sub": "tester", "role": "operator"},
    )

    resp = await list_updates(status=None, update_type=None, agent=None, limit=50)
    assert [u.update_id for u in resp.updates] == ["upd_sync1"]
