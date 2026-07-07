"""Unit tests for LearningSignalsFeedbackStore.

The store adapts real cognitive-workflow reward signals (``learning_signals``
dspy rows) into the feedback-item contract the feedback_learner collector
expects. These tests pin the provenance predicates (synthetic rows must NEVER
feed the learner), the reward→rating mapping, and the agent attribution rules.

Uses the self-chaining recorder idiom from test_agent_activity.py (#894).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.learning_signals_feedback import (
    LearningSignalsFeedbackStore,
    get_learning_signals_feedback_store,
)


class _RecordingQuery:
    """supabase-style fluent builder: records calls, chains itself."""

    def __init__(self, data: list | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict]] = []
        self._data = data or []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> "_RecordingQuery":
        self.calls.append((name, args, kwargs))
        return self

    def select(self, *a: Any, **kw: Any) -> "_RecordingQuery":
        return self._record("select", *a, **kw)

    def eq(self, *a: Any) -> "_RecordingQuery":
        return self._record("eq", *a)

    def gte(self, *a: Any) -> "_RecordingQuery":
        return self._record("gte", *a)

    def lte(self, *a: Any) -> "_RecordingQuery":
        return self._record("lte", *a)

    def order(self, *a: Any, **kw: Any) -> "_RecordingQuery":
        return self._record("order", *a, **kw)

    def limit(self, *a: Any) -> "_RecordingQuery":
        return self._record("limit", *a)

    def execute(self) -> Any:
        result = MagicMock()
        result.data = list(self._data)
        return AsyncMock(return_value=result)()

    def named(self, name: str) -> list[tuple[Any, ...]]:
        return [args for (n, args, _kw) in self.calls if n == name]


class _RecordingClient:
    def __init__(self, data: list | None = None) -> None:
        self._data = data or []
        self.tables: list[str] = []
        self.query: _RecordingQuery | None = None

    def table(self, name: str) -> _RecordingQuery:
        self.tables.append(name)
        self.query = _RecordingQuery(self._data)
        return self.query


def _row(
    component: str = "agent",
    reward: Any = 0.8,
    routed: list | None = None,
    signal_id: str = "sig-1",
) -> dict:
    return {
        "signal_id": signal_id,
        "created_at": "2026-07-04T22:51:56+00:00",
        "signal_details": {
            "type": component,
            "query": "Intent: GAP_ANALYSIS, Evidence: 4 items",
            "response": "Remibrutinib strategy...",
            "reward": reward,
            "feedback": None,
            "metadata": {
                "routed_agents": routed if routed is not None else [],
                "conversation_id": "conv-1",
            },
            "domain_signal": "dspy_signal",
        },
    }


@pytest.mark.unit
class TestLearningSignalsFeedbackStore:
    @pytest.mark.asyncio
    async def test_maps_rows_to_collector_contract(self):
        client = _RecordingClient(
            [
                _row("agent", 0.8, routed=["gap_analyzer", "causal_impact"], signal_id="s1"),
                _row("investigator", 0.767, signal_id="s2"),
                _row("summarizer", 1.0, signal_id="s3"),
            ]
        )
        store = LearningSignalsFeedbackStore(client)
        items = await store.get_feedback()

        assert [i["id"] for i in items] == ["s1", "s2", "s3"]
        # type=agent → first routed agent; components → cognitive_<type>
        assert items[0]["agent"] == "gap_analyzer"
        assert items[1]["agent"] == "cognitive_investigator"
        assert items[2]["agent"] == "cognitive_summarizer"
        # reward 0..1 → rating on the analyzer's 1-5 scale (0.8 → 4.2); the raw
        # reward survives in metadata for insight math
        assert items[0]["rating"] == pytest.approx(4.2)
        assert items[0]["metadata"]["reward"] == pytest.approx(0.8)
        assert isinstance(items[1]["rating"], float)
        assert items[0]["query"].startswith("Intent:")
        assert items[0]["metadata"]["source"] == "learning_signals"
        assert items[0]["metadata"]["conversation_id"] == "conv-1"

    @pytest.mark.asyncio
    async def test_rating_scale_matches_analyzer_contract(self):
        """pattern_analyzer._rating_to_numeric passes numerics through and flags
        avg < 3.0 as a low-ratings pattern. Raw 0..1 rewards would ALWAYS read
        as abysmal ratings and fabricate that pattern every cycle — the store
        must emit the same 1-5 scale the thumbs map onto (up→5, down→1)."""
        rows = [
            _row("agent", 0.0, routed=["a"], signal_id="worst"),
            _row("agent", 0.5, routed=["a"], signal_id="mid"),
            _row("agent", 1.0, routed=["a"], signal_id="best"),
        ]
        store = LearningSignalsFeedbackStore(_RecordingClient(rows))
        items = {i["id"]: i["rating"] for i in await store.get_feedback()}
        assert items["worst"] == pytest.approx(1.0)
        assert items["mid"] == pytest.approx(3.0)
        assert items["best"] == pytest.approx(5.0)
        # A healthy real stream (avg reward ~0.87) must sit ABOVE the 3.0 gate
        assert 1.0 + 4.0 * 0.87 > 3.0

    @pytest.mark.asyncio
    async def test_pins_provenance_and_dspy_filters(self):
        """Synthetic showcase rows must NEVER feed the learner, and only
        dspy_signal-shaped rows are graded feedback."""
        client = _RecordingClient([])
        store = LearningSignalsFeedbackStore(client)
        await store.get_feedback(start_time="2026-06-30T00:00:00+00:00", limit=100)

        assert client.tables == ["learning_signals"]
        q = client.query
        assert ("is_synthetic", False) in q.named("eq")
        assert ("signal_details->>domain_signal", "dspy_signal") in q.named("eq")
        assert ("created_at", "2026-06-30T00:00:00+00:00") in q.named("gte")
        assert (100,) in q.named("limit")

    @pytest.mark.asyncio
    async def test_skips_rows_without_numeric_reward(self):
        rows = [_row("agent", None, signal_id="bad"), _row("agent", "high", signal_id="bad2")]
        rows.append(_row("agent", 0.6, routed=["explainer"], signal_id="good"))
        store = LearningSignalsFeedbackStore(_RecordingClient(rows))
        items = await store.get_feedback()
        assert [i["id"] for i in items] == ["good"]

    @pytest.mark.asyncio
    async def test_agent_without_routed_agents_credits_orchestrator(self):
        store = LearningSignalsFeedbackStore(_RecordingClient([_row("agent", 0.8, routed=[])]))
        items = await store.get_feedback()
        assert items[0]["agent"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_agents_filter_applies_to_mapped_attribution(self):
        rows = [
            _row("agent", 0.8, routed=["gap_analyzer"], signal_id="s1"),
            _row("investigator", 0.7, signal_id="s2"),
        ]
        store = LearningSignalsFeedbackStore(_RecordingClient(rows))
        items = await store.get_feedback(agents=["gap_analyzer"])
        assert [i["id"] for i in items] == ["s1"]

    @pytest.mark.asyncio
    async def test_no_client_returns_empty(self):
        store = get_learning_signals_feedback_store(supabase_client=None)
        assert await store.get_feedback() == []
