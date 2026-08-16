"""#1661: the optimizer gate must be measurable from outside the beat.

Two things are pinned here.

1. **The measured cause of "8 signals".** A learning cycle that detects zero
   patterns is *mathematically* incapable of producing a signal that clears the
   optimizer's ``reward >= 0.5`` floor: with ``patterns_detected == 0`` the
   coverage term is 0 and the actionability term is 0, so even a perfect 5.0
   rubric on a perfectly efficient cycle tops out at exactly 0.5 (0.3 with no
   rubric at all). Since the pattern analyzer only emits patterns when it finds
   real quality defects, eligibility is coupled to the platform behaving
   *badly*. Measured in prod 2026-08-16: 218 feedback_learner signals, 203 with
   zero patterns, none of which ever reached 0.5; all 8 eligible signals came
   from cycles that found >= 2 patterns, inside the single 2026-08-05..08-08
   window where user-reward ratings dipped below 3.0.

2. **The gate's constants are one SSOT**, and a status reader derives the
   operator-visible numbers from that same SSOT — so the health surface can
   never quietly disagree with the beat that does the skipping.
"""

from __future__ import annotations

import pytest

from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal


def _signal(**kw) -> FeedbackLearnerTrainingSignal:
    base = {
        "batch_id": "b",
        "feedback_count": 100,
        "time_range_start": "2026-08-16T00:00:00Z",
        "time_range_end": "2026-08-16T06:00:00Z",
    }
    base.update(kw)
    return FeedbackLearnerTrainingSignal(**base)  # type: ignore[arg-type]


# =============================================================================
# 1. The measured ceiling — why the eligible count is 8 and not 20
# =============================================================================


def test_zero_patterns_caps_reward_at_the_gate_floor():
    """A best-possible pattern-free cycle lands exactly ON the 0.5 floor.

    Perfect rubric (5.0) + perfect efficiency + zero patterns == 0.5. Anything
    less than a flawless rubric falls below it, which is why 203 of 218 real
    signals never qualified (observed max 0.4100 at rubric 3.92).
    """
    from src.agents.feedback_learner.signal_store import OPTIMIZER_MIN_REWARD

    best = _signal(
        patterns_detected=0,
        recommendation_actionability=0.0,
        total_latency_ms=1.0,
        rubric_weighted_score=5.0,
    )
    assert best.compute_reward() == OPTIMIZER_MIN_REWARD

    # One notch off a flawless rubric already drops below the floor.
    realistic = _signal(
        patterns_detected=0,
        recommendation_actionability=0.0,
        total_latency_ms=1.0,
        rubric_weighted_score=4.5,
    )
    assert realistic.compute_reward() < OPTIMIZER_MIN_REWARD


def test_zero_patterns_without_rubric_caps_far_below_the_floor():
    best = _signal(
        patterns_detected=0,
        recommendation_actionability=0.0,
        total_latency_ms=1.0,
    )
    assert best.compute_reward() == pytest.approx(0.3)


def test_replays_two_real_production_rows_exactly():
    """Guard the model of the reward against the stored values it explains.

    Both rows are real ``dspy_agent_training_signals`` rows read on 2026-08-16;
    if compute_reward ever stops reproducing them, the ceiling analysis above
    has gone stale and must be re-measured before it is trusted.
    """
    # Pattern-free cycle, rubric 3.92 -> stored reward 0.4100 (never eligible).
    ineligible = _signal(
        feedback_count=74,
        patterns_detected=0,
        recommendation_actionability=0.0,
        total_latency_ms=69.0,
        rubric_weighted_score=3.92,
    )
    assert ineligible.compute_reward() == pytest.approx(0.4100)

    # Cycle that found 2 patterns -> stored reward 0.6507 (one of the 8).
    eligible = _signal(
        feedback_count=26,
        patterns_detected=2,
        recommendation_actionability=0.4,
        total_latency_ms=32.0,
        rubric_weighted_score=3.67,
    )
    assert eligible.compute_reward() == pytest.approx(0.6507)


# =============================================================================
# 2. Gate constants are a single source of truth
# =============================================================================


def test_beat_reads_the_gate_constants_from_signal_store():
    """The beat must not re-declare the floor/threshold as bare literals."""
    from src.agents.feedback_learner import signal_store
    from src.tasks import dspy_optimization_tasks as task

    assert task.OPTIMIZER_MIN_REWARD is signal_store.OPTIMIZER_MIN_REWARD
    assert task.optimizer_min_signals is signal_store.optimizer_min_signals


def test_min_signals_honours_the_env_override(monkeypatch):
    from src.agents.feedback_learner.signal_store import DEFAULT_MIN_SIGNALS, optimizer_min_signals

    monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
    assert optimizer_min_signals() == DEFAULT_MIN_SIGNALS
    monkeypatch.setenv("DSPY_MIN_SIGNALS", "7")
    assert optimizer_min_signals() == 7
    # A garbled value must not crash the health surface.
    monkeypatch.setenv("DSPY_MIN_SIGNALS", "not-a-number")
    assert optimizer_min_signals() == DEFAULT_MIN_SIGNALS


# =============================================================================
# 3. The status reader — real queries, no fabricated numbers
# =============================================================================


class _FakeQuery:
    def __init__(self, table: str, sink: dict, data, count: int):
        self.table = table
        self.sink = sink
        self._data = data
        self._count = count
        self.filters: list = []

    def select(self, *a, **k):
        self.sink.setdefault("selects", []).append((self.table, a, k))
        return self

    def eq(self, col, val):
        self.filters.append(("eq", col, val))
        return self

    def gte(self, col, val):
        self.filters.append(("gte", col, val))
        return self

    def order(self, col, desc=False):
        self.filters.append(("order", col, desc))
        return self

    def limit(self, n):
        self.filters.append(("limit", n))
        return self

    async def execute(self):
        self.sink.setdefault("queries", []).append((self.table, list(self.filters)))
        return type("R", (), {"data": self._data, "count": self._count})()


class _FakeClient:
    """Stand-in that answers the three reads the status needs."""

    def __init__(self, sink: dict, *, eligible: int, total: int, runs: int, last: str | None):
        self.sink = sink
        self.eligible = eligible
        self.total = total
        self.runs = runs
        self.last = last

    def table(self, name):
        if name == "prompt_optimization_runs":
            return _FakeQuery(name, self.sink, [], self.runs)

        # dspy_agent_training_signals: the eligible read carries a gte filter,
        # the total read does not. Decide by inspecting the built query.
        client = self

        class _Dispatch(_FakeQuery):
            async def execute(self):
                client.sink.setdefault("queries", []).append((name, list(self.filters)))
                is_eligible_read = any(f[0] == "gte" for f in self.filters)
                if is_eligible_read:
                    rows = [{"created_at": client.last}] if client.last else []
                    return type("R", (), {"data": rows, "count": client.eligible})()
                return type("R", (), {"data": [], "count": client.total})()

        return _Dispatch(name, self.sink, [], 0)


@pytest.mark.asyncio
async def test_gate_status_reports_real_counts_and_the_gate_verdict():
    from src.agents.feedback_learner.signal_store import (
        OPTIMIZER_MIN_REWARD,
        get_optimizer_gate_status,
    )

    sink: dict = {}
    client = _FakeClient(
        sink, eligible=8, total=218, runs=0, last="2026-08-08T07:09:02.686027+00:00"
    )
    status = await get_optimizer_gate_status(client=client)

    assert status["eligible_signals"] == 8
    assert status["total_signals"] == 218
    assert status["optimization_runs"] == 0
    assert status["min_reward"] == OPTIMIZER_MIN_REWARD
    assert status["min_signals"] == 20
    assert status["would_trigger"] is False
    assert status["last_eligible_signal_at"] == "2026-08-08T07:09:02.686027+00:00"
    # The denominator is what stops "8 < 20" reading as a volume problem.
    assert "218" in status["reason"] and "8" in status["reason"]

    # The eligible read must use the gate's own filters, not a re-invention.
    signal_queries = [f for t, f in sink["queries"] if t == "dspy_agent_training_signals"]
    eligible_filters = next(f for f in signal_queries if any(x[0] == "gte" for x in f))
    assert ("eq", "source_agent", "feedback_learner") in eligible_filters
    assert ("gte", "reward", OPTIMIZER_MIN_REWARD) in eligible_filters


@pytest.mark.asyncio
async def test_gate_status_flips_to_would_trigger_when_supply_clears_threshold():
    from src.agents.feedback_learner.signal_store import get_optimizer_gate_status

    status = await get_optimizer_gate_status(
        client=_FakeClient({}, eligible=25, total=300, runs=3, last="2026-08-16T00:00:00+00:00")
    )
    assert status["would_trigger"] is True
    assert status["optimization_runs"] == 3


@pytest.mark.asyncio
async def test_gate_status_degrades_honestly_without_a_client(monkeypatch):
    """No client -> no numbers. Never a fabricated zero that reads as measured."""
    from src.agents.feedback_learner import signal_store

    async def _none():
        return None

    monkeypatch.setattr("src.memory.services.factories.get_supabase_client", _none, raising=False)
    status = await signal_store.get_optimizer_gate_status(client=None)
    assert status["eligible_signals"] is None
    assert status["total_signals"] is None
    assert status["would_trigger"] is None
    assert "unavailable" in status["reason"].lower()
