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
    """The beat must not re-declare the floor/limit/state path as bare literals."""
    from src.agents.feedback_learner import signal_store
    from src.tasks import dspy_optimization_tasks as task

    assert task.OPTIMIZER_MIN_REWARD is signal_store.OPTIMIZER_MIN_REWARD
    assert task.OPTIMIZER_SIGNAL_LIMIT is signal_store.OPTIMIZER_SIGNAL_LIMIT
    assert task._STATE_PATH is signal_store.TRIGGER_STATE_PATH


def test_beat_and_health_surface_share_one_trigger_decision():
    """The health surface must not re-implement a subset of the gate.

    The real trigger checks cooldown BEFORE the signal count, then a forced
    interval, then a reward delta. A status field that models only "count >=
    threshold" would report Ready while the beat still skips — reproducing, one
    layer up, exactly the false-green this issue is about.
    """
    from src.agents.feedback_learner import signal_store
    from src.tasks import dspy_optimization_tasks as task

    assert task._decide_trigger is signal_store.decide_optimizer_trigger
    assert task._load_trigger_state is signal_store.load_trigger_state


def test_cooldown_binds_even_when_the_signal_gate_is_satisfied():
    """Plenty of signals + a recent optimization must NOT read as Ready."""
    from datetime import datetime, timedelta, timezone

    from src.agents.feedback_learner.signal_store import decide_optimizer_trigger

    signals = [{"reward": 0.9}] * 50
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    should, reason = decide_optimizer_trigger(signals, {"last_optimization": recent})
    assert should is False
    assert "Cooldown active" in reason


def test_signal_gate_is_what_binds_with_no_prior_optimization():
    """Today's real state: no trigger file, 8 eligible signals."""
    from src.agents.feedback_learner.signal_store import decide_optimizer_trigger

    should, reason = decide_optimizer_trigger([{"reward": 0.6}] * 8, {})
    assert should is False
    assert reason == "Insufficient signals: 8 < 20"


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

        # dspy_agent_training_signals: the eligible read carries a gte filter
        # (and returns rows, because the trigger needs their mean reward); the
        # total read does not. Decide by inspecting the built query.
        client = self

        class _Dispatch(_FakeQuery):
            async def execute(self):
                client.sink.setdefault("queries", []).append((name, list(self.filters)))
                if any(f[0] == "gte" for f in self.filters):
                    rows = [
                        {"reward": 0.6, "created_at": client.last} for _ in range(client.eligible)
                    ]
                    # Honour the limit the way PostgREST does — a fake that
                    # ignored it could not catch a count/verdict mismatch.
                    cap = next((f[1] for f in self.filters if f[0] == "limit"), None)
                    if cap is not None:
                        rows = rows[:cap]
                    return type("R", (), {"data": rows, "count": client.eligible})()
                return type("R", (), {"data": [], "count": client.total})()

        return _Dispatch(name, self.sink, [], 0)


@pytest.mark.asyncio
async def test_gate_status_reports_real_counts_and_the_gate_verdict(monkeypatch):
    from src.agents.feedback_learner import signal_store

    monkeypatch.setattr(signal_store, "load_trigger_state", dict)

    sink: dict = {}
    client = _FakeClient(
        sink, eligible=8, total=218, runs=0, last="2026-08-08T07:09:02.686027+00:00"
    )
    status = await signal_store.get_optimizer_gate_status(client=client)

    assert status["eligible_signals"] == 8
    assert status["total_signals"] == 218
    assert status["optimization_runs"] == 0
    assert status["min_reward"] == signal_store.OPTIMIZER_MIN_REWARD
    assert status["min_signals"] == 20
    assert status["would_trigger"] is False
    assert status["last_eligible_signal_at"] == "2026-08-08T07:09:02.686027+00:00"
    # Verbatim from the REAL trigger, not a re-worded copy.
    assert status["reason"] == "Insufficient signals: 8 < 20"
    # The denominator is what stops "8 < 20" reading as a volume problem.
    assert status["total_signals"] == 218

    # The eligible read must use the gate's own filters, not a re-invention.
    signal_queries = [f for t, f in sink["queries"] if t == "dspy_agent_training_signals"]
    eligible_filters = next(f for f in signal_queries if any(x[0] == "gte" for x in f))
    assert ("eq", "source_agent", "feedback_learner") in eligible_filters
    assert ("gte", "reward", signal_store.OPTIMIZER_MIN_REWARD) in eligible_filters
    assert ("limit", signal_store.OPTIMIZER_SIGNAL_LIMIT) in eligible_filters


@pytest.mark.asyncio
async def test_gate_status_flips_to_would_trigger_when_supply_clears_threshold(monkeypatch):
    from src.agents.feedback_learner import signal_store

    # Baseline 0.0 with mean reward 0.6 -> the reward-delta branch fires once
    # the count gate is open, which is what the beat itself would do.
    monkeypatch.setattr(signal_store, "load_trigger_state", dict)
    status = await signal_store.get_optimizer_gate_status(
        client=_FakeClient({}, eligible=25, total=300, runs=3, last="2026-08-16T00:00:00+00:00")
    )
    assert status["would_trigger"] is True
    assert status["optimization_runs"] == 3


@pytest.mark.asyncio
async def test_eligible_count_is_what_the_gate_actually_counted(monkeypatch):
    """Report the rows the trigger saw, not a wider exact count.

    The eligible read is capped at ``OPTIMIZER_SIGNAL_LIMIT``. A PostgREST
    ``count="exact"`` would keep counting past that cap, so a card reading
    "3000 / 2500" could sit beside a reason saying "Insufficient signals:
    2000 < 2500". The count and the verdict must describe the same row set.
    """
    from src.agents.feedback_learner import signal_store

    monkeypatch.setattr(signal_store, "load_trigger_state", dict)
    monkeypatch.setattr(signal_store, "OPTIMIZER_SIGNAL_LIMIT", 5)
    monkeypatch.setenv("DSPY_MIN_SIGNALS", "10")

    # 12 rows exist; the capped read returns 5. Both numbers must say 5.
    client = _FakeClient({}, eligible=12, total=99, runs=0, last="2026-08-16T00:00:00+00:00")
    status = await signal_store.get_optimizer_gate_status(client=client)

    assert status["eligible_signals"] == 5
    assert status["reason"] == "Insufficient signals: 5 < 10"


def test_trigger_state_write_is_atomic_and_leaves_no_temp_behind(tmp_path, monkeypatch):
    """A reader in another container must never see a half-written file."""
    from src.agents.feedback_learner import signal_store
    from src.tasks import dspy_optimization_tasks as task

    path = tmp_path / "optimized_modules" / ".trigger_state.json"
    monkeypatch.setattr(task, "_STATE_PATH", path)
    monkeypatch.setattr(signal_store, "TRIGGER_STATE_PATH", path)

    task._save_trigger_state({"last_optimization": "2026-08-16T00:00:00+00:00"})
    assert signal_store.load_trigger_state() == {"last_optimization": "2026-08-16T00:00:00+00:00"}
    # Overwriting must not leave a partial file or temp litter beside it.
    task._save_trigger_state({"baseline_reward": 0.6})
    assert signal_store.load_trigger_state() == {"baseline_reward": 0.6}
    assert [p.name for p in path.parent.iterdir()] == [".trigger_state.json"]


def test_beat_signal_read_is_deterministically_ordered():
    """Both readers must take the SAME slice when the limit binds.

    The beat reads eligible signals through SignalCollectorAdapter and the
    health surface reads them directly. An unordered ``limit`` makes the row
    set arbitrary, so the two could compute different mean rewards from the
    same database and disagree on the reward-delta branch. Newest-first is the
    right slice for training on recent signals, and it makes the two identical.
    """
    import asyncio

    from src.rag.memory_adapters import SignalCollectorAdapter

    sink: dict = {}

    class _SyncQuery(_FakeQuery):
        def execute(self):
            self.sink.setdefault("queries", []).append((self.table, list(self.filters)))
            return type("R", (), {"data": []})()

    class _SyncClient:
        def table(self, name):
            return _SyncQuery(name, sink, [], 0)

    asyncio.run(
        SignalCollectorAdapter(supabase_client=_SyncClient()).get_signals_for_optimization(
            source_agent="feedback_learner", min_reward=0.5, limit=2000
        )
    )
    filters = sink["queries"][0][1]
    assert ("order", "created_at", True) in filters
    # created_at is NOT unique: without a PK tiebreak, rows tied at the limit
    # boundary can come back in different physical orders across plans, and the
    # two readers' slices diverge again.
    assert ("order", "signal_id", True) in filters
    # ...and the order must be applied BEFORE the limit, or it sorts a slice.
    assert filters.index(("order", "created_at", True)) < filters.index(("limit", 2000))
    assert filters.index(("order", "signal_id", True)) < filters.index(("limit", 2000))


@pytest.mark.asyncio
async def test_gate_status_agrees_with_the_beat_once_the_signal_gate_opens(monkeypatch):
    """The status must report what the BEAT would decide — the #1661 invariant.

    Superseded the earlier form of this test, which asserted that a recent
    ``last_optimization`` surfaces as ``"Cooldown active"`` once the signal gate
    opens. That was correct while the cooldown bound the beat. #1656 established
    that it must NOT: ``last_optimization`` is a COMPLETION stamp, so on a
    ``crontab(hour=6)`` entry any nonzero runtime leaves the next fire inside the
    24h window and the daily task silently runs every OTHER day. The beat now
    passes ``scheduled=True``, so the cooldown no longer binds it.

    Asserting "Cooldown active" here would therefore pin the status surface to a
    gate the beat does not apply — reporting Skipped while the beat runs, which
    is #1661's own defect with the sign flipped. So this now asserts the
    invariant directly (status == beat decision) rather than one instance of it,
    which is the property #1661 existed to protect.
    """
    from datetime import datetime, timedelta, timezone

    from src.agents.feedback_learner import signal_store

    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    state = {"last_optimization": recent}
    monkeypatch.setattr(signal_store, "load_trigger_state", lambda: state)
    status = await signal_store.get_optimizer_gate_status(
        client=_FakeClient({}, eligible=25, total=300, runs=1, last="2026-08-16T00:00:00+00:00")
    )

    # What the beat itself would decide. Reward differs from the fake client's
    # rows, so the reason STRINGS differ in their numbers — the invariant is the
    # decision and the gate that produced it, not the formatted text.
    beat_should, beat_reason = signal_store.decide_optimizer_trigger(
        [{"reward": 0.9}] * 25, state, scheduled=True
    )
    assert status["would_trigger"] == beat_should
    assert ("Cooldown" in status["reason"]) == ("Cooldown" in beat_reason)
    # ...and concretely: a run 2h ago no longer suppresses the scheduled path.
    assert status["would_trigger"] is True
    assert "Cooldown" not in status["reason"]


@pytest.mark.asyncio
async def test_event_triggered_path_still_surfaces_the_cooldown(monkeypatch):
    """The cooldown is not gone — it still binds runs the crontab does not pace.

    Retains the coverage the superseded test above provided, on the path where
    the cooldown remains load-bearing (#1656 dropped it for the cron path only).
    """
    from datetime import datetime, timedelta, timezone

    from src.agents.feedback_learner import signal_store

    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    should, reason = signal_store.decide_optimizer_trigger(
        [{"reward": 0.9}] * 25, {"last_optimization": recent}, scheduled=False
    )
    assert should is False
    assert "Cooldown active" in reason


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
