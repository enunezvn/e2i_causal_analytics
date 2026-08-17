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

    The 0.5 is a literal here, not ``OPTIMIZER_MIN_REWARD``. #1668 removed that
    constant: the gate no longer applies a reward floor at all (it counts label
    classes), so the number below is a property of ``compute_reward`` — which is
    what this test is about — and importing a gate constant to express it would
    re-couple the two after they were deliberately separated.
    """
    reward_ceiling_for_a_pattern_free_cycle = 0.5

    best = _signal(
        patterns_detected=0,
        recommendation_actionability=0.0,
        total_latency_ms=1.0,
        rubric_weighted_score=5.0,
    )
    assert best.compute_reward() == reward_ceiling_for_a_pattern_free_cycle

    # One notch off a flawless rubric already drops below it.
    realistic = _signal(
        patterns_detected=0,
        recommendation_actionability=0.0,
        total_latency_ms=1.0,
        rubric_weighted_score=4.5,
    )
    assert realistic.compute_reward() < reward_ceiling_for_a_pattern_free_cycle


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
    """The beat must not re-declare the pool/state path as bare literals.

    #1668 narrowed what there is to share. The beat used to build its own read
    from ``OPTIMIZER_MIN_REWARD`` + ``OPTIMIZER_SIGNAL_LIMIT``; it now calls
    ``read_optimizer_signal_pool``, so the SSOT is the FUNCTION rather than the
    two constants it was assembled from — one fewer way for a caller to compose
    a slightly different pool.
    """
    import inspect

    from src.agents.feedback_learner import signal_store
    from src.tasks import dspy_optimization_tasks as task

    assert task._STATE_PATH is signal_store.TRIGGER_STATE_PATH
    source = inspect.getsource(task._run)
    assert "read_optimizer_signal_pool()" in source
    # ...and it must not have reintroduced a hand-rolled read beside it.
    assert "get_feedback_learner_training_signals" not in source


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

    from ._gate_supply_fixtures import balanced_pool

    signals = balanced_pool(50)
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    should, reason = decide_optimizer_trigger(signals, {"last_optimization": recent})
    assert should is False
    assert "Cooldown active" in reason


def test_signal_gate_is_what_binds_with_no_prior_optimization():
    """Today's real state: no trigger file, supply below the threshold.

    #1668: the rows are real-shaped rather than ``{"reward": 0.6}``. The gate
    counts the scarcer label class now, so a row carrying only a reward is not
    a signal the optimizer could train on and must not read as one.
    """
    from src.agents.feedback_learner.signal_store import decide_optimizer_trigger

    from ._gate_supply_fixtures import balanced_pool

    should, reason = decide_optimizer_trigger(balanced_pool(8), {})
    assert should is False
    assert reason == "Insufficient trainset: 16 < 40 examples"


def test_the_threshold_honours_the_env_override(monkeypatch):
    """The override moved to DSPY_MIN_TRAINSET_EXAMPLES with the unit.

    The old name is deliberately not honoured — see
    ``test_gate_threshold_unit_1668`` for why reading it would halve the gate.
    """
    from src.agents.feedback_learner.dspy_integration import GEPAOptimizationTrigger
    from src.agents.feedback_learner.signal_store import optimizer_min_trainset_examples

    default = GEPAOptimizationTrigger.min_trainset_examples
    monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
    monkeypatch.delenv("DSPY_MIN_TRAINSET_EXAMPLES", raising=False)
    assert optimizer_min_trainset_examples() == default
    monkeypatch.setenv("DSPY_MIN_TRAINSET_EXAMPLES", "14")
    assert optimizer_min_trainset_examples() == 14
    # A garbled value must not crash the health surface.
    monkeypatch.setenv("DSPY_MIN_TRAINSET_EXAMPLES", "not-a-number")
    assert optimizer_min_trainset_examples() == default


# =============================================================================
# 3. The status reader — real queries, no fabricated numbers
# =============================================================================
#
# #1668 rewrote this section's stand-ins. The status used to hand-roll its own
# eligible-signals query, so the fake client had to answer it and the tests
# asserted on the filters it built. It now reads the pool through
# ``read_optimizer_signal_pool`` — the same call the beat makes — so what these
# tests must pin is that the pool the status reports on IS that pool, and that
# the numbers it publishes are the ones the beat's decision function derived
# from it. The filters themselves are pinned once, at the adapter, by
# ``test_beat_signal_read_is_deterministically_ordered`` below.


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


class _CountClient:
    """Answers the two exact-count reads the status makes beside the pool."""

    def __init__(self, *, total: int, runs: int, sink=None):
        self.total = total
        self.runs = runs
        self.sink = sink if sink is not None else {}

    def table(self, name):
        count = self.runs if name == "prompt_optimization_runs" else self.total
        return _FakeQuery(name, self.sink, [], count)


@pytest.mark.asyncio
async def test_gate_status_reports_real_counts_and_the_gate_verdict(monkeypatch):
    from src.agents.feedback_learner import signal_store

    from ._gate_supply_fixtures import negative, positive

    # The real shape measured 2026-08-17: 15 positives, 60 negatives (plus 148
    # empty-input rows that are neither class and are therefore left out of the
    # balance). 15 + 60 + 148 == the 223 `total_signals` asserted below.
    pool = [positive(f"p{i}", created_at="2026-08-08T07:09:02.686027+00:00") for i in range(15)] + [
        negative(f"n{i}") for i in range(60)
    ]
    called: dict = {}

    async def _pool(client=None):
        called["client"] = client
        return pool

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _pool)
    monkeypatch.setattr(signal_store, "load_trigger_state", dict)

    client = _CountClient(total=223, runs=0)
    status = await signal_store.get_optimizer_gate_status(client=client)

    assert status["trainset_examples"] == 30
    assert status["positive_signals"] == 15
    assert status["negative_signals"] == 60
    assert status["governing_phase"] == "pattern"
    assert status["total_signals"] == 223
    assert status["optimization_runs"] == 0
    assert status["min_trainset_examples"] == 40
    assert status["would_trigger"] is False
    assert status["last_trainable_signal_at"] == "2026-08-08T07:09:02.686027+00:00"
    # Verbatim from the REAL trigger, not a re-worded copy.
    assert status["reason"] == "Insufficient trainset: 30 < 40 examples"
    # The pool read must go through the caller's client, not a second factory
    # lookup that could resolve to a different database.
    assert called["client"] is client


@pytest.mark.asyncio
async def test_gate_status_flips_to_would_trigger_when_supply_clears_threshold(monkeypatch):
    from src.agents.feedback_learner import signal_store

    from ._gate_supply_fixtures import balanced_pool

    # Baseline 0.0 with a positive mean reward -> the reward-delta branch fires
    # once the count gate is open, which is what the beat itself would do.
    monkeypatch.setattr(signal_store, "load_trigger_state", dict)

    async def _pool(client=None):
        return balanced_pool(25)

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _pool)
    status = await signal_store.get_optimizer_gate_status(client=_CountClient(total=300, runs=3))
    assert status["would_trigger"] is True
    assert status["trainset_examples"] == 50
    assert status["optimization_runs"] == 3


@pytest.mark.asyncio
async def test_reported_supply_is_what_the_gate_actually_counted(monkeypatch):
    """Report the rows the trigger saw, not a wider exact count.

    The pool read is capped at ``OPTIMIZER_SIGNAL_LIMIT``. ``total_signals`` is
    a PostgREST ``count="exact"`` that keeps counting past that cap, so it can
    legitimately exceed the pool — but the GATE's number must come from the
    capped rows, or a card reading "3000 / 2500" could sit beside a reason
    saying "Insufficient trainset: 4000 < 5000 examples".
    """
    from src.agents.feedback_learner import signal_store

    from ._gate_supply_fixtures import balanced_pool

    monkeypatch.setattr(signal_store, "load_trigger_state", dict)
    monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
    monkeypatch.setenv("DSPY_MIN_TRAINSET_EXAMPLES", "20")

    async def _pool(client=None):
        return balanced_pool(5)  # what the capped read returned

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _pool)
    status = await signal_store.get_optimizer_gate_status(client=_CountClient(total=999, runs=0))

    assert status["trainset_examples"] == 10
    assert status["total_signals"] == 999
    assert status["reason"] == "Insufficient trainset: 10 < 20 examples"


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
    """Every reader must take the SAME slice when the limit binds.

    The beat, the runner and the health surface now all read through
    ``read_optimizer_signal_pool`` -> ``SignalCollectorAdapter``. An unordered
    ``limit`` makes the row set arbitrary, so two readers could compute
    different label balances and different mean rewards from the same database.
    Newest-first is the right slice for training on recent signals, and it makes
    them identical.
    """
    import asyncio

    from src.agents.feedback_learner import signal_store

    sink: dict = {}

    class _SyncQuery(_FakeQuery):
        def execute(self):
            self.sink.setdefault("queries", []).append((self.table, list(self.filters)))
            return type("R", (), {"data": []})()

    class _SyncClient:
        def table(self, name):
            return _SyncQuery(name, sink, [], 0)

    asyncio.run(signal_store.read_optimizer_signal_pool(client=_SyncClient()))
    # Positive control: the assertions below would pass vacuously on an empty
    # list, so prove a query was issued at all before inspecting its filters.
    assert sink["queries"], "the pool helper issued no query"
    filters = sink["queries"][0][1]
    assert ("eq", "source_agent", "feedback_learner") in filters
    # #1668: NO reward floor on the pool. A correct abstention scores near zero
    # by construction, so a floor and the negative class are the same set —
    # filtering by reward starves the class the balance needs.
    assert ("gte", "reward", signal_store.OPTIMIZER_POOL_MIN_REWARD) in filters
    assert ("limit", signal_store.OPTIMIZER_SIGNAL_LIMIT) in filters
    assert ("order", "created_at", True) in filters
    # created_at is NOT unique: without a PK tiebreak, rows tied at the limit
    # boundary can come back in different physical orders across plans, and the
    # readers' slices diverge again.
    assert ("order", "signal_id", True) in filters
    # ...and the order must be applied BEFORE the limit, or it sorts a slice.
    limit_at = filters.index(("limit", signal_store.OPTIMIZER_SIGNAL_LIMIT))
    assert filters.index(("order", "created_at", True)) < limit_at
    assert filters.index(("order", "signal_id", True)) < limit_at


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

    from ._gate_supply_fixtures import balanced_pool

    pool = balanced_pool(25)
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    state = {"last_optimization": recent}
    monkeypatch.setattr(signal_store, "load_trigger_state", lambda: state)

    async def _pool(client=None):
        return pool

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _pool)
    status = await signal_store.get_optimizer_gate_status(client=_CountClient(total=300, runs=1))

    # What the beat itself would decide, over the SAME rows and the same state.
    # #1668 made this an exact string match: both sides now count the same pool
    # through the same function, so a difference in the numbers inside the
    # reason is a real disagreement rather than a fixture artefact.
    beat_should, beat_reason = signal_store.decide_optimizer_trigger(pool, state, scheduled=True)
    assert status["would_trigger"] == beat_should
    assert status["reason"] == beat_reason
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

    from ._gate_supply_fixtures import balanced_pool

    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    should, reason = signal_store.decide_optimizer_trigger(
        balanced_pool(25), {"last_optimization": recent}, scheduled=False
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
    assert status["trainset_examples"] is None
    assert status["positive_signals"] is None
    assert status["negative_signals"] is None
    assert status["total_signals"] is None
    assert status["would_trigger"] is None
    assert "unavailable" in status["reason"].lower()
