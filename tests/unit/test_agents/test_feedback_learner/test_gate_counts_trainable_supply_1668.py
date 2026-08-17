"""#1668: the optimizer gate must count what the trainset builder can use.

#1675 stopped the TRAINSET being selected on defect yield. The GATE was left
measuring it: BEFORE this change the beat counted ``feedback_learner`` rows at
``reward >= 0.5``, which is **8** on the production table. That number is a
defect-yield proxy — ``compute_reward`` gives the coverage and actionability
terms zero on a cycle that detected no patterns, so a row clears 0.5 essentially
only when the cycle found patterns — and it measured a different quantity from
the one the builder consumes.

(The gate has since moved once more: it counts TRAINSET EXAMPLES, and the
threshold is stated in that unit too. The tests below assert the current
behaviour; the paragraphs describe the defect they were written against.)

Measured 2026-08-17 against the 223 real ``feedback_learner`` rows, read-only:

    A  eligible reward >= 0.5                       8    <- the ORIGINAL gate
    B  informative pool (non-empty feedback_batch) 75
    C  minority label class (pattern phase)        15
    D  built pattern examples                      30    == 2 * C, THE GATE'S UNIT

and, decisively:

    pattern examples built from the BEAT's own reward>=0.5 pool:  0

The eight rows the OLD gate counted are 100% positive (all from cycles that
found patterns), so they are single-class and ``_signals_to_examples`` refuses
them outright. That gate and the builder were not merely different numbers; the
gate's own row set trained nothing. Twenty such rows would have satisfied it and
still been untrainable.

``_interleave`` caps the trainset at ``k = min(n_pos, n_neg)`` pairs, so the
minority label class IS the supply constraint and ``len(trainset) == 2 * k``
exactly. That identity is asserted here, over the real builder, because it is
the mechanism that stops the two from drifting apart again.

No mocks of anything under test: real signal rows through the real builder and
the real trigger.
"""

from __future__ import annotations

import pytest

from src.agents.feedback_learner.dspy_integration import DSPY_AVAILABLE, FeedbackLearnerOptimizer

from ._gate_supply_fixtures import PATTERNS, RECS, degenerate, negative, positive, signal_row


def _positive(tag: str, reward: float = 0.9) -> dict:
    return positive(tag, reward=reward)


def _negative(tag: str, reward: float = 0.0) -> dict:
    return negative(tag, reward=reward)


def _degenerate(tag: str) -> dict:
    return degenerate(tag)


def _signal(**kw) -> dict:
    return signal_row(**kw)


def _optimizer() -> FeedbackLearnerOptimizer:
    return FeedbackLearnerOptimizer(optimizer_type="gepa")


# =============================================================================
# 1. The gate counts the trainset the builder produces, not reward-eligible rows
# =============================================================================


def test_all_positive_pool_that_builds_nothing_must_not_open_the_gate():
    """The sharpest form of the defect: 25 rows the builder refuses outright.

    Every one clears ``reward >= 0.5``, so the PRE-#1668 gate read 25 >= 20 and
    fired. ``_signals_to_examples`` then finds a single-class pool and returns
    ZERO examples — the beat would run and compile nothing. The gate must read
    the trainset as 0.
    """
    pool = [_positive(f"p{i}", reward=0.9) for i in range(25)]

    from src.agents.feedback_learner.signal_store import decide_optimizer_trigger

    should, reason = decide_optimizer_trigger(pool, {}, scheduled=True)
    assert should is False, reason
    assert reason == "Insufficient trainset: 0 < 40 examples"

    # Positive control: the probe is not blind. Add the missing class and the
    # same function opens.
    balanced = pool + [_negative(f"n{i}") for i in range(20)]
    should_b, reason_b = decide_optimizer_trigger(balanced, {}, scheduled=True)
    assert should_b is True, reason_b


def test_gate_counts_the_minority_class_not_the_reward_eligible_count():
    """30 eligible positives + 5 ineligible negatives -> a 10-example trainset, not 30."""
    from src.agents.feedback_learner.signal_store import decide_optimizer_trigger

    pool = [_positive(f"p{i}", reward=0.9) for i in range(30)]
    pool += [_negative(f"n{i}", reward=0.1) for i in range(5)]

    eligible = [s for s in pool if float(s["reward"]) >= 0.5]
    assert len(eligible) == 30  # what the old gate would have counted

    should, reason = decide_optimizer_trigger(pool, {}, scheduled=True)
    assert should is False
    assert reason == "Insufficient trainset: 10 < 40 examples"


def test_degenerate_rows_are_neither_class():
    """The 148 empty-input rows must not inflate the negative class."""
    from src.agents.feedback_learner.dspy_integration import label_class_counts

    pool = [_positive(f"p{i}") for i in range(3)]
    pool += [_negative(f"n{i}") for i in range(4)]
    pool += [_degenerate(f"d{i}") for i in range(100)]

    assert label_class_counts(pool, "pattern") == (3, 4)


def test_gate_supply_is_the_best_supplied_phase():
    """A run is worth doing if ANY phase can be trained, so the gate takes the max.

    The recommendation phase conditions on ``detected_patterns``, so its pool is
    the cycles that FOUND patterns and its classes split on whether
    recommendations were produced. A pattern phase starved of negatives must not
    hide a well-supplied recommendation phase.
    """
    from src.agents.feedback_learner.dspy_integration import gate_supply

    # 8 pattern-positives, of which 4 produced recommendations and 4 did not,
    # plus 2 pattern-negatives.
    pool = [_signal(tag=f"a{i}", patterns=PATTERNS, recommendations=RECS) for i in range(4)]
    pool += [_signal(tag=f"b{i}", patterns=PATTERNS, recommendations=[]) for i in range(4)]
    pool += [_negative(f"n{i}") for i in range(2)]

    phase, supply = gate_supply(pool)
    assert (phase, supply) == ("recommendation", 4)  # pattern minority is only 2


def test_gate_supply_is_zero_and_phaseless_on_an_empty_pool():
    from src.agents.feedback_learner.dspy_integration import gate_supply

    assert gate_supply([]) == (None, 0)


# =============================================================================
# 2. THE anti-divergence mechanism: one classifier, asserted end to end
# =============================================================================


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy required to build examples")
@pytest.mark.parametrize("phase", ["pattern", "recommendation"])
@pytest.mark.parametrize(
    "pool_name",
    ["balanced", "positive_heavy", "negative_heavy", "single_class", "degenerate_only", "empty"],
)
def test_builder_output_is_exactly_twice_the_counted_supply(phase: str, pool_name: str):
    """``len(trainset) == 2 * trainable_supply`` — for every phase, every pool.

    This is the mechanism that stops the gate and the builder desynchronising:
    both route through ``classify_signal_for_phase``, and this asserts the
    consequence against the REAL builder rather than trusting the refactor. An
    edit to either side that changes which rows are usable, or which label they
    carry, breaks this immediately.
    """
    from src.agents.feedback_learner.dspy_integration import trainable_supply

    pools = {
        "balanced": [_positive(f"p{i}") for i in range(6)]
        + [_signal(tag=f"b{i}", patterns=PATTERNS, recommendations=[]) for i in range(6)]
        + [_negative(f"n{i}") for i in range(6)],
        "positive_heavy": [_positive(f"p{i}") for i in range(9)] + [_negative("n0")],
        "negative_heavy": [_positive("p0")]
        + [_signal(tag="b0", patterns=PATTERNS, recommendations=[])]
        + [_negative(f"n{i}") for i in range(9)],
        "single_class": [_positive(f"p{i}") for i in range(7)],
        "degenerate_only": [_degenerate(f"d{i}") for i in range(11)],
        "empty": [],
    }
    pool = pools[pool_name]

    built = _optimizer()._signals_to_examples(pool, phase)
    assert len(built) == 2 * trainable_supply(pool, phase)


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy required to build examples")
def test_the_identity_would_catch_a_desynchronised_builder():
    """Positive control for the test above: break one side, it must fail.

    A "no matches" style guarantee is worthless without evidence it CAN match.
    Here the builder is given a pool whose usable rows the classifier and a
    hypothetical divergent filter disagree about, and the identity is shown to
    be sensitive to that disagreement.
    """
    from src.agents.feedback_learner.dspy_integration import trainable_supply

    pool = [_positive(f"p{i}") for i in range(4)] + [_negative(f"n{i}") for i in range(4)]
    assert len(_optimizer()._signals_to_examples(pool, "pattern")) == 2 * trainable_supply(
        pool, "pattern"
    )
    # Drop one positive from the classifier's view ONLY (simulating a builder
    # that admitted a row the counter does not). The identity must break.
    starved = pool[1:]
    assert len(_optimizer()._signals_to_examples(pool, "pattern")) != 2 * trainable_supply(
        starved, "pattern"
    )


# =============================================================================
# 3. The beat, the runner and the status all read ONE pool
# =============================================================================


def test_optimizer_pool_has_no_reward_floor():
    """The pool the gate counts must be the pool the builder trains on.

    A reward floor on the pool starves the negative class by construction: a
    correct abstention scores near zero. #1675 already moved the runner to
    ``min_reward=0.0``; the gate reading a different pool is the divergence this
    issue is about.
    """
    from src.agents.feedback_learner import signal_store

    assert signal_store.OPTIMIZER_POOL_MIN_REWARD == 0.0


@pytest.mark.asyncio
async def test_beat_hands_the_runner_the_rows_it_counted():
    """Not "the same query" — the same LIST OBJECT.

    Two reads seconds apart can differ (a learning cycle writes every 6h), so
    fetching twice leaves the gate's verdict describing a different row set from
    the one that gets trained on.
    """
    from src.tasks import dspy_optimization_tasks as task

    pool = [_positive(f"p{i}") for i in range(20)] + [_negative(f"n{i}") for i in range(20)]
    seen: dict = {}

    async def _fake_pool(client=None):
        return pool

    async def _fake_run(**kwargs):
        seen["signals"] = kwargs.get("signals")
        return {"status": "completed_no_modules", "phases": {}}

    import src.agents.feedback_learner.optimization_runner as runner_mod
    import src.agents.feedback_learner.signal_store as store_mod

    orig_pool = store_mod.read_optimizer_signal_pool
    orig_run = runner_mod.run_feedback_learner_optimization
    store_mod.read_optimizer_signal_pool = _fake_pool  # type: ignore[assignment]
    runner_mod.run_feedback_learner_optimization = _fake_run  # type: ignore[assignment]
    try:
        import src.agents.feedback_learner.prompt_bundles as bundles

        orig_install = bundles.install_all_prompt_bundles
        orig_factories = bundles.RECIPIENT_FACTORIES
        bundles.install_all_prompt_bundles = lambda: {}  # type: ignore[assignment]
        bundles.RECIPIENT_FACTORIES = {}  # type: ignore[assignment]

        async def _no_rag():
            return {"status": "skipped"}

        orig_rag = task._run_rag_leg_guarded
        orig_save = task._save_trigger_state
        task._run_rag_leg_guarded = _no_rag  # type: ignore[assignment]
        task._save_trigger_state = lambda state: None  # type: ignore[assignment]
        try:
            await task._run("t1", force=True, budget="light")
        finally:
            task._run_rag_leg_guarded = orig_rag  # type: ignore[assignment]
            task._save_trigger_state = orig_save  # type: ignore[assignment]
            bundles.install_all_prompt_bundles = orig_install  # type: ignore[assignment]
            bundles.RECIPIENT_FACTORIES = orig_factories  # type: ignore[assignment]
    finally:
        store_mod.read_optimizer_signal_pool = orig_pool  # type: ignore[assignment]
        runner_mod.run_feedback_learner_optimization = orig_run  # type: ignore[assignment]

    assert seen["signals"] is pool


# =============================================================================
# 4. The operator surface reports the quantity the beat decides on
# =============================================================================


class _PoolClient:
    """Answers the two count reads; the pool itself is injected separately."""

    def __init__(self, *, total: int, runs: int):
        self.total = total
        self.runs = runs

    def table(self, name):
        outer = self

        class _Q:
            def select(self, *a, **k):
                return self

            def eq(self, *a):
                return self

            def limit(self, *a):
                return self

            async def execute(self):
                count = outer.runs if name == "prompt_optimization_runs" else outer.total
                return type("R", (), {"data": [], "count": count})()

        return _Q()


@pytest.mark.asyncio
async def test_status_reports_the_number_the_beat_decides_on(monkeypatch):
    """#1666's invariant, preserved: the surface calls the beat's decision.

    Before this change the surface reported 8 (reward-eligible) while the beat
    would decide on a different quantity. The two must be one number.
    """
    from src.agents.feedback_learner import signal_store

    pool = [_positive(f"p{i}") for i in range(15)] + [_negative(f"n{i}") for i in range(60)]

    async def _pool(client=None):
        return pool

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _pool)
    monkeypatch.setattr(signal_store, "load_trigger_state", dict)

    status = await signal_store.get_optimizer_gate_status(client=_PoolClient(total=223, runs=0))

    beat_should, beat_reason = signal_store.decide_optimizer_trigger(pool, {}, scheduled=True)
    assert status["would_trigger"] == beat_should
    assert status["reason"] == beat_reason

    assert status["trainset_examples"] == 30
    assert status["positive_signals"] == 15
    assert status["negative_signals"] == 60
    assert status["governing_phase"] == "pattern"
    assert status["total_signals"] == 223
    assert status["optimization_runs"] == 0
    assert status["min_trainset_examples"] == 40
    assert status["reason"] == "Insufficient trainset: 30 < 40 examples"


@pytest.mark.asyncio
async def test_status_no_longer_reports_the_reward_eligible_count(monkeypatch):
    """ "8 of 218" was the #1661 surface. Leaving it beside a gate that decides
    on a different number is the disagreement this issue exists to remove."""
    from src.agents.feedback_learner import signal_store

    async def _pool(client=None):
        return [_positive("p0"), _negative("n0")]

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _pool)
    monkeypatch.setattr(signal_store, "load_trigger_state", dict)
    status = await signal_store.get_optimizer_gate_status(client=_PoolClient(total=2, runs=0))

    assert "eligible_signals" not in status
    assert "last_eligible_signal_at" not in status
    assert "min_reward" not in status


@pytest.mark.asyncio
async def test_status_last_trainable_signal_is_the_scarcer_class(monkeypatch):
    """Supply moves when the SCARCE class moves; a fresh negative changes nothing.

    Reporting the newest row of either class would show a date moving daily
    while the gate stayed frozen — the shape of false green #1661 removed. This
    is production's actual shape: 15 positives against 60 negatives, and the
    negatives are the ones arriving.

    The fixture used to be ONE positive and ONE negative, which is a TIE — and
    the iter-3 fix established that a tie means something different (neither
    class alone raises ``min``, so the newest row of either class is what
    completed the pair). It was asserting the unequal rule on a tied pool. Made
    genuinely unequal; the tie is covered by
    ``test_last_trainable_signal_at_is_the_newest_pair_completer_on_a_tie``.
    """
    from src.agents.feedback_learner import signal_store

    old_pos = _positive("p1")
    old_pos["created_at"] = "2026-08-08T07:09:02+00:00"
    newer_negatives = []
    for i in range(4):
        neg = _negative(f"n{i}")
        neg["created_at"] = f"2026-08-1{i + 3}T05:00:00+00:00"
        newer_negatives.append(neg)

    async def _pool(client=None):
        # newest-first, as the adapter returns
        return list(reversed(newer_negatives)) + [old_pos]

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _pool)
    monkeypatch.setattr(signal_store, "load_trigger_state", dict)
    status = await signal_store.get_optimizer_gate_status(client=_PoolClient(total=5, runs=0))

    assert (status["positive_signals"], status["negative_signals"]) == (1, 4)
    assert status["last_trainable_signal_at"] == "2026-08-08T07:09:02+00:00"


@pytest.mark.asyncio
async def test_status_degrades_honestly_without_a_client(monkeypatch):
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


# =============================================================================
# 5. codex iter-1 findings, verified before fixing
# =============================================================================


@pytest.mark.asyncio
async def test_a_failed_pool_read_reports_unknown_not_a_measured_zero(monkeypatch):
    """codex iter-1 MEDIUM — verified, and it is a regression this PR introduced.

    #1661's contract is that every count is ``None``, never 0, when the read
    fails: "a fabricated zero on a health surface is indistinguishable from a
    measured one". The status used to issue its OWN query and catch the error.
    It now reads through ``SignalCollectorAdapter.get_signals_for_optimization``,
    which SWALLOWS read failures and returns ``[]`` (memory_adapters.py:844) —
    deliberately, because a DB outage must not crash the Celery beat. So a pool
    read failure would publish ``trainset_examples: 0`` and "Insufficient
    trainset: 0 < 40 examples", which reads as a measurement of an outage.

    Worse than cosmetic: 0 is also the value a genuinely single-class corpus
    produces, so the one number an operator would use to tell "the loop is
    starved" from "the database is down" reads identically in both.
    """
    from src.agents.feedback_learner import signal_store

    class _ExplodingPoolClient:
        """ONLY the pool SELECT raises; both exact counts succeed.

        The distinction matters: if the count queries failed too, the status
        would return `unavailable` through its existing guard and this test
        would pass without exercising the defect at all. The adapter's read is
        the one issued as ``select("*")``; the status's counts carry
        ``count="exact"``.
        """

        def __init__(self, *, total: int, runs: int):
            self.total = total
            self.runs = runs

        def table(self, name):
            outer = self

            class _Q:
                def __init__(self):
                    self.exact = False

                def select(self, *a, **k):
                    self.exact = k.get("count") == "exact"
                    return self

                def eq(self, *a):
                    return self

                def gte(self, *a):
                    return self

                def order(self, *a, **k):
                    return self

                def limit(self, *a):
                    return self

                def execute(self):
                    if self.exact:
                        count = outer.runs if name == "prompt_optimization_runs" else outer.total
                        return type("R", (), {"data": [], "count": count})()
                    raise RuntimeError("connection reset by peer")

            return _Q()

    monkeypatch.setattr(signal_store, "load_trigger_state", dict)
    client = _ExplodingPoolClient(total=223, runs=0)

    # Positive control on the FAKE itself: the counts really do succeed, so a
    # `None` below is caused by the pool read and not by a client that fails at
    # everything.
    from src.agents.feedback_learner.signal_store import TABLE

    probe = client.table(TABLE).select("signal_id", count="exact").limit(1).execute()
    assert probe.count == 223

    status = await signal_store.get_optimizer_gate_status(client=client)

    assert status["trainset_examples"] is None
    assert status["positive_signals"] is None
    assert status["negative_signals"] is None
    assert status["total_signals"] is None
    assert status["would_trigger"] is None
    assert "unavailable" in status["reason"].lower()


@pytest.mark.asyncio
async def test_a_genuinely_empty_corpus_still_reports_a_measured_zero(monkeypatch):
    """Positive control for the test above.

    Making a failed read report ``None`` is only correct if a SUCCESSFUL read of
    an empty/single-class corpus still reports a measured 0 — otherwise the fix
    would just move the ambiguity to the other side.
    """
    from src.agents.feedback_learner import signal_store

    async def _empty(client=None):
        return []

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _empty)
    monkeypatch.setattr(signal_store, "load_trigger_state", dict)
    status = await signal_store.get_optimizer_gate_status(client=_PoolClient(total=0, runs=0))

    assert status["trainset_examples"] == 0
    assert status["would_trigger"] is False
    assert status["reason"] == "Insufficient trainset: 0 < 40 examples"


@pytest.mark.asyncio
async def test_a_run_that_compiled_nothing_does_not_move_the_reward_baseline(monkeypatch):
    """codex iter-1 HIGH — verified, and #1668 makes it imminent rather than theoretical.

    ``_run`` stamped ``baseline_reward`` after every triggered run, including one
    where ``run_feedback_learner_optimization`` returned ``completed_no_modules``
    or ``skipped_no_lm``. ``baseline_reward`` means "the reward level of the
    prompt that is currently installed", so pinning it to a prompt that was
    never installed makes the next beat compute ``delta ~= 0`` and return
    "No trigger" — a loop that goes quiet after ONE no-op run.

    Why it matters now: supply is at 15 against a threshold of 20, so the first
    time this gate ever opens is a foreseeable near-term event, and the first
    run is exactly when "no LM configured" / "no phase built a trainset" is most
    likely.

    ``last_optimization`` IS still stamped: a run really did execute and spend
    recipient budget, and on the event-triggered path that stamp is the only
    thing bounding how often it re-fires. The two keys record different facts.
    """
    from src.tasks import dspy_optimization_tasks as task

    from ._gate_supply_fixtures import balanced_pool

    saved: list = []
    pool = balanced_pool(25)

    async def _fake_pool(client=None):
        return pool

    async def _no_modules(**kwargs):
        return {"status": "completed_no_modules", "phases": {"pattern": {"status": "no_module"}}}

    async def _no_rag():
        return {"status": "skipped"}

    import src.agents.feedback_learner.optimization_runner as runner_mod
    import src.agents.feedback_learner.prompt_bundles as bundles
    import src.agents.feedback_learner.signal_store as store_mod

    monkeypatch.setattr(store_mod, "read_optimizer_signal_pool", _fake_pool)
    monkeypatch.setattr(runner_mod, "run_feedback_learner_optimization", _no_modules)
    monkeypatch.setattr(bundles, "install_all_prompt_bundles", lambda: {})
    monkeypatch.setattr(bundles, "RECIPIENT_FACTORIES", {})
    monkeypatch.setattr(task, "_run_rag_leg_guarded", _no_rag)
    monkeypatch.setattr(task, "_load_trigger_state", dict)
    monkeypatch.setattr(task, "_save_trigger_state", lambda state: saved.append(state))

    result = await task._run("t-nomod", force=True, budget="light")

    assert saved, "the run still records that it executed"
    assert "last_optimization" in saved[-1]
    assert "baseline_reward" not in saved[-1]
    assert result["status"] == "completed_no_modules"

    # Positive control: a run that DID install a module must move the baseline,
    # or the assertion above would pass on a beat that never stamps anything.
    saved.clear()

    async def _optimized(**kwargs):
        return {
            "status": "completed",
            "phases": {"pattern": {"status": "optimized", "version_id": "v1"}},
        }

    monkeypatch.setattr(runner_mod, "run_feedback_learner_optimization", _optimized)
    result = await task._run("t-mod", force=True, budget="light")
    assert "baseline_reward" in saved[-1]
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_status_attributes_its_class_counts_even_when_no_phase_is_trainable(monkeypatch):
    """codex iter-2 LOW — verified: the breakdown was left describing nothing.

    With a single-class pool no phase has both labels, so ``gate_supply``
    honestly returns ``(None, 0)``. The status still published
    ``positive_signals``/``negative_signals`` — documented as being "for the
    governing phase" — beside ``governing_phase: null``. That is the starved
    case, i.e. exactly when an operator most needs to know WHICH class is
    missing and on which phase, and it is the case #1668's whole argument turns
    on (an all-positive pool builds zero examples). The phase must be named.
    """
    from src.agents.feedback_learner import signal_store

    async def _pool(client=None):
        return [_positive(f"p{i}") for i in range(15)]  # no negatives at all

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _pool)
    monkeypatch.setattr(signal_store, "load_trigger_state", dict)
    status = await signal_store.get_optimizer_gate_status(client=_PoolClient(total=15, runs=0))

    assert status["trainset_examples"] == 0
    assert status["governing_phase"] == "pattern"
    assert status["positive_signals"] == 15
    assert status["negative_signals"] == 0
    assert status["reason"] == "Insufficient trainset: 0 < 40 examples"


def test_gate_supply_breakdown_names_a_phase_on_an_empty_pool():
    """Degenerate input must still attribute its zeros rather than crash."""
    from src.agents.feedback_learner.dspy_integration import (
        OPTIMIZABLE_PHASES,
        gate_supply_breakdown,
    )

    phase, supply, positives, negatives = gate_supply_breakdown([])
    assert phase in OPTIMIZABLE_PHASES
    assert (supply, positives, negatives) == (0, 0, 0)


def test_beat_and_status_report_the_same_supply_breakdown():
    """Both surfaces must derive the breakdown from ONE function, not two."""
    import inspect

    from src.tasks import dspy_optimization_tasks as task

    source = inspect.getsource(task._run)
    assert "gate_supply_breakdown" in source


@pytest.mark.asyncio
async def test_last_trainable_signal_at_is_the_newest_pair_completer_on_a_tie(monkeypatch):
    """codex iter-3 LOW — verified, and the reasoning is sharper than "ambiguous".

    ``supply = min(positives, negatives)``. When the classes are UNEQUAL only the
    scarcer one can raise it, so "when supply last moved" is that class's newest
    row. When they are EQUAL at k, adding either class leaves ``min`` at k — so
    supply last moved when the PAIR completed, which is the newest usable row of
    either class. Picking positives on the tie (``positives <= negatives``)
    reported a stale date whenever the newest row was the negative that closed
    the pair.
    """
    from src.agents.feedback_learner import signal_store

    old_pos = _positive("p1")
    old_pos["created_at"] = "2026-08-01T00:00:00+00:00"
    new_neg = _negative("n1")
    new_neg["created_at"] = "2026-08-17T05:00:00+00:00"

    async def _tied(client=None):
        return [new_neg, old_pos]  # newest-first, 1 positive / 1 negative

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _tied)
    monkeypatch.setattr(signal_store, "load_trigger_state", dict)
    status = await signal_store.get_optimizer_gate_status(client=_PoolClient(total=2, runs=0))

    assert (status["positive_signals"], status["negative_signals"]) == (1, 1)
    assert status["trainset_examples"] == 2
    # The NEGATIVE closed the pair, so it is what moved supply from 0 to 1.
    assert status["last_trainable_signal_at"] == "2026-08-17T05:00:00+00:00"

    # Positive control lives in ``test_status_last_trainable_signal_is_the_
    # scarcer_class`` above: on an UNEQUAL pool the newest abundant-class row is
    # still ignored. Without it, "use the newest usable row" could silently
    # become the rule everywhere and the date would advance daily while the gate
    # stayed frozen — the false green #1661 removed.
