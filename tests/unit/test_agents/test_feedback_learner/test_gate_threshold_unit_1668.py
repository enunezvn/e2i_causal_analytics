"""#1668 follow-up: the gate's THRESHOLD must be stated in the unit it gates.

#1677 made the gate count trainable supply and pinned the *supply's* unit
(``len(_signals_to_examples(pool, phase)) == 2 * trainable_supply(pool, phase)``).
It left the *threshold* in the old unit: a constant named ``min_signals``,
compared against half a trainset, carrying a comment justifying 20 on a
reachability argument about a quantity it no longer counts.

The tests here pin the threshold's unit. The load-bearing one is
:func:`test_the_gate_opens_exactly_where_the_builder_reaches_the_threshold`:
it walks real pools through the real builder and asserts that the pool at which
``decide_optimizer_trigger`` flips is the pool whose built trainset is exactly
the threshold. That is the invariant a future semantic change cannot satisfy by
accident — if the gate's quantity moves again without the threshold moving, the
flip point stops matching the builder's output and this test fails.
"""

from __future__ import annotations

import pytest

from ._gate_supply_fixtures import balanced_pool, negative, positive

# --------------------------------------------------------------------------
# 1. The constant names its unit
# --------------------------------------------------------------------------


def test_the_threshold_is_named_and_typed_in_trainset_examples():
    """``min_signals`` gated half a trainset. The name must say which unit."""
    from src.agents.feedback_learner.dspy_integration import GEPAOptimizationTrigger

    trigger = GEPAOptimizationTrigger()
    assert hasattr(trigger, "min_trainset_examples")
    assert not hasattr(trigger, "min_signals"), (
        "a constant called min_signals that gates trainset examples is the defect"
    )


def test_the_threshold_comment_no_longer_claims_reachability():
    """The justification in the source was measured false; it must not survive.

    Measured on the production table 2026-08-17: 8 rows at ``reward >= 0.5``
    over a 68.8-day span (0.116/day), and 0 positives in the last 8 recorded
    days. "20 ≈ reachable in normal operation" is not a true statement about
    either the old quantity or the new one.
    """
    import inspect

    from src.agents.feedback_learner import dspy_integration

    src = inspect.getsource(dspy_integration.GEPAOptimizationTrigger)
    assert "reachable in normal operation" not in src


# --------------------------------------------------------------------------
# 2. THE unit-pinning test
# --------------------------------------------------------------------------


def _first_pool_that_opens_the_gate(max_supply: int = 60):
    """Smallest balanced pool for which the beat's own decision flips to True."""
    from src.agents.feedback_learner.signal_store import decide_optimizer_trigger

    state = {"baseline_reward": 0.0}
    for k in range(1, max_supply + 1):
        pool = balanced_pool(k)
        should, _reason = decide_optimizer_trigger(pool, state, scheduled=True)
        if should:
            return k, pool
    return None, None


def test_the_gate_opens_exactly_where_the_builder_reaches_the_threshold():
    """The threshold's unit == the builder's output unit, measured end to end.

    No ``2 *`` anywhere in the assertion: the pool at which the gate flips is
    fed to the REAL trainset builder, and the number of examples it produces
    must equal the threshold the gate published. Before this change the gate
    flipped at a 40-example trainset while publishing a threshold of 20.
    """
    pytest.importorskip("dspy")
    from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer
    from src.agents.feedback_learner.signal_store import optimizer_min_trainset_examples

    k, pool = _first_pool_that_opens_the_gate()
    assert k is not None, "the gate never opened — the sweep is not measuring anything"

    built = FeedbackLearnerOptimizer(optimizer_type="gepa")._signals_to_examples(pool, "pattern")
    assert len(built) == optimizer_min_trainset_examples()

    # Positive control for the flip point: one class-pair fewer must NOT open it,
    # so the equality above is the boundary and not an accident of the sweep.
    from src.agents.feedback_learner.signal_store import decide_optimizer_trigger

    smaller = balanced_pool(k - 1)
    assert decide_optimizer_trigger(smaller, {"baseline_reward": 0.0}, scheduled=True)[0] is False


def test_the_reason_string_states_the_unit():
    from src.agents.feedback_learner.signal_store import decide_optimizer_trigger

    should, reason = decide_optimizer_trigger(balanced_pool(3), {}, scheduled=True)
    assert should is False
    assert "6" in reason and "example" in reason.lower(), reason


# --------------------------------------------------------------------------
# 3. The derived constants, each pinned to the thing it was derived FROM
# --------------------------------------------------------------------------


def test_the_feasibility_floor_is_pinned_to_the_gepa_guard_it_came_from():
    """``MIN_FEASIBLE_TRAINSET_EXAMPLES`` is not a chosen number.

    ``_optimize_with_gepa`` splits via ``gepa_split_sizes`` and returns None
    when ``len(trainset) < 5``. The floor is the smallest EVEN n (the builder
    emits pairs) that clears it. Asserted through the SAME split function the
    optimizer uses rather than a literal ``int(n * 0.8)``, so moving the cut
    fails here instead of silently invalidating the floor.
    """
    from src.agents.feedback_learner.dspy_integration import (
        MIN_FEASIBLE_TRAINSET_EXAMPLES,
        gepa_split_sizes,
    )

    n = MIN_FEASIBLE_TRAINSET_EXAMPLES
    assert n % 2 == 0, "the builder emits balanced pairs, so the floor is even"
    assert gepa_split_sizes(n)[0] >= 5, "at the floor, GEPA's own trainset guard must pass"
    assert gepa_split_sizes(n - 2)[0] < 5, "one pair below the floor, the guard must reject"


def test_the_budget_ladder_is_derived_from_dspys_own_candidate_counts():
    """``*2``/``*3`` are replaced by the size each preset's ranking needs.

    dspy's ``AUTO_RUN_SETTINGS`` says how many candidate programs each preset
    explores (measured: light 6, medium 12, heavy 18). A candidate can only be
    ranked if the valset can express a distinct level for it; on the gold-empty
    half the metric is binary (measured), so a V-example valset expresses at
    least V+1 levels, and V is ``n - int(0.8n)``.
    """
    dspy_gepa = pytest.importorskip("dspy.teleprompt.gepa.gepa")
    from src.agents.feedback_learner.dspy_integration import BUDGET_MIN_TRAINSET_EXAMPLES

    for preset, cfg in dspy_gepa.AUTO_RUN_SETTINGS.items():
        candidates = cfg["n"]
        expected = 2
        while (expected - int(expected * 0.8)) + 1 < candidates:
            expected += 2
        assert BUDGET_MIN_TRAINSET_EXAMPLES[preset] == expected, preset


def test_the_threshold_clears_the_lightest_presets_ranking_floor():
    """The gate must never open below the size the budget it spends can use.

    Production spends ``budget="light"`` (``run_feedback_learner_optimization``
    default), so the gate opening below light's ranking floor would authorise a
    run whose candidate selection the valset cannot support.
    """
    from src.agents.feedback_learner.dspy_integration import (
        BUDGET_MIN_TRAINSET_EXAMPLES,
        GEPAOptimizationTrigger,
    )

    assert GEPAOptimizationTrigger().min_trainset_examples >= BUDGET_MIN_TRAINSET_EXAMPLES["light"]


def test_budget_escalates_on_the_derived_thresholds_not_on_multiples_of_the_gate():
    from src.agents.feedback_learner.dspy_integration import (
        BUDGET_MIN_TRAINSET_EXAMPLES,
        GEPAOptimizationTrigger,
    )

    trigger = GEPAOptimizationTrigger()
    heavy = BUDGET_MIN_TRAINSET_EXAMPLES["heavy"]
    medium = BUDGET_MIN_TRAINSET_EXAMPLES["medium"]

    assert trigger.get_recommended_budget(heavy, hours_since_last=1.0) == "heavy"
    assert trigger.get_recommended_budget(heavy - 2, hours_since_last=1.0) == "medium"
    assert trigger.get_recommended_budget(medium, hours_since_last=1.0) == "medium"
    assert trigger.get_recommended_budget(medium - 2, hours_since_last=1.0) == "light"


def test_the_critical_override_relaxes_adequacy_but_not_selectability():
    """Urgency cannot create data, and it cannot make a choice measurable.

    The override used to accept ``min_signals // 2`` — a fraction of a number
    whose unit changed underneath it, which in the new unit is "half a
    trainset". It now bounds at the lightest preset's ranking floor: below it
    the valset cannot express a distinct level per candidate, so the argmax
    that installs a prompt is arbitrary and urgency buys only spend.

    That bound is 22, which is STRICTER than the 20 examples ``// 2`` allowed —
    asserted here, because a critical-pattern path that fires where the
    pre-change one stayed quiet would be a behaviour change this PR does not
    claim.
    """
    from src.agents.feedback_learner.dspy_integration import (
        BUDGET_MIN_TRAINSET_EXAMPLES,
        GEPAOptimizationTrigger,
    )

    trigger = GEPAOptimizationTrigger()
    floor = BUDGET_MIN_TRAINSET_EXAMPLES["light"]

    # The old bar, in the new unit: `min_signals // 2` == 10 per class == 20.
    assert floor >= 20, "the override must not fire below where it fired before"

    fires, reason = trigger.should_trigger(
        trainset_examples=floor, current_reward=0.0, has_critical_patterns=True
    )
    assert fires is True, reason

    quiet, reason = trigger.should_trigger(
        trainset_examples=floor - 2, current_reward=0.0, has_critical_patterns=True
    )
    assert quiet is False, reason
    assert "example" in reason.lower()


# --------------------------------------------------------------------------
# 4. The env override moved with the unit
# --------------------------------------------------------------------------


def test_the_env_override_uses_the_new_name(monkeypatch):
    from src.agents.feedback_learner.signal_store import optimizer_min_trainset_examples

    monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
    monkeypatch.setenv("DSPY_MIN_TRAINSET_EXAMPLES", "44")
    assert optimizer_min_trainset_examples() == 44


def test_the_old_env_name_is_TRANSLATED_not_read_at_face_value(monkeypatch, caplog):
    """``DSPY_MIN_SIGNALS=N`` meant N per class, which is a 2N-example trainset.

    Reading N as examples would HALVE whatever the operator asked for. IGNORING
    it — the first version of this — is only fail-closed for N <= 20: a host
    with N=100 meant a 200-example bar and would have been silently relaxed to
    the 40-example default. Translating is the only reading that changes
    nothing at any value, which is what this PR claims of itself.
    """
    import logging

    from src.agents.feedback_learner.signal_store import optimizer_min_trainset_examples

    monkeypatch.delenv("DSPY_MIN_TRAINSET_EXAMPLES", raising=False)

    monkeypatch.setenv("DSPY_MIN_SIGNALS", "20")
    with caplog.at_level(logging.WARNING):
        assert optimizer_min_trainset_examples() == 40
    assert "DSPY_MIN_SIGNALS" in caplog.text

    # The case that makes IGNORING it wrong: a legacy value STRICTER than the
    # default must stay stricter.
    monkeypatch.setenv("DSPY_MIN_SIGNALS", "100")
    assert optimizer_min_trainset_examples() == 200


def test_the_new_env_name_wins_when_both_are_set(monkeypatch, caplog):
    """It needs no interpretation — it is already in the gate's unit."""
    import logging

    from src.agents.feedback_learner.signal_store import optimizer_min_trainset_examples

    monkeypatch.setenv("DSPY_MIN_SIGNALS", "100")
    monkeypatch.setenv("DSPY_MIN_TRAINSET_EXAMPLES", "44")
    with caplog.at_level(logging.WARNING):
        assert optimizer_min_trainset_examples() == 44
    assert "Both" in caplog.text


def test_an_override_below_the_feasibility_floor_is_clamped(monkeypatch, caplog):
    """Below the floor every run provably compiles nothing but still spends state."""
    import logging

    from src.agents.feedback_learner.dspy_integration import MIN_FEASIBLE_TRAINSET_EXAMPLES
    from src.agents.feedback_learner.signal_store import optimizer_min_trainset_examples

    monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
    monkeypatch.setenv("DSPY_MIN_TRAINSET_EXAMPLES", "4")
    with caplog.at_level(logging.WARNING):
        assert optimizer_min_trainset_examples() == MIN_FEASIBLE_TRAINSET_EXAMPLES
    assert "DSPY_MIN_TRAINSET_EXAMPLES" in caplog.text


def test_the_default_has_ONE_definition(monkeypatch):
    """signal_store must not carry a second copy of the dataclass default."""
    from src.agents.feedback_learner import signal_store
    from src.agents.feedback_learner.dspy_integration import GEPAOptimizationTrigger

    monkeypatch.delenv("DSPY_MIN_TRAINSET_EXAMPLES", raising=False)
    monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
    assert (
        signal_store.optimizer_min_trainset_examples()
        == GEPAOptimizationTrigger().min_trainset_examples
    )
    assert not hasattr(signal_store, "DEFAULT_MIN_SIGNALS")


def test_the_override_is_documented_as_not_reaching_the_containers():
    """The comment claims the knob is inert in prod. Pin the claim to reality.

    ``x-common-env`` in docker/docker-compose.yml is a WHITELIST, and neither
    the new env name nor the ``DSPY_MIN_SIGNALS`` it replaces is in it — so the
    in-code default governs every containerised run. That gap predates this
    change and #1489 recorded the decision to leave it alone ("forwarding
    DSPY_MIN_SIGNALS would change when the nightly optimization triggers — a
    behavioral change that needs its own decision, not a drive-by").

    This test exists so the two cannot silently disagree in EITHER direction:
    wire the var into compose and this fails, forcing whoever does it to delete
    the comment that says it is not wired — which is the moment that behavioural
    decision gets made deliberately.
    """
    import pathlib

    from src.agents.feedback_learner import signal_store

    compose = pathlib.Path("docker/docker-compose.yml")
    if not compose.exists():  # pragma: no cover - repo layout guard
        pytest.skip("compose file not present in this checkout")
    text = compose.read_text()

    # Positive control: the file really is the env-carrying one, so "absent"
    # below is a property of the variable rather than of an unread file.
    assert "x-common-env" in text and "DSPY_LM_MODEL" in text

    assert signal_store.MIN_TRAINSET_EXAMPLES_ENV not in text
    assert signal_store.LEGACY_MIN_SIGNALS_ENV not in text

    import inspect

    doc = inspect.getsource(signal_store).split("def optimizer_min_trainset_examples")[0]
    assert "NOT FORWARDED TO CONTAINERS" in doc


# --------------------------------------------------------------------------
# 5. The gate is still closed on the real corpus shape
# --------------------------------------------------------------------------


def test_the_gate_is_closed_at_the_production_supply_shape(monkeypatch):
    """15 positives / 60 negatives — the measured corpus — must stay CLOSED.

    Equivalently 30 examples < 40. Stated in the new unit so a future edit that
    changes the unit again cannot keep this passing by coincidence.
    """
    pytest.importorskip("dspy")
    monkeypatch.delenv("DSPY_MIN_TRAINSET_EXAMPLES", raising=False)
    monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)

    from src.agents.feedback_learner.dspy_integration import FeedbackLearnerOptimizer
    from src.agents.feedback_learner.signal_store import (
        decide_optimizer_trigger,
        optimizer_min_trainset_examples,
    )

    pool = [positive(f"p{i}") for i in range(15)] + [negative(f"n{i}") for i in range(60)]
    built = FeedbackLearnerOptimizer(optimizer_type="gepa")._signals_to_examples(pool, "pattern")
    assert len(built) == 30

    should, reason = decide_optimizer_trigger(pool, {}, scheduled=True)
    assert should is False, reason
    assert f"30 < {optimizer_min_trainset_examples()}" in reason, reason


# --------------------------------------------------------------------------
# 6. The status surface publishes the gate's own unit
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_publishes_the_number_the_gate_compares(monkeypatch):
    pytest.importorskip("dspy")
    monkeypatch.delenv("DSPY_MIN_TRAINSET_EXAMPLES", raising=False)
    monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)

    from src.agents.feedback_learner import signal_store

    pool = [positive(f"p{i}") for i in range(15)] + [negative(f"n{i}") for i in range(60)]

    async def _pool(_client=None, **_kw):
        return pool

    monkeypatch.setattr(signal_store, "read_optimizer_signal_pool", _pool)

    class _Res:
        count = 223

    class _Table:
        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            return _Res()

    class _Client:
        def table(self, *_a, **_k):
            return _Table()

    status = await signal_store.get_optimizer_gate_status(_Client())
    assert status["trainset_examples"] == 30
    assert status["min_trainset_examples"] == 40
    assert "trainable_signals" not in status
    assert status["would_trigger"] is False
    assert f"30 < {status['min_trainset_examples']}" in status["reason"]


# --------------------------------------------------------------------------
# 7. The persisted run row records the sets the optimizer TRAINS AND VALIDATES on
# --------------------------------------------------------------------------


def test_gepa_split_has_one_definition():
    """``_optimize_with_gepa``'s 80/20 cut must not be re-derived by its readers.

    The first attempt at fixing ``record_run_started(trainset_size=...)``
    replaced the POOL size (223) with the EXAMPLE count (30) — still not the
    trainset, because GEPA trains on ``examples[: int(0.8n)]`` = 24 and holds 6
    back. Fixing a unit error by landing on the next one along is the failure
    this file exists to stop, so the cut has exactly one definition and both
    the optimizer and the recorder read it.
    """
    import inspect

    from src.agents.feedback_learner.dspy_integration import (
        FeedbackLearnerOptimizer,
        gepa_split_sizes,
    )

    assert gepa_split_sizes(30) == (24, 6)
    assert gepa_split_sizes(40) == (32, 8)
    assert sum(gepa_split_sizes(37)) == 37, "the split must be exhaustive at any size"

    src = inspect.getsource(FeedbackLearnerOptimizer._optimize_with_gepa)
    assert "gepa_split_sizes" in src, "the optimizer must use the shared split, not re-derive it"
    assert "int(len(examples) * 0.8)" not in src


def test_the_feasibility_floor_is_pinned_to_the_shared_split():
    """The floor tracks the real split function, not a literal copy of it."""
    from src.agents.feedback_learner.dspy_integration import (
        MIN_FEASIBLE_TRAINSET_EXAMPLES,
        gepa_split_sizes,
    )

    train, _val = gepa_split_sizes(MIN_FEASIBLE_TRAINSET_EXAMPLES)
    assert train >= 5
    train_below, _ = gepa_split_sizes(MIN_FEASIBLE_TRAINSET_EXAMPLES - 2)
    assert train_below < 5


def test_recorded_run_sizes_are_the_sets_the_optimizer_trains_and_validates_on():
    """The two paths split in OPPOSITE directions from the same example list.

    Both siblings (`recipient_optimizer`, the RAG leg) record
    ``trainset_size=len(trainset)`` / ``valset_size=len(valset)`` for their
    post-split sets — which, because they pass an explicit valset, is also what
    they hand to ``compile()``. This path does not: `_optimize_with_miprov2`
    passes the WHOLE list with no valset and dspy re-splits it internally, so
    "what is passed to compile" and "what is trained on" are 30 and 6. The
    column is named trainset_size, so it takes the latter — otherwise the
    feedback-learner runner would record a third quantity under a name its
    siblings already use for something else.
    """
    from src.agents.feedback_learner.dspy_integration import recorded_set_sizes

    assert recorded_set_sizes(30, "gepa") == (24, 6)
    # MIPROv2 is handed all 30 and splits the OTHER WAY internally: dspy keeps
    # the 20% prefix as the trainset. 30 is the ARGUMENT, 6 is the trainset.
    assert recorded_set_sizes(30, "miprov2") == (6, 24)
    assert recorded_set_sizes(0, "gepa") == (0, 0)


def test_the_miprov2_split_matches_the_INSTALLED_dspy_not_our_reading_of_it():
    """Pinned against dspy's real function, so a version bump fails loudly here.

    `miprov2_split_sizes` restates arithmetic that lives in the dependency
    (`_set_and_validate_datasets`). Restating a dependency's internals is a
    drift risk by definition, so it is asserted against the dependency itself
    rather than against the numbers I read out of it. Spends no tokens: this is
    argument validation, no rollout.
    """
    mipro = pytest.importorskip("dspy.teleprompt.mipro_optimizer_v2")
    dspy = pytest.importorskip("dspy")

    from src.agents.feedback_learner.dspy_integration import miprov2_split_sizes

    optimizer = object.__new__(mipro.MIPROv2)  # no LM needed for the split
    for n in (2, 5, 8, 30, 40, 44, 100):
        examples = [dspy.Example(q=str(i)).with_inputs("q") for i in range(n)]
        real_train, real_val = mipro.MIPROv2._set_and_validate_datasets(optimizer, examples, None)
        assert miprov2_split_sizes(n) == (len(real_train), len(real_val)), n
