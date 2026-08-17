"""Shard 05: optimizer save fix + orchestrator wiring."""

from __future__ import annotations

import inspect

import pytest

from src.agents.feedback_learner import dspy_integration as fdi

dspy = pytest.importorskip("dspy")


def test_gepa_no_broken_await_save():
    """The broken `await save_optimized_module(...)` call must be gone (sync fn)."""
    src = inspect.getsource(fdi.FeedbackLearnerOptimizer._optimize_with_gepa)
    assert "await save_optimized_module" not in src
    assert "optimized_module=" not in src  # the wrong kwarg name


def test_gepa_passes_auto_not_budget_kwarg():
    """create_gepa_optimizer maps budget->auto; passing budget= would TypeError GEPA()."""
    src = inspect.getsource(fdi.FeedbackLearnerOptimizer._optimize_with_gepa)
    assert "budget=budget" not in src
    assert "auto=budget" in src


def test_miprov2_construction_is_accepted_by_real_dspy_validation():
    """#1668 (codex iter-3): the MIPROv2 fallback could never compile.

    dspy 3.1.0 defaults ``MIPROv2.auto`` to ``"light"``
    (mipro_optimizer_v2.py:56) and ``compile`` rejects an explicit
    ``num_candidates``/``num_trials`` while ``auto`` is set (:151). The fallback
    constructed ``MIPROv2(num_candidates=10)`` and then called
    ``compile(num_trials=budget)``, so **every** MIPROv2 run raised ValueError
    before a single rollout. Making its metric gold-aware would have been moot
    on a path that cannot run at all.

    This exercises dspy's REAL validation rather than asserting our own source:
    the construction must get PAST the auto gate. No LM call is made — both the
    old and the new error are raised during argument validation.
    """
    from dspy.teleprompt import MIPROv2

    # MIPROv2.__init__ requires a default LM to exist. It is never called.
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="not-used-no-call-is-made"))
    student = dspy.Predict("question -> answer")

    def _metric(example, prediction, trace=None) -> float:
        return 1.0

    # RED (the old construction): auto stays "light" and compile refuses.
    old = MIPROv2(metric=_metric, num_candidates=10, max_bootstrapped_demos=4, num_threads=4)
    assert old.auto == "light"
    with pytest.raises(ValueError, match="If auto is not None"):
        old.compile(student, trainset=[], num_trials=50)

    # GREEN: the construction the code now uses clears that gate — the next
    # failure is the unrelated dataset check, which is how we know we got past.
    src = inspect.getsource(fdi.FeedbackLearnerOptimizer._optimize_with_miprov2)
    assert "auto=None" in src, "the MIPROv2 fallback must pin auto=None"

    fixed = MIPROv2(
        metric=_metric, auto=None, num_candidates=10, max_bootstrapped_demos=4, num_threads=4
    )
    with pytest.raises(ValueError) as excinfo:
        fixed.compile(student, trainset=[], num_trials=50)
    assert "If auto is not None" not in str(excinfo.value)
    assert "Trainset cannot be empty" in str(excinfo.value)


def test_miprov2_clears_the_minibatch_gate_at_realistic_trainset_sizes():
    """#1668 (codex iter-4): the SECOND pre-rollout gate, at the sizes we build.

    dspy derives ``valset = int(0.80 * len(trainset))`` when none is passed
    (mipro_optimizer_v2.py:311-317) and refuses to minibatch when the default
    ``minibatch_size=35`` exceeds it (:201). With ``minibatch`` defaulting to
    True, EVERY trainset below 44 examples raised before any rollout — and the
    #1668 balanced builder produces 30 from today's 220 real signals.

    Spends no tokens: the first LM-touching step, ``_bootstrap_fewshot_examples``,
    is replaced with a sentinel, so reaching it is positive proof that both
    validation gates were cleared rather than an absence of errors.
    """
    from dspy.teleprompt import MIPROv2

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="not-used-no-call-is-made"))
    student = dspy.Predict("question -> answer")
    trainset = [
        dspy.Example(question=f"q{i}", answer=f"a{i}").with_inputs("question") for i in range(30)
    ]

    def _metric(example, prediction, trace=None) -> float:
        return 1.0

    def _new():
        return MIPROv2(
            metric=_metric, auto=None, num_candidates=10, max_bootstrapped_demos=4, num_threads=4
        )

    # RED: dspy's default minibatch on a 30-example trainset (valset 24 < 35).
    with pytest.raises(ValueError, match="Minibatch size cannot exceed"):
        _new().compile(student, trainset=trainset, num_trials=6)

    class _ReachedBootstrapping(RuntimeError):
        pass

    def _sentinel(*_args, **_kwargs):
        raise _ReachedBootstrapping

    fixed = _new()
    fixed._bootstrap_fewshot_examples = _sentinel  # type: ignore[method-assign]
    with pytest.raises(_ReachedBootstrapping):
        fixed.compile(student, trainset=trainset, num_trials=6, minibatch=False)

    src = inspect.getsource(fdi.FeedbackLearnerOptimizer._optimize_with_miprov2)
    assert "minibatch=False" in src, "the MIPROv2 fallback must disable minibatching"


def test_save_load_roundtrip_offline(tmp_path):
    """A real ChainOfThought saves and loads on dspy 3.1.0 without an LLM call."""
    from src.agents.feedback_learner.dspy_integration import PatternDetectionSignature
    from src.optimization.gepa import load_optimized_module, save_optimized_module

    module = dspy.ChainOfThought(PatternDetectionSignature)
    info = save_optimized_module(
        module,
        agent_name="feedback_learner_pattern",
        output_dir=str(tmp_path),
        metadata={"phase": "pattern", "budget": "light"},
    )
    assert "version_id" in info and "path" in info
    # load_optimized_module calls module_cls() (versioning.py:160), so pass a
    # zero-arg FACTORY that supplies the signature — NOT the bare class.
    loaded, meta = load_optimized_module(
        lambda: dspy.ChainOfThought(PatternDetectionSignature),
        agent_name="feedback_learner_pattern",
        version_id=info["version_id"],
        input_dir=str(tmp_path),
    )
    # Loaded object is a usable dspy module.
    assert hasattr(loaded, "predictors") or hasattr(loaded, "forward")


@pytest.mark.asyncio
async def test_orchestrator_skips_when_insufficient_signals():
    from src.agents.feedback_learner.optimization_runner import (
        run_feedback_learner_optimization,
    )

    class _EmptyClient:
        def table(self, *_):
            return self

        def select(self, *_):
            return self

        def eq(self, *_):
            return self

        def gte(self, *_):
            return self

        def limit(self, *_):
            return self

        def execute(self):
            return type("R", (), {"data": []})()

    result = await run_feedback_learner_optimization(client=_EmptyClient())
    assert result["status"] == "skipped_insufficient_signals"
    assert result["signals_used"] == 0


@pytest.mark.asyncio
async def test_orchestrator_discards_run_on_pre_compile_skip(monkeypatch):
    """optimize() returns None ONLY from pre-compile guards (both optimizer
    paths: dspy/GEPA unavailable, <5 examples, unavailable phase — compile
    failures raise). No budget was spent, so the provisional run row must be
    DISCARDED, not persisted as a failed run. codex iter-1 HIGH finding."""
    import src.repositories.prompt_optimization as rec
    from src.agents.feedback_learner.optimization_runner import (
        run_feedback_learner_optimization,
    )

    calls = {"started": 0, "failed": 0, "discarded": []}

    async def fake_started(**kwargs):
        calls["started"] += 1
        return "run-prov-1"

    async def fake_failed(run_id, *args, **kwargs):
        calls["failed"] += 1
        return True

    async def fake_discarded(run_id, client=None):
        calls["discarded"].append(run_id)
        return True

    monkeypatch.setattr(rec, "record_run_started", fake_started)
    monkeypatch.setattr(rec, "record_run_failed", fake_failed)
    monkeypatch.setattr(rec, "record_run_discarded", fake_discarded, raising=False)

    async def fake_signals(client=None, min_reward=0.5):
        # 6 signals pass the runner's MIN_SIGNALS=5 guard, yet GEPA's own
        # trainset guard (int(6 * 0.8) = 4 < 5) skips pre-compile — the
        # common real shape of this bug.
        return [{"source_agent": "feedback_learner", "reward": 0.9}] * 6

    monkeypatch.setattr(
        "src.agents.feedback_learner.signal_store.get_feedback_learner_training_signals",
        fake_signals,
    )
    monkeypatch.setattr("src.optimization.dspy_lm.ensure_dspy_configured", lambda: True)

    class _SkippingOptimizer:
        optimizer_type = "gepa"

        def __init__(self, optimizer_type="gepa"):
            pass

        async def optimize(self, phase, signals, budget="light"):
            return None

    monkeypatch.setattr(fdi, "FeedbackLearnerOptimizer", _SkippingOptimizer)

    result = await run_feedback_learner_optimization(phases=("pattern",))

    assert result["phases"]["pattern"]["status"] == "no_module"
    assert calls["started"] == 1
    assert calls["discarded"] == ["run-prov-1"]
    assert calls["failed"] == 0


@pytest.mark.asyncio
async def test_run_that_compiles_nothing_is_not_reported_as_completed(monkeypatch, tmp_path):
    """#1668: a run where every phase built no trainset must not look like success.

    This is the issue's own acceptance item — "a daily task that has never run
    should not look identical to one that ran and found nothing to do" — and the
    #1668 trainset fix makes the no-module outcome MORE likely, not less: a
    single-class signal pool is now an explicit skip rather than a silently
    biased trainset. Reporting ``completed`` for that would relocate the silent
    inertness from the beat down into the runner.

    No LM is reached and the optimizer is the REAL one:
    ``_signals_to_examples`` returns ``[]`` for a single-class pool, so
    ``_optimize_with_gepa`` returns None at its ``< 5`` guard long before
    ``compile``. Only the dspy-configured probe and the signal read are stubbed.
    """
    from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal
    from src.agents.feedback_learner.optimization_runner import (
        run_feedback_learner_optimization,
    )

    sig = FeedbackLearnerTrainingSignal(
        batch_id="allpos",
        feedback_count=6,
        time_range_start="t0",
        time_range_end="t1",
        patterns_detected=1,
        recommendations_generated=1,
        feedback_batch=[{"feedback_id": "f1", "feedback_type": "rating", "user_feedback": 2}],
        patterns=[{"pattern_type": "accuracy_issue", "severity": "high"}],
        recommendations=[{"category": "prompt_update", "expected_impact": "x"}],
        learning_summary="Learning cycle complete. Processed 6 feedback items.",
        total_latency_ms=900.0,
    )
    rows = []
    for _ in range(8):  # all POSITIVE -> single class -> no trainset for any phase
        d = sig.to_dict()
        d["reward"] = 0.9
        rows.append(d)

    async def fake_signals(client=None, min_reward=0.0, limit=1000):
        return rows

    monkeypatch.setattr(
        "src.agents.feedback_learner.signal_store.get_feedback_learner_training_signals",
        fake_signals,
    )
    monkeypatch.setattr("src.optimization.dspy_lm.ensure_dspy_configured", lambda: True)
    monkeypatch.chdir(tmp_path)

    result = await run_feedback_learner_optimization()

    assert all(p.get("status") == "no_module" for p in result["phases"].values()), result["phases"]
    assert result["status"] != "completed", (
        "a run that compiled nothing reports the same status as one that did"
    )
    assert result["status"] == "completed_no_modules"
