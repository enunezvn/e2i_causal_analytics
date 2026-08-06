"""The nightly RAG-prompt GEPA leg (issue #1486).

The leg is deliberately OPPORTUNISTIC. The replay that produces its input is
manual-only, and #1485 measured that only 3 of 10 replayed turns retrieved any
evidence at all — a turn that retrieved nothing records an empty ``contexts``
list on purpose. So "no usable examples, do nothing" is the EXPECTED nightly
outcome, not an edge case, and the no-op path must cost zero API calls.

Cost is the other constraint. Measured against installed dspy 3.1.0,
``auto="light"`` resolves to ~384-396 metric calls almost independently of
dataset size (5 examples -> 384, 20 -> 396: the budget is driven by
num_candidates=6, not by len(trainset)). At four RAGAS sub-metrics per metric
call that is 1,500+ judge LLM calls per run, which #504's calibration (~96 min
for a 30-sample RAGAS eval) puts in the many-hours range. Capping examples does
NOT bound that; only an explicit ``max_metric_calls`` does. Hence the tests
below pin an explicit budget rather than ``auto``.
"""

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.xdist_group(name="gepa_metrics")


def _record(query: str, contexts: list[str], answer: str = "an answer", **over: Any) -> dict:
    """A replay record in build_cognitive_record's shape (#1485).

    Consumed as PURE JSON — this lane never imports #1485 code.
    """
    rec = {
        "query_id": "q1",
        "query": query,
        "conversation_id": "c1",
        "target": "cognitive",
        "response_text": answer,
        "contexts": contexts,
        "evidence_count": len(contexts),
        "detected_intent": "CAUSAL_ANALYSIS",
        "error": None,
    }
    rec.update(over)
    return rec


def _records_file(tmp_path: Path, records: list[dict], wrap: bool = True) -> Path:
    path = tmp_path / "goldset_records.json"
    path.write_text(json.dumps({"records": records} if wrap else records))
    return path


class TestRecordParsing:
    def test_reads_the_replay_record_shape(self, tmp_path: Path) -> None:
        from src.tasks.dspy_optimization_tasks import load_rag_examples_from_records

        path = _records_file(
            tmp_path, [_record("why did TRx drop?", ["Q4 report: payer mix"], "TRx fell 12%")]
        )
        examples = load_rag_examples_from_records(str(path))

        assert len(examples) == 1
        ex = examples[0]
        assert ex.user_query == "why did TRx drop?"
        assert ex.retrieved_contexts == ["Q4 report: payer mix"]

    def test_accepts_a_bare_list_as_well_as_the_wrapper(self, tmp_path: Path) -> None:
        from src.tasks.dspy_optimization_tasks import load_rag_examples_from_records

        path = _records_file(tmp_path, [_record("q", ["ctx"])], wrap=False)
        assert len(load_rag_examples_from_records(str(path))) == 1

    def test_drops_records_with_no_retrieved_contexts(self, tmp_path: Path) -> None:
        """#1485: only ~3 of 10 turns retrieve anything; the rest are unjudgeable.

        The RAGAS metric refuses a no-context example, and a refusal inside GEPA
        becomes failure_score 0.0 — so these must be filtered out here, before
        they can fabricate a bad-quality signal.
        """
        from src.tasks.dspy_optimization_tasks import load_rag_examples_from_records

        path = _records_file(
            tmp_path,
            [_record("has ctx", ["ctx"]), _record("no ctx", []), _record("also none", [])],
        )
        examples = load_rag_examples_from_records(str(path))
        assert [e.user_query for e in examples] == ["has ctx"]

    def test_drops_failed_and_answerless_records(self, tmp_path: Path) -> None:
        from src.tasks.dspy_optimization_tasks import load_rag_examples_from_records

        path = _records_file(
            tmp_path,
            [
                _record("ok", ["ctx"]),
                _record("errored", ["ctx"], error="CognitiveSearchError: boom"),
                _record("empty answer", ["ctx"], answer=""),
            ],
        )
        assert [e.user_query for e in load_rag_examples_from_records(str(path))] == ["ok"]


class TestOpportunisticNoOp:
    """Every no-op path must cost ZERO API calls and say why, loudly."""

    @pytest.fixture(autouse=True)
    def _forbid_optimizer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any evaluator/optimizer construction in a no-op path is a bug."""
        import src.optimization.gepa.metrics as metrics_module

        def _explode(*a: Any, **k: Any) -> Any:
            raise AssertionError("no-op path constructed a metric/optimizer (API cost)")

        monkeypatch.setattr(metrics_module, "get_metric_for_agent", _explode, raising=True)

    @pytest.mark.asyncio
    async def test_no_env_var_configured_is_a_loud_skip(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from src.tasks.dspy_optimization_tasks import (
            RAG_RECORDS_PATH_ENV,
            run_rag_prompt_optimization,
        )

        monkeypatch.delenv(RAG_RECORDS_PATH_ENV, raising=False)
        with caplog.at_level("INFO", logger="src.tasks.dspy_optimization_tasks"):
            result = await run_rag_prompt_optimization()

        assert result["status"] == "skipped"
        assert RAG_RECORDS_PATH_ENV in result["reason"]
        # The operator must be able to act on the log alone.
        blob = " ".join(r.message for r in caplog.records)
        assert RAG_RECORDS_PATH_ENV in blob
        assert "replay_golden_set" in blob
        # Codex iter-1 F5 residual: #1485's parser rejects --record-out unless
        # --target cognitive accompanies it, so a message an operator can follow
        # verbatim must name it.
        assert "--target cognitive" in blob

    @pytest.mark.asyncio
    async def test_missing_file_is_a_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.tasks.dspy_optimization_tasks import (
            RAG_RECORDS_PATH_ENV,
            run_rag_prompt_optimization,
        )

        monkeypatch.setenv(RAG_RECORDS_PATH_ENV, str(tmp_path / "absent.json"))
        result = await run_rag_prompt_optimization()
        assert result["status"] == "skipped"
        assert "not found" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_too_few_usable_examples_is_a_skip_naming_the_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The expected nightly outcome — so it must report usable vs total."""
        from src.tasks.dspy_optimization_tasks import (
            RAG_RECORDS_PATH_ENV,
            run_rag_prompt_optimization,
        )

        records = [_record(f"q{i}", ["ctx"]) for i in range(2)] + [
            _record(f"n{i}", []) for i in range(8)
        ]
        monkeypatch.setenv(RAG_RECORDS_PATH_ENV, str(_records_file(tmp_path, records)))

        result = await run_rag_prompt_optimization()
        assert result["status"] == "skipped"
        assert result["usable_examples"] == 2
        assert result["total_records"] == 10

    @pytest.mark.asyncio
    async def test_already_optimized_records_are_skipped_as_stale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dedup on the records fingerprint, mirroring the .trigger_state.json pattern.

        Without this the beat would re-spend the full judge budget every 24h on
        a records file that has not changed.
        """
        from src.tasks.dspy_optimization_tasks import (
            RAG_RECORDS_PATH_ENV,
            _rag_max_metric_calls,
            _rag_records_fingerprint,
            _save_trigger_state,
            run_rag_prompt_optimization,
        )

        monkeypatch.chdir(tmp_path)
        records = [_record(f"q{i}", ["ctx"]) for i in range(8)]
        path = _records_file(tmp_path, records)
        monkeypatch.setenv(RAG_RECORDS_PATH_ENV, str(path))

        # Fingerprint covers records content AND budget (F4a).
        _save_trigger_state(
            {
                "rag_records_fingerprint": _rag_records_fingerprint(
                    str(path), _rag_max_metric_calls()
                )
            }
        )

        result = await run_rag_prompt_optimization()
        assert result["status"] == "skipped"
        assert "already" in result["reason"].lower() or "stale" in result["reason"].lower()


class TestBudgetIsExplicitNotAuto:
    def test_leg_uses_explicit_max_metric_calls(self) -> None:
        """auto="light" is ~384-396 metric calls regardless of dataset size.

        At 4 RAGAS sub-metrics per call that is 1,500+ judge calls — capping
        examples cannot bound it, so the leg must set max_metric_calls itself.
        """
        import inspect

        from src.tasks.dspy_optimization_tasks import run_rag_prompt_optimization

        # Scoped to the function body: the module docstring/comments discuss
        # auto="light" precisely to explain why it is NOT used.
        source = inspect.getsource(run_rag_prompt_optimization)
        assert "max_metric_calls=budget" in source
        assert "auto=None" in source
        assert 'auto="light"' not in source

    def test_budget_default_is_conservative_and_env_tunable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.tasks.dspy_optimization_tasks import (
            RAG_MAX_METRIC_CALLS_ENV,
            _rag_max_metric_calls,
        )

        monkeypatch.delenv(RAG_MAX_METRIC_CALLS_ENV, raising=False)
        assert _rag_max_metric_calls() <= 60

        monkeypatch.setenv(RAG_MAX_METRIC_CALLS_ENV, "12")
        assert _rag_max_metric_calls() == 12


class TestArtifactConsumerChain:
    """The chain rounds 1-2 proved missing: leg -> artifact -> runtime."""

    def test_save_name_and_load_name_are_one_constant(self) -> None:
        """Two literals that can drift is how the 6-week unshipped artifact happened."""
        import inspect

        import src.tasks.dspy_optimization_tasks as task_module
        from src.rag.cognitive_rag_dspy import OPTIMIZED_SYNTHESIS_AGENT_NAME

        source = inspect.getsource(task_module)
        assert "OPTIMIZED_SYNTHESIS_AGENT_NAME" in source, (
            "the leg hardcodes its save name instead of importing the runtime's constant"
        )
        assert f'"{OPTIMIZED_SYNTHESIS_AGENT_NAME}"' not in source, (
            "the leg re-declares the agent name as a literal; it must import the constant"
        )

    def test_runtime_loads_what_the_leg_saves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end on the artifact contract, without running GEPA.

        Saves through the leg's own save path and loads through the runtime's
        loader, so a drift in either name or directory breaks this test.
        """
        import dspy

        from src.optimization.gepa.versioning import save_optimized_module
        from src.rag.cognitive_rag_dspy import (
            OPTIMIZED_SYNTHESIS_AGENT_NAME,
            EvidenceSynthesisSignature,
            load_optimized_synthesis_module,
        )

        monkeypatch.chdir(tmp_path)
        marker = "CHAIN-1486"
        save_optimized_module(
            module=dspy.ChainOfThought(EvidenceSynthesisSignature.with_instructions(marker)),
            agent_name=OPTIMIZED_SYNTHESIS_AGENT_NAME,
            output_dir="./optimized_modules",
        )

        loaded = load_optimized_synthesis_module(reset=True)
        assert loaded is not None
        assert any(
            marker in (getattr(getattr(p, "signature", None), "instructions", "") or "")
            for p in loaded.predictors()
        )


class _FakeOptimizer:
    """Stands in for a compiled GEPA run, without spending a judge call."""

    def __init__(self, instructions: str, degraded: int = 0, metric: Any = None) -> None:
        self._instructions = instructions
        self._degraded = degraded
        self._metric = metric

    def compile(self, module: Any, trainset: Any = None, valset: Any = None) -> Any:
        import dspy

        from src.rag.cognitive_rag_dspy import EvidenceSynthesisSignature

        if self._metric is not None:
            self._metric._degraded_examples = self._degraded
        return dspy.ChainOfThought(EvidenceSynthesisSignature.with_instructions(self._instructions))


def _arm_leg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    instructions: str = "AN IMPROVED PROMPT",
    degraded: int = 0,
) -> Any:
    """Point the leg at a usable records file with the judge/optimizer stubbed."""
    import src.tasks.dspy_optimization_tasks as task_module

    monkeypatch.chdir(tmp_path)
    records = [_record(f"q{i}", [f"ctx{i}"], f"answer {i}") for i in range(8)]
    path = _records_file(tmp_path, records)
    monkeypatch.setenv(task_module.RAG_RECORDS_PATH_ENV, str(path))

    class _Metric:
        """Callable, so it satisfies GEPA's metric contract if ever constructed."""

        _degraded_examples = 0

        def __call__(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - never scored here
            raise AssertionError("the stubbed optimizer must not score examples")

        @property
        def degraded_examples(self) -> int:
            return self._degraded_examples

    metric = _Metric()
    monkeypatch.setattr(
        "src.optimization.gepa.metrics.get_metric_for_agent",
        lambda *a, **k: metric,
        raising=True,
    )
    # The leg imports create_gepa_optimizer from the PACKAGE, so patch it there —
    # patching optimizer_setup leaves the already-bound package alias untouched.
    monkeypatch.setattr(
        "src.optimization.gepa.create_gepa_optimizer",
        lambda **k: _FakeOptimizer(instructions, degraded, metric),
        raising=True,
    )
    monkeypatch.setattr(
        "src.optimization.dspy_lm.ensure_dspy_configured", lambda *a, **k: True, raising=True
    )
    return path


def _artifact_count(tmp_path: Path) -> int:
    from src.rag.cognitive_rag_dspy import OPTIMIZED_SYNTHESIS_AGENT_NAME

    root = tmp_path / "optimized_modules" / OPTIMIZED_SYNTHESIS_AGENT_NAME
    return len(list(root.glob("gepa_*.json"))) if root.exists() else 0


class TestJudgeDegradationBlocksPersistence:
    """Codex iter-1 F2: a run selected on noise must not be saved or deduped."""

    @pytest.mark.asyncio
    async def test_degraded_run_saves_no_artifact_and_no_fingerprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.tasks.dspy_optimization_tasks as task_module

        _arm_leg(tmp_path, monkeypatch, degraded=3)
        result = await task_module.run_rag_prompt_optimization()

        assert result["status"] == "failed"
        assert "degrad" in result["reason"].lower()
        assert "3" in result["reason"]
        assert _artifact_count(tmp_path) == 0, "a noise-selected candidate was persisted"
        assert "rag_records_fingerprint" not in task_module._load_trigger_state(), (
            "records were marked done despite the judge degrading mid-run"
        )

    @pytest.mark.asyncio
    async def test_clean_run_does_save_and_fingerprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: the degradation guard must not block a healthy run."""
        import src.tasks.dspy_optimization_tasks as task_module

        _arm_leg(tmp_path, monkeypatch, degraded=0)
        result = await task_module.run_rag_prompt_optimization()

        assert result["status"] == "completed", result
        assert _artifact_count(tmp_path) == 1
        assert "rag_records_fingerprint" in task_module._load_trigger_state()


class TestBudgetExhaustionIsNotSuccess:
    """Codex iter-1 F4, confirmed against the gepa source.

    dspy/teleprompt/gepa/gepa.py:553 seeds with the base program; gepa
    core/state.py:54 makes it candidate 0; core/result.py:77 picks argmax over
    val_aggregate_scores, and `max` resolves ties to the FIRST index — so when
    no candidate beats the seed, compile() hands back the seed unchanged
    (rebuilt at gepa.py:592). Saving that as an optimized artifact and marking
    the records done would silently skip a later budget increase.
    """

    @pytest.mark.asyncio
    async def test_seed_identical_result_is_not_saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.tasks.dspy_optimization_tasks as task_module
        from src.rag.cognitive_rag_dspy import EvidenceSynthesisSignature

        seed_instructions = EvidenceSynthesisSignature.instructions
        _arm_leg(tmp_path, monkeypatch, instructions=seed_instructions)

        result = await task_module.run_rag_prompt_optimization()

        assert result["status"] == "skipped"
        assert "budget" in result["reason"].lower()
        assert _artifact_count(tmp_path) == 0, "a baseline-identical artifact was saved"

    @pytest.mark.asyncio
    async def test_no_improvement_still_fingerprints_so_it_does_not_re_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A no-improvement result is a completed measurement, not a failure.

        Without persisting the fingerprint, identical records at an identical
        budget would re-run the FULL compile every triggered beat — a nightly
        re-spend of the whole judge budget at steady state, the exact class the
        dedup exists to prevent. Persisting is safe only because the budget is
        part of the key (F4a): raising DSPY_RAG_MAX_METRIC_CALLS changes the key
        and re-runs. Contrast the DEGRADED path, which must NOT fingerprint —
        there the judge failed transiently and a retry is wanted.
        """
        import src.tasks.dspy_optimization_tasks as task_module
        from src.rag.cognitive_rag_dspy import EvidenceSynthesisSignature

        seed_instructions = EvidenceSynthesisSignature.instructions
        _arm_leg(tmp_path, monkeypatch, instructions=seed_instructions)

        first = await task_module.run_rag_prompt_optimization()
        assert first["status"] == "skipped"
        assert "budget" in first["reason"].lower()
        assert "rag_records_fingerprint" in task_module._load_trigger_state(), (
            "a no-improvement run left the records un-fingerprinted, so the next "
            "beat re-spends the whole judge budget on identical inputs"
        )

        # The second call must short-circuit on the fingerprint, BEFORE building
        # an optimizer — otherwise the dedup saved nothing.
        def _explode(*a: Any, **k: Any) -> Any:
            raise AssertionError("second run built an optimizer despite the dedup")

        monkeypatch.setattr("src.optimization.gepa.create_gepa_optimizer", _explode, raising=True)
        second = await task_module.run_rag_prompt_optimization()
        assert second["status"] == "skipped"
        assert "already optimized" in second["reason"].lower()

    @pytest.mark.asyncio
    async def test_fingerprint_covers_the_budget_so_a_raise_reruns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same records at a HIGHER budget must not be treated as already done."""
        import src.tasks.dspy_optimization_tasks as task_module

        path = _arm_leg(tmp_path, monkeypatch)
        monkeypatch.setenv(task_module.RAG_MAX_METRIC_CALLS_ENV, "20")
        first = await task_module.run_rag_prompt_optimization()
        assert first["status"] == "completed", first

        # Unchanged budget -> deduped (no nightly re-spend).
        assert (await task_module.run_rag_prompt_optimization())["status"] == "skipped"

        # Raised budget -> the same records are worth re-running.
        monkeypatch.setenv(task_module.RAG_MAX_METRIC_CALLS_ENV, "60")
        assert (await task_module.run_rag_prompt_optimization())["status"] == "completed"
        assert path.exists()


class TestBeatPreservesLegState:
    """The trigger-state file has two writers now; neither may clobber the other.

    The RAG leg persists ``rag_records_fingerprint`` mid-beat, and the beat's
    final save used to write a FRESH two-key dict — erasing the fingerprint five
    lines after it was written. That made the dedup dead on arrival: every
    triggered beat would re-optimize identical records and re-spend the judge
    budget, which is precisely what the fingerprint exists to prevent.
    """

    @pytest.fixture
    def _stubbed_beat(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the beat's collaborators so _run reaches its final state save."""
        import src.agents.feedback_learner.optimization_runner as runner_module
        import src.agents.feedback_learner.prompt_bundles as bundles_module
        import src.agents.feedback_learner.signal_store as signal_module

        monkeypatch.chdir(tmp_path)

        async def _no_signals(*a: Any, **k: Any) -> list:
            return []

        async def _no_optimization(*a: Any, **k: Any) -> dict:
            return {"status": "stubbed"}

        monkeypatch.setattr(
            signal_module, "get_feedback_learner_training_signals", _no_signals, raising=True
        )
        monkeypatch.setattr(
            runner_module, "run_feedback_learner_optimization", _no_optimization, raising=True
        )
        monkeypatch.setattr(bundles_module, "RECIPIENT_FACTORIES", {}, raising=True)
        monkeypatch.setattr(
            bundles_module, "install_all_prompt_bundles", lambda *a, **k: {}, raising=True
        )

    @pytest.mark.asyncio
    async def test_fingerprint_survives_the_beats_final_save(
        self, _stubbed_beat: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.tasks.dspy_optimization_tasks as task_module

        async def _leg_that_persists(*a: Any, **k: Any) -> dict:
            # Exactly what the real leg does on success.
            state = task_module._load_trigger_state()
            state["rag_records_fingerprint"] = "FINGERPRINT-1486"
            task_module._save_trigger_state(state)
            return {"status": "completed"}

        monkeypatch.setattr(
            task_module, "run_rag_prompt_optimization", _leg_that_persists, raising=True
        )

        result = await task_module._run("task-1", force=True, budget="light")
        assert result["status"] == "completed"

        state = task_module._load_trigger_state()
        assert state.get("rag_records_fingerprint") == "FINGERPRINT-1486", (
            "the beat's final save clobbered the leg's fingerprint; dedup can never "
            "survive a beat, so every run re-spends the judge budget"
        )
        # The beat's own keys must still be written.
        assert "last_optimization" in state
        assert "baseline_reward" in state

    @pytest.mark.asyncio
    async def test_unrelated_pre_existing_keys_also_survive(
        self, _stubbed_beat: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the general two-writer fix, not a special case for one key."""
        import src.tasks.dspy_optimization_tasks as task_module

        task_module._save_trigger_state({"some_other_writer": "keep-me"})

        async def _noop_leg(*a: Any, **k: Any) -> dict:
            return {"status": "skipped", "reason": "stubbed"}

        monkeypatch.setattr(task_module, "run_rag_prompt_optimization", _noop_leg, raising=True)

        await task_module._run("task-2", force=True, budget="light")

        assert task_module._load_trigger_state().get("some_other_writer") == "keep-me"


class TestLegIsolation:
    """One leg must never abort the nightly beat."""

    @pytest.mark.asyncio
    async def test_a_raising_leg_does_not_abort_the_beat(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Also covers the metric's construction-time raise on a keyless box.

        get_metric_for_agent("cognitive_rag") raises when the RAGAS judge cannot
        run (by design, #1486 round 1). That must surface as a logged per-leg
        skip, never as a failed nightly run.
        """
        import src.tasks.dspy_optimization_tasks as task_module

        async def _boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("RAGAS judged path is unavailable")

        monkeypatch.setattr(task_module, "run_rag_prompt_optimization", _boom, raising=True)

        with caplog.at_level("ERROR", logger="src.tasks.dspy_optimization_tasks"):
            result = await task_module._run_rag_leg_guarded()

        assert result["status"] == "failed"
        assert "RAGAS judged path is unavailable" in result["reason"]
        assert any(r.levelname == "ERROR" for r in caplog.records)
