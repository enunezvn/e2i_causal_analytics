"""Red-first tests for #1475 target 3: synthetic-LLM warm legs.

The #1454 warm deliberately made no LLM calls, so the first real request per
worker still paid the first-LLM-call machinery. Measured (2026-08-05, real
calls on the prod box):

* classify first call 2.5-3.7s vs novel-query steady state ~1.2-1.5s — the
  prod spans' "8ms warm classify" was a DSPy cache hit on a repeated probe
  query, not a warm module.
* After ONE first call in the process, other signatures' first calls sit
  ~<=1s above their own steady state (the penalty is process-shared litellm
  init, not per-module).
* 120s idle then a novel call: 1183ms == steady state — the warmed state does
  NOT decay per-connection.

So exactly two narrow synthetic calls (classify signature + RAG query
rewriter) recover the warmable part. These tests pin:

1.  The two LLM legs exist, run LAST (they need the singletons the
    construction legs build), and appear in the warm report.
2.  ``CHATBOT_STARTUP_WARM_LLM_ENABLED``: explicit value wins; default ON
    outside pytest, OFF inside (same reasoning as the parent flag: a real
    LLM call must never fire from a test that opted into the parent warm).
3.  The legs call the RAW DSPy modules — bypassing classify_intent_dspy /
    rewrite_query_dspy keeps the training-signal buffers and
    classification_logs untouched.
4.  The warm query is recognized-shaped (kpi vocabulary), and the
    conversation_context carries a per-process cache-buster so a persistent
    DSPy disk cache can never turn the warm call into a no-op cache hit.
5.  A ``None`` module getter (feature off / build already reported failed by
    the construction leg) is a silent skip, not a failure.
6.  A garbage prediction (empty intent / empty rewritten_query) raises — a
    warm that got nothing back must report failure, not fabricate success.
7.  A hung LLM call times out (fail-open) instead of pinning the warm task.
8.  The event loop keeps serving while an LLM leg blocks its worker thread.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import src.api.chatbot_warmup as warmup
from src.api.routes import chatbot_dspy

# =============================================================================
# Helpers
# =============================================================================


@pytest.fixture(autouse=True)
def _opt_in_to_warming(monkeypatch):
    """Opt into BOTH flags — inside pytest each defaults OFF."""
    monkeypatch.setenv(warmup.WARM_ENABLED_ENV, "true")
    monkeypatch.setenv(warmup.WARM_LLM_ENABLED_ENV, "true")


def _recording_step(calls, name):
    async def step():
        calls.append(name)

    return step


def _stub_construction_legs(calls):
    return (
        patch.object(warmup, "_warm_dspy_config", _recording_step(calls, "dspy_config")),
        patch.object(warmup, "_warm_rag", _recording_step(calls, "rag")),
        patch.object(warmup, "_warm_orchestrator", _recording_step(calls, "orchestrator")),
        patch.object(warmup, "_warm_classify", _recording_step(calls, "classify")),
    )


class _StubPrediction:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


class _RecordingModule:
    """Stands in for the raw DSPy module: records call kwargs, returns a
    prediction. No LLM is ever reached from this test file."""

    def __init__(self, prediction, delay_s: float = 0.0):
        self.prediction = prediction
        self.delay_s = delay_s
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.prediction


# =============================================================================
# 1. Steps present, last, and reported
# =============================================================================


class TestLlmStepsInWarmRoutine:
    @pytest.mark.asyncio
    async def test_llm_legs_run_last_and_report(self, monkeypatch):
        calls = []
        stubs = _stub_construction_legs(calls)
        with stubs[0], stubs[1], stubs[2], stubs[3]:
            monkeypatch.setattr(
                warmup, "_warm_classify_llm", _recording_step(calls, "classify_llm")
            )
            monkeypatch.setattr(
                warmup, "_warm_rag_rewrite_llm", _recording_step(calls, "rag_rewrite_llm")
            )
            result = await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        assert calls == [
            "dspy_config",
            "rag",
            "orchestrator",
            "classify",
            "classify_llm",
            "rag_rewrite_llm",
        ], "LLM legs must run AFTER every construction leg (they consume the singletons)"
        assert "classify_llm" in result["steps_ms"]
        assert "rag_rewrite_llm" in result["steps_ms"]
        assert result["failed"] == {}

    @pytest.mark.asyncio
    async def test_llm_leg_failure_is_fail_open_and_other_llm_leg_still_runs(self, monkeypatch):
        calls = []

        async def boom():
            calls.append("classify_llm")
            raise RuntimeError("llm boom")

        stubs = _stub_construction_legs(calls)
        with stubs[0], stubs[1], stubs[2], stubs[3]:
            monkeypatch.setattr(warmup, "_warm_classify_llm", boom)
            monkeypatch.setattr(
                warmup, "_warm_rag_rewrite_llm", _recording_step(calls, "rag_rewrite_llm")
            )
            result = await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        assert "classify_llm" in result["failed"]
        assert "llm boom" in result["failed"]["classify_llm"]
        assert "rag_rewrite_llm" in calls, "one failed LLM leg must not stop the other"


# =============================================================================
# 2. Feature flag
# =============================================================================


class TestLlmFlag:
    @pytest.mark.asyncio
    async def test_flag_off_omits_llm_steps_entirely(self, monkeypatch):
        monkeypatch.setenv(warmup.WARM_LLM_ENABLED_ENV, "false")
        calls = []
        stubs = _stub_construction_legs(calls)
        with stubs[0], stubs[1], stubs[2], stubs[3]:
            result = await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        assert "classify_llm" not in result["steps_ms"]
        assert "rag_rewrite_llm" not in result["steps_ms"]
        assert calls == ["dspy_config", "rag", "orchestrator", "classify"]

    def test_default_on_outside_pytest(self, monkeypatch):
        monkeypatch.delenv(warmup.WARM_LLM_ENABLED_ENV, raising=False)
        monkeypatch.delitem(sys.modules, "pytest")
        assert warmup.chatbot_warm_llm_enabled() is True

    def test_default_off_inside_pytest(self, monkeypatch):
        """A test that opts into the PARENT warm (as the #1454 test file does)
        must not thereby fire real LLM calls."""
        monkeypatch.delenv(warmup.WARM_LLM_ENABLED_ENV, raising=False)
        assert warmup.chatbot_warm_llm_enabled() is False

    def test_explicit_flag_wins_inside_pytest(self, monkeypatch):
        monkeypatch.setenv(warmup.WARM_LLM_ENABLED_ENV, "true")
        assert warmup.chatbot_warm_llm_enabled() is True


# =============================================================================
# 3. Raw-module calls: shape, cache-buster, no signal pollution
# =============================================================================


class TestClassifyLlmLeg:
    @pytest.mark.asyncio
    async def test_calls_raw_classifier_with_recognized_query_and_cache_buster(self, monkeypatch):
        stub = _RecordingModule(_StubPrediction(intent="kpi_query", confidence=0.9))
        monkeypatch.setattr(chatbot_dspy, "_get_dspy_classifier", lambda: stub)
        signals_before = len(chatbot_dspy.get_signal_collector())

        await warmup._warm_classify_llm()

        assert len(stub.calls) == 1
        kwargs = stub.calls[0]
        assert kwargs["query"] == warmup.WARM_LLM_QUERY
        assert warmup._WARM_LLM_CACHE_BUSTER in kwargs["conversation_context"], (
            "the context must carry the per-process cache-buster, else a persistent "
            "DSPy disk cache turns the warm call into a no-op cache hit"
        )
        assert len(chatbot_dspy.get_signal_collector()) == signals_before, (
            "raw-module warm calls must not enter the intent training-signal buffer"
        )

    def test_warm_query_is_recognized_shaped(self):
        """The warm must exercise the real kpi_query path, not the
        CLARIFICATION fallback (#1478 measured warm probes landing there)."""
        intent, _confidence, _reasoning = chatbot_dspy.classify_intent_hardcoded(
            warmup.WARM_LLM_QUERY
        )
        assert intent == chatbot_dspy.IntentType.KPI_QUERY

    @pytest.mark.asyncio
    async def test_none_classifier_is_skip_not_failure(self, monkeypatch):
        monkeypatch.setattr(chatbot_dspy, "_get_dspy_classifier", lambda: None)
        await warmup._warm_classify_llm()  # must not raise

    @pytest.mark.asyncio
    async def test_empty_intent_raises(self, monkeypatch):
        stub = _RecordingModule(_StubPrediction(intent=""))
        monkeypatch.setattr(chatbot_dspy, "_get_dspy_classifier", lambda: stub)
        with pytest.raises(RuntimeError):
            await warmup._warm_classify_llm()


class TestRagRewriteLlmLeg:
    @pytest.mark.asyncio
    async def test_calls_raw_rewriter_with_domain_vocabulary_and_cache_buster(self, monkeypatch):
        stub = _RecordingModule(
            _StubPrediction(rewritten_query="Kisqali TRx trend quarterly", search_keywords="trx")
        )
        monkeypatch.setattr(chatbot_dspy, "_get_dspy_query_rewriter", lambda: stub)
        rag_signals_before = len(chatbot_dspy.get_rag_signal_collector())

        await warmup._warm_rag_rewrite_llm()

        assert len(stub.calls) == 1
        kwargs = stub.calls[0]
        assert kwargs["original_query"] == warmup.WARM_LLM_QUERY
        assert kwargs["domain_vocabulary"] is chatbot_dspy.E2I_DOMAIN_VOCABULARY
        assert warmup._WARM_LLM_CACHE_BUSTER in kwargs["conversation_context"]
        assert len(chatbot_dspy.get_rag_signal_collector()) == rag_signals_before

    @pytest.mark.asyncio
    async def test_none_rewriter_is_skip_not_failure(self, monkeypatch):
        monkeypatch.setattr(chatbot_dspy, "_get_dspy_query_rewriter", lambda: None)
        await warmup._warm_rag_rewrite_llm()  # must not raise

    @pytest.mark.asyncio
    async def test_empty_rewritten_query_raises(self, monkeypatch):
        stub = _RecordingModule(_StubPrediction(rewritten_query="   "))
        monkeypatch.setattr(chatbot_dspy, "_get_dspy_query_rewriter", lambda: stub)
        with pytest.raises(RuntimeError):
            await warmup._warm_rag_rewrite_llm()


# =============================================================================
# 4. Timeout + loop liveness
# =============================================================================


class TestLlmLegRuntimeDiscipline:
    @pytest.mark.asyncio
    async def test_hung_llm_call_times_out_fail_open(self, monkeypatch):
        stub = _RecordingModule(_StubPrediction(intent="kpi_query"), delay_s=0.5)
        monkeypatch.setattr(chatbot_dspy, "_get_dspy_classifier", lambda: stub)
        monkeypatch.setattr(warmup, "WARM_LLM_TIMEOUT_SECONDS", 0.05)

        with pytest.raises(asyncio.TimeoutError):
            await warmup._warm_classify_llm()

    @pytest.mark.asyncio
    async def test_loop_keeps_serving_while_llm_leg_blocks(self, monkeypatch):
        """The raw module call is sync — it must run off-loop (to_thread)."""
        stub = _RecordingModule(_StubPrediction(intent="kpi_query"), delay_s=0.3)
        monkeypatch.setattr(chatbot_dspy, "_get_dspy_classifier", lambda: stub)

        beats = 0
        stop = asyncio.Event()

        async def heartbeat():
            nonlocal beats
            while not stop.is_set():
                beats += 1
                await asyncio.sleep(0.02)

        hb = asyncio.create_task(heartbeat())
        await warmup._warm_classify_llm()
        stop.set()
        await hb

        assert beats >= 5, f"loop starved during the LLM leg (only {beats} heartbeats)"


# =============================================================================
# 5. Cache-buster is per-process
# =============================================================================


class TestCacheBuster:
    def test_cache_buster_differs_across_processes(self):
        """The property that defeats a persistent disk cache across boots."""
        cmd = [
            sys.executable,
            "-c",
            "import src.api.chatbot_warmup as w; print(w._WARM_LLM_CACHE_BUSTER)",
        ]
        root = Path(warmup.__file__).resolve().parents[2]
        out1 = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=root
        ).stdout.strip()
        out2 = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=root
        ).stdout.strip()
        assert out1 and out2
        assert out1 != out2

    def test_cache_buster_nonempty_hex(self):
        assert len(warmup._WARM_LLM_CACHE_BUSTER) >= 8
        int(warmup._WARM_LLM_CACHE_BUSTER, 16)  # raises if not hex
