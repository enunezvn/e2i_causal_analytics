"""Red-first tests for #1454: chatbot cold-start warming.

PR #1471's span instrumentation measured the cold chatbot request at 48.1s vs a
16.7-18.6s warm floor. The ~30s delta is process-scoped lazy init paid by
whichever user happens to send the first request to a fresh worker:
``retrieve_rag`` 17.0s vs ~2.1s warm, ``orchestrator`` 25.1s vs 14.4-16.5s,
``classify_intent`` 5.5s vs ~8ms.

These tests pin the warm routine that moves that init to worker startup:

1. All four legs run, DSPy config FIRST, then the rest in measured-value order.
2. Each leg is individually fail-open — one failing leg neither stops the others
   nor propagates out of the warm task.
3. ``CHATBOT_STARTUP_WARM_ENABLED=false`` skips every leg.
4. The completion log line carries the pid (default JSON logging has no process
   id, so warm completion is otherwise uncorrelatable to a worker) plus per-step
   and total wall time.
5. Thread affinity: the DSPy config leg runs ON the event loop thread (dspy
   3.1.0 permanently binds an owner thread at first ``configure``); every heavy
   leg runs OFF it.
6. The loop keeps serving during a slow warm (the #1406 heartbeat method) —
   "background task" is not by itself a non-blocking guarantee.
7. Shutdown cancels the warm task; no further steps start after cancellation.
8. DSPy 3.1.0 ownership semantics (real package, no mocks, in a subprocess so
   the process-global owner state cannot poison sibling tests) — these are the
   WHY behind constraint 5, and fail loudly if a dspy upgrade changes them.
"""

import asyncio
import json
import logging
import subprocess
import sys
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.api.chatbot_warmup as warmup
from src.api import main

# =============================================================================
# Helpers
# =============================================================================


@pytest.fixture(autouse=True)
def _opt_in_to_warming(monkeypatch):
    """pytest sets ``E2I_TESTING_MODE=true``, which defaults the warm OFF so no
    test can accidentally schedule a real one. This module exercises the warm
    itself, so it opts in explicitly; the flag tests delete or override it."""
    monkeypatch.setenv(warmup.WARM_ENABLED_ENV, "true")


def _recording_step(calls, name, delay=0.0):
    """An async warm leg that records its call order and thread."""

    async def step():
        calls.append((name, threading.get_ident()))
        if delay:
            await asyncio.sleep(delay)

    return step


def _failing_step(calls, name, exc):
    async def step():
        calls.append((name, threading.get_ident()))
        raise exc

    return step


@contextmanager
def _stub_all_legs(calls):
    """Replace every warm leg with a recorder — keeps these tests off the heavy
    singletons; the legs' real wiring is pinned by the thread-affinity tests and
    proven end-to-end by the post-deploy live probe."""
    with (
        patch.object(warmup, "_warm_dspy_config", _recording_step(calls, "dspy_config")),
        patch.object(warmup, "_warm_rag", _recording_step(calls, "rag")),
        patch.object(warmup, "_warm_orchestrator", _recording_step(calls, "orchestrator")),
        patch.object(warmup, "_warm_classify", _recording_step(calls, "classify")),
    ):
        yield


# =============================================================================
# 1-2. Step order + per-step fail-open
# =============================================================================


class TestWarmStepOrderAndFailOpen:
    @pytest.mark.asyncio
    async def test_runs_four_legs_with_dspy_config_first(self):
        calls = []
        with _stub_all_legs(calls):
            result = await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        assert [name for name, _ in calls] == [
            "dspy_config",
            "rag",
            "orchestrator",
            "classify",
        ], "DSPy config must run first (owner-thread binding), then measured-value order"
        assert result["skipped"] is False
        assert set(result["steps_ms"]) == {"dspy_config", "rag", "orchestrator", "classify"}
        assert result["failed"] == {}

    @pytest.mark.asyncio
    async def test_failing_leg_does_not_stop_later_legs_or_raise(self, caplog):
        caplog.set_level(logging.WARNING)
        calls = []
        with (
            patch.object(warmup, "_warm_dspy_config", _recording_step(calls, "dspy_config")),
            patch.object(warmup, "_warm_rag", _failing_step(calls, "rag", RuntimeError("boom"))),
            patch.object(warmup, "_warm_orchestrator", _recording_step(calls, "orchestrator")),
            patch.object(warmup, "_warm_classify", _recording_step(calls, "classify")),
        ):
            result = await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        assert [name for name, _ in calls] == [
            "dspy_config",
            "rag",
            "orchestrator",
            "classify",
        ], "a failed leg must not abort the remaining legs"
        assert "rag" in result["failed"]
        assert "boom" in result["failed"]["rag"]
        assert "rag" in result["steps_ms"], "a failed leg still reports its wall time"
        warn = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("rag" in r.getMessage() for r in warn), "failure must be an honest WARNING"

    @pytest.mark.asyncio
    async def test_every_leg_failing_still_completes(self):
        calls = []
        with (
            patch.object(
                warmup, "_warm_dspy_config", _failing_step(calls, "dspy_config", ValueError("a"))
            ),
            patch.object(warmup, "_warm_rag", _failing_step(calls, "rag", ValueError("b"))),
            patch.object(
                warmup, "_warm_orchestrator", _failing_step(calls, "orchestrator", ValueError("c"))
            ),
            patch.object(
                warmup, "_warm_classify", _failing_step(calls, "classify", ValueError("d"))
            ),
        ):
            result = await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        assert len(result["failed"]) == 4
        assert result["skipped"] is False


# =============================================================================
# 3. Feature flag
# =============================================================================


class TestWarmFlag:
    def test_enabled_by_default_in_production(self, monkeypatch):
        monkeypatch.delenv(warmup.WARM_ENABLED_ENV, raising=False)
        monkeypatch.delenv("E2I_TESTING_MODE", raising=False)
        assert warmup.chatbot_warm_enabled() is True

    def test_disabled_by_default_under_the_pytest_harness(self, monkeypatch):
        """A test driving the real lifespan must not schedule a real warm: the
        0-5s jitter can draw ~0, and the executor thread outlives cancellation."""
        monkeypatch.delenv(warmup.WARM_ENABLED_ENV, raising=False)
        monkeypatch.setenv("E2I_TESTING_MODE", "true")
        assert warmup.chatbot_warm_enabled() is False

    def test_explicit_flag_wins_over_the_harness_default(self, monkeypatch):
        monkeypatch.setenv("E2I_TESTING_MODE", "true")
        monkeypatch.setenv(warmup.WARM_ENABLED_ENV, "true")
        assert warmup.chatbot_warm_enabled() is True

    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no"])
    def test_disabled_values(self, monkeypatch, value):
        monkeypatch.setenv(warmup.WARM_ENABLED_ENV, value)
        assert warmup.chatbot_warm_enabled() is False

    @pytest.mark.asyncio
    async def test_flag_off_skips_every_leg(self, monkeypatch):
        monkeypatch.setenv(warmup.WARM_ENABLED_ENV, "false")
        calls = []
        with _stub_all_legs(calls):
            result = await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        assert calls == []
        assert result["skipped"] is True
        assert result["steps_ms"] == {}


# =============================================================================
# 4. Completion log line
# =============================================================================


class TestWarmLogLine:
    @pytest.mark.asyncio
    async def test_completion_line_carries_pid_marker_and_step_times(self, caplog):
        caplog.set_level(logging.INFO)
        calls = []
        with _stub_all_legs(calls):
            result = await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        lines = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.INFO and warmup.WARM_COMPLETE_MARKER in r.getMessage()
        ]
        assert len(lines) == 1, "exactly one completion marker line per warm"
        msg = lines[0]
        assert f"pid={result['pid']}" in msg, "pid is what makes completion worker-attributable"
        assert "total_ms=" in msg
        for step in ("dspy_config", "rag", "orchestrator", "classify"):
            assert step in msg, f"per-step time for {step} missing: {msg}"

    @pytest.mark.asyncio
    async def test_skip_line_is_logged_when_flag_off(self, monkeypatch, caplog):
        monkeypatch.setenv(warmup.WARM_ENABLED_ENV, "false")
        caplog.set_level(logging.INFO)
        with _stub_all_legs([]):
            await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        assert any(
            warmup.WARM_LOG_PREFIX in r.getMessage() and "skipped" in r.getMessage()
            for r in caplog.records
        )


# =============================================================================
# 5. Thread affinity
# =============================================================================


class TestThreadAffinity:
    @pytest.mark.asyncio
    async def test_dspy_config_leg_runs_on_the_event_loop_thread(self):
        """dspy 3.1.0 binds the owner thread on the FIRST configure; running the
        config leg on the loop thread arms every ``settings.lm is not None``
        guard before any worker thread can reach ``dspy.configure``."""
        import dspy

        seen = {}

        def fake_ensure():
            seen["thread"] = threading.get_ident()

        with (
            patch("src.api.routes.chatbot_dspy._ensure_dspy_configured", fake_ensure),
            # satisfy the postcondition; this test is about WHERE it ran
            patch.object(dspy, "settings", types.SimpleNamespace(lm=object())),
        ):
            await warmup._warm_dspy_config()

        assert seen["thread"] == threading.get_ident(), "config leg must NOT be to_thread'd"

    @pytest.mark.asyncio
    async def test_orchestrator_leg_runs_off_the_loop_thread(self):
        seen = {}

        def fake_get_orchestrator():
            seen["thread"] = threading.get_ident()
            return object()

        with patch("src.api.routes.cognitive.get_orchestrator", fake_get_orchestrator):
            await warmup._warm_orchestrator()

        assert seen["thread"] != threading.get_ident(), "heavy sync build must be to_thread'd"

    @pytest.mark.asyncio
    async def test_classify_leg_runs_off_the_loop_thread(self):
        seen = {}

        def fake_classifier():
            seen["thread"] = threading.get_ident()
            return object()

        with patch("src.api.routes.chatbot_dspy._get_dspy_classifier", fake_classifier):
            await warmup._warm_classify()

        assert seen["thread"] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_rag_leg_exercises_both_retrieval_legs(self):
        """``hybrid_search`` has a dense AND a sparse leg against different
        Supabase RPCs; warming only the dense one leaves the sparse path cold."""
        calls = []

        class _StubConnector:
            async def vector_search_by_text(self, query, k=10, **kwargs):
                calls.append(("vector", query, k, threading.get_ident()))
                return []

            async def fulltext_search(self, query, k=10, **kwargs):
                calls.append(("fulltext", query, k, threading.get_ident()))
                return []

        await warmup._rag_warm(_StubConnector())

        assert [c[0] for c in calls] == ["vector", "fulltext"]
        assert all(c[2] == 1 for c in calls), "warm calls must be tiny (k=1)"

    @pytest.mark.asyncio
    async def test_rag_leg_runs_off_the_loop_thread(self):
        """The retrieval calls are async-wrapping-SYNC (sync openai client for
        the embedding, sync supabase client for the RPC) — awaiting them on the
        loop would block it for the full external-HTTP duration."""
        seen = {}

        def fake_rag_sync():
            seen["rag"] = threading.get_ident()

        def fake_dspy_modules():
            seen["dspy_modules"] = threading.get_ident()

        with (
            patch.object(warmup, "_rag_warm_sync", fake_rag_sync),
            patch.object(warmup, "_warm_rag_dspy_modules", fake_dspy_modules),
        ):
            await warmup._warm_rag()

        loop_thread = threading.get_ident()
        assert seen["rag"] != loop_thread
        assert seen["dspy_modules"] != loop_thread

    @pytest.mark.asyncio
    async def test_rag_dspy_modules_leg_builds_rewriter_and_hop_decider(self):
        built = []

        def _rewriter():
            built.append("rewriter")
            return object()

        def _hop():
            built.append("hop")
            return object()

        with (
            patch("src.api.routes.chatbot_dspy._get_dspy_query_rewriter", _rewriter),
            patch("src.api.routes.chatbot_dspy._get_dspy_hop_decider", _hop),
        ):
            warmup._warm_rag_dspy_modules()

        assert built == ["rewriter", "hop"]


class TestSwallowedFailureHonesty:
    """Every warm leg calls a request-path helper that FAILS SOFT by design:
    ``_ensure_dspy_configured`` catches and warns (chatbot_dspy.py:75), and the
    three ``_get_dspy_*`` getters return ``None`` after a swallowed construction
    error (chatbot_dspy.py:517, 1271, 1577). That is correct for a request (fall
    back to the non-DSPy path) but it means the warm sees no exception — so
    without a postcondition the completion line would claim ``failed=[]`` for a
    leg that warmed nothing. The postcondition must distinguish "failed" from
    "feature is switched off", which is also a legitimate ``None``."""

    @pytest.mark.asyncio
    async def test_dspy_config_leg_fails_when_no_lm_is_configured(self):
        import dspy

        with (
            patch("src.api.routes.chatbot_dspy._ensure_dspy_configured", lambda: None),
            patch.object(dspy, "settings", types.SimpleNamespace(lm=None)),
        ):
            with pytest.raises(RuntimeError, match="DSPy"):
                await warmup._warm_dspy_config()

    @pytest.mark.asyncio
    async def test_dspy_config_leg_is_clean_when_dspy_is_unavailable(self):
        with (
            patch("src.api.routes.chatbot_dspy.DSPY_AVAILABLE", False),
            patch("src.api.routes.chatbot_dspy._ensure_dspy_configured", lambda: None),
        ):
            await warmup._warm_dspy_config()  # must not raise

    @pytest.mark.asyncio
    async def test_classify_leg_fails_when_the_enabled_classifier_is_none(self):
        with (
            patch("src.api.routes.chatbot_dspy.DSPY_AVAILABLE", True),
            patch("src.api.routes.chatbot_dspy.CHATBOT_DSPY_INTENT_ENABLED", True),
            patch("src.api.routes.chatbot_dspy._get_dspy_classifier", lambda: None),
        ):
            with pytest.raises(RuntimeError, match="classifier"):
                await warmup._warm_classify()

    @pytest.mark.asyncio
    async def test_classify_leg_is_clean_when_the_feature_is_off(self):
        """Flag off -> the getter returns None BY DESIGN. Reporting that as a
        warm failure would be a fabricated failure."""
        with (
            patch("src.api.routes.chatbot_dspy.DSPY_AVAILABLE", True),
            patch("src.api.routes.chatbot_dspy.CHATBOT_DSPY_INTENT_ENABLED", False),
            patch("src.api.routes.chatbot_dspy._get_dspy_classifier", lambda: None),
        ):
            await warmup._warm_classify()  # must not raise

    def test_rag_dspy_modules_fail_when_an_enabled_singleton_is_none(self):
        with (
            patch("src.api.routes.chatbot_dspy.DSPY_AVAILABLE", True),
            patch("src.api.routes.chatbot_dspy.CHATBOT_COGNITIVE_RAG_ENABLED", True),
            patch("src.api.routes.chatbot_dspy._get_dspy_query_rewriter", lambda: object()),
            patch("src.api.routes.chatbot_dspy._get_dspy_hop_decider", lambda: None),
        ):
            with pytest.raises(RuntimeError, match="hop decider"):
                warmup._warm_rag_dspy_modules()

    def test_rag_dspy_modules_are_clean_when_cognitive_rag_is_off(self):
        with (
            patch("src.api.routes.chatbot_dspy.DSPY_AVAILABLE", True),
            patch("src.api.routes.chatbot_dspy.CHATBOT_COGNITIVE_RAG_ENABLED", False),
            patch("src.api.routes.chatbot_dspy._get_dspy_query_rewriter", lambda: None),
            patch("src.api.routes.chatbot_dspy._get_dspy_hop_decider", lambda: None),
        ):
            warmup._warm_rag_dspy_modules()  # must not raise


class TestRagWarmFailureHonesty:
    """``_rag_warm_sync`` is called INSIDE a ``to_thread`` worker (no running
    loop), which is why these tests are sync — ``asyncio.run`` on the test's own
    loop would be a harness artefact, not the production shape."""

    def test_rag_warm_surfaces_a_swallowed_rpc_failure(self):
        """``vector_search``/``fulltext_search`` catch RPC exceptions, log ERROR
        and return ``[]`` (src/rag/memory_connector.py:174,245). An empty list is
        a legitimate result for the "warmup" query, so the connector's own ERROR
        log is the only honest failure signal — without it the warm would report
        success for a leg that never reached Supabase."""

        class _FailingConnector:
            async def vector_search_by_text(self, query, k=10, **kwargs):
                logging.getLogger("src.rag.memory_connector").error(
                    "Vector search failed: connection refused"
                )
                return []

            async def fulltext_search(self, query, k=10, **kwargs):
                return []

        with patch("src.rag.memory_connector.get_memory_connector", lambda: _FailingConnector()):
            with pytest.raises(RuntimeError, match="connection refused"):
                warmup._rag_warm_sync()

    @pytest.mark.asyncio
    async def test_rag_warm_failure_is_reported_by_the_warm_routine(self):
        class _FailingConnector:
            async def vector_search_by_text(self, query, k=10, **kwargs):
                return []

            async def fulltext_search(self, query, k=10, **kwargs):
                logging.getLogger("src.rag.memory_connector").error(
                    "Fulltext search failed: relation does not exist"
                )
                return []

        calls = []
        with (
            patch("src.rag.memory_connector.get_memory_connector", lambda: _FailingConnector()),
            patch.object(warmup, "_warm_dspy_config", _recording_step(calls, "dspy_config")),
            patch.object(warmup, "_warm_rag_dspy_modules", lambda: None),
            patch.object(warmup, "_warm_orchestrator", _recording_step(calls, "orchestrator")),
            patch.object(warmup, "_warm_classify", _recording_step(calls, "classify")),
        ):
            result = await warmup.warm_chatbot_stack(jitter_seconds=0.0)

        assert "rag" in result["failed"], "a degraded RAG warm must not report success"
        assert "relation does not exist" in result["failed"]["rag"]

    def test_quiet_connector_is_a_clean_rag_warm(self):
        class _QuietConnector:
            async def vector_search_by_text(self, query, k=10, **kwargs):
                return []  # no rows for "warmup" is NORMAL, not a failure

            async def fulltext_search(self, query, k=10, **kwargs):
                return []

        with patch("src.rag.memory_connector.get_memory_connector", lambda: _QuietConnector()):
            warmup._rag_warm_sync()  # must not raise

    def test_another_threads_error_does_not_fail_the_warm(self):
        """The capture is attached to the process-global connector logger, so a
        real request racing startup would otherwise emit "Vector search failed"
        into it and fabricate a warm failure. Only records from the warm's OWN
        thread count."""

        class _QuietConnectorWithNoisyNeighbour:
            async def vector_search_by_text(self, query, k=10, **kwargs):
                noisy = threading.Thread(
                    target=lambda: logging.getLogger("src.rag.memory_connector").error(
                        "Vector search failed: a CONCURRENT REQUEST's error"
                    )
                )
                noisy.start()
                noisy.join()
                return []

            async def fulltext_search(self, query, k=10, **kwargs):
                return []

        with patch(
            "src.rag.memory_connector.get_memory_connector",
            lambda: _QuietConnectorWithNoisyNeighbour(),
        ):
            warmup._rag_warm_sync()  # must not raise


# =============================================================================
# 6. The loop keeps serving during a slow warm (#1406 heartbeat method)
# =============================================================================


class TestLoopStaysResponsive:
    @pytest.mark.asyncio
    async def test_heartbeat_keeps_ticking_during_a_slow_warm(self):
        ticks = 0
        stop = asyncio.Event()

        async def heartbeat():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.005)
                ticks += 1

        def slow_sync_build():
            time.sleep(0.5)  # a blocking build, as the real ones are
            return object()

        beat = asyncio.create_task(heartbeat())
        with patch("src.api.routes.cognitive.get_orchestrator", slow_sync_build):
            await warmup._warm_orchestrator()
        stop.set()
        await beat

        # A loop blocked for 0.5s would tick ~0 times; a free loop ticks ~100.
        assert ticks >= 10, f"event loop was blocked during warm (only {ticks} ticks)"


# =============================================================================
# 7. Cancellation
# =============================================================================


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancelling_the_warm_task_stops_further_steps(self):
        calls = []
        started = asyncio.Event()

        async def blocking_rag():
            calls.append(("rag", threading.get_ident()))
            started.set()
            await asyncio.sleep(60)

        with (
            patch.object(warmup, "_warm_dspy_config", _recording_step(calls, "dspy_config")),
            patch.object(warmup, "_warm_rag", blocking_rag),
            patch.object(warmup, "_warm_orchestrator", _recording_step(calls, "orchestrator")),
            patch.object(warmup, "_warm_classify", _recording_step(calls, "classify")),
        ):
            task = asyncio.create_task(warmup.warm_chatbot_stack(jitter_seconds=0.0))
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        names = [name for name, _ in calls]
        assert names == ["dspy_config", "rag"], "no further step may start after cancellation"

    @pytest.mark.asyncio
    async def test_cancellation_during_jitter_starts_no_step(self):
        calls = []
        with _stub_all_legs(calls):
            task = asyncio.create_task(warmup.warm_chatbot_stack(jitter_seconds=30.0))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert calls == []


# =============================================================================
# 7b. Lifespan wiring (real src.api.main.lifespan)
# =============================================================================


def _fake_app():
    return types.SimpleNamespace(state=types.SimpleNamespace())


@contextmanager
def _hermetic_lifespan_io():
    """Patch out every networky lifespan init/cleanup so the lifespan runs
    hermetically, leaving ONLY the chatbot-warm wiring live. Mirrors
    tests/unit/test_api/test_audit_chain_lifespan_wiring.py."""
    feast = MagicMock()
    feast.initialize = AsyncMock()
    feast._initialized = False

    with (
        patch("src.api.main.get_bentoml_client", new=AsyncMock()),
        patch("src.api.main.configure_bentoml_endpoints", new=MagicMock()),
        patch("src.api.main.init_redis", new=AsyncMock()),
        patch("src.api.main.init_falkordb", new=AsyncMock()),
        patch("src.api.main.init_supabase", return_value=None),
        patch("src.api.main.get_mlflow_connector", new=MagicMock()),
        patch("src.api.main.get_opik_connector", new=MagicMock()),
        patch("src.api.main.FeastClient", new=MagicMock(return_value=feast)),
        patch("src.api.main.close_bentoml_client", new=AsyncMock()),
        patch("src.api.main.close_redis", new=AsyncMock()),
        patch("src.api.main.close_falkordb", new=AsyncMock()),
        patch("src.api.main.close_supabase", new=MagicMock()),
        patch("src.api.main.shutdown_opentelemetry", new=MagicMock()),
        patch(
            "src.memory.sentinels.config_loader.load_sentinels_from_yaml",
            new=AsyncMock(return_value=0),
        ),
        patch.dict("os.environ", {"HEALTH_HISTORY_HEARTBEAT": "false"}),
    ):
        yield


class TestLifespanWiring:
    @pytest.mark.asyncio
    async def test_lifespan_starts_the_warm_task_and_cancels_it_on_shutdown(self, monkeypatch):
        monkeypatch.setenv(warmup.WARM_ENABLED_ENV, "true")
        state = {"started": asyncio.Event(), "cancelled": False}

        async def fake_warm(*args, **kwargs):
            state["started"].set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
            return {}

        with _hermetic_lifespan_io(), patch.object(main, "warm_chatbot_stack", fake_warm):
            async with main.lifespan(_fake_app()):
                await asyncio.wait_for(state["started"].wait(), timeout=5)

        assert state["cancelled"] is True, "shutdown must cancel the in-flight warm task"

    @pytest.mark.asyncio
    async def test_lifespan_skips_the_warm_task_when_flag_off(self, monkeypatch):
        monkeypatch.setenv(warmup.WARM_ENABLED_ENV, "false")
        called = []

        async def fake_warm(*args, **kwargs):
            called.append(True)
            return {}

        with _hermetic_lifespan_io(), patch.object(main, "warm_chatbot_stack", fake_warm):
            async with main.lifespan(_fake_app()):
                await asyncio.sleep(0.05)

        assert called == [], "flag off must not even create the task"

    @pytest.mark.asyncio
    async def test_lifespan_survives_a_warm_task_that_raises(self, monkeypatch, caplog):
        """A warm crash must be logged, never propagated into startup/shutdown."""
        monkeypatch.setenv(warmup.WARM_ENABLED_ENV, "true")
        caplog.set_level(logging.WARNING)

        async def exploding_warm(*args, **kwargs):
            raise RuntimeError("warm exploded")

        with _hermetic_lifespan_io(), patch.object(main, "warm_chatbot_stack", exploding_warm):
            async with main.lifespan(_fake_app()):
                await asyncio.sleep(0.05)

        assert any("warm exploded" in r.getMessage() for r in caplog.records)


# =============================================================================
# 8. DSPy 3.1.0 ownership pins (REAL dspy, subprocess-isolated)
# =============================================================================

_PIN_SCRIPT = r"""
import json, threading

import dspy

from src.api.routes import chatbot_dspy

out = {}

# (a) global-read pin: an LM configured on this thread is readable elsewhere.
chatbot_dspy._ensure_dspy_configured()
out["configured_lm_is_set"] = dspy.settings.lm is not None

seen = {}
def _read():
    seen["lm_visible"] = dspy.settings.lm is not None
t = threading.Thread(target=_read)
t.start(); t.join()
out["lm_visible_from_other_thread"] = seen.get("lm_visible")

# (b) guard pin: the guarded helper is a no-op from another thread.
guard = {}
def _guarded():
    try:
        chatbot_dspy._ensure_dspy_configured()
        guard["error"] = None
    except Exception as e:  # noqa: BLE001
        guard["error"] = f"{type(e).__name__}: {e}"
t = threading.Thread(target=_guarded)
t.start(); t.join()
out["guarded_reconfigure_error"] = guard["error"]

# (c) ownership pin: a RAW dspy.configure from another thread must raise.
raw = {}
def _raw():
    try:
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
        raw["error"] = None
    except Exception as e:  # noqa: BLE001
        raw["error"] = type(e).__name__
t = threading.Thread(target=_raw)
t.start(); t.join()
out["raw_configure_error"] = raw["error"]

print("PIN_RESULT " + json.dumps(out))
"""


@pytest.fixture(scope="module")
def dspy_pin_results():
    """Run the dspy semantics probes in a SUBPROCESS.

    ``dspy.configure`` permanently binds process-global owner state
    (``config_owner_thread_id`` / ``config_owner_async_task`` /
    ``main_thread_config``); a subprocess is the only isolation that cannot
    poison sibling tests in the same worker.
    """
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-c", _PIN_SCRIPT],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("PIN_RESULT ")]
    assert marker, f"dspy pin subprocess failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return json.loads(marker[-1][len("PIN_RESULT ") :])


@pytest.mark.timeout(180)  # whichever test runs first pays the subprocess (~12s warm) in setup
class TestDspyOwnershipPins:
    def test_configured_lm_is_readable_from_another_thread(self, dspy_pin_results):
        """dspy 3.1.0 reads fall back to the GLOBAL main_thread_config, so an LM
        configured on the loop thread is visible to every executor thread."""
        assert dspy_pin_results["configured_lm_is_set"] is True
        assert dspy_pin_results["lm_visible_from_other_thread"] is True

    def test_guarded_reconfigure_from_another_thread_is_a_noop(self, dspy_pin_results):
        """``_ensure_dspy_configured`` early-returns once an LM is set — this
        guard is what keeps worker threads away from ``dspy.configure``."""
        assert dspy_pin_results["guarded_reconfigure_error"] is None

    def test_raw_configure_from_another_thread_raises(self, dspy_pin_results):
        """The armed trap the config-leg ordering exists to avoid. If a dspy
        upgrade drops this, this test fails loudly and the ordering rationale
        must be re-derived."""
        assert dspy_pin_results["raw_configure_error"] == "RuntimeError"
