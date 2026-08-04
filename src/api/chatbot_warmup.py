"""#1454: chatbot cold-start warming — a fail-open startup warm task.

PR #1471's span instrumentation measured the first chatbot request into a fresh
worker at 48.1s against a 16.7-18.6s warm floor. The ~30s delta is process-scoped
lazy init that one unlucky user pays for everybody:

    retrieve_rag       17.0s cold / ~2.1s warm   (embedding service first init,
                                                  Supabase client first touch,
                                                  DSPy rewriter + hop-decider builds)
    orchestrator       25.1s cold / 14.4-16.5s   (agent registry build ~3.4-4.0s
                                                  + OrchestratorAgent construction)
    classify_intent     5.5s cold / ~8ms         (ChatbotIntentClassifier build)

This module pre-builds those singletons in a background task at worker startup.

Honest limits — this is NOT a "warm = fast" guarantee:

* It removes construction, import and first-connection cost only. Each module's
  first REAL LLM call (classify, rewrite, evidence scoring) still lands in the
  first user request by design: no synthetic token spend at startup.
* It is NOT a readiness gate. A request that races the warm task takes exactly
  today's lazy path — correct, just unimproved.
* Warm failure is a WARNING and the normal lazy path, never a fabricated
  "warmed" claim.

Thread discipline (the two constraints this module exists to honour):

* The DSPy config leg runs FIRST and ON the event loop thread. dspy 3.1.0
  ``configure`` writes the GLOBAL ``main_thread_config`` (readable from every
  thread) but permanently binds an owner thread id — and, inside a task, an
  owner async task. Any later ``configure`` from a different thread/task raises
  RuntimeError. Running it first arms every ``settings.lm is not None`` guard
  before any ``to_thread`` step, so no worker thread ever reaches
  ``dspy.configure``. Pinned by the dspy tests in
  tests/unit/test_api/test_chatbot_warmup_1454.py.
* Every other leg runs OFF the loop via ``asyncio.to_thread``. The retrieval
  calls are async functions wrapping SYNC clients (sync ``openai.OpenAI`` for
  the embedding at src/memory/services/factories.py:155, sync Supabase
  ``.rpc().execute()`` at src/rag/memory_connector.py:134), so awaiting them
  here would block the loop for the full external-HTTP duration. "Background
  task" is not by itself a non-blocking guarantee.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

WARM_ENABLED_ENV = "CHATBOT_STARTUP_WARM_ENABLED"
WARM_LOG_PREFIX = "[Chatbot] startup warm"
WARM_COMPLETE_MARKER = f"{WARM_LOG_PREFIX} complete"

# Per-worker start jitter. On deploy every worker boots inside the same second;
# without a spread they build agent registries (CPU) and call the embedding /
# Supabase endpoints (quota) in lockstep.
WARM_JITTER_MAX_SECONDS = 5.0

_TRUTHY = ("1", "true", "yes", "on")


def chatbot_warm_enabled() -> bool:
    """Whether the startup warm is enabled (production default: yes).

    Default OFF under the pytest harness. Any test that drives the real lifespan
    would otherwise schedule a real warm, and the jitter is
    ``random.uniform(0, 5)`` — it can draw ~0, in which case the agent registry
    build and live Supabase/embedding calls start inside the test process, and
    the executor thread outlives the lifespan cancellation. An explicit
    ``CHATBOT_STARTUP_WARM_ENABLED`` always wins, so the warm's own tests still
    opt in.
    """
    explicit = os.getenv(WARM_ENABLED_ENV)
    if explicit is not None:
        return explicit.strip().lower() in _TRUTHY
    if os.getenv("E2I_TESTING_MODE", "").strip().lower() in _TRUTHY:
        return False
    return True


# =============================================================================
# Warm legs
# =============================================================================


async def _warm_dspy_config() -> None:
    """Configure the DSPy LM — FIRST, and on the event loop thread.

    Cheap (``dspy.LM()`` is a no-I/O litellm wrapper, ``configure`` writes a
    dict) and deliberately not ``to_thread``'d: see the module docstring.

    ``_ensure_dspy_configured`` catches its own errors and only warns
    (chatbot_dspy.py:75) — right for a request, but it would let the warm report
    success for a leg that configured nothing, so the postcondition is checked.
    """
    from src.api.routes import chatbot_dspy

    chatbot_dspy._ensure_dspy_configured()

    if chatbot_dspy.DSPY_AVAILABLE:
        import dspy

        if dspy.settings.lm is None:
            raise RuntimeError("DSPy LM still unconfigured after _ensure_dspy_configured()")


async def _rag_warm(connector: Any) -> None:
    """Exercise BOTH retrieval legs with the smallest possible real calls.

    ``hybrid_search`` fans out to a dense RPC and a separate sparse RPC; warming
    only the dense leg leaves the sparse path cold.
    """
    await connector.vector_search_by_text("warmup", k=1)
    await connector.fulltext_search("warmup", k=1)


class _ErrorCapture(logging.Handler):
    """Collects ERROR records emitted from THIS thread while attached.

    The handler sits on the process-global connector logger, so a real request
    racing startup would otherwise leak its own "Vector search failed" into the
    warm's report and fabricate a failure. The warm's retrieval calls all run on
    the thread that attaches this handler (its private ``asyncio.run`` loop runs
    there too), so the thread id is an exact filter. A record with no thread id
    (``logging.logThreads`` disabled) cannot be attributed and is kept: over-
    reporting a real error beats silently claiming a warm that did not happen.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.thread_id = threading.get_ident()
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread is None or record.thread == self.thread_id:
            self.messages.append(record.getMessage())


def _rag_warm_sync() -> None:
    """Run both retrieval legs on a private loop inside this worker thread.

    The clients underneath are sync objects holding no loop-bound state, so a
    short-lived ``asyncio.run`` loop is safe and closes its own resources.

    Both connector methods CATCH their RPC exceptions, log ERROR and return
    ``[]`` (src/rag/memory_connector.py:174 and :245). An empty list is a
    legitimate result for the "warmup" query, so the return value cannot tell us
    whether Supabase was reached — the connector's own ERROR log is the only
    honest signal. Capture it and raise, so the step is reported as failed
    instead of a warm that never happened being reported as success. (A real
    request erroring concurrently would be attributed here too; that only makes
    the warm report more pessimistic, and only about an error that did occur.)
    """
    import src.rag.retriever  # noqa: F401  # the request path's lazy import (chatbot_dspy.py:1622)
    from src.rag.memory_connector import get_memory_connector

    capture = _ErrorCapture()
    connector_logger = logging.getLogger("src.rag.memory_connector")
    connector_logger.addHandler(capture)
    try:
        asyncio.run(_rag_warm(get_memory_connector()))
    finally:
        connector_logger.removeHandler(capture)

    if capture.messages:
        raise RuntimeError("retrieval RPC error: " + "; ".join(capture.messages[:3]))


def _warm_rag_dspy_modules() -> None:
    """Build the cognitive-RAG DSPy singletons (construction only).

    Their internal ``_ensure_dspy_configured()`` calls no-op because the config
    leg already armed the guard on the loop thread.

    Both getters return ``None`` on a swallowed construction failure — and also
    when the feature is switched off, which is not a failure. Only the first
    case is reported.
    """
    from src.api.routes import chatbot_dspy

    rewriter = chatbot_dspy._get_dspy_query_rewriter()
    decider = chatbot_dspy._get_dspy_hop_decider()

    if not (chatbot_dspy.DSPY_AVAILABLE and chatbot_dspy.CHATBOT_COGNITIVE_RAG_ENABLED):
        return
    if rewriter is None:
        raise RuntimeError("cognitive-RAG query rewriter did not build")
    if decider is None:
        raise RuntimeError("cognitive-RAG hop decider did not build")


async def _warm_rag() -> None:
    await asyncio.to_thread(_rag_warm_sync)
    await asyncio.to_thread(_warm_rag_dspy_modules)


def _build_orchestrator() -> Any:
    from src.api.routes.cognitive import get_orchestrator

    return get_orchestrator()


async def _warm_orchestrator() -> None:
    """Build the orchestrator singleton (agent registry + graph compile)."""
    await asyncio.to_thread(_build_orchestrator)


def _build_classifier() -> Any:
    """Build the intent classifier, reporting a swallowed build failure.

    ``None`` means "feature off" as well as "build failed"; only the latter is a
    warm failure.
    """
    from src.api.routes import chatbot_dspy

    classifier = chatbot_dspy._get_dspy_classifier()
    if (
        classifier is None
        and chatbot_dspy.DSPY_AVAILABLE
        and chatbot_dspy.CHATBOT_DSPY_INTENT_ENABLED
    ):
        raise RuntimeError("DSPy intent classifier did not build")
    return classifier


async def _warm_classify() -> None:
    """Build the intent classifier. Construction only — no synthetic chat call."""
    await asyncio.to_thread(_build_classifier)


# =============================================================================
# Warm routine
# =============================================================================


async def _run_step(
    name: str,
    step: Callable[[], Awaitable[None]],
    steps_ms: Dict[str, float],
    failed: Dict[str, str],
) -> None:
    """Run one leg, fail-open. ``CancelledError`` is a BaseException and so
    propagates — cancellation must stop the warm, not be logged as a failure."""
    start = time.perf_counter()
    try:
        await step()
    except Exception as e:  # noqa: BLE001 - a warm leg must never break startup
        failed[name] = f"{type(e).__name__}: {e}"
        logger.warning("%s step=%s FAILED (lazy path unchanged): %s", WARM_LOG_PREFIX, name, e)
    finally:
        steps_ms[name] = round((time.perf_counter() - start) * 1000.0, 1)


async def warm_chatbot_stack(*, jitter_seconds: float | None = None) -> Dict[str, Any]:
    """Pre-build the chatbot's process-scoped singletons. Never raises.

    Args:
        jitter_seconds: start delay. ``None`` (production) draws a random
            0-``WARM_JITTER_MAX_SECONDS`` spread; tests pass 0.0.

    Returns:
        A report dict: ``skipped``, ``pid``, ``jitter_s``, ``steps_ms``,
        ``failed`` and ``total_ms``.
    """
    pid = os.getpid()
    if not chatbot_warm_enabled():
        logger.info("%s skipped pid=%d (%s is off)", WARM_LOG_PREFIX, pid, WARM_ENABLED_ENV)
        return {
            "skipped": True,
            "pid": pid,
            "jitter_s": 0.0,
            "steps_ms": {},
            "failed": {},
            "total_ms": 0.0,
        }

    delay = (
        random.uniform(0.0, WARM_JITTER_MAX_SECONDS) if jitter_seconds is None else jitter_seconds
    )
    if delay > 0:
        await asyncio.sleep(delay)

    # Built here (not at module level) so the legs resolve at call time.
    # DSPy config FIRST (owner-thread binding), then measured-value order.
    steps: List[Tuple[str, Callable[[], Awaitable[None]]]] = [
        ("dspy_config", _warm_dspy_config),
        ("rag", _warm_rag),
        ("orchestrator", _warm_orchestrator),
        ("classify", _warm_classify),
    ]

    steps_ms: Dict[str, float] = {}
    failed: Dict[str, str] = {}
    start = time.perf_counter()
    for name, step in steps:
        await _run_step(name, step, steps_ms, failed)
    total_ms = round((time.perf_counter() - start) * 1000.0, 1)

    logger.info(
        "%s pid=%d total_ms=%.1f jitter_s=%.2f steps_ms=%s failed=%s",
        WARM_COMPLETE_MARKER,
        pid,
        total_ms,
        delay,
        json.dumps(steps_ms, sort_keys=True),
        json.dumps(sorted(failed)),
    )
    return {
        "skipped": False,
        "pid": pid,
        "jitter_s": round(delay, 2),
        "steps_ms": steps_ms,
        "failed": failed,
        "total_ms": total_ms,
    }


def log_warm_task_outcome(task: "asyncio.Task[Any]") -> None:
    """Done-callback: surface a warm task that died instead of losing it."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("%s task failed: %r", WARM_LOG_PREFIX, exc)
