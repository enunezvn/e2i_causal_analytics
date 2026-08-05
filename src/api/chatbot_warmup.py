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

* The construction legs remove construction, import and first-connection cost.
  The two synthetic-LLM legs (#1475 target 3, flag-gated) additionally prepay
  the process-shared first-LLM-call machinery with one real classify call and
  one real rewriter call (~2 small completions per worker per boot). What
  remains is genuinely unwarmable: novel-query steady-state LLM latency
  (~1.2-2.4s per call, measured) and multi-hop retrieval chains.
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
* Heavy sync construction legs run OFF the loop via ``asyncio.to_thread``.
* The RAG retrieval leg runs ON the loop (#1475). It used to be async
  functions wrapping SYNC clients, so PR #1474 pushed it into a ``to_thread``
  worker with a private ``asyncio.run`` loop. After #1475's migration
  (``openai.AsyncOpenAI`` + async Supabase client) the internals are genuinely
  async — and the old workaround would be actively harmful: the private loop
  would create the loop-affine async client singletons
  (``_async_supabase_client``, the service's ``AsyncOpenAI``), then close,
  handing the request path pooled connections bound to a DEAD loop. Warming
  on the main loop creates them exactly where requests will use them.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import random
import sys
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

WARM_ENABLED_ENV = "CHATBOT_STARTUP_WARM_ENABLED"
WARM_LOG_PREFIX = "[Chatbot] startup warm"
WARM_COMPLETE_MARKER = f"{WARM_LOG_PREFIX} complete"

# Per-worker start jitter. On deploy every worker boots inside the same second;
# without a spread they build agent registries (CPU) and call the embedding /
# Supabase endpoints (quota) in lockstep.
WARM_JITTER_MAX_SECONDS = 5.0

# --- #1475 target 3: synthetic-LLM warm legs -------------------------------
#
# The construction legs above leave each worker's FIRST real LLM call to the
# first user request. Measured (2026-08-05, real calls on the prod box): that
# first call costs 2.5-3.7s for classify vs a ~1.2-1.5s novel-query steady
# state, the penalty is process-shared litellm init (after one first call,
# other signatures sit <=~1s above their own steady state), and it does NOT
# decay — 120s idle then a novel call was 1183ms, indistinguishable from
# steady state. Two narrow synthetic calls (classify + the RAG query
# rewriter) therefore move ~3-6s off the first request, durably. A blanket
# per-module sweep would buy nothing extra: the remaining first-RAG cost is
# novel-query steady state and multi-hop chains, which no warm can remove.
WARM_LLM_ENABLED_ENV = "CHATBOT_STARTUP_WARM_LLM_ENABLED"

# A hung provider call must fail the leg (fail-open), not pin the warm task.
# wait_for cancels the awaiting task; the worker thread finishes quietly.
WARM_LLM_TIMEOUT_SECONDS = 30.0

# Recognized-shaped on purpose: "trx" hits the kpi_query pattern, so the warm
# exercises the path real KPI asks take. #1478 measured generic warm probes
# landing in the CLARIFICATION/confidence-0.0 fallback — warming the WRONG
# path. Pinned by test_warm_query_is_recognized_shaped.
WARM_LLM_QUERY = "What is the TRx trend for Kisqali this quarter?"

# DSPy caches responses by full prompt (disk + memory, shared across
# processes where the cache dir persists). A cache hit would turn the warm
# call into a no-op that warms no network path, so every process salts the
# conversation_context with a fresh token.
_WARM_LLM_CACHE_BUSTER = uuid.uuid4().hex

_TRUTHY = ("1", "true", "yes", "on")


def chatbot_warm_enabled() -> bool:
    """Whether the startup warm is enabled (production default: yes).

    Default OFF inside a pytest process. Any test that drives the real lifespan
    would otherwise schedule a real warm, and the jitter is
    ``random.uniform(0, 5)`` — it can draw ~0, in which case the agent registry
    build and live Supabase/embedding calls start inside the test process, and
    the executor thread outlives the lifespan cancellation. The signal is
    ``sys.modules`` deliberately, not an env var: a test-env variable copied into
    a container would silently reintroduce the #1454 cold start, whereas pytest
    is never imported in the API image. An explicit
    ``CHATBOT_STARTUP_WARM_ENABLED`` always wins, so the warm's own tests opt in.
    """
    explicit = os.getenv(WARM_ENABLED_ENV)
    if explicit is not None:
        return explicit.strip().lower() in _TRUTHY
    return "pytest" not in sys.modules


def chatbot_warm_llm_enabled() -> bool:
    """Whether the synthetic-LLM warm legs run (production default: yes).

    Same shape as ``chatbot_warm_enabled`` and for the same reason, but this
    flag matters even to tests that opt INTO the parent warm: the #1454 test
    file sets ``CHATBOT_STARTUP_WARM_ENABLED=true`` and stubs the four
    construction legs — without a separate pytest-off default here, those
    tests would fire two real LLM calls per run.
    """
    explicit = os.getenv(WARM_LLM_ENABLED_ENV)
    if explicit is not None:
        return explicit.strip().lower() in _TRUTHY
    return "pytest" not in sys.modules


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


# True inside the warm's OWN task context (and any task it spawns —
# ``create_task`` copies the context). A real request racing startup runs in
# its own task with the default False, so its connector errors are never
# attributed to the warm. Replaces the pre-#1475 thread-id filter: the
# retrieval legs now run ON the loop thread, which a racing request shares.
_warm_rag_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "chatbot_warm_rag_active", default=False
)


class _ErrorCapture(logging.Handler):
    """Collects ERROR records emitted from the warm's task context.

    The handler sits on the process-global connector logger, so a real request
    racing startup would otherwise leak its own "Vector search failed" into the
    warm's report and fabricate a failure. ``Handler.emit`` runs synchronously
    in the logging caller's own context, so the ``_warm_rag_active`` contextvar
    is an exact per-task filter.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if _warm_rag_active.get():
            self.messages.append(record.getMessage())


async def _rag_warm_probe() -> None:
    """Run both retrieval legs ON the event loop (#1475).

    The internals are genuinely async now (``openai.AsyncOpenAI`` +
    ``get_async_supabase_client``), so awaiting them here no longer blocks the
    loop — and running them on the MAIN loop is required, not just allowed:
    this first touch creates the loop-affine async client singletons on the
    loop the request path uses (see module docstring).

    Both connector methods CATCH their RPC exceptions, log ERROR and return
    ``[]``. An empty list is a legitimate result for the "warmup" query, so
    the return value cannot tell us whether Supabase was reached — the
    connector's own ERROR log is the only honest signal. Capture it and raise,
    so the step is reported as failed instead of a warm that never happened
    being reported as success.
    """
    import src.rag.retriever  # noqa: F401  # the request path's lazy import (chatbot_dspy.py:1622)
    from src.rag.memory_connector import get_memory_connector

    capture = _ErrorCapture()
    connector_logger = logging.getLogger("src.rag.memory_connector")
    token = _warm_rag_active.set(True)
    connector_logger.addHandler(capture)
    try:
        await _rag_warm(get_memory_connector())
    finally:
        connector_logger.removeHandler(capture)
        _warm_rag_active.reset(token)

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
    # Retrieval ON the loop (#1475 — async clients are loop-affine);
    # DSPy module construction stays off-loop (sync build work).
    await _rag_warm_probe()
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


def _warm_llm_context() -> str:
    return f"[startup warm {_WARM_LLM_CACHE_BUSTER}]"


def _classify_llm_call() -> None:
    """One real call through the raw classify module (sync — runs off-loop).

    RAW module deliberately: ``classify_intent_dspy`` would push a synthetic
    row into the intent training-signal buffer that feedback_learner optimizes
    prompts from. (``classification_logs`` is never at risk either way — only
    the orchestrator's 4-stage shadow node writes it.)

    ``None`` is a skip, not a failure: it means feature-off or a build failure
    the ``classify`` construction leg already reported.
    """
    from src.api.routes import chatbot_dspy

    classifier = chatbot_dspy._get_dspy_classifier()
    if classifier is None:
        return
    result = classifier(query=WARM_LLM_QUERY, conversation_context=_warm_llm_context())
    if not str(getattr(result, "intent", "") or "").strip():
        raise RuntimeError("classify warm call returned no intent")


async def _warm_classify_llm() -> None:
    await asyncio.wait_for(asyncio.to_thread(_classify_llm_call), WARM_LLM_TIMEOUT_SECONDS)


def _rag_rewrite_llm_call() -> None:
    """One real call through the raw RAG query-rewriter module (sync, off-loop).

    Same raw-module reasoning as ``_classify_llm_call`` (the RAG signal
    collector lives in ``cognitive_rag_retrieve``, above this seam). The
    evidence scorer and hop decider are deliberately NOT warmed: measured,
    their first calls after any one first call in the process sit at their
    own steady state — extra synthetic calls would spend tokens on nothing.
    """
    from src.api.routes import chatbot_dspy

    rewriter = chatbot_dspy._get_dspy_query_rewriter()
    if rewriter is None:
        return
    result = rewriter(
        original_query=WARM_LLM_QUERY,
        conversation_context=_warm_llm_context(),
        domain_vocabulary=chatbot_dspy.E2I_DOMAIN_VOCABULARY,
    )
    if not str(getattr(result, "rewritten_query", "") or "").strip():
        raise RuntimeError("rewrite warm call returned no rewritten_query")


async def _warm_rag_rewrite_llm() -> None:
    await asyncio.wait_for(asyncio.to_thread(_rag_rewrite_llm_call), WARM_LLM_TIMEOUT_SECONDS)


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
    # The synthetic-LLM legs run LAST: they consume the singletons the
    # construction legs build, and a failed/slow provider call must never
    # delay the construction warm.
    steps: List[Tuple[str, Callable[[], Awaitable[None]]]] = [
        ("dspy_config", _warm_dspy_config),
        ("rag", _warm_rag),
        ("orchestrator", _warm_orchestrator),
        ("classify", _warm_classify),
    ]
    if chatbot_warm_llm_enabled():
        steps.append(("classify_llm", _warm_classify_llm))
        steps.append(("rag_rewrite_llm", _warm_rag_rewrite_llm))

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
