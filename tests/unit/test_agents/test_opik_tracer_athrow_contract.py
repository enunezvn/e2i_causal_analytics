"""Opik tracer async-context-manager cancellation/athrow contract — issue #606.

PROD-LATENT BUG. Several per-agent Opik tracers expose an ``@asynccontextmanager``
(``trace_orchestration``, ``trace_health_check``, ``trace_synthesis``,
``trace_composition``) that yields once on the happy path and then **yields a
second time inside an ``except`` block** (or swallows the thrown exception and
falls through to a second yield). When an exception propagates out of the
``async with`` body, ``__aexit__`` calls ``gen.athrow(exc)``; a generator-based
context manager MUST stop after ``athrow`` — yielding again raises
``RuntimeError("generator didn't stop after athrow()")`` which **masks the real
exception**.

This surfaced in the Tier 1-5 harness (#606): the orchestrator's missing-LLM-key
``ValueError`` was reported as the cryptic ``athrow`` ``RuntimeError`` instead.
But the bug is reachable in **production** under any cancellation (client
disconnect, request timeout) during a traced operation — so it is fixed in the
tracer code, not worked around in the harness.

``heterogeneous_optimizer/opik_tracer.py`` already uses the correct single-yield
pattern (see its comment "Single yield point - avoids 'generator didn't stop
after athrow()' errors"); this test pins every other tracer to the same contract.

Pure asyncio + a fake Opik client — no services, no LLM, no heavy ML.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest


class _FakeTrace:
    """Minimal stand-in for an Opik trace object (has .end()/.update()/.span())."""

    def end(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def span(self, *a, **k):
        return _FakeTrace()


class _FakeClient:
    def trace(self, *a, **k):
        return _FakeTrace()

    def flush(self, *a, **k):
        pass


# (module path, class name, context-manager method, call kwargs)
# These three share the identical `yield ctx` -> `except Exception: yield null`
# -> `finally` shape driven by enabled + _get_client + _should_sample.
_TRIO = [
    (
        "src.agents.orchestrator.opik_tracer",
        "OrchestratorOpikTracer",
        "trace_orchestration",
        {"query_id": "q-606"},
    ),
    (
        "src.agents.health_score.opik_tracer",
        "HealthScoreOpikTracer",
        "trace_health_check",
        {},
    ),
    (
        "src.agents.prediction_synthesizer.opik_tracer",
        "PredictionSynthesizerOpikTracer",
        "trace_synthesis",
        {},
    ),
]


def _force_traced_path(tracer):
    """Force the enabled+client+sampled path so the buggy except-yield is reached."""
    tracer.enabled = True
    tracer._should_sample = lambda: True
    tracer._get_client = lambda: _FakeClient()
    return tracer


@pytest.mark.parametrize("modpath,clsname,method,kwargs", _TRIO)
def test_tracer_propagates_body_exception_not_athrow_runtimeerror(modpath, clsname, method, kwargs):
    mod = importlib.import_module(modpath)
    tracer = _force_traced_path(getattr(mod, clsname)(enabled=True))
    cm = getattr(tracer, method)

    async def run():
        async with cm(**kwargs):
            raise ValueError("boom-from-body-606")

    # Today: RuntimeError("generator didn't stop after athrow()") masks the real
    # error. After the single-yield fix: the original ValueError propagates.
    with pytest.raises(ValueError, match="boom-from-body-606"):
        asyncio.run(run())


class _FakeAsyncSpanCM:
    """Async CM standing in for OpikConnector.trace_agent(...)."""

    async def __aenter__(self):
        return object()  # the opik span

    async def __aexit__(self, *exc):
        return False


class _FakeConnector:
    is_enabled = True

    def trace_agent(self, *a, **k):
        return _FakeAsyncSpanCM()


def test_tool_composer_tracer_propagates_body_exception_not_athrow_runtimeerror():
    """tool_composer's trace_composition used a nested `async with ... yield` whose
    inner `except` swallowed an athrow'd body exception and fell through to a
    second yield. The manual-enter single-yield fix must let the body exception
    propagate."""
    from src.agents.tool_composer.opik_tracer import ToolComposerOpikTracer

    tracer = ToolComposerOpikTracer(enabled=True)
    tracer._ensure_initialized = lambda: None  # don't overwrite our fake connector
    tracer._opik_connector = _FakeConnector()
    tracer._should_trace = lambda: True

    async def run():
        async with tracer.trace_composition(query="q-606"):
            raise ValueError("boom-from-body-606")

    with pytest.raises(ValueError, match="boom-from-body-606"):
        asyncio.run(run())
