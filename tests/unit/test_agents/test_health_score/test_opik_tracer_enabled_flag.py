"""Red-first (#952): the health_score Opik tracer must honor OPIK_ENABLED.

Opik is intentionally STOPPED on the prod droplet (memory relief) and OFF by
design in CI (no opik container; ``OPIK_URL`` points at a dead port). Against
that dead endpoint the Opik SDK's background uploader thread does NOT no-op — it
raises ``httpx.ConnectTimeout`` and RETRIES, spawning persistent threads that
(a) leak on every health check in prod and (b) DETERMINISTICALLY hang the
serviceless CI unit shard's xdist worker (``node down`` -> 20-min job cancel;
issue #952 / PR #948).

The fix mirrors ``src/mlops/opik_connector.py`` (``OPIK_ENABLED``, opt-out,
default ``"true"``): when ``OPIK_ENABLED=false`` the health tracer must be
disabled — no client constructed, no trace emitted — while leaving the default
(env unset) enabled so real deployments with Opik running keep tracing. An
explicit ``enabled=`` argument always wins, so callers can still force a state.
"""

from __future__ import annotations

import pytest

from src.agents.health_score import opik_tracer as health_score_opik_module
from src.agents.health_score.opik_tracer import (
    HealthScoreOpikTracer,
    get_health_score_tracer,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """The tracer is a process-wide singleton; reset it so each test resolves
    OPIK_ENABLED freshly (mirrors test_opik_tracer.py)."""
    HealthScoreOpikTracer._instance = None
    HealthScoreOpikTracer._initialized = False
    health_score_opik_module._tracer_instance = None
    yield
    HealthScoreOpikTracer._instance = None
    HealthScoreOpikTracer._initialized = False
    health_score_opik_module._tracer_instance = None


def test_default_enabled_when_env_unset(monkeypatch):
    """Env unset => enabled (opt-out; matches opik_connector default 'true')."""
    monkeypatch.delenv("OPIK_ENABLED", raising=False)
    assert get_health_score_tracer().enabled is True


def test_enabled_when_env_true(monkeypatch):
    monkeypatch.setenv("OPIK_ENABLED", "true")
    assert get_health_score_tracer().enabled is True


def test_factory_disabled_when_opik_env_false(monkeypatch):
    """OPIK_ENABLED=false disables the singleton built by the agent's
    get_health_score_tracer() (called with no args on the run path)."""
    monkeypatch.setenv("OPIK_ENABLED", "false")
    assert get_health_score_tracer().enabled is False


def test_constructor_disabled_when_opik_env_false(monkeypatch):
    """Direct construction with no explicit `enabled` also honors the env."""
    monkeypatch.setenv("OPIK_ENABLED", "false")
    assert HealthScoreOpikTracer().enabled is False


@pytest.mark.parametrize("val", ["false", "False", "FALSE", " false "])
def test_falsey_values_disable(monkeypatch, val):
    monkeypatch.setenv("OPIK_ENABLED", val)
    assert HealthScoreOpikTracer().enabled is False


def test_explicit_enabled_arg_overrides_env(monkeypatch):
    """An explicit enabled= argument wins over the env (callers can force)."""
    monkeypatch.setenv("OPIK_ENABLED", "false")
    assert HealthScoreOpikTracer(enabled=True).enabled is True


@pytest.mark.asyncio
async def test_trace_health_check_noop_when_disabled(monkeypatch):
    """Disabled => trace_health_check yields a no-op context WITHOUT building an
    Opik client. Client construction + emission is exactly what spawns the
    forever-retrying background uploader against the dead endpoint."""
    monkeypatch.setenv("OPIK_ENABLED", "false")
    tracer = get_health_score_tracer()

    calls = {"n": 0}
    real = tracer._get_client

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(tracer, "_get_client", _spy)
    async with tracer.trace_health_check(check_scope="full") as ctx:
        assert ctx.trace is None
    assert calls["n"] == 0, "disabled tracer must not construct an Opik client"
