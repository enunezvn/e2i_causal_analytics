"""Red-first tests for #609: wire the global audit-chain service in the API lifespan.

Issue #609: ``set_audit_chain_service`` / ``init_audit_chain_service`` were never
called anywhere in ``src/``, so ``get_audit_chain_service()`` was always ``None``
and every Tier 1-5 agent's ``audit_init`` node (``create_workflow_initializer``)
was a silent no-op — zero ``audit_chain_entries`` rows were ever written on the
``/api/*`` path despite the platform's regulatory-compliance audit rationale.

These tests drive the REAL FastAPI lifespan (``src.api.main.lifespan``) with the
networky init/cleanup blocks patched out (they are all ``try/except``-guarded and
irrelevant to the audit wiring), asserting that:

  1. when Supabase is available, the lifespan wires the global ``AuditChainService``
     REUSING the lifespan's existing client (not a second ``create_client``);
  2. a representative Tier 1-5 agent's ``audit_init`` then fires ``start_workflow``,
     attempting an insert into ``audit_chain_entries`` — the CI-faithful proxy for
     "a representative agent run produces an audit_chain_entries row" (no live DB
     in CI, so we assert the insert CALL against a mock client, not a physical row);
  3. the lifespan degrades safely (global stays ``None``) when Supabase is absent;
  4. the lifespan resets the global on shutdown (symmetric lifecycle, no leak).
"""

import types
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from src.agents.base.audit_chain_mixin import (
    create_workflow_initializer,
    get_audit_chain_service,
    set_audit_chain_service,
)
from src.api import main
from src.utils.audit_chain import AgentTier, AuditChainService


@pytest.fixture(autouse=True)
def reset_global_service():
    """Reset the global audit service before and after each test.

    Mirrors the codebase convention (tests/unit/test_agents/test_base/
    test_audit_chain_mixin.py:85-90 and tests/integration/
    test_audit_chain_integration.py:75-80). The lifespan SETS the module-global
    ``_audit_service``; without this reset it would leak into sibling tests.
    """
    set_audit_chain_service(None)
    yield
    set_audit_chain_service(None)


def _fake_app():
    """Minimal stand-in for the FastAPI app — the lifespan only touches app.state."""
    return types.SimpleNamespace(state=types.SimpleNamespace())


@contextmanager
def _hermetic_lifespan_io():
    """Patch out every networky lifespan init/cleanup so the lifespan runs
    hermetically and fast, leaving ONLY the Supabase + audit-wiring path live.

    Each patch targets the symbol in ``src.api.main``'s namespace (where the
    lifespan resolves it). ``init_supabase`` is patched per-test by the caller.
    """
    feast = MagicMock()
    feast.initialize = AsyncMock()
    feast._initialized = False
    feast_cls = MagicMock(return_value=feast)

    with (
        patch("src.api.main.get_bentoml_client", new=AsyncMock()),
        patch("src.api.main.configure_bentoml_endpoints", new=MagicMock()),
        patch("src.api.main.init_redis", new=AsyncMock()),
        patch("src.api.main.init_falkordb", new=AsyncMock()),
        patch("src.api.main.get_mlflow_connector", new=MagicMock()),
        patch("src.api.main.get_opik_connector", new=MagicMock()),
        patch("src.api.main.FeastClient", new=feast_cls),
        patch("src.api.main.close_bentoml_client", new=AsyncMock()),
        patch("src.api.main.close_redis", new=AsyncMock()),
        patch("src.api.main.close_falkordb", new=AsyncMock()),
        patch("src.api.main.close_supabase", new=MagicMock()),
        patch("src.api.main.shutdown_opentelemetry", new=MagicMock()),
        patch(
            "src.memory.sentinels.config_loader.load_sentinels_from_yaml",
            new=AsyncMock(return_value=0),
        ),
    ):
        yield


def _representative_audit_init_state():
    """Input state for the causal_impact ``audit_init`` node (the literal node
    wired at src/agents/causal_impact/graph.py:425)."""
    return {
        "query": "What is the impact of treatment on outcome?",
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "session_id": None,
    }


@pytest.mark.asyncio
async def test_lifespan_wires_audit_service_reusing_supabase_client():
    """When Supabase is available, the lifespan wires a global AuditChainService
    that REUSES the lifespan's existing client (not a second create_client).

    RED before #609: the lifespan never calls set_audit_chain_service, so the
    global stays None.
    """
    mock_client = MagicMock(name="supabase_client")

    with _hermetic_lifespan_io(), patch("src.api.main.init_supabase", return_value=mock_client):
        async with main.lifespan(_fake_app()):
            service = get_audit_chain_service()
            assert service is not None, "lifespan must wire the audit-chain global"
            assert isinstance(service, AuditChainService)
            # Reuse, not re-create: the wired service wraps the SAME client.
            assert service.db is mock_client


@pytest.mark.asyncio
async def test_lifespan_wiring_makes_agent_audit_init_emit_audit_chain_entry():
    """Issue acceptance: with the service configured by the lifespan, a
    representative Tier 1-5 agent's audit_init fires start_workflow and attempts
    an insert into 'audit_chain_entries'.

    RED before #609: global is None -> audit_init returns state unchanged with no
    audit_workflow_id and no DB insert.
    """
    mock_client = MagicMock(name="supabase_client")

    with _hermetic_lifespan_io(), patch("src.api.main.init_supabase", return_value=mock_client):
        async with main.lifespan(_fake_app()):
            initializer = create_workflow_initializer("causal_impact", AgentTier.CAUSAL_ANALYTICS)
            result = initializer(_representative_audit_init_state())

            # The genesis block fired: state gained a real workflow id.
            assert "audit_workflow_id" in result
            assert isinstance(result["audit_workflow_id"], UUID)

            # CI-faithful proxy for a written row: an insert into the
            # audit_chain_entries table was attempted on the wired client.
            tables = [c.args[0] for c in mock_client.table.call_args_list if c.args]
            assert "audit_chain_entries" in tables

            # ...and the insert carried a well-formed genesis audit row (not just
            # a table reference): correct agent, genesis action, matching id, and
            # the tamper-evident hash.
            insert_mock = mock_client.table.return_value.insert
            assert insert_mock.called
            payload = insert_mock.call_args.args[0]
            assert payload["agent_name"] == "causal_impact"
            assert payload["action_type"] == "workflow_start"
            assert payload["workflow_id"] == str(result["audit_workflow_id"])
            assert payload["entry_hash"]


@pytest.mark.asyncio
async def test_lifespan_degrades_safely_when_supabase_unavailable():
    """When Supabase is not configured (init_supabase -> None), the lifespan must
    NOT wire the audit chain and must not crash; the global stays None."""
    with _hermetic_lifespan_io(), patch("src.api.main.init_supabase", return_value=None):
        app = _fake_app()
        async with main.lifespan(app):
            assert get_audit_chain_service() is None
            assert app.state.supabase_available is False


@pytest.mark.asyncio
async def test_lifespan_resets_audit_service_on_shutdown():
    """The lifespan must reset the global on shutdown so it does not leak across
    app reloads / tests (symmetric lifecycle).

    RED before #609 adds the shutdown reset: with startup wiring present, the
    global would remain set after the context exits.
    """
    mock_client = MagicMock(name="supabase_client")

    with _hermetic_lifespan_io(), patch("src.api.main.init_supabase", return_value=mock_client):
        async with main.lifespan(_fake_app()):
            assert get_audit_chain_service() is not None  # wired at startup
        # Context exited -> shutdown ran.
        assert get_audit_chain_service() is None  # reset at shutdown


@pytest.mark.asyncio
async def test_lifespan_resets_audit_service_on_exceptional_shutdown():
    """The audit global must be reset even when the application body raises during
    the lifespan (exceptional shutdown) — otherwise the wired singleton leaks into
    a subsequent same-process lifespan (uvicorn --reload, tests).

    RED before the try/finally fix: the shutdown reset sits after ``yield`` outside
    any try/finally, so an exception thrown into the lifespan skips it and the
    global stays set.
    """
    mock_client = MagicMock(name="supabase_client")

    with _hermetic_lifespan_io(), patch("src.api.main.init_supabase", return_value=mock_client):
        with pytest.raises(RuntimeError, match="app crashed during run"):
            async with main.lifespan(_fake_app()):
                assert get_audit_chain_service() is not None  # wired at startup
                raise RuntimeError("app crashed during run")
        # Shutdown was exceptional, but the global must still be reset.
        assert get_audit_chain_service() is None


@pytest.mark.asyncio
async def test_lifespan_runs_resource_cleanup_on_exceptional_shutdown():
    """All resource-cleanup blocks (redis/falkordb/bentoml/supabase/feast/opik/
    otel) must run on EXCEPTIONAL shutdown too, not just on normal shutdown.

    RED before this fix: the cleanup blocks lived AFTER ``yield`` but OUTSIDE the
    ``finally`` that held the audit-chain reset, so an exception thrown into the
    lifespan body skipped every close()/flush() — leaking connections, sockets
    and unflushed traces on any abnormal startup/run failure.

    Each cleanup is individually try/except-wrapped, so moving them into the
    finally cannot make shutdown crash; this test asserts they actually fire.
    """
    mock_client = MagicMock(name="supabase_client")

    # Track the cleanup hooks we expect to fire on shutdown.
    close_redis = AsyncMock(name="close_redis")
    close_falkordb = AsyncMock(name="close_falkordb")
    close_bentoml = AsyncMock(name="close_bentoml_client")
    close_supabase = MagicMock(name="close_supabase")
    shutdown_otel = MagicMock(name="shutdown_opentelemetry")

    feast = MagicMock()
    feast.initialize = AsyncMock()
    feast._initialized = False
    feast.close = AsyncMock(name="feast_close")
    feast_cls = MagicMock(return_value=feast)

    with (
        patch("src.api.main.get_bentoml_client", new=AsyncMock()),
        patch("src.api.main.configure_bentoml_endpoints", new=MagicMock()),
        patch("src.api.main.init_redis", new=AsyncMock()),
        patch("src.api.main.init_falkordb", new=AsyncMock()),
        patch("src.api.main.get_mlflow_connector", new=MagicMock()),
        patch("src.api.main.get_opik_connector", new=MagicMock()),
        patch("src.api.main.FeastClient", new=feast_cls),
        patch("src.api.main.close_bentoml_client", new=close_bentoml),
        patch("src.api.main.close_redis", new=close_redis),
        patch("src.api.main.close_falkordb", new=close_falkordb),
        patch("src.api.main.close_supabase", new=close_supabase),
        patch("src.api.main.shutdown_opentelemetry", new=shutdown_otel),
        patch(
            "src.memory.sentinels.config_loader.load_sentinels_from_yaml",
            new=AsyncMock(return_value=0),
        ),
        patch("src.api.main.init_supabase", return_value=mock_client),
    ):
        with pytest.raises(RuntimeError, match="boom during run"):
            async with main.lifespan(_fake_app()):
                raise RuntimeError("boom during run")

    # Every cleanup must have fired despite the exceptional shutdown.
    close_redis.assert_awaited_once()
    close_falkordb.assert_awaited_once()
    close_bentoml.assert_awaited_once()
    close_supabase.assert_called_once()
    feast.close.assert_awaited_once()
    shutdown_otel.assert_called_once()
