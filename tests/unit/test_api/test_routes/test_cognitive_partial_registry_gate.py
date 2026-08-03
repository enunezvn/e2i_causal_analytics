"""#1448 — ``get_orchestrator`` must not swallow the strict partial-registry gate.

``src/api/routes/cognitive.get_orchestrator`` is the ONLY production construction
site for the real agent registry (grep: ``create_agent_registry`` under ``src/``).
It wraps construction in a broad ``except Exception`` that logs a warning and
returns ``None``.

That is the right behaviour for a genuinely unavailable orchestrator, but it would
turn ``E2I_REQUIRE_FULL_AGENT_REGISTRY=true`` into a *trap*: the strict gate would
raise, get swallowed, and every request would silently lose orchestration entirely
— strictly worse than the 18/21 partial registry it was meant to flag. The gate
must propagate.

Default (strict OFF) behaviour is unchanged and is asserted here as a control.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.agents.orchestrator as orchestrator_pkg
import src.api.routes.cognitive as cognitive
from src.agents import factory


def _boom(module_path: str, class_name: str):  # noqa: ANN202 - test stub
    raise RuntimeError("simulated missing project-root marker")


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    monkeypatch.setattr(cognitive, "_orchestrator_instance", None, raising=False)
    monkeypatch.delenv(factory.REQUIRE_FULL_REGISTRY_ENV, raising=False)


def test_get_orchestrator_propagates_partial_registry_error(monkeypatch):
    """With the gate armed, a degraded registry raises out of get_orchestrator."""
    monkeypatch.setenv(factory.REQUIRE_FULL_REGISTRY_ENV, "true")

    with patch.object(factory, "_create_agent", side_effect=_boom):
        with pytest.raises(factory.PartialAgentRegistryError):
            cognitive.get_orchestrator()

    assert cognitive._orchestrator_instance is None, (
        "a failed gate must not cache a half-built orchestrator"
    )


def test_get_orchestrator_still_degrades_gracefully_by_default(monkeypatch):
    """Control: with the gate OFF (the default), a partial registry still yields a
    working orchestrator over the agents that DID construct — the #814 fail-closed
    dispatch property is preserved."""
    built = MagicMock(name="OrchestratorAgent")
    monkeypatch.setattr(orchestrator_pkg, "OrchestratorAgent", MagicMock(return_value=built))

    def _only_causal_impact(module_path: str, class_name: str):  # noqa: ANN202
        if class_name == "CausalImpactAgent":
            return object()
        raise RuntimeError("simulated missing project-root marker")

    with patch.object(factory, "_create_agent", side_effect=_only_causal_impact):
        result = cognitive.get_orchestrator()

    assert result is built
    registry = orchestrator_pkg.OrchestratorAgent.call_args.kwargs["agent_registry"]
    assert set(registry) == {"causal_impact"}
