"""#814: ``create_agent_registry`` must surface a PARTIAL registry loudly.

When an enabled agent fails to instantiate (e.g. missing credentials), the
factory keeps building a degraded registry (it does NOT raise unless
``fail_on_import_error``). Because the dispatcher now fails closed for a missing
agent, a *silent* drop would look like a routing bug rather than a
misconfiguration — so the factory must log a prominent partial-registry summary
naming the dropped agents.

These tests patch the per-agent instantiation boundary only; the factory's own
selection/logging logic runs for real.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from src.agents import factory


def test_partial_registry_logs_dropped_agents(caplog):
    """An enabled agent that fails to instantiate is dropped AND named in a
    loud PARTIAL-registry warning (operator visibility)."""

    def _boom(module_path: str, class_name: str):  # noqa: ANN202 - test stub
        raise RuntimeError("simulated missing creds")

    with patch.object(factory, "_create_agent", side_effect=_boom):
        with caplog.at_level(logging.WARNING):
            registry = factory.create_agent_registry(include_agents=["causal_impact"])

    assert registry == {}, "a failed agent must not appear in the registry"
    partial = [r for r in caplog.records if "PARTIAL registry" in r.getMessage()]
    assert partial, "expected a PARTIAL-registry warning when an agent is dropped"
    assert any("causal_impact" in r.getMessage() for r in partial)


def test_full_registry_emits_no_partial_warning(caplog):
    """When every selected agent instantiates, there is NO partial warning."""

    def _ok(module_path: str, class_name: str):  # noqa: ANN202 - test stub
        return object()

    with patch.object(factory, "_create_agent", side_effect=_ok):
        with caplog.at_level(logging.WARNING):
            registry = factory.create_agent_registry(
                include_agents=["causal_impact", "gap_analyzer"]
            )

    assert set(registry) == {"causal_impact", "gap_analyzer"}
    assert not any("PARTIAL registry" in r.getMessage() for r in caplog.records)


def test_none_instance_is_treated_as_dropped(caplog):
    """An agent whose constructor returns None (no exception) is also a drop and
    must be surfaced — not silently skipped."""

    def _none(module_path: str, class_name: str):  # noqa: ANN202 - test stub
        return None

    with patch.object(factory, "_create_agent", side_effect=_none):
        with caplog.at_level(logging.WARNING):
            registry = factory.create_agent_registry(include_agents=["causal_impact"])

    assert registry == {}
    assert any("PARTIAL registry" in r.getMessage() for r in caplog.records)
