"""#814: ``create_agent_registry`` must surface a PARTIAL registry loudly.

When an enabled agent fails to instantiate (e.g. missing credentials), the
factory keeps building a degraded registry (it does NOT raise unless
``fail_on_import_error``). Because the dispatcher now fails closed for a missing
agent, a *silent* drop would look like a routing bug rather than a
misconfiguration — so the factory must log a prominent partial-registry summary
naming the dropped agents.

#1448 raises the bar: a WARNING that fires on every registry build (once per
gunicorn worker, again after every ``--max-requests`` recycle) trains readers to
ignore it. The partial-registry summary must be emitted at ERROR severity with
machine-readable fields, and an opt-in strict mode
(``E2I_REQUIRE_FULL_AGENT_REGISTRY``) must turn the degradation into a hard,
named failure so it can be used as a deploy gate.

These tests patch the per-agent instantiation boundary only; the factory's own
selection/logging logic runs for real.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

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


# ---------------------------------------------------------------------------
# #1448 — loud failure, not a forever-warning
# ---------------------------------------------------------------------------


def _boom(module_path: str, class_name: str):  # noqa: ANN202 - test stub
    raise RuntimeError("simulated missing creds")


def _ok(module_path: str, class_name: str):  # noqa: ANN202 - test stub
    return object()


@pytest.fixture(autouse=True)
def _no_ambient_strict_mode(monkeypatch):
    """Strict mode must be opt-in per test, never inherited from the shell/.env."""
    monkeypatch.delenv(factory.REQUIRE_FULL_REGISTRY_ENV, raising=False)


def test_partial_registry_is_reported_at_error_severity(caplog):
    """#1448: the partial-registry summary is a high-severity alert (>= ERROR),
    not a WARNING lost among per-request noise."""
    with patch.object(factory, "_create_agent", side_effect=_boom):
        with caplog.at_level(logging.DEBUG):
            factory.create_agent_registry(include_agents=["causal_impact"])

    partial = [r for r in caplog.records if "PARTIAL registry" in r.getMessage()]
    assert partial, "expected a PARTIAL-registry record"
    assert all(r.levelno >= logging.ERROR for r in partial), (
        f"PARTIAL registry must be logged at ERROR or higher; got {[r.levelname for r in partial]}"
    )


def test_partial_registry_record_carries_structured_alert_fields(caplog):
    """Operators alert on fields, not on prose. The record must expose the dropped
    agents, the expected count and the realised registry size."""
    with patch.object(factory, "_create_agent", side_effect=_boom):
        with caplog.at_level(logging.DEBUG):
            factory.create_agent_registry(include_agents=["causal_impact", "gap_analyzer"])

    partial = [r for r in caplog.records if "PARTIAL registry" in r.getMessage()]
    assert partial
    record = partial[0]
    assert getattr(record, "dropped_agents", None) == ["causal_impact", "gap_analyzer"]
    assert getattr(record, "expected_agent_count", None) == 2
    assert getattr(record, "registry_size", None) == 0


def test_require_full_raises_partial_agent_registry_error():
    """#1448 strict mode: a partial registry is a hard, named failure naming the
    dropped agents — usable as a deploy gate."""
    with patch.object(factory, "_create_agent", side_effect=_boom):
        with pytest.raises(factory.PartialAgentRegistryError) as excinfo:
            factory.create_agent_registry(include_agents=["causal_impact"], require_full=True)

    assert excinfo.value.dropped == ["causal_impact"]
    assert "causal_impact" in str(excinfo.value)


def test_require_full_defaults_from_env(monkeypatch):
    """Strict mode can be switched on for a deployment without a code change."""
    monkeypatch.setenv(factory.REQUIRE_FULL_REGISTRY_ENV, "true")

    with patch.object(factory, "_create_agent", side_effect=_boom):
        with pytest.raises(factory.PartialAgentRegistryError):
            factory.create_agent_registry(include_agents=["causal_impact"])


def test_require_full_argument_overrides_env(monkeypatch):
    """An explicit ``require_full=False`` beats the env default (benchmark/CLI
    callers that deliberately build a subset must not be broken by the flag)."""
    monkeypatch.setenv(factory.REQUIRE_FULL_REGISTRY_ENV, "true")

    with patch.object(factory, "_create_agent", side_effect=_boom):
        registry = factory.create_agent_registry(
            include_agents=["causal_impact"], require_full=False
        )

    assert registry == {}


def test_require_full_is_silent_when_registry_is_complete(monkeypatch, caplog):
    """No drop -> no alert, no raise, even in strict mode."""
    monkeypatch.setenv(factory.REQUIRE_FULL_REGISTRY_ENV, "1")

    with patch.object(factory, "_create_agent", side_effect=_ok):
        with caplog.at_level(logging.DEBUG):
            registry = factory.create_agent_registry(
                include_agents=["causal_impact", "gap_analyzer"]
            )

    assert set(registry) == {"causal_impact", "gap_analyzer"}
    assert not any("PARTIAL registry" in r.getMessage() for r in caplog.records)


def test_assert_full_agent_registry_is_the_deploy_gate():
    """The gate helper raises regardless of the env default, so an operator can run
    it against a running container without mutating its environment."""
    with patch.object(factory, "_create_agent", side_effect=_boom):
        with pytest.raises(factory.PartialAgentRegistryError):
            factory.assert_full_agent_registry(include_agents=["causal_impact"])

    with patch.object(factory, "_create_agent", side_effect=_ok):
        registry = factory.assert_full_agent_registry(include_agents=["causal_impact"])

    assert set(registry) == {"causal_impact"}
