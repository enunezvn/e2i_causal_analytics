"""WS-SYNTH: the dispatcher synthetic opt-in resolver honors the deployment
showcase flag (``E2I_INCLUDE_SYNTHETIC``).

On a synthetic-gold showcase instance synthetic is a *badge*, not a *gate*, so
``_resolve_include_synthetic_opt_in`` must default to include — consistent with
``apply_provenance_filter`` — while staying strict (real-mode default-exclude,
explicit-opt-in respected) when the env is unset so a real-data prod is unchanged.
"""

import pytest

from src.agents.orchestrator.nodes.dispatcher import _resolve_include_synthetic_opt_in


@pytest.mark.unit
def test_resolver_default_real_mode_when_flag_unset(monkeypatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    assert _resolve_include_synthetic_opt_in({}, {}) is False


@pytest.mark.unit
def test_resolver_explicit_opt_in_respected_when_flag_unset(monkeypatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    assert _resolve_include_synthetic_opt_in({}, {"include_synthetic": True}) is True


@pytest.mark.unit
def test_resolver_loose_value_fails_closed_when_flag_unset(monkeypatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    # An ambiguous "false" opt-OUT must stay real-mode (the strict #874 contract).
    assert _resolve_include_synthetic_opt_in({}, {"include_synthetic": "false"}) is False


@pytest.mark.unit
def test_showcase_flag_forces_include_over_explicit_optout(monkeypatch):
    """The deployment showcase flag wins: synthetic-gold powers every dispatch."""
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    assert _resolve_include_synthetic_opt_in({}, {"include_synthetic": False}) is True
    assert _resolve_include_synthetic_opt_in({}, {}) is True
