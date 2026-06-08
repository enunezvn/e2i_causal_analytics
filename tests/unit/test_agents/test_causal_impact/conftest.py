"""Causal-impact unit-test conftest.

``CausalImpactAgent.run()`` now contributes results to the tri-memory architecture
(episodic 1536-dim + semantic CausalPath) on success (#788). That is a real, networked
side effect (OpenAI embed + Supabase insert + FalkorDB write). The many synthetic
``agent.run()`` unit tests in this directory care about the ANALYSIS, not memory — left
unguarded they would write to the shared dev store on every run (non-hermetic, slow, and
they would pollute the very episodic/CausalPath counts #785 measures).

This autouse fixture no-ops the run()-triggered memory contribution by default, keeping
the unit sweep hermetic. The dedicated wiring tests
(``test_episodic_wiring_788.py``) override ``_contribute_to_memory`` per-test to assert on
it, and the faithful end-to-end proof lives in
``tests/integration/test_causal_impact_episodic_embedding_788.py`` (real embed + real
Supabase round-trip), so the real behavior is verified without polluting unit runs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agents.causal_impact.agent import CausalImpactAgent


@pytest.fixture(autouse=True)
def _neutralize_run_memory_contribution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Autouse: make ``CausalImpactAgent.run()`` not write to memory in unit tests.

    Per-test ``patch.object(agent, "_contribute_to_memory", ...)`` (the wiring tests)
    shadows this instance-side during the test, so assertions on the contribution still
    work; everything else gets a harmless no-op.
    """
    monkeypatch.setattr(
        CausalImpactAgent,
        "_contribute_to_memory",
        AsyncMock(return_value=None),
        raising=True,
    )
