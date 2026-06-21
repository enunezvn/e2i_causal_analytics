"""T1 — provenance/warnings must not duplicate through the LangGraph state.

Regression (found via a faithful live `/optimize` call): the ``warnings`` channel
used ``operator.add`` while every node returns ``{**state, ...}`` — re-emitting the
whole accumulated list. ``operator.add`` therefore CONCATENATED the accumulated
warnings once per node, doubling a single seeded provenance line 1->2->4->...->32
across the audit_init/formulate/optimize/scenario/project chain. The real response
carried 32 identical "SYNTHETIC DATA:" lines, rendered as a wall of warnings.

The fix is a dedup-union reducer on the ``warnings`` channel: re-emitting the
accumulated list is idempotent, and genuinely-identical warnings collapse to one.
This test exercises the REAL graph (scipy solver, no mocks) end-to-end.
"""

from __future__ import annotations

import pytest

from src.agents.resource_optimizer import build_resource_optimizer_graph

_PROVENANCE = (
    "SYNTHETIC DATA: no real per-entity budget source is wired, so this "
    "optimization ran on synthetic territory_metrics."
)


@pytest.mark.asyncio
async def test_seeded_provenance_warning_is_not_duplicated(state_with_scenarios):
    """A single seeded provenance warning survives the full scenario graph exactly
    once — no operator.add doubling across the {**state}-returning nodes."""
    state = {**state_with_scenarios, "warnings": [_PROVENANCE]}
    graph = build_resource_optimizer_graph()

    result = await graph.ainvoke(state)

    # Sanity: we exercised the real optimize + scenario + project path.
    assert result["status"] == "completed"
    # The provenance line must appear exactly once, not 2/4/.../32 times.
    assert result["warnings"].count(_PROVENANCE) == 1, result["warnings"]
    # And no warning is duplicated.
    assert len(result["warnings"]) == len(set(result["warnings"])), result["warnings"]
