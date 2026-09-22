"""DoWhy effect identification under the agent's time budget.

``optimize_backdoor`` (dowhy's path-based ``Backdoor`` search) instead of the
default candidate enumeration WHEN there is an adjustment set to find. The
default accepts the full common-cause set on its first candidate and then
re-runs as a minimal-set search from the SMALLEST subset upward; on a
data-built graph the minimal valid set IS the full set, so that pass burns its
100,000-iteration cap of d-separation checks before giving up -- k=12: 0.65 s,
k>=17: the cap, k=77 (Optum biologic persistence, 2026-09-22): 467 s of a
478 s reconstruction, past the agent's 900 s hard cap. Same adjustment set (the
full common-cause set) and a byte-identical estimate on k=8 / k=12 synthetic
frames, 0.01 s at every k measured.

NOT for an EMPTY adjustment set (a validated-RCT / negative-control rebuild):
there the path-based search returns no set at all, ``estimands["backdoor"]`` is
None and DoWhy's estimate carries ``value=None`` (codex r2 HIGH), while the
default search returns the explicit empty set in 0.0 s -- so the empty case
keeps the default. Both shapes are pinned by
``tests/unit/test_agents/test_causal_impact/test_refutation_identify_budget.py``.
"""

from __future__ import annotations

from typing import Any, Sequence


def identify_effect(model: Any, common_causes: Sequence[str]) -> Any:
    """``model.identify_effect`` with the budget-safe identifier for a non-empty set."""
    return model.identify_effect(
        proceed_when_unidentifiable=True,
        optimize_backdoor=bool(common_causes),
    )
