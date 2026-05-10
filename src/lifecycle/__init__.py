"""Lifecycle-state primitives for gate-relevant code paths.

Plan v4 Gate N2 (codex-rescue MEDIUM-1): the advisory invariant must guard
BEHAVIOR, not SPELLING. A naive "scan for ``advisory_mode: true``" misses
configs that use other names (``mode: advisory``, ``enforcement: false``,
``shadow: true``, ``warn_only: true``, env vars, JSON, TOML, nested YAML,
code constants).

This module exposes:

* :class:`GateLifecycleState` — the canonical 5-value enum that every
  gate-relevant location MUST declare (via YAML key ``lifecycle_state`` or
  Python module constant ``LIFECYCLE_STATE_*``).
* :class:`LifecycleDeclaration` — the Pydantic model that validates a
  gate's lifecycle declaration + optional metadata (calibration window,
  signing reviewer, etc.).
"""

from src.lifecycle.gate_lifecycle import (
    GateLifecycleState,
    LifecycleDeclaration,
)

__all__ = [
    "GateLifecycleState",
    "LifecycleDeclaration",
]
