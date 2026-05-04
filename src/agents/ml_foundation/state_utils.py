"""Runtime state-contract validation for ml_foundation agents.

Companion to the ``TypedDict`` definitions in each agent's ``state.py``.
Provides a lightweight runtime check that bridges the gap between
``TypedDict(total=False)`` — declaration-only, no runtime enforcement —
and a full pydantic conversion (deferred per ``mypy-type-debt.md``
baseline of 2,676 mypy errors).

Used by ``tests/integration/test_agent_output_contracts.py`` for the 2
light agents (``scope_definer``, ``observability_connector``) where node
invocation is cheap. The 4 heavy agents (``feature_analyzer``,
``model_trainer``, ``model_deployer``, ``model_selector``) are guarded at
the shape level only — see the test module's docstring for the
infrastructure-cost rationale.

Design selected by codex consult 2026-05-04 (agent ``ae2d78db4919ac47e``)
as the cheap middle path between declaration-only TypedDict and a full
pydantic refactor.
"""

from __future__ import annotations

from typing import Any


def validate_state(
    state: dict[str, Any],
    cls: type,
    required_keys: list[str],
) -> None:
    """Assert that a live state dict contains the required keys for ``cls``.

    Raises ``ValueError`` if any required key is missing. ``TypedDict(total=False)``
    silently allows missing fields; this helper turns a missing-key bug into
    a loud failure at runtime.

    Parameters
    ----------
    state
        Live state dict produced by an agent node.
    cls
        The ``TypedDict`` class for the agent state. Only the class name is
        used in error messages; no isinstance check is performed (TypedDicts
        cannot be instantiated, only annotated).
    required_keys
        Keys that MUST be present in ``state`` for the contract to hold.

    Raises
    ------
    ValueError
        If ``state`` is missing one or more keys from ``required_keys``.
    """
    missing = [k for k in required_keys if k not in state]
    if missing:
        raise ValueError(
            f"{cls.__name__} state missing required keys: {missing!r}. "
            f"Present keys: {sorted(state.keys())!r}"
        )
