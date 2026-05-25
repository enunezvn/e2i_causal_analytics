"""Issue #501 — M-structure structural-remediation gate (plan §8.4).

Unit tests for the hoisted role→remediation constants and the pure
``apply_structural_remediation_gate`` helper. The gate operates on the REACHABLE
remediation seam (collider narrows remediation to {drop}); severity is never
touched (the info→moderate route is unreachable for intra-LEAK-role
disagreements — codex iter-0 HIGH-1).
"""

from __future__ import annotations

import pytest

from src.data.kg.ensemble_voter import (
    ROLE_DEFAULT_REMEDIATION,
    ROLE_VALID_REMEDIATIONS,
    _role_to_remediation,
    apply_structural_remediation_gate,
    structural_gate_enabled,
)

ALL_ROLES = ("mediator", "descendant", "collider", "ancestor", "confounder", "instrument")
ALL_REMEDIATIONS = ("window", "transform", "drop", "keep_with_caveat", "keep", "review", None)


def test_role_to_remediation_map_hoist_is_behavior_identical() -> None:
    """After hoisting the maps to module constants, ``_role_to_remediation``
    returns the SAME output for every (role, llm_remediation) pair as the
    pre-refactor function-local dicts.

    The pre-refactor behaviour is reconstructed inline (the exact dicts that
    lived in the function) and compared cell-by-cell.
    """
    pre_default = {
        "mediator": "window",
        "descendant": "drop",
        "collider": "drop",
        "ancestor": "keep_with_caveat",
        "confounder": "keep_with_caveat",
        "instrument": "keep_with_caveat",
    }
    pre_valid = {
        "mediator": frozenset({"window", "transform", "drop"}),
        "descendant": frozenset({"drop", "transform"}),
        "collider": frozenset({"drop"}),
        "ancestor": frozenset({"keep_with_caveat", "keep"}),
        "confounder": frozenset({"keep_with_caveat", "keep"}),
        "instrument": frozenset({"keep_with_caveat", "keep"}),
    }
    # The hoisted constants must equal the pre-refactor dicts exactly.
    assert ROLE_DEFAULT_REMEDIATION == pre_default
    assert ROLE_VALID_REMEDIATIONS == pre_valid

    for role in ALL_ROLES:
        for rem in ALL_REMEDIATIONS:
            got = _role_to_remediation(role, rem)  # type: ignore[arg-type]
            # Reconstruct pre-refactor logic.
            default = pre_default[role]
            if rem is None:
                expected = default
            elif rem in pre_valid[role]:
                expected = rem
            else:
                expected = default
            assert got == expected, f"({role}, {rem}) -> {got}, expected {expected}"


def test_structural_gate_collider_narrows_remediation_to_drop(monkeypatch) -> None:
    """structural=collider, llm=mediator, LLM proposed window → forced to drop.

    A true collider mis-read by the LLM as a mediator would wrongly permit a
    permissive window/transform; the structural role narrows it to {drop}.
    """
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    assert structural_gate_enabled() is True
    override = apply_structural_remediation_gate(
        structural_role="collider",
        llm_role="mediator",
        current_remediation="window",
        llm_remediation="window",
    )
    assert override == "drop"


def test_structural_gate_descendant_vs_collider_llm_widens_safely(monkeypatch) -> None:
    """structural=descendant, llm=collider, LLM proposed transform → permitted.

    A transformable descendant must NOT be over-restricted: transform is in
    descendant's valid set {drop, transform}. The override changes the
    remediation from the collider's drop back to the descendant-permitted
    transform.
    """
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    override = apply_structural_remediation_gate(
        structural_role="descendant",
        llm_role="collider",
        current_remediation="drop",
        llm_remediation="transform",
    )
    assert override == "transform"


def test_structural_gate_no_override_when_roles_agree(monkeypatch) -> None:
    """structural == llm → no disagreement → no override (None)."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    override = apply_structural_remediation_gate(
        structural_role="collider",
        llm_role="collider",
        current_remediation="drop",
        llm_remediation="drop",
    )
    assert override is None


def test_structural_gate_no_override_when_role_absent(monkeypatch) -> None:
    """Missing structural OR llm role → no override."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    assert (
        apply_structural_remediation_gate(
            structural_role=None,
            llm_role="mediator",
            current_remediation="window",
            llm_remediation="window",
        )
        is None
    )
    assert (
        apply_structural_remediation_gate(
            structural_role="collider",
            llm_role=None,
            current_remediation="window",
            llm_remediation="window",
        )
        is None
    )


def test_structural_gate_no_op_when_override_equals_current(monkeypatch) -> None:
    """If the computed remediation equals the current one, return None (no-op)."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    # structural=collider forces drop; current already drop → no change.
    override = apply_structural_remediation_gate(
        structural_role="collider",
        llm_role="mediator",
        current_remediation="drop",
        llm_remediation="drop",
    )
    assert override is None


def test_structural_gate_disabled_by_default() -> None:
    """Without the env var set, the gate is disabled."""
    # (No monkeypatch.setenv — relies on the var being unset in the test env.)
    import os

    if "ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED" in os.environ:
        pytest.skip("env var set externally")
    assert structural_gate_enabled() is False
