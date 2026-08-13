"""Shared exception types for the tool composer reference/input contract (#1573).

These live in their own module so both the executor and the tool
registrations can import them without a cycle: ``executor`` imports
``tool_registrations`` at module level (for the ``@composable_tool``
registration side-effects), so neither of those modules can host a symbol
the other needs.
"""

from __future__ import annotations


class ReferenceResolutionError(Exception):
    """A plan reference (``$step_X.field`` / ``$context.field``) cannot be resolved.

    Raised by the executor's reference resolver when the planner emitted a
    reference to an unknown source (e.g. the invented ``$dataset``) or to a
    field the referenced output does not carry. This is a PLAN defect, not a
    tool failure: the referencing step is deterministically doomed, so the
    executor fails it fast — with this error's message as the explicit,
    synthesis-visible reason — instead of degrading to a silent ``None``
    (issue #1573, q08 ``NoneType * float`` crash).
    """

    def __init__(self, reference: str, reason: str):
        self.reference = reference
        self.reason = reason
        super().__init__(f"reference '{reference}' is unresolvable: {reason}")


class ToolInputError(ValueError):
    """A composable tool deterministically rejects its input.

    Raised by a tool when an input value violates the tool's contract in a
    way that no retry can fix (e.g. ``counterfactual_simulator`` receiving
    ``expected_effect=None``). The executor treats this as non-retryable:
    the step fails once, with the tool's stated reason, instead of being
    retried identically (#1573 acceptance: no ``NoneType`` retry loops).
    """
