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


class ToolRefusalError(RuntimeError):
    """A composable tool deterministically REFUSES to produce a result (#1600).

    The distinction from :class:`ToolInputError` is what the tool is objecting
    to. ``ToolInputError`` says *this value is not a legal input* (a ``None``
    where a float is required). ``ToolRefusalError`` says *these inputs are
    structurally fine, but the data they carry cannot answer the question* —
    a single-brand frame asked for a brand-vs-brand gap (#1574), a metric
    column that is entirely null within every group (#1599), a treatment
    column with one class. Both are deterministic over the step's resolved
    inputs, so the executor handles them identically: fail the step ONCE with
    the tool's own reason, and do NOT record it against the circuit breaker
    (a plan/data defect is not a signal about the tool's health, and must not
    open the circuit for other, valid steps that use the same tool).

    **Why this subclasses ``RuntimeError`` rather than reusing
    ``ToolInputError``.** The fail-closed contract of every guard in
    ``tool_registrations`` is documented as ``RuntimeError`` — in the tools'
    own docstrings and ``Raises:`` sections, and pinned by ~35
    ``pytest.raises(RuntimeError)`` assertions. ``ToolInputError`` is a
    ``ValueError``, so converting those guards to it would be a breaking
    change to a published contract in exchange for nothing functional. This
    type keeps the contract exactly as documented while adding the one
    property #1600 needs: non-retryability.

    Raise this ONLY when the refusal is a property of the resolved inputs, so
    that re-running the identical call is futile BY CONSTRUCTION. Failures
    that report the OUTCOME of a computation (a DoWhy pipeline that returned
    ``status='failed'``, a refutation suite that produced no verdict) stay
    plain ``RuntimeError`` and keep retrying: that machinery is genuinely
    stochastic (bootstrap resampling, placebo simulations, no pinned
    ``random_state``), so a second attempt is not futile by construction.
    """
