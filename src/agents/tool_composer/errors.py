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


class PlanArgumentError(TypeError):
    """A step's PLANNED arguments cannot be bound to its tool's signature (#2045).

    The sibling of :class:`ReferenceResolutionError`: there the planner named a
    source that does not exist, here it omitted (or misnamed) arguments the tool
    declares as required. Both are defects of the PLAN, detectable from the plan
    and the signature alone, before the tool is ever called — so the executor
    fails the step once with an explicit reason, does not retry, and does not
    charge the tool's circuit breaker.

    Before this type existed the mismatch surfaced as the plain ``TypeError``
    CPython raises at call time, which is indistinguishable from a tool's own
    internal ``TypeError``. It therefore landed in the executor's generic retry
    arm: measured live on image ``dde03e0b9`` (Kisqali, tool_composer path),
    three ``gap_calculator`` steps each missing ``metric``/``entity_type``/
    ``entities`` were re-dispatched three times apiece — nine doomed calls, each
    holding a bounded heavy-compute slot for a sync tool — and the nine recorded
    failures opened the breaker, which then blocked a LATER, well-formed
    ``gap_calculator`` step that would have succeeded.

    Subclasses ``TypeError`` deliberately: it IS the failure Python would raise,
    just raised earlier, with the missing parameters named, and off the retry
    path. Anything that already handles the call-time ``TypeError`` keeps working.
    """

    def __init__(
        self,
        tool_name: str,
        missing: tuple[str, ...] = (),
        unexpected: tuple[str, ...] = (),
        supplied: tuple[str, ...] = (),
    ):
        self.tool_name = tool_name
        self.missing = missing
        self.unexpected = unexpected
        self.supplied = supplied
        problems = []
        if missing:
            problems.append(f"missing required argument(s) {_quoted(missing)}")
        if unexpected:
            problems.append(f"unexpected argument(s) {_quoted(unexpected)}")
        super().__init__(
            f"plan defect: tool '{tool_name}' cannot be called with the planned "
            f"arguments — {'; '.join(problems)}. The plan supplied: "
            f"{_quoted(supplied) if supplied else 'no arguments'}"
        )


def _quoted(names: tuple[str, ...]) -> str:
    return ", ".join(f"'{n}'" for n in names)
