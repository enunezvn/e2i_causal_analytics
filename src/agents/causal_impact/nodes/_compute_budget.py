"""Budget-guarded submission to the bounded agent-compute pool (#1601).

The causal_impact nodes off-load heavy sync compute to a BOUNDED pool
(``run_in_agent_compute_executor``). A bounded pool *queues*, and that makes a
pre-submit deadline check insufficient on its own: a call can pass the check on
the event loop, then sit in the queue behind another agent-compute task until
its cooperative ``compute_deadline`` has already expired, and start anyway. It
would then hold a scarce pool slot doing work whose turn is already lost —
while the caller's own ``wait_for`` may have cancelled the awaiting coroutine,
leaving the thread running with nothing to return to. That is precisely the
orphaned-uncancellable-thread failure the deadline mechanism exists to prevent.

So the budget is re-checked ON the worker thread, at the instant the callable
actually starts. The pre-submit check is still worth keeping: it fails faster
and avoids occupying a queue slot at all.

This is deliberately NOT an ``asyncio.wait_for`` envelope. ``wait_for`` cancels
the future, never the thread, so it would abandon a running call that keeps
holding the slot. Refusing to *start* is the only bound that a thread-based
pool can actually honour.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class ComputeBudgetExpired(Exception):
    """The cooperative budget lapsed before the pooled call could start.

    A dedicated type so each node can translate it into its own structured
    fail-closed error (``EstimationError`` / ``RefutationError``) without the
    generic ``except Exception`` arms mislabeling a budget refusal as a
    computation failure.
    """


async def run_bounded_with_budget(
    func: Callable[..., T],
    *args: Any,
    budget_deadline: Optional[float],
    **kwargs: Any,
) -> T:
    """Run ``func`` on the bounded agent-compute pool, honouring the budget.

    :param budget_deadline: absolute ``time.monotonic()`` seconds, or ``None``
        for an unbounded run (no caller budget was supplied). Named distinctly
        from ``deadline`` because ``RefutationRunner.run_all_tests`` takes a
        ``deadline`` of its own that must still be forwarded through ``kwargs``.
    :raises ComputeBudgetExpired: when the budget is already spent by the time
        the worker thread picks the call up — i.e. the work never started.
        Anything ``func`` itself raises propagates unchanged.
    """
    # Function-local import (the #1590 / #1598 precedent) keeps the
    # ``src.api.dependencies`` package off the agent modules' import path.
    from src.api.dependencies.compute import run_in_agent_compute_executor

    def _guarded(*inner_args: Any, **inner_kwargs: Any) -> T:
        if budget_deadline is not None and time.monotonic() >= budget_deadline:
            raise ComputeBudgetExpired(
                "compute budget lapsed while the call waited for a bounded "
                "agent-compute slot; refusing to start"
            )
        return func(*inner_args, **inner_kwargs)

    return await run_in_agent_compute_executor(_guarded, *args, **kwargs)
