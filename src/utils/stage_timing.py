"""Per-request agent stage-timing ledger (#1475).

#1454 / PR #1471 attributed chat latency down to the chatbot graph's node
boundary; the orchestrator node was still a ~14.9s warm black box. This module
is the seam that opens it WITHOUT inventing a parallel timing wrapper: the
shared ``audited_node`` wrapper (src/agents/base/audit_chain_mixin.py) already
times every orchestrator graph node with ``time.perf_counter()`` — it simply
records ``{agent_name}.{node_name}`` wall time here as well, whenever a ledger
is active.

The ledger is a CONTEXTVAR, activated by the chatbot's ``orchestrator_node``
around ``orchestrator.run`` and read back into the trace context afterwards:

* per-request timings must never ride a graph state channel — the Redis
  checkpointer would serialize and replay them across turns (bug class #1442);
* ``contextvars`` propagate into ``asyncio`` tasks (``create_task`` copies the
  context), so stages recorded by DISPATCHED agents' own audited nodes land on
  the same request's ledger, name-spaced by their agent name;
* with no ledger active (every non-chat caller of any audited graph),
  ``record_stage_wall_time`` is a no-op — zero behavior change elsewhere.

This module deliberately imports nothing from ``src`` so both the agents layer
and the API layer can use it without layering inversions.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Dict, Optional, Tuple

_active_stage_ledger: contextvars.ContextVar[Optional[Dict[str, float]]] = contextvars.ContextVar(
    "agent_stage_ledger", default=None
)

# codex iter-1 LOW: today every recording for a given ledger lands on that
# graph's event-loop thread (the audited_node wrapper records in its coroutine
# finally, AFTER any to_thread hop returns) — but context propagation means a
# worker thread running its own loop COULD share the ledger object. The
# accumulation is a read-modify-write, so guard it. One process-wide lock is
# fine: the critical section is sub-microsecond and uncontended in practice.
_record_lock = threading.Lock()


def activate_stage_ledger() -> Tuple[Dict[str, float], "contextvars.Token"]:
    """Activate a fresh ledger for the current context.

    Returns the ledger dict (the caller keeps a direct reference — recordings
    from child tasks land on this same object) and the token for
    :func:`deactivate_stage_ledger`.
    """
    ledger: Dict[str, float] = {}
    token = _active_stage_ledger.set(ledger)
    return ledger, token


def deactivate_stage_ledger(token: "contextvars.Token") -> None:
    """Restore the previously active ledger (usually ``None``)."""
    _active_stage_ledger.reset(token)


def get_active_stage_ledger() -> Optional[Dict[str, float]]:
    """The ledger active in this context, or ``None``."""
    return _active_stage_ledger.get()


def record_stage_wall_time(stage: str, duration_ms: float) -> None:
    """Accumulate ``duration_ms`` against ``stage`` on the active ledger.

    Stage names are ``{agent_name}.{node_name}`` for graph nodes (recorded by
    ``audited_node``) or bare labels for non-graph legs (e.g.
    ``orchestrator.memory_read``, ``get_orchestrator``). Repeat visits
    accumulate, mirroring ``ChatbotTraceContext.node_wall_ms``. No-op when no
    ledger is active. Callers must measure with ``time.perf_counter()`` — the
    consumer subtracts these from a perf_counter total, and a wall-clock
    source would fabricate or hide untimed overhead.
    """
    ledger = _active_stage_ledger.get()
    if ledger is not None:
        with _record_lock:
            ledger[stage] = ledger.get(stage, 0.0) + duration_ms


__all__ = [
    "activate_stage_ledger",
    "deactivate_stage_ledger",
    "get_active_stage_ledger",
    "record_stage_wall_time",
]
