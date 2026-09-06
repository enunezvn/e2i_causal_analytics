"""Execution outcome of ``audit_chain_entries`` rows.

The ONE definition of "did this agent run succeed" shared by the agent-health
reader (``/health-score`` -> /system-health) and the analytics readers
(``/analytics`` -> Analytics dashboard, ``/analytics/agents/{name}``).

Why a row's ``validation_passed`` is NOT that definition (2026-09-06): the
audited-node wrapper (``src.agents.base.audit_chain_mixin``) records under that
column whatever a node returns as ``validation_passed`` / ``overall_robust`` —
the heterogeneous optimizer's EconML<->CausalML cross-library agreement, or
causal_impact's refutation verdict. Those are SCIENTIFIC results about the
data, and downstream nodes returning ``{**state}`` re-record the same verdict
once per node. Counting them as failed invocations made /system-health warn
"heterogeneous_optimizer has low success rate (89.7%)" over a 30-day window in
which no node of any agent raised. Verdicts belong on the analysis pages
(Library Validation / Refutation cards), where they already surface.

Execution failure has exactly one marker: the wrapper writes an
``action_type = "<node>_error"`` row when a node raises. And the unit of an
"invocation" is the workflow run (``workflow_id``), not the row — a run writes
one genesis row plus one row per node.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

ERROR_ACTION_SUFFIX = "_error"


def is_execution_failure(entry: Mapping[str, Any]) -> bool:
    """True when the row records a node that RAISED (``<node>_error``).

    A ``validation_passed = False`` on an ordinary node row is a verdict about
    the analysis, never an execution failure.
    """
    return str(entry.get("action_type") or "").endswith(ERROR_ACTION_SUFFIX)


class WorkflowOutcomeTally:
    """Count workflow runs (not rows) and how many of them failed.

    Rows carrying a ``workflow_id`` are grouped into one run each; a run failed
    when ANY of its rows is an error row, counted once however many rows it
    wrote. Legacy rows without a ``workflow_id`` (pre-instrumentation) have no
    other unit, so each such row counts as one run.

    ``recent`` counts runs with at least one row inside the caller's recency
    window (``add(..., recent=True)``) — for "invocations in the last 24h".
    """

    def __init__(self) -> None:
        self._ids: set[Any] = set()
        self._failed_ids: set[Any] = set()
        self._recent_ids: set[Any] = set()
        self._legacy_total = 0
        self._legacy_failed = 0
        self._legacy_recent = 0

    def add(self, entry: Mapping[str, Any], *, recent: bool = False) -> None:
        failed = is_execution_failure(entry)
        wid = entry.get("workflow_id")
        if wid:
            self._ids.add(wid)
            if failed:
                self._failed_ids.add(wid)
            if recent:
                self._recent_ids.add(wid)
            return
        self._legacy_total += 1
        if failed:
            self._legacy_failed += 1
        if recent:
            self._legacy_recent += 1

    @property
    def total(self) -> int:
        return len(self._ids) + self._legacy_total

    @property
    def failed(self) -> int:
        return len(self._failed_ids) + self._legacy_failed

    @property
    def successful(self) -> int:
        return self.total - self.failed

    @property
    def recent(self) -> int:
        return len(self._recent_ids) + self._legacy_recent

    @property
    def success_rate(self) -> Optional[float]:
        """successful / total, or None when nothing ran (unmeasured, not 0)."""
        return self.successful / self.total if self.total else None
