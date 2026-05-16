"""Centralized self-dispatch strip helper for Issue #251 F1.

The orchestrator routes to OTHER agents and must never appear in either:

* the dispatch plan emitted by :class:`RouterNode`
* the ``agents_dispatched`` list serialized to API consumers
* any ``agent_used`` field surfaced to clients

A single source of truth (``SELF_AGENT_NAME``) is exposed here so all
producers and consumers of agent-name strings agree on the forbidden
literal. This sidesteps the codex MED-1 finding that
``_build_output`` (``agent.py:216``) was rebuilding ``agents_dispatched``
from ``agent_results.keys()`` *after* the router's strip ran, and that
the chatbot serializers at ``chatbot_graph.py:1001`` and
``chatbot_tools.py:1072`` passed the raw value through without
re-applying the guard.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

logger = logging.getLogger(__name__)

# Issue #251 F1: invariant — the orchestrator routes to OTHER agents.
# This name must never appear in ``dispatch_plan`` or ``agents_dispatched``
# nor be returned from any API ``agent_used`` field.
SELF_AGENT_NAME = "orchestrator"

# Recognisable marker for the F2 degraded path. When the orchestrator
# is unavailable / threw / returned empty dispatch, callers must NOT
# substitute the API-default route (which maps GENERAL→"orchestrator")
# and must instead surface this string.
SELF_DEGRADED_MARKER = "orchestrator_degraded"


def strip_self_dispatch(
    agent_names: Iterable[str],
    *,
    context: str = "unknown",
) -> List[str]:
    """Remove ``SELF_AGENT_NAME`` from a list of agent-name strings.

    Logs a WARNING when an entry is stripped so the upstream caller that
    produced the F1-invariant violation can be tracked down.

    Parameters
    ----------
    agent_names:
        Any iterable yielding agent-name strings.
    context:
        Free-form caller tag for the warning log line (e.g.
        ``"agent._build_output"``).

    Returns
    -------
    A list with all ``SELF_AGENT_NAME`` entries removed, order preserved.
    """
    cleaned: List[str] = []
    stripped: List[str] = []
    for name in agent_names:
        if name == SELF_AGENT_NAME:
            stripped.append(name)
            continue
        cleaned.append(name)
    if stripped:
        logger.warning(
            "Stripped self-dispatch entries from agents list at %s: %s. "
            "Issue #251 F1 invariant violation — investigate upstream producer.",
            context,
            stripped,
        )
    return cleaned


def is_self_dispatch(agent_name: object) -> bool:
    """Return True iff ``agent_name`` is the forbidden self literal.

    Use at boundaries where a single string (not a list) crosses out of
    orchestrator-owned code (e.g. ``agent_used`` selection in
    ``api/routes/cognitive.py``).
    """
    return agent_name == SELF_AGENT_NAME
