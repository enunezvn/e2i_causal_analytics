"""Ownership-tagged merging for the ``warnings`` state channel (#2290).

``DataPreparerState.warnings`` is ``Optional[List[Dict[str, Any]]]`` with no
reducer, so — exactly like ``blocking_issues`` (#2283) — the last node to
write it replaces the whole channel. The same contract applies, for the same
reasons (see ``blocking_issues.py``): no ``operator.add`` (re-entrant nodes
would duplicate their entries and nothing could ever be retracted), and no
plain ``incoming + own`` (the QC retry edge re-runs ``run_quality_checks``).

Entries are dicts, so ownership cannot ride on a ``"<kind>: "`` string prefix;
it is an explicit ``kind`` field instead, using the same kind names as
``blocking_issues.py``. Entries without a ``kind`` are treated as foreign and
preserved.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

__all__ = ["WARNING_KIND_KEY", "merge_warnings"]

#: Field on each warning dict naming the node that produced it.
WARNING_KIND_KEY = "kind"


def merge_warnings(
    incoming: Optional[Iterable[Dict[str, Any]]],
    own: Iterable[Dict[str, Any]],
    *,
    kind: str,
) -> List[Dict[str, Any]]:
    """Replace this node's own warnings, preserve every other node's.

    Args:
        incoming: ``state["warnings"]`` as the node received it (``None`` is empty).
        own: the warning dicts this node computed on this pass. They are
            COPIED before tagging, so a dict shared with another channel
            (``quality_checker`` puts the same result dicts in
            ``expectation_results``) is not mutated.
        kind: this node's kind, e.g. ``blocking_issues.KIND_QUALITY_CHECK``.

    Returns:
        A new list, never ``None``.
    """
    preserved = [w for w in (incoming or []) if w.get(WARNING_KIND_KEY) != kind]
    return preserved + [{**w, WARNING_KIND_KEY: kind} for w in own]
