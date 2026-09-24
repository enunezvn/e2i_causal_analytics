"""Ownership-tagged merging for the ``blocking_issues`` state channel.

``DataPreparerState.blocking_issues`` (``state.py``) is a plain
``Optional[List[str]]``: it has **no reducer**, so LangGraph gives it
``LastValue`` semantics — the last node to write the channel replaces whatever
was there. Any node that returns ``blocking_issues`` therefore owns the whole
channel for that superstep, and must reproduce the entries it did not create.

Issue #2283: ``quality_checker`` started from a fresh local ``[]`` and
``ge_validator`` returned ``None`` on its happy path, so a failed Pandera
schema validation never reached the QC gate and training proceeded with
``gate_passed=True``.

Why not ``Annotated[List[str], operator.add]``
----------------------------------------------
An additive reducer is the obvious fix and the wrong one here. It re-creates
the #2238 / PR #2251 failure mode: nodes that echo state back into an
``operator.add`` channel accumulate duplicates, and a channel that can only
grow can never retract an issue that was subsequently remediated.

Why not a plain ``incoming + own`` merge
----------------------------------------
Most of the writers are **re-entrant**. ``graph.py`` routes ``finalize_output
-> qc_remediation --retry--> run_quality_checks``, and the retry then follows
the *entire* downstream chain again — GE, feature engineering, leakage,
transform, Feast, baseline, sufficiency. The nodes upstream of
``run_quality_checks`` — ``load_data``, ``audit_sampling_frame`` and
``run_schema_validation`` — are the ones that run only once. A plain concatenation would duplicate each re-entrant node's own entries
on the second pass, and — worse — would make them permanently sticky: an issue
that remediation actually fixed would still be in the channel, so the gate
would stay blocked and the remediation loop would be pointless.

Adoption status (be precise about this)
---------------------------------------
``quality_checker``, ``ge_validator``, ``data_loader``, ``data_transformer``
and ``leakage_detector`` route through this helper, and
``leakage_remediation`` prunes by kind. ``feast_registrar`` and
``sufficiency_check`` still concatenate naively; on a QC retry they can
duplicate and fail to retract their own entries. That is a *fail-closed*
defect (a stale blocker keeps the gate shut) of a different class from the
fail-open bug #2283 fixed, and it is tracked separately rather than folded in
here. This module does NOT yet describe every writer.

The contract implemented here
-----------------------------
Each producing node declares a **kind**. On every pass it drops the entries
carrying its own kind (its previous contribution, now recomputed) and keeps
every other entry untouched, then appends its freshly computed issues. The
result is idempotent under re-entry, self-cleaning when an issue is resolved,
and lossless for other nodes' entries.

``nodes/sampling_frame_audit.py`` already used this prefix shape
(``"sampling_frame_drift: ..."``); this module generalises it.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

__all__ = [
    "KIND_DATA_LOADING",
    "KIND_DATA_TRANSFORM",
    "KIND_GE_VALIDATION",
    "KIND_LEAKAGE",
    "KIND_QUALITY_CHECK",
    "KIND_SEPARATOR",
    "merge_blocking_issues",
    "tag_blocking_issue",
    "untag_blocking_issue",
]

#: Separator between an entry's kind and its message. Matches the shape
#: ``sampling_frame_audit`` has emitted since ``5749b974c``.
KIND_SEPARATOR = ": "

KIND_QUALITY_CHECK = "quality_check"
KIND_GE_VALIDATION = "ge_validation"
KIND_DATA_LOADING = "data_loading"
KIND_DATA_TRANSFORM = "data_transform"
KIND_LEAKAGE = "leakage"


def tag_blocking_issue(kind: str, message: str) -> str:
    """Prefix ``message`` with its producing node's ``kind``."""
    return f"{kind}{KIND_SEPARATOR}{message}"


def untag_blocking_issue(kind: str, issue: str) -> Optional[str]:
    """The message inside ``issue`` if it carries ``kind``, else ``None``.

    Callers that match on an entry's CONTENT must use this rather than
    searching the whole tagged string: the kind prefix is part of that string
    and can produce false hits. ``leakage: `` contains ``age``, so a leaked
    feature named ``age`` matched every leakage entry — including unrelated
    ones — when the prune searched the tagged form (codex r2 HIGH on #2283).
    """
    prefix = f"{kind}{KIND_SEPARATOR}"
    return issue[len(prefix) :] if issue.startswith(prefix) else None


def merge_blocking_issues(
    incoming: Optional[Iterable[str]],
    own_messages: Iterable[str],
    *,
    kind: str,
) -> List[str]:
    """Replace this node's own entries, preserve every other node's.

    Args:
        incoming: ``state["blocking_issues"]`` as the node received it. ``None``
            (the channel's initial value) is treated as empty.
        own_messages: the UNTAGGED messages this node computed on this pass.
        kind: this node's kind, e.g. :data:`KIND_QUALITY_CHECK`.

    Returns:
        A new list: ``incoming`` minus this kind's entries, followed by
        ``own_messages`` tagged with ``kind``. Never ``None``, so the caller
        cannot wipe the channel by returning it.
    """
    prefix = f"{kind}{KIND_SEPARATOR}"
    preserved = [issue for issue in (incoming or []) if not issue.startswith(prefix)]
    return preserved + [tag_blocking_issue(kind, message) for message in own_messages]
