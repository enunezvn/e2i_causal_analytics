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
Writers of the channel, and how (re-verified by grep for #2294):

* Merge through this helper under their own kind: ``data_loader``
  (``data_loading``), ``run_schema_validation`` (``schema``),
  ``run_quality_checks`` (``quality_check``), ``run_ge_validation``
  (``ge_validation``), ``detect_leakage`` (``leakage``), ``transform_data``
  (``data_transform``), ``register_features_in_feast`` (``feast_freshness``)
  and ``run_sufficiency_check`` (``data_sufficiency``). Each writes the
  channel on every pass that computes its verdict, so a re-run replaces its
  entry and a resolved condition is retracted; paths that compute NO verdict
  omit the key and leave an earlier pass's entry in place (sufficiency
  ``SKIPPED``; Feast's early returns and ``except`` on a run that does not
  train on Feast-served features — on one that does, they block as
  "unverifiable").
* ``audit_sampling_frame`` copies the incoming list and appends its
  ``sampling_frame_drift:`` entry. It runs once (upstream of the QC retry
  edge), so it cannot duplicate or go stale.
* ``leakage_remediation`` does NOT write the channel (#2294). Its
  ``leakage:`` entries are rebuilt by the ``detect_leakage`` recheck that
  ``graph._route_after_leakage_remediation`` now takes after EVERY applied
  pass; it used to retract entries by matching feature names in free text.
* ``finalize_output`` echoes the channel it gated on.

NOT fixed here, and why: ``adaptive_validity_check`` writes no entry. It
escalates ``leakage_severity`` (routing to remediation) and ``finalize_output``
reads no severity, so a feature it still flags after the final recheck, or one
whose Layer-3 scoring raised, does not block the gate. Layer 3 flags on
SIGNIFICANCE (z > 5 sigma over a permutation null), which a legitimately
predictive feature clears at production n — the reason the FDR and delta-AUC
effect floor exist — and its documented role is to route features to
remediation review, not to gate. Making it gate would change what a Layer-3
flag means, at a false-block rate that cannot be measured without the real
cohorts; that is an owner decision, raised in the #2294 PR (a Layer-1 manifest
violation, which is definitional rather than statistical, is the strongest
candidate to block).

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
    "KIND_DATA_SUFFICIENCY",
    "KIND_DATA_TRANSFORM",
    "KIND_FEAST_FRESHNESS",
    "KIND_GE_VALIDATION",
    "KIND_LEAKAGE",
    "KIND_QUALITY_CHECK",
    "KIND_SCHEMA_VALIDATION",
    "KIND_SEPARATOR",
    "merge_blocking_issues",
    "tag_blocking_issue",
]

#: Separator between an entry's kind and its message. Matches the shape
#: ``sampling_frame_audit`` has emitted since ``5749b974c``.
KIND_SEPARATOR = ": "

KIND_QUALITY_CHECK = "quality_check"
KIND_GE_VALIDATION = "ge_validation"
KIND_DATA_LOADING = "data_loading"
KIND_DATA_TRANSFORM = "data_transform"
KIND_LEAKAGE = "leakage"
KIND_SCHEMA_VALIDATION = "schema"
#: ``sufficiency_check`` has emitted ``"data_sufficiency: ..."`` since PR #462;
#: the kind reuses that prefix so its entries are unchanged on the wire.
KIND_DATA_SUFFICIENCY = "data_sufficiency"
KIND_FEAST_FRESHNESS = "feast_freshness"


def tag_blocking_issue(kind: str, message: str) -> str:
    """Prefix ``message`` with its producing node's ``kind``."""
    return f"{kind}{KIND_SEPARATOR}{message}"


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
