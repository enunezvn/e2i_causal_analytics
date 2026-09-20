"""Causal-path and refutation-evidence shaping for the chat tools (#2115 extraction).

Three pure helpers, lifted out of ``chatbot_tools.py`` unchanged. They turn a
``causal_paths`` registry row and its refutation rows into the shape a chat answer
carries, and they decide what a FAILED evidence lookup is allowed to look like -- an
explicit ``lookup_failed`` marker, never an empty summary that would read as "no
evidence exists".

Why they moved: ``chatbot_tools.py`` is size-ratchet pinned
(tests/unit/test_tests_meta/test_module_size_ratchet.py), and registering
``forecast_kpi_tool`` there costs lines the pin does not have. The ratchet only ever
moves DOWN, so the honest way to add a tool is to take a coherent cluster out rather
than to raise the pin. This is that cluster: three functions with one subject, no
Supabase access and no dependency on anything else in the module. The pin drops with
them.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _format_causal_path(
    row: Dict[str, Any], refutation_evidence: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Map a ``causal_paths`` registry row onto the chat-facing chain shape.

    ``confidence`` here is the registry's method-attributed ``confidence_level``
    (a real 0-1 causal confidence) — never a retrieval similarity score.

    #1352: ``validation_status`` carries the migration-119 pinned semantics
    ('validated' == "RefutationSuite evidence exists and passed"), and
    ``refutation_evidence`` is the per-path summary from
    :func:`_summarize_refutation_rows` — ``None`` means the evidence lookup
    succeeded and found nothing on record (see
    :func:`_refutation_evidence_entry` for the lookup-failed state).
    """
    return {
        "path_id": row.get("path_id"),
        "cause": row.get("start_node"),
        "effect": row.get("end_node"),
        "via": list(row.get("intermediate_nodes") or []),
        "effect_size": row.get("causal_effect_size"),
        "confidence": row.get("confidence_level"),
        "method": row.get("method_used"),
        "time_lag_days": row.get("time_lag_days"),
        "business_impact_estimate": row.get("business_impact_estimate"),
        "brand": row.get("brand"),
        "validation_status": row.get("validation_status"),
        "refutation_evidence": refutation_evidence,
    }


_REFUTATION_GATES = frozenset({"proceed", "review", "block"})


def _summarize_refutation_rows(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Aggregate one path's ``causal_validations`` rows into a chat summary.

    The block > review > proceed ORDER mirrors
    ``CausalValidationRepository.get_gate_decision``, but the fail-closed
    behaviour below does not: block wins if any row reads block, then review
    if any row reads review, and proceed ONLY when every row is readable (a
    known ``gate_decision`` value); otherwise the gate is ``unknown``. The
    column holds ONLY the refutation vocabulary (proceed / review / block);
    any other value, including NULL, is counted in ``gate_unreadable_rows``
    and never mapped to proceed (#1991 debt 4) — the repository method
    itself still defaults an unreadable value to proceed, which is out of
    scope for this lane.
    ``evidence_is_synthetic`` reads the migration-119 provenance label
    (``details_json.is_synthetic``) so seeded synthetic evidence can never
    masquerade as real RefutationSuite output in an answer.
    """
    if not rows:
        return None

    def _status_count(status: str) -> int:
        return sum(1 for r in rows if r.get("status") == status)

    gates = [r.get("gate_decision") for r in rows]
    known = {g for g in gates if g in _REFUTATION_GATES}
    unreadable = sum(1 for g in gates if g not in _REFUTATION_GATES)
    if "block" in known:
        gate = "block"
    elif "review" in known:
        gate = "review"
    elif known and not unreadable:
        gate = "proceed"
    else:
        # Fail closed: a row we cannot read is not evidence of robustness.
        gate = "unknown"

    confidences = [
        float(r["confidence_score"]) for r in rows if r.get("confidence_score") is not None
    ]

    def _details(r: Dict[str, Any]) -> Dict[str, Any]:
        raw = r.get("details_json")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except (ValueError, TypeError):
                return {}
        return {}

    timestamps = [str(r["created_at"]) for r in rows if r.get("created_at")]
    summary: Dict[str, Any] = {
        "tests_total": len(rows),
        "tests_passed": _status_count("passed"),
        "tests_failed": _status_count("failed"),
        "tests_warning": _status_count("warning"),
        "gate_decision": gate,
        "gate_unreadable_rows": unreadable,
        "confidence_score": (sum(confidences) / len(confidences)) if confidences else None,
        "evidence_is_synthetic": any(bool(_details(r).get("is_synthetic")) for r in rows),
        "latest_test_at": max(timestamps) if timestamps else None,
    }
    if unreadable:
        summary["note"] = (
            f"{unreadable} of {len(rows)} persisted refutation rows carry a gate value "
            "outside proceed/review/block and could not be read; block or review still "
            "wins if any readable row says so, but proceed is never reported while any "
            "row is unreadable — the gate reads 'unknown' instead."
        )
    return summary


def _refutation_evidence_entry(
    path_id: Optional[str], summaries: Optional[Dict[str, Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """Resolve one path's refutation-evidence entry, keeping three states
    honestly distinct:

    * summary dict — evidence rows exist for this path;
    * ``None`` — the lookup succeeded and there is genuinely no refutation
      evidence on record;
    * lookup-failed marker — the evidence query errored (``summaries is
      None``); this must never be presented as absence of evidence.
    """
    if summaries is None:
        return {
            "lookup_failed": True,
            "note": (
                "refutation-evidence lookup unavailable for this answer — "
                "do not read this as 'no evidence exists'"
            ),
        }
    if not path_id:
        return None
    return summaries.get(path_id)
