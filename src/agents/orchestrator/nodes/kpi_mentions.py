"""Deciding what a KPI mention MEANS before any of it is masked (#2114 codex r9).

The multi-KPI veto masks the recognized span and rescans; whatever is still found
is a SECOND metric and the ask fails closed. Task 11a made that masking cover
every mention the resolved KPI owns, which fixed a false refusal on a repeated
panel phrase — and introduced two regressions that failed OPEN, because

    MASKING IS DESTRUCTIVE, AND IT DESTROYS THE EVIDENCE LATER CHECKS NEED.

11a decided everything from the FIRST mention and then erased them all. Two
per-occurrence facts were gone by the time anything wanted them:

* **who OWNS this occurrence** — "trx share" (WS3-BI-008) was masked out of the
  MIDDLE of "trx share panel" (WS3-BI-014), leaving an orphan "panel" the
  scanner could not recognise, so a genuine two-KPI ask answered with ONE
  number. Fixed in :func:`~src.services.kpi_resolution.vocabulary_occurrences`,
  which settles ownership longest-wins on the INTACT string first.
* **what GOVERNS this occurrence** — the #1475 head guards ran on the first
  match only, so "What is NRx panel and the cost of NRx panel?" masked the
  later mention as a redundant repeat and answered a question nobody asked.
  Fixed here: every owned occurrence is head-checked, not just the first.

Both failures were worse in kind than the one 11a fixed. That one failed CLOSED
— a refusal the user did not deserve. These returned a plausible value for a
question that was never asked.

Lives beside ``dispatcher`` rather than inside it because that module is
ratchet-pinned (tests/unit/test_tests_meta/test_module_size_ratchet.py).
"""

from __future__ import annotations

from typing import Optional


def owned_mask(normalized_query: str, kpi_id: str, start: int, end: int) -> str:
    """Blank every span ``kpi_id`` owns, length-preserving.

    Ownership comes from the longest-wins occurrence map over the intact string,
    so a phrase belonging to ANOTHER KPI is never carved out from under it.
    """
    from src.services.kpi_resolution import mask_spans, owned_mention_spans

    return mask_spans(normalized_query, owned_mention_spans(normalized_query, kpi_id, start, end))


def masked_or_refusal(normalized_query: str, kpi_id: str, start: int, end: int) -> Optional[str]:
    """The head-checked mask for the VALUE-lookup path, or ``None`` to refuse.

    A repeated mention of the same KPI binds only when the repeat is genuinely
    REDUNDANT. "cost of X" and "X drivers" are not redundant: they carry a
    sub-ask a bare value does not answer. So every owned occurrence gets the
    same two #1475 guards the first one gets, and any one of them refusing
    refuses the whole ask.

    The dispatcher's head helpers are imported lazily here because dispatcher
    imports this module — the cycle is real, and every import in that module is
    function-local for the same reason.
    """
    from src.agents.orchestrator.nodes.dispatcher import (
        _CAUSAL_OF_HEADS,
        _VALUE_OF_HEADS,
        _kpi_governing_of_head,
        _kpi_right_head,
    )
    from src.services.kpi_resolution import mask_spans, owned_mention_spans

    spans = owned_mention_spans(normalized_query, kpi_id, start, end)
    for span_start, span_end in spans:
        of_head = _kpi_governing_of_head(normalized_query, span_start)
        if of_head is not None and of_head not in _VALUE_OF_HEADS:
            return None
        if _kpi_right_head(normalized_query, span_end) in _CAUSAL_OF_HEADS:
            return None
    return mask_spans(normalized_query, spans)
