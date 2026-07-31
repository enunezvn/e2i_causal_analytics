"""Routing-telemetry aggregation + threshold-proposal maths (#1341 Phases 2-3).

Pure, deterministic, DB-free functions over labeled ``classification_logs`` rows.
Phase 1 (PR #1342) populates ``was_correct`` / ``correct_pattern`` /
``feedback_notes``; these turn that raw label stream into:

- **Phase 2 — ``compute_run_metrics``**: standing safety telemetry the nightly
  labeler emits each run (per-pattern accuracy with counts, engagement rate,
  abstention correctness, LLM-layer share, label-source breakdown). This is the
  signal any future active-mode promotion would be judged against.
- **Phase 3 — ``compute_threshold_proposals``**: offline ``MIN_ACTIVE_CONFIDENCE``
  retune *proposals* compiled from the accumulated labeled set — expected
  engagement flips and their judged accuracy at each candidate floor. Proposals
  only; never applied to routing (authority changes stay human-gated, #1341).

Kept as pure functions (no repo, no client, no env) so the maths is unit-tested
red-first and reused by both the nightly task (Phase 2) and the manual proposal
script (Phase 3). ``pipeline-vs-legacy`` agreement is deliberately absent: the
shadow writer (PR #1330) records only the pipeline's decision, so legacy-vs-
pipeline is not recoverable from this table — the computable safety signal is
pipeline-vs-judge agreement (overall accuracy below).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

# The router's active-mode floor (RouterNode.MIN_ACTIVE_CONFIDENCE). Duplicated
# as a default here to keep this module import-light (no orchestrator/langgraph
# pull-in); the value is pinned identical by
# tests/unit/test_tasks/test_routing_metrics.py::test_default_floor_matches_router.
DEFAULT_ACTIVE_FLOOR = 0.5

ABSTAIN_PATTERN = "CLARIFICATION_NEEDED"

VALID_PATTERNS = (
    "SINGLE_AGENT",
    "PARALLEL_DELEGATION",
    "TOOL_COMPOSER",
    "CLARIFICATION_NEEDED",
)

# Label channels written into feedback_notes.source by the Phase-1 labeler.
LABEL_SOURCES = (
    "explicit_feedback",
    "implicit_outcome",
    "llm_judge",
    "llm_judge_abstain",
)


def _label_source(row: Dict[str, Any]) -> Optional[str]:
    """Extract the ``source`` field the Phase-1 labeler stored in feedback_notes."""
    notes = row.get("feedback_notes")
    if not notes:
        return None
    if isinstance(notes, dict):
        return notes.get("source")
    try:
        parsed = json.loads(notes)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed.get("source") if isinstance(parsed, dict) else None


def _is_engaged(row: Dict[str, Any], floor: float) -> bool:
    """Whether the pipeline would take active-mode routing authority for a row.

    Mirrors RouterNode._dispatch_from_classification's abstain rule: a
    CLARIFICATION_NEEDED pattern or a confidence below the floor abstains.
    """
    if row.get("routing_pattern") == ABSTAIN_PATTERN:
        return False
    try:
        return float(row.get("confidence") or 0.0) >= floor
    except (TypeError, ValueError):
        return False


def compute_run_metrics(
    rows: Sequence[Dict[str, Any]], *, active_floor: float = DEFAULT_ACTIVE_FLOOR
) -> Dict[str, Any]:
    """Aggregate a window of classification_logs rows into run telemetry.

    ``rows`` may be labeled or not; each needs routing_pattern, confidence,
    used_llm_layer, was_correct, feedback_notes. All rates are computed over the
    denominator they are meaningful for (accuracy over LABELED rows only, so an
    unlabeled backlog never dilutes it).
    """
    total = len(rows)
    per_pattern: Dict[str, Dict[str, Any]] = {
        p: {"total": 0, "correct": 0, "incorrect": 0, "awaiting": 0, "accuracy_pct": None}
        for p in VALID_PATTERNS
    }
    labeled = correct = incorrect = 0
    engaged = llm_layer = 0
    abstain_total = abstain_correct = abstain_incorrect = 0
    label_sources: Dict[str, int] = dict.fromkeys(LABEL_SOURCES, 0)

    for row in rows:
        pattern = row.get("routing_pattern")
        bucket = per_pattern.get(pattern) if pattern in per_pattern else None
        if bucket is not None:
            bucket["total"] += 1

        was_correct = row.get("was_correct")
        if was_correct is True:
            labeled += 1
            correct += 1
            if bucket is not None:
                bucket["correct"] += 1
        elif was_correct is False:
            labeled += 1
            incorrect += 1
            if bucket is not None:
                bucket["incorrect"] += 1
        else:
            if bucket is not None:
                bucket["awaiting"] += 1

        if _is_engaged(row, active_floor):
            engaged += 1
        if row.get("used_llm_layer") is True:
            llm_layer += 1

        if pattern == ABSTAIN_PATTERN:
            abstain_total += 1
            if was_correct is True:
                abstain_correct += 1
            elif was_correct is False:
                abstain_incorrect += 1

        source = _label_source(row)
        if source in label_sources:
            label_sources[source] += 1

    for bucket in per_pattern.values():
        denom = bucket["correct"] + bucket["incorrect"]
        bucket["accuracy_pct"] = round(100.0 * bucket["correct"] / denom, 2) if denom else None

    judged = correct + incorrect
    return {
        "total": total,
        "labeled": labeled,
        "awaiting_feedback": total - labeled,
        "overall_accuracy_pct": round(100.0 * correct / judged, 2) if judged else None,
        "engagement_rate": round(engaged / total, 4) if total else None,
        "active_floor": active_floor,
        "llm_layer_share": round(llm_layer / total, 4) if total else None,
        "abstention": {
            "total": abstain_total,
            "judged_correct": abstain_correct,
            "judged_incorrect": abstain_incorrect,
            # Of the abstentions the judge ruled on, the share that were RIGHT to
            # abstain (genuinely ambiguous). Low -> the classifier over-abstains
            # (the #1337 finding), the standing safety signal for active-mode.
            "correctness_pct": (
                round(100.0 * abstain_correct / (abstain_correct + abstain_incorrect), 2)
                if (abstain_correct + abstain_incorrect)
                else None
            ),
        },
        "per_pattern": per_pattern,
        "label_sources": label_sources,
    }


def compute_threshold_proposals(
    rows: Sequence[Dict[str, Any]],
    *,
    current_floor: float = DEFAULT_ACTIVE_FLOOR,
    candidates: Optional[Sequence[float]] = None,
    min_evidence: int = 20,
) -> Dict[str, Any]:
    """Offline MIN_ACTIVE_CONFIDENCE retune proposals from the labeled set.

    For each candidate floor, count the rows whose ENGAGEMENT decision would
    flip vs ``current_floor`` and, among the LABELED flips, how many the judge
    ruled correctly routed. A lower floor engages more rows: the proposal is
    worthwhile only if those newly-engaged rows are mostly judged-correct.

    Returns a data-only artifact (per-candidate evidence + at most one
    ``recommended`` floor). It NEVER mutates routing config — Phase 3 surfaces a
    proposal a human turns into a PR; authority changes stay human-gated (#1341).
    ``recommended`` is set only when a candidate has >= ``min_evidence`` labeled
    flips AND improves judged accuracy of the engaged set without lowering it.
    """
    if candidates is None:
        candidates = (0.40, 0.45, 0.55, 0.60, 0.65)

    labeled = [r for r in rows if r.get("was_correct") in (True, False)]
    # Baseline engaged/correct at the current floor over labeled rows.
    base_engaged = [r for r in labeled if _is_engaged(r, current_floor)]
    base_correct = sum(1 for r in base_engaged if r.get("was_correct") is True)
    base_acc = (100.0 * base_correct / len(base_engaged)) if base_engaged else None

    proposals: List[Dict[str, Any]] = []
    for cand in candidates:
        cand_engaged = [r for r in labeled if _is_engaged(r, cand)]
        cand_correct = sum(1 for r in cand_engaged if r.get("was_correct") is True)
        cand_acc = (100.0 * cand_correct / len(cand_engaged)) if cand_engaged else None
        # Rows that flip engagement between current_floor and cand.
        flips = [r for r in labeled if _is_engaged(r, current_floor) != _is_engaged(r, cand)]
        flip_correct = sum(1 for r in flips if r.get("was_correct") is True)
        proposals.append(
            {
                "candidate_floor": cand,
                "direction": "lower" if cand < current_floor else "raise",
                "engaged_n": len(cand_engaged),
                "engaged_accuracy_pct": round(cand_acc, 2) if cand_acc is not None else None,
                "labeled_flips": len(flips),
                "flips_judged_correct": flip_correct,
                "flips_judged_incorrect": len(flips) - flip_correct,
                "accuracy_delta_pct": (
                    round(cand_acc - base_acc, 2)
                    if (cand_acc is not None and base_acc is not None)
                    else None
                ),
            }
        )

    recommended = _pick_recommended(proposals, base_acc, min_evidence)
    return {
        "current_floor": current_floor,
        "baseline_engaged_n": len(base_engaged),
        "baseline_accuracy_pct": round(base_acc, 2) if base_acc is not None else None,
        "labeled_rows": len(labeled),
        "min_evidence": min_evidence,
        "candidates": proposals,
        "recommended_floor": recommended,
        "note": (
            "PROPOSAL ONLY — human-gated. Compiled offline from judged labels; "
            "never auto-applied to routing (RouterNode.MIN_ACTIVE_CONFIDENCE is "
            "unchanged). Insufficient labeled evidence yields no recommendation."
        ),
    }


def _pick_recommended(
    proposals: List[Dict[str, Any]], base_acc: Optional[float], min_evidence: int
) -> Optional[float]:
    """Best candidate that has enough labeled flips and does not worsen accuracy."""
    if base_acc is None:
        return None
    eligible = [
        p
        for p in proposals
        if p["labeled_flips"] >= min_evidence
        and p["accuracy_delta_pct"] is not None
        and p["accuracy_delta_pct"] > 0
    ]
    if not eligible:
        return None
    best = max(eligible, key=lambda p: (p["accuracy_delta_pct"], p["engaged_n"]))
    return float(best["candidate_floor"])
