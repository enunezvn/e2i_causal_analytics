"""Unit tests for routing-telemetry aggregation + threshold proposals (#1341 Phases 2-3).

Pure-function tests with hand-computed expected values (no DB, no LLM). The
Phase-2 metrics and Phase-3 proposals are the standing-safety-telemetry and
human-gated-retune halves of the routing learning loop; these pin the maths and,
critically, that computing a proposal NEVER mutates routing config (#1341:
authority changes stay human-gated).
"""

from __future__ import annotations

import json
from datetime import timedelta

from src.tasks.routing_metrics import (
    CLASSIFIER_BASELINE_EPOCH,
    DEFAULT_ACTIVE_FLOOR,
    compute_run_metrics,
    compute_threshold_proposals,
)

_ON_CURRENT_BASELINE = (CLASSIFIER_BASELINE_EPOCH + timedelta(days=1)).isoformat()


def _row(pattern, conf, was_correct, *, used_llm=False, source=None):
    notes = json.dumps({"source": source}) if source else None
    return {
        "routing_pattern": pattern,
        "confidence": conf,
        "used_llm_layer": used_llm,
        "was_correct": was_correct,
        "feedback_notes": notes,
    }


# 7-row window: 6 labeled (1 awaiting), 2 abstentions (1 right, 1 wrong).
_WINDOW = [
    _row("SINGLE_AGENT", 0.9, True, source="explicit_feedback"),
    _row("SINGLE_AGENT", 0.7, False, used_llm=True, source="llm_judge"),
    _row("TOOL_COMPOSER", 0.85, True, used_llm=True, source="llm_judge"),
    _row("CLARIFICATION_NEEDED", 0.0, False, source="llm_judge"),  # over-abstained
    _row("CLARIFICATION_NEEDED", 0.3, True, source="llm_judge"),  # correct abstain
    _row("SINGLE_AGENT", 0.4, None),  # awaiting
    _row("PARALLEL_DELEGATION", 0.55, True, source="implicit_outcome"),
]


class TestComputeRunMetrics:
    def test_totals_and_accuracy(self):
        m = compute_run_metrics(_WINDOW)
        assert m["total"] == 7
        assert m["labeled"] == 6
        assert m["awaiting_feedback"] == 1
        # 4 correct / 6 judged
        assert m["overall_accuracy_pct"] == 66.67

    def test_engagement_and_llm_share(self):
        m = compute_run_metrics(_WINDOW)
        # engaged = pattern != CLARIFICATION and conf >= 0.5: rows 1,2,3,7 -> 4/7
        assert m["engagement_rate"] == 0.5714
        assert m["active_floor"] == 0.5
        # used_llm_layer on rows 2,3 -> 2/7
        assert m["llm_layer_share"] == 0.2857

    def test_abstention_correctness(self):
        m = compute_run_metrics(_WINDOW)
        ab = m["abstention"]
        assert ab["total"] == 2
        assert ab["judged_correct"] == 1
        assert ab["judged_incorrect"] == 1
        assert ab["correctness_pct"] == 50.0

    def test_per_pattern_counts(self):
        pp = compute_run_metrics(_WINDOW)["per_pattern"]
        assert pp["SINGLE_AGENT"] == {
            "total": 3,
            "correct": 1,
            "incorrect": 1,
            "awaiting": 1,
            "accuracy_pct": 50.0,
        }
        assert pp["TOOL_COMPOSER"]["accuracy_pct"] == 100.0
        assert pp["CLARIFICATION_NEEDED"]["accuracy_pct"] == 50.0

    def test_label_sources(self):
        ls = compute_run_metrics(_WINDOW)["label_sources"]
        assert ls == {
            "explicit_feedback": 1,
            "implicit_outcome": 1,
            "llm_judge": 4,
            "llm_judge_abstain": 0,
        }

    def test_empty_window_is_safe(self):
        m = compute_run_metrics([])
        assert m["total"] == 0
        assert m["overall_accuracy_pct"] is None
        assert m["engagement_rate"] is None

    def test_default_floor_matches_router(self):
        # The active floor must track RouterNode.MIN_ACTIVE_CONFIDENCE so the
        # engagement metric mirrors the real abstain rule.
        from src.agents.orchestrator.nodes.router import RouterNode

        assert DEFAULT_ACTIVE_FLOOR == RouterNode.MIN_ACTIVE_CONFIDENCE


class TestThresholdProposals:
    # Two rows sit in the (0.4, 0.5) band: both judged-correct, so LOWERING the
    # floor to 0.4 engages them profitably.
    _PROP_ROWS = [
        _row("SINGLE_AGENT", 0.9, True),  # engaged at both
        _row("SINGLE_AGENT", 0.7, True),  # engaged at both
        _row("SINGLE_AGENT", 0.45, True),  # abstain@0.5, engage@0.4 (correct flip)
        _row("TOOL_COMPOSER", 0.42, True),  # abstain@0.5, engage@0.4 (correct flip)
        _row("SINGLE_AGENT", 0.8, False),  # engaged, wrong (drags accuracy)
        _row("CLARIFICATION_NEEDED", 0.0, True),  # never engages
    ]

    def test_lowering_floor_surfaces_correct_flips(self):
        # #1593: a floor recommendation now requires rows provably on the
        # CURRENT classifier baseline — engagement is a property of the
        # classifier, so pooled-across-flip rows cannot justify a floor. The
        # maths under test is unchanged; the rows are stamped so the
        # precondition is visible rather than incidental. The withholding
        # directions are pinned in test_routing_metrics_baseline_1593.py.
        rows = [dict(r, created_at=_ON_CURRENT_BASELINE) for r in self._PROP_ROWS]
        out = compute_threshold_proposals(
            rows, current_floor=0.5, candidates=[0.40], min_evidence=2
        )
        assert out["current_floor"] == 0.5
        # baseline engaged @0.5: rows with conf>=0.5 & not CLARIFICATION = 3
        # (0.9T, 0.7T, 0.8F) -> 2 correct / 3 = 66.67
        assert out["baseline_engaged_n"] == 3
        assert out["baseline_accuracy_pct"] == 66.67
        cand = out["candidates"][0]
        assert cand["candidate_floor"] == 0.40
        assert cand["direction"] == "lower"
        assert cand["labeled_flips"] == 2  # the 0.45 and 0.42 rows
        assert cand["flips_judged_correct"] == 2
        # engaged @0.4 = 5 rows, 4 correct -> 80.0
        assert cand["engaged_n"] == 5
        assert cand["engaged_accuracy_pct"] == 80.0
        assert cand["accuracy_delta_pct"] == 13.33
        assert out["recommended_floor"] == 0.40

    def test_insufficient_evidence_no_recommendation(self):
        out = compute_threshold_proposals(
            self._PROP_ROWS, current_floor=0.5, candidates=[0.40], min_evidence=99
        )
        assert out["recommended_floor"] is None

    def test_never_mutates_router_config(self):
        # The whole point of Phase 3: proposals are advisory. Computing them must
        # not touch the live routing floor.
        from src.agents.orchestrator.nodes.router import RouterNode

        before = RouterNode.MIN_ACTIVE_CONFIDENCE
        compute_threshold_proposals(self._PROP_ROWS)
        assert RouterNode.MIN_ACTIVE_CONFIDENCE == before == 0.5
