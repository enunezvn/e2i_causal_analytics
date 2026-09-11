"""A similar-composition reference recommends only the steps that worked (spec §7.3).

`find_similar_compositions` keeps rows whose `raw_content.success` is true, and that is true for
a PARTIAL run, so the rendered "Tools used" list named tools that had failed — both live
pre-loop rows repeat `gap_calculator`, which succeeded in neither.

Now the hook hydrates each reference with its recorded steps and the formatter renders:

- "Tools that worked": the `succeeded` / `cache_hit` steps, in step order, as the sequence to reuse;
- "Did not work for that question": the rest, each with what happened to it.

A reference with no succeeded step is dropped. A reference with NO recorded steps (the live
pre-loop rows, and any row whose step writes were lost) renders as today only when every tool
worked (`tools_succeeded == tools_executed`); otherwise it cannot say which tools failed, so it
is dropped rather than recommended.

The formatter is pure: these tests build reference rows in the shapes the hook produces, and the
three legacy shapes are copied verbatim from the live `episodic_memories` rows (read-only,
2026-09-11).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.tool_composer.planner import ToolPlanner

# The three live `composition_completed` rows, verbatim (raw_content only, query text elided to
# its first words). Two are PARTIAL and must be dropped; the third worked end to end.
LIVE_PARTIAL_1_OF_6: Dict[str, Any] = {
    "query": "Compare TRx market share for Kisqali vs its competitors over the last 6 months",
    "success": True,
    "confidence": 0.35,
    "tool_sequence": [
        "gap_calculator",
        "cohort_statistics",
        "gap_calculator",
        "gap_calculator",
        "causal_effect_estimator",
        "cate_analyzer",
    ],
    "composition_id": "comp_9aa4c262",
    "tools_executed": 6,
    "tools_succeeded": 1,
    "total_duration_ms": 49518,
    "sub_questions_count": 6,
}
LIVE_PARTIAL_5_OF_6: Dict[str, Any] = {
    "query": "Full causal impact analysis of Q2 commercial arms on Kisqali TRx",
    "success": True,
    "confidence": 0.62,
    "tool_sequence": [
        "gap_calculator",
        "causal_effect_estimator",
        "cate_analyzer",
        "gap_calculator",
        "refutation_runner",
        "sensitivity_analyzer",
    ],
    "composition_id": "comp_023ff592",
    "tools_executed": 6,
    "tools_succeeded": 5,
    "total_duration_ms": 49171,
    "sub_questions_count": 5,
}
LIVE_ALL_SUCCESS: Dict[str, Any] = {
    "query": "For Kisqali, what is the causal effect of treatment_arm on adherence_rate",
    "success": True,
    "confidence": 0.88,
    "tool_sequence": [
        "cohort_builder",
        "cohort_statistics",
        "causal_effect_estimator",
        "sensitivity_analyzer",
        "refutation_runner",
    ],
    "composition_id": "comp_016c9c4b",
    "tools_executed": 5,
    "tools_succeeded": 5,
    "total_duration_ms": 52624,
    "sub_questions_count": 5,
}


def _formatter() -> ToolPlanner:
    """The formatter uses no planner state (as tests/integration/…_889.py already relies on)."""
    return object.__new__(ToolPlanner)


def _step(number: int, tool: str, outcome: str) -> Dict[str, Any]:
    return {"step_number": number, "tool_name": tool, "outcome_class": outcome}


def _reference(raw: Dict[str, Any], steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A row as ``find_similar_compositions`` returns it, hydrated with its recorded steps."""
    return {
        "memory_id": raw.get("composition_id", "m"),
        "raw_content": raw,
        "recorded_steps": steps or [],
    }


def _worked_line(block: str) -> str:
    return next(line for line in block.splitlines() if line.startswith("- Tools that worked:"))


def _failed_line(block: str) -> str:
    return next(line for line in block.splitlines() if line.startswith("- Did not work"))


HYDRATED_PARTIAL = _reference(
    LIVE_PARTIAL_5_OF_6,
    [
        _step(0, "causal_effect_estimator", "succeeded"),
        _step(1, "gap_calculator", "refused"),
        _step(2, "cate_analyzer", "succeeded"),
        _step(3, "rank_drivers", "dependency_unmet"),
    ],
)


def test_partial_reference_recommends_only_succeeded():
    block = _formatter()._format_episodic_context([HYDRATED_PARTIAL])

    worked = _worked_line(block)
    assert "causal_effect_estimator" in worked and "cate_analyzer" in worked
    assert "gap_calculator" not in worked and "rank_drivers" not in worked
    # Step order, not the order the outcome classes happen to appear in.
    assert worked.index("causal_effect_estimator") < worked.index("cate_analyzer")

    failed = _failed_line(block)
    assert "gap_calculator" in failed and "refused" in failed
    assert "rank_drivers" in failed and "dependency unmet" in failed


def test_cache_hit_steps_count_as_worked():
    reference = _reference(
        LIVE_ALL_SUCCESS,
        [
            _step(0, "cohort_builder", "cache_hit"),
            _step(1, "causal_effect_estimator", "succeeded"),
        ],
    )
    worked = _worked_line(_formatter()._format_episodic_context([reference]))
    assert worked.index("cohort_builder") < worked.index("causal_effect_estimator")


def test_all_cache_hit_reference_renders():
    reference = _reference(
        LIVE_ALL_SUCCESS,
        [_step(0, "cohort_builder", "cache_hit"), _step(1, "cohort_statistics", "cache_hit")],
    )
    block = _formatter()._format_episodic_context([reference])
    assert block, "an all-cache_hit reference is a composition that worked; it must not be dropped"
    assert "cohort_builder" in _worked_line(block)


def test_zero_success_reference_dropped():
    reference = _reference(
        LIVE_PARTIAL_1_OF_6,
        [
            _step(0, "gap_calculator", "refused"),
            _step(1, "cohort_statistics", "error"),
            _step(2, "cate_analyzer", "dependency_unmet"),
        ],
    )
    assert _formatter()._format_episodic_context([reference]) == ""


def test_legacy_partial_references_dropped():
    """Both live pre-loop rows: PARTIAL with no recorded steps. Backfill is impossible."""
    references = [_reference(LIVE_PARTIAL_1_OF_6), _reference(LIVE_PARTIAL_5_OF_6)]
    assert _formatter()._format_episodic_context(references) == ""


def test_legacy_all_success_reference_renders_as_today():
    block = _formatter()._format_episodic_context([_reference(LIVE_ALL_SUCCESS)])
    assert "cohort_builder, cohort_statistics, causal_effect_estimator" in block
    assert "0.88" in block
    assert "52624ms" in block


def test_a_dropped_reference_does_not_take_the_others_with_it():
    block = _formatter()._format_episodic_context(
        [_reference(LIVE_PARTIAL_1_OF_6), HYDRATED_PARTIAL, _reference(LIVE_ALL_SUCCESS)]
    )
    assert "comp_9aa4c262" not in block
    assert "causal_effect_estimator" in _worked_line(block)
    assert "cohort_builder" in block  # the legacy all-success reference still renders


def test_no_references_and_no_renderable_references_both_yield_no_context():
    assert _formatter()._format_episodic_context([]) == ""
    assert _formatter()._format_episodic_context([_reference({})]) == ""
