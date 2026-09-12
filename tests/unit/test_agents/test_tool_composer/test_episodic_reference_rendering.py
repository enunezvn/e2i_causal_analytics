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
from unittest.mock import AsyncMock

import pytest

from src.agents.tool_composer import memory_hooks as hooks_module
from src.agents.tool_composer.memory_hooks import select_renderable
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


# ---------------------------------------------------------------------------
# Selection: dropping happens before the limit, not after
# ---------------------------------------------------------------------------


def test_dropped_candidates_do_not_starve_a_renderable_one():
    """Three un-renderable candidates must not consume all three reference slots."""
    candidates = [
        _reference(LIVE_PARTIAL_1_OF_6),
        _reference(LIVE_PARTIAL_5_OF_6),
        _reference(LIVE_PARTIAL_1_OF_6),
        _reference(LIVE_ALL_SUCCESS),
    ]
    kept = select_renderable(candidates, 3)
    assert [r["raw_content"]["composition_id"] for r in kept] == ["comp_016c9c4b"]


def test_selection_keeps_search_order_and_respects_the_limit():
    candidates = [HYDRATED_PARTIAL, _reference(LIVE_ALL_SUCCESS), HYDRATED_PARTIAL]
    kept = select_renderable(candidates, 2)
    assert len(kept) == 2
    assert kept[0] is HYDRATED_PARTIAL and kept[1]["raw_content"] is LIVE_ALL_SUCCESS


# ---------------------------------------------------------------------------
# Malformed rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "not a dict", ["gap_calculator"], 7, {}])
def test_a_malformed_reference_does_not_take_a_valid_one_with_it(raw):
    malformed = {"memory_id": "m", "raw_content": raw, "recorded_steps": []}

    block = _formatter()._format_episodic_context([malformed, HYDRATED_PARTIAL])

    assert "causal_effect_estimator" in _worked_line(block)
    assert select_renderable([malformed], 3) == []


def test_an_out_of_range_confidence_does_not_abort_the_other_references():
    """10**400 is JSON-representable and numeric, but formatting it with ':.2f' overflows."""
    huge = _reference({**LIVE_ALL_SUCCESS, "confidence": 10**400})

    block = _formatter()._format_episodic_context([huge, HYDRATED_PARTIAL])

    assert block.count("### Reference") == 2
    assert "causal_effect_estimator" in _worked_line(block)


def test_a_step_backed_reference_renders_even_without_raw_content():
    """The steps say what worked; the counts are only the fallback for rows without them."""
    reference = {
        "memory_id": "m",
        "raw_content": None,
        "recorded_steps": [_step(0, "cohort_builder", "succeeded")],
    }

    block = _formatter()._format_episodic_context([reference])

    assert "cohort_builder" in _worked_line(block)
    assert select_renderable([reference], 3) == [reference]


@pytest.mark.parametrize("sequence", [7, "gap_calculator", {"a": 1}])
def test_a_legacy_reference_whose_tool_sequence_is_unusable_is_dropped_not_raised(sequence):
    broken = _reference({"tools_executed": 5, "tools_succeeded": 5, "tool_sequence": sequence})

    block = _formatter()._format_episodic_context([broken, HYDRATED_PARTIAL])

    assert "causal_effect_estimator" in _worked_line(block)
    assert select_renderable([broken], 3) == []


@pytest.mark.parametrize("steps", [7, "steps", {"step_number": 0}])
def test_unusable_recorded_steps_drop_the_reference_without_raising(steps):
    broken = {"memory_id": "m", "raw_content": {}, "recorded_steps": steps}

    block = _formatter()._format_episodic_context([broken, HYDRATED_PARTIAL])

    assert "causal_effect_estimator" in _worked_line(block)
    assert select_renderable([broken], 3) == []


def test_an_unhashable_outcome_class_neither_raises_nor_counts_as_worked():
    reference = {
        "memory_id": "m",
        "raw_content": {},
        "recorded_steps": [
            {"step_number": 0, "tool_name": "gap_calculator", "outcome_class": {"weird": 1}},
            _step(1, "cate_analyzer", "succeeded"),
        ],
    }

    worked = _worked_line(_formatter()._format_episodic_context([reference]))

    assert "cate_analyzer" in worked and "gap_calculator" not in worked


# ---------------------------------------------------------------------------
# The hook itself drops before it limits
# ---------------------------------------------------------------------------


async def test_find_similar_compositions_drops_before_limiting(monkeypatch):
    """Restoring the premature slice would leave a usable reference unrendered (codex iter-1)."""
    candidates = [
        {"memory_id": "1", "raw_content": dict(LIVE_PARTIAL_1_OF_6)},
        {"memory_id": "2", "raw_content": dict(LIVE_PARTIAL_5_OF_6)},
        {"memory_id": "3", "raw_content": dict(LIVE_PARTIAL_1_OF_6)},
        {"memory_id": "4", "raw_content": dict(LIVE_ALL_SUCCESS)},
    ]

    async def fake_search(**_: Any) -> List[Dict[str, Any]]:
        return [dict(row) for row in candidates]

    async def fake_hydrate_raw(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return results

    monkeypatch.setattr("src.memory.episodic_memory.search_episodic_by_text", fake_search)
    monkeypatch.setattr("src.memory.episodic_memory.hydrate_raw_content", fake_hydrate_raw)

    class StepsPort:
        def __init__(self) -> None:
            self.calls: List[Any] = []

        async def call(self, name: str, params: Dict[str, Any]) -> Any:
            self.calls.append((name, params))
            return {}

    port = StepsPort()
    real_hydrate = hooks_module.hydrate_reference_steps

    async def hydrate_with_port(references: List[Dict[str, Any]], **_: Any):
        return await real_hydrate(references, port=port)

    monkeypatch.setattr(hooks_module, "hydrate_reference_steps", hydrate_with_port)

    kept = await hooks_module.ToolComposerMemoryHooks().find_similar_compositions("q", limit=3)

    assert [r["raw_content"]["composition_id"] for r in kept] == ["comp_016c9c4b"]
    ((name, params),) = port.calls
    assert name == "composer_steps_for"
    # One read, for the WHOLE candidate set — not just the first three.
    assert set(params["p_composition_ids"]) == {
        "comp_9aa4c262",
        "comp_023ff592",
        "comp_016c9c4b",
    }


async def test_hydration_survives_a_malformed_payload_for_one_composition():
    """A per-composition value that is not a list, and steps whose order keys do not compare."""

    class Port:
        async def call(self, name: str, params: Dict[str, Any]) -> Any:
            return {
                "comp_9aa4c262": 7,
                "comp_016c9c4b": [
                    {
                        "step_number": "x",
                        "tool_name": "cohort_builder",
                        "outcome_class": "succeeded",
                    },
                    _step(0, "cohort_statistics", "succeeded"),
                ],
            }

    rows = [_reference(LIVE_PARTIAL_1_OF_6), _reference(LIVE_ALL_SUCCESS)]

    hydrated = await hooks_module.hydrate_reference_steps(rows, port=Port())

    assert hydrated[0]["recorded_steps"] == []
    # The comparable key sorts first; the unusable one keeps its place rather than raising.
    assert [s["tool_name"] for s in hydrated[1]["recorded_steps"]] == [
        "cohort_statistics",
        "cohort_builder",
    ]


async def test_a_malformed_hydration_payload_does_not_lose_the_other_references(monkeypatch):
    candidates = [{"memory_id": "4", "raw_content": dict(LIVE_ALL_SUCCESS)}]

    async def fake_search(**_: Any) -> List[Dict[str, Any]]:
        return [dict(row) for row in candidates]

    async def fake_hydrate_raw(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return results

    monkeypatch.setattr("src.memory.episodic_memory.search_episodic_by_text", fake_search)
    monkeypatch.setattr("src.memory.episodic_memory.hydrate_raw_content", fake_hydrate_raw)

    class NonsensePort:
        async def call(self, name: str, params: Dict[str, Any]) -> Any:
            return {"comp_016c9c4b": 7}

    real_hydrate = hooks_module.hydrate_reference_steps

    async def hydrate_with_port(references: List[Dict[str, Any]], **_: Any):
        return await real_hydrate(references, port=NonsensePort())

    monkeypatch.setattr(hooks_module, "hydrate_reference_steps", hydrate_with_port)

    kept = await hooks_module.ToolComposerMemoryHooks().find_similar_compositions("q", limit=3)

    # It falls back to the legacy reading (all tools worked), rather than losing the reference.
    assert [r["raw_content"]["composition_id"] for r in kept] == ["comp_016c9c4b"]


# ---------------------------------------------------------------------------
# Only a usable tool name can be recommended
# ---------------------------------------------------------------------------


def test_a_worked_step_without_a_usable_name_is_not_recommended():
    reference = {
        "memory_id": "m",
        "raw_content": {},
        "recorded_steps": [
            {"step_number": 0, "tool_name": None, "outcome_class": "succeeded"},
            {"step_number": 1, "tool_name": {"a": 1}, "outcome_class": "succeeded"},
            _step(2, "cate_analyzer", "succeeded"),
        ],
    }

    worked = _worked_line(_formatter()._format_episodic_context([reference]))

    assert "cate_analyzer" in worked
    assert "None" not in worked and "{" not in worked


def test_a_reference_whose_worked_steps_have_no_usable_name_is_dropped():
    reference = {
        "memory_id": "m",
        "raw_content": {},
        "recorded_steps": [{"step_number": 0, "tool_name": None, "outcome_class": "succeeded"}],
    }

    assert _formatter()._format_episodic_context([reference]) == ""
    assert select_renderable([reference], 3) == []


def test_an_unregistered_tool_is_not_recommended():
    """A tool that no longer exists cannot be reused; recommending it plans a failing step."""
    import src.agents.tool_composer.composer  # noqa: F401 - registers the real tools

    reference = {
        "memory_id": "m",
        "raw_content": {},
        "recorded_steps": [
            _step(0, "retired_tool", "succeeded"),
            _step(1, "cate_analyzer", "succeeded"),
        ],
    }

    worked = _worked_line(_formatter()._format_episodic_context([reference]))

    assert "cate_analyzer" in worked and "retired_tool" not in worked


def test_a_legacy_sequence_of_unusable_names_is_dropped():
    broken = _reference(
        {"tools_executed": 2, "tools_succeeded": 2, "tool_sequence": [None, {"a": 1}]}
    )

    assert select_renderable([broken], 3) == []
    assert _formatter()._format_episodic_context([broken]) == ""


async def test_check_episodic_memory_survives_a_malformed_confidence():
    """The debug log must not format a caller's value numerically: it would discard every row."""
    rows = [
        {"memory_id": "bad", "raw_content": {**LIVE_ALL_SUCCESS, "confidence": "high"}},
        {"memory_id": "ok", "raw_content": dict(LIVE_ALL_SUCCESS)},
    ]
    hooks = AsyncMock()
    hooks.find_similar_compositions = AsyncMock(return_value=rows)
    planner = _formatter()
    planner.use_episodic_memory = True
    planner.memory_hooks = hooks

    similar = await planner._check_episodic_memory("q")

    assert len(similar) == 2
    assert planner._format_episodic_context(similar).count("### Reference") == 2
