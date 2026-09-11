"""The reliability caveat reaches the planning prompt only behind its flag (spec §7.2).

Whether a caveat line changes the LLM's tool choice has never been measured, so the integration
ships default-off and an experiment decides it. Two things must hold until then:

- with the flag unset the prompt is BYTE-IDENTICAL to what it renders today, pinned against golden
  fixtures generated from the pre-change code (`fixtures/*_56f8b8589.txt`);
- with the flag set, a caveated tool adds exactly ONE line, and the declared "Avg execution" line
  is untouched — measured latency is never substituted into it.

The planner and the DSPy formatter share one block formatter, so both are pinned here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.agents.tool_composer.dspy_integration import format_available_tools_for_planning
from src.agents.tool_composer.planner import ToolPlanner
from src.agents.tool_composer.reliability import ToolReliability
from src.tool_registry.registry import get_registry

FIXTURES = Path(__file__).parent / "fixtures"
PLANNER_GOLDEN = FIXTURES / "planner_tools_prompt_56f8b8589.txt"
DSPY_GOLDEN = FIXTURES / "dspy_tools_block_56f8b8589.txt"


@pytest.fixture
def registry():
    """The real global registry, with every composable tool registered."""
    import src.agents.tool_composer.composer  # noqa: F401 - registers the tools

    return get_registry()


def _planner(registry) -> ToolPlanner:
    planner = object.__new__(ToolPlanner)
    planner.registry = registry
    return planner


def _row(name: str, **over: Any) -> Dict[str, Any]:
    row = {
        "tool_name": name,
        "category": "CAUSAL",
        "source_agent": "causal_impact",
        "version": "1.0.0",
        "declared_latency_ms": 5000.0,
        "n_invoked": 30,
        "n_succeeded": 18,
        "n_refused": 6,
        "n_health_failures": 6,
        "n_health": 24,
        "n_retried": 0,
        "n_synthetic": 0,
        "p50_latency_ms": 900.0,
        "p95_latency_ms": 2500.0,
        "last_executed_at": "2026-09-11T00:00:00+00:00",
        "most_common_health_error": "timeout",
    }
    row.update(over)
    return row


def _caveated(name: str) -> Dict[str, ToolReliability]:
    tool = ToolReliability.from_row(_row(name))
    assert tool.verdict == "caveat", "fixture must actually earn a caveat"
    return {name: tool}


def _first_tool_name(registry) -> str:
    return registry.get_schemas_for_planning()[0]["name"]


# ---------------------------------------------------------------------------
# Flag off: nothing changes
# ---------------------------------------------------------------------------


def test_flag_off_planner_prompt_is_byte_identical(registry, monkeypatch):
    monkeypatch.delenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", raising=False)
    assert _planner(registry)._format_tools_for_prompt() == PLANNER_GOLDEN.read_text()


def test_flag_off_dspy_block_is_byte_identical(registry, monkeypatch):
    monkeypatch.delenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", raising=False)
    assert format_available_tools_for_planning() == DSPY_GOLDEN.read_text()


def test_verdicts_are_ignored_while_the_flag_is_off(registry, monkeypatch):
    """Even handed verdicts, the flag decides. Nothing else in the pipeline may add the line."""
    monkeypatch.delenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", raising=False)
    name = _first_tool_name(registry)

    planner_block = _planner(registry)._format_tools_for_prompt(verdicts=_caveated(name))
    dspy_block = format_available_tools_for_planning(verdicts=_caveated(name))

    assert planner_block == PLANNER_GOLDEN.read_text()
    assert dspy_block == DSPY_GOLDEN.read_text()


# ---------------------------------------------------------------------------
# Flag on: exactly one line per caveated tool
# ---------------------------------------------------------------------------


def _added_lines(before: str, after: str) -> List[str]:
    kept = list(before.splitlines())
    added: List[str] = []
    for line in after.splitlines():
        if kept and line == kept[0]:
            kept.pop(0)
        else:
            added.append(line)
    assert not kept, "the flag removed or reordered a line; it may only ADD"
    return added


def test_flag_on_adds_one_caveat_line_per_caveated_tool(registry, monkeypatch):
    monkeypatch.setenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", "1")
    name = _first_tool_name(registry)
    golden = PLANNER_GOLDEN.read_text()

    block = _planner(registry)._format_tools_for_prompt(verdicts=_caveated(name))

    added = _added_lines(golden, block)
    assert len(added) == 1
    assert added[0].startswith("Reliability caveat:")
    assert "6 of 24" in added[0] and "timeout" in added[0]
    # The declared number the LLM already reads is untouched; measured latency is not substituted.
    assert [line for line in block.splitlines() if line.startswith("Avg execution:")] == [
        line for line in golden.splitlines() if line.startswith("Avg execution:")
    ]


def test_flag_on_says_nothing_about_tools_without_a_caveat(registry, monkeypatch):
    monkeypatch.setenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", "1")
    name = _first_tool_name(registry)
    reliable = {name: ToolReliability.from_row(_row(name, n_succeeded=40, n_health_failures=0, n_health=40))}
    assert reliable[name].verdict == "reliable"

    block = _planner(registry)._format_tools_for_prompt(verdicts=reliable)

    assert block == PLANNER_GOLDEN.read_text()


def test_flag_on_with_no_verdicts_changes_nothing(registry, monkeypatch):
    """A failed reliability read is a fail-open empty dict; the prompt must not change."""
    monkeypatch.setenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", "1")
    assert _planner(registry)._format_tools_for_prompt(verdicts={}) == PLANNER_GOLDEN.read_text()


def test_flag_on_dspy_block_gets_the_same_line(registry, monkeypatch):
    monkeypatch.setenv("TOOL_COMPOSER_RELIABILITY_IN_PLANNER", "1")
    name = _first_tool_name(registry)

    block = format_available_tools_for_planning(verdicts=_caveated(name))

    added = _added_lines(DSPY_GOLDEN.read_text(), block)
    assert len(added) == 1 and added[0].startswith("Reliability caveat:")
