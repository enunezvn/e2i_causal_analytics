"""The reliability caveat reaches the planning prompt only behind its flag (spec §7.2).

Whether a caveat line changes the LLM's tool choice has never been measured, so the integration
ships default-off and an experiment decides it. Two things must hold until then:

- with the flag unset, verdicts change nothing — the flag is the only thing that can alter the
  block;
- with the flag set, a caveated tool adds exactly ONE line, and the declared "Avg execution" line
  is untouched — measured latency is never substituted into it.

Both are DELTA properties, so the baseline is rendered at run time from the same registry by the
code under test. An earlier version of this file compared against goldens captured from one main
sha (`fixtures/*_56f8b8589.txt`). Those files embedded every other tool's prose, so when #2014
rewrote `causal_effect_estimator`'s description every assertion here failed with nothing wrong with
the flag. A byte-golden pinned to a past sha cannot express "the flag adds one line";
``test_the_delta_survives_a_description_change`` pins that this file now can.

The planner and the DSPy formatter share one block formatter, so both are covered.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import pytest

from src.agents.tool_composer.dspy_integration import format_available_tools_for_planning
from src.agents.tool_composer.planner import ToolPlanner
from src.agents.tool_composer.reliability import ToolReliability
from src.tool_registry.registry import get_registry

FLAG = "TOOL_COMPOSER_RELIABILITY_IN_PLANNER"
CAVEAT_PREFIX = "Reliability caveat:"


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


def _baseline(render: Callable[..., str], monkeypatch: pytest.MonkeyPatch) -> str:
    """What the code under test renders with the flag off, right now.

    Asserting the baseline carries no caveat line keeps every delta assertion below honest: if the
    line were already present with the flag off, "adds exactly one" could pass while saying nothing.
    """
    monkeypatch.delenv(FLAG, raising=False)
    text = render()
    assert CAVEAT_PREFIX not in text, "the flag is off; the block must not already carry a caveat"
    assert text.strip(), "empty baseline would make every comparison below vacuous"
    return text


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


# ---------------------------------------------------------------------------
# Flag off: nothing changes
# ---------------------------------------------------------------------------


def test_verdicts_are_ignored_while_the_flag_is_off(registry, monkeypatch):
    """Even handed verdicts, the flag decides. Nothing else in the pipeline may add the line."""
    planner = _planner(registry)
    name = _first_tool_name(registry)

    planner_baseline = _baseline(planner._format_tools_for_prompt, monkeypatch)
    dspy_baseline = _baseline(format_available_tools_for_planning, monkeypatch)

    assert planner._format_tools_for_prompt(verdicts=_caveated(name)) == planner_baseline
    assert format_available_tools_for_planning(verdicts=_caveated(name)) == dspy_baseline


# ---------------------------------------------------------------------------
# Flag on: exactly one line per caveated tool
# ---------------------------------------------------------------------------


def test_flag_on_adds_one_caveat_line_per_caveated_tool(registry, monkeypatch):
    planner = _planner(registry)
    name = _first_tool_name(registry)
    baseline = _baseline(planner._format_tools_for_prompt, monkeypatch)

    monkeypatch.setenv(FLAG, "1")
    block = planner._format_tools_for_prompt(verdicts=_caveated(name))

    added = _added_lines(baseline, block)
    assert len(added) == 1
    assert added[0].startswith(CAVEAT_PREFIX)
    assert "6 of 24" in added[0] and "timeout" in added[0]
    # The declared number the LLM already reads is untouched; measured latency is not substituted.
    assert [line for line in block.splitlines() if line.startswith("Avg execution:")] == [
        line for line in baseline.splitlines() if line.startswith("Avg execution:")
    ]


def test_flag_on_says_nothing_about_tools_without_a_caveat(registry, monkeypatch):
    planner = _planner(registry)
    name = _first_tool_name(registry)
    baseline = _baseline(planner._format_tools_for_prompt, monkeypatch)

    reliable = {
        name: ToolReliability.from_row(_row(name, n_succeeded=40, n_health_failures=0, n_health=40))
    }
    assert reliable[name].verdict == "reliable"

    monkeypatch.setenv(FLAG, "1")
    assert planner._format_tools_for_prompt(verdicts=reliable) == baseline


def test_flag_on_with_no_verdicts_changes_nothing(registry, monkeypatch):
    """A failed reliability read is a fail-open empty dict; the prompt must not change."""
    planner = _planner(registry)
    baseline = _baseline(planner._format_tools_for_prompt, monkeypatch)

    monkeypatch.setenv(FLAG, "1")
    assert planner._format_tools_for_prompt(verdicts={}) == baseline


def test_flag_on_dspy_block_gets_the_same_line(registry, monkeypatch):
    name = _first_tool_name(registry)
    baseline = _baseline(format_available_tools_for_planning, monkeypatch)

    monkeypatch.setenv(FLAG, "1")
    block = format_available_tools_for_planning(verdicts=_caveated(name))

    added = _added_lines(baseline, block)
    assert len(added) == 1 and added[0].startswith(CAVEAT_PREFIX)


# ---------------------------------------------------------------------------
# The property must survive another lane editing a tool's prose
# ---------------------------------------------------------------------------


def test_the_delta_survives_a_description_change(registry, monkeypatch):
    """A tool description rewrite is someone else's business; the flag delta is ours.

    This is the case the golden fixtures could not express: #2014 rewrote
    `causal_effect_estimator`'s description and every assertion in this file failed under the
    merge, though the flag behaved correctly throughout.
    """
    planner = _planner(registry)
    name = _first_tool_name(registry)
    rewritten = "REWRITTEN BY ANOTHER LANE - describes the same tool in different words"

    schema = registry.get_schema(name)
    assert schema is not None, f"{name} must be registered for this test to mean anything"
    monkeypatch.setattr(schema, "description", rewritten)

    baseline = _baseline(planner._format_tools_for_prompt, monkeypatch)
    # Positive control: the rewrite really did reach the rendered block, so a pass below is the
    # delta surviving a changed description rather than the mutation quietly not applying.
    assert rewritten in baseline

    monkeypatch.setenv(FLAG, "1")
    block = planner._format_tools_for_prompt(verdicts=_caveated(name))

    added = _added_lines(baseline, block)
    assert len(added) == 1 and added[0].startswith(CAVEAT_PREFIX)
    assert rewritten in block
