"""The reliability caveat reaches the planning prompt only behind its flag (spec §7.2).

Whether a caveat line changes the LLM's tool choice has never been measured, so the integration
ships default-off and an experiment decides it. Two things must hold until then:

- with the flag unset, verdicts change nothing — the flag is the only thing that can alter the
  block;
- with the flag set, each caveated tool adds exactly ONE line INSIDE ITS OWN block, and the
  declared "Avg execution" line is untouched — measured latency is never substituted into it.

Both are DELTA properties, so each test renders its own baseline at run time, from the same
registry, with the flag off. An earlier version of this file compared the whole block against
goldens captured from main 56f8b8589. Those files carried every *other* tool's prose, so when
#2014 rewrote `causal_effect_estimator`'s description all seven assertions failed with nothing
wrong with the flag.

A delta alone is not enough, though: deleting the "Avg execution" line outright would empty both
sides of the latency comparison and every delta would still hold. So the expected latency lines are
derived independently from the registry, and `test_rendering_contract_on_a_synthetic_schema` pins
the exact block format against a tool invented here. That restores what the goldens really bought
— an unconditional rendering contract — without pinning any real tool's prose.

The planner and the DSPy formatter share one block formatter, so both are covered, and
`test_dspy_and_planner_render_the_same_block` pins that they cannot drift apart.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pytest

from src.agents.tool_composer.dspy_integration import format_available_tools_for_planning
from src.agents.tool_composer.planner import ToolPlanner
from src.agents.tool_composer.reliability import ToolReliability
from src.tool_registry.registry import get_registry

FLAG = "TOOL_COMPOSER_RELIABILITY_IN_PLANNER"
CAVEAT_PREFIX = "Reliability caveat:"
LATENCY_PREFIX = "Avg execution:"
HEADER_PREFIX = "### "


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


def _verdict_of(name: str, expected: str, **over: Any) -> ToolReliability:
    """A reliability whose verdict is asserted, so a fixture can never drift off its own case."""
    tool = ToolReliability.from_row(_row(name, **over))
    assert tool.verdict == expected, (
        f"fixture must actually earn {expected!r}, got {tool.verdict!r}"
    )
    return tool


def _caveated(name: str) -> Dict[str, ToolReliability]:
    return {name: _verdict_of(name, "caveat")}


def _tool_names(registry) -> List[str]:
    return [schema["name"] for schema in registry.get_schemas_for_planning()]


def _expected_latency_lines(registry) -> List[str]:
    """Derived from the registry, not from the render, so an absent line cannot pass unnoticed."""
    lines = [f"{LATENCY_PREFIX} {s['avg_ms']}ms" for s in registry.get_schemas_for_planning()]
    assert lines, "no tools registered; every comparison below would be vacuous"
    return lines


def _lines_with(text: str, prefix: str) -> List[str]:
    return [line for line in text.splitlines() if line.startswith(prefix)]


def _block_of(text: str, tool_name: str) -> List[str]:
    """The lines of one tool's block: its header up to the next header (or the end)."""
    out: List[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith(HEADER_PREFIX):
            if inside:
                break
            inside = line.startswith(f"{HEADER_PREFIX}{tool_name} (")
        if inside:
            out.append(line)
    assert out, f"{tool_name} has no block in the render"
    return out


def _baseline(render: Callable[..., str], monkeypatch: pytest.MonkeyPatch, registry) -> str:
    """What the code under test renders with the flag off, right now.

    The guards keep every delta below honest. Without them "adds exactly one line" could pass
    against a baseline that already carried the caveat, was empty, or had lost the very latency
    line the test claims is untouched.
    """
    monkeypatch.delenv(FLAG, raising=False)
    text = render()
    assert not _lines_with(text, CAVEAT_PREFIX), "flag off: the block must carry no caveat line"
    assert _lines_with(text, LATENCY_PREFIX) == _expected_latency_lines(registry)
    assert len(_lines_with(text, HEADER_PREFIX)) == len(_tool_names(registry))
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
# The rendering contract itself, on a tool invented here
# ---------------------------------------------------------------------------

SYNTHETIC = {
    "name": "synthetic_tool",
    "description": "A synthetic tool, so this contract owns no real tool's prose",
    "source": "test_agent",
    "inputs": ["alpha: str - first input", "beta: int - second input"],
    "output": "SyntheticOutput",
    "output_fields": ["value", "note"],
    "avg_ms": 1234.5,
}

SYNTHETIC_BLOCK = (
    "### synthetic_tool (test_agent)\n"
    "Description: A synthetic tool, so this contract owns no real tool's prose\n"
    "Inputs: alpha: str - first input, beta: int - second input\n"
    "Output: SyntheticOutput (fields: value, note)\n"
    "Avg execution: 1234.5ms\n"
)

SYNTHETIC_BLOCK_NO_FIELDS = SYNTHETIC_BLOCK.replace(
    "Output: SyntheticOutput (fields: value, note)", "Output: SyntheticOutput"
)


@pytest.mark.parametrize(
    "output_fields, expected",
    [(["value", "note"], SYNTHETIC_BLOCK), ([], SYNTHETIC_BLOCK_NO_FIELDS)],
    ids=["with_output_fields", "without_output_fields"],
)
@pytest.mark.parametrize("flag", [None, "1"], ids=["flag_off", "flag_on_no_verdicts"])
def test_rendering_contract_on_a_synthetic_schema(monkeypatch, output_fields, expected, flag):
    """Every line of a tool block, pinned literally — and no real tool's words in sight.

    This is what the deleted goldens were actually protecting: an unconditional change to the
    ordinary rendering (dropping the declared latency line, say) is invisible to a pure delta,
    because it changes both sides equally.
    """
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    schema = dict(SYNTHETIC, output_fields=output_fields)

    planner = _planner(SimpleNamespace(get_schemas_for_planning=lambda: [schema]))

    assert planner._format_tools_for_prompt() == expected
    assert format_available_tools_for_planning(schemas=[schema]) == expected


# ---------------------------------------------------------------------------
# Flag off: nothing changes
# ---------------------------------------------------------------------------


def test_verdicts_are_ignored_while_the_flag_is_off(registry, monkeypatch):
    """Even handed verdicts, the flag decides. Nothing else in the pipeline may add the line."""
    planner = _planner(registry)
    name = _tool_names(registry)[0]

    planner_baseline = _baseline(planner._format_tools_for_prompt, monkeypatch, registry)
    dspy_baseline = _baseline(format_available_tools_for_planning, monkeypatch, registry)

    assert planner._format_tools_for_prompt(verdicts=_caveated(name)) == planner_baseline
    assert format_available_tools_for_planning(verdicts=_caveated(name)) == dspy_baseline


# ---------------------------------------------------------------------------
# Flag on: exactly one line, in the caveated tool's own block
# ---------------------------------------------------------------------------


def test_flag_on_adds_one_caveat_line_inside_each_caveated_tools_block(registry, monkeypatch):
    """Two caveated tools, distinct counts, and the second is NOT the first tool in the block.

    A caveat that landed on the wrong tool, or in a trailing clump after every block, would still
    add the right number of lines; only its placement and its counts distinguish the two.
    """
    planner = _planner(registry)
    names = _tool_names(registry)
    assert len(names) >= 2, "this test needs a second tool to prove placement"
    first, other = names[0], names[-1]

    baseline = _baseline(planner._format_tools_for_prompt, monkeypatch, registry)
    verdicts = {
        first: _verdict_of(first, "caveat"),
        # Distinct counts, so a caveat rendered against the wrong tool cannot pass.
        other: _verdict_of(other, "caveat", n_invoked=40, n_health=30, n_health_failures=9),
    }

    monkeypatch.setenv(FLAG, "1")
    block = planner._format_tools_for_prompt(verdicts=verdicts)

    added = _added_lines(baseline, block)
    assert len(added) == 2
    assert all(line.startswith(CAVEAT_PREFIX) for line in added)

    first_caveat = _lines_with("\n".join(_block_of(block, first)), CAVEAT_PREFIX)
    other_caveat = _lines_with("\n".join(_block_of(block, other)), CAVEAT_PREFIX)
    assert len(first_caveat) == 1 and len(other_caveat) == 1
    assert "6 of 24" in first_caveat[0] and "timeout" in first_caveat[0]
    assert "9 of 30" in other_caveat[0]
    assert first_caveat[0] != other_caveat[0]

    # The declared number the LLM already reads is untouched, and it is still actually there:
    # the expected list comes from the registry, not from the render.
    assert _lines_with(block, LATENCY_PREFIX) == _expected_latency_lines(registry)


@pytest.mark.parametrize(
    "expected_verdict, over",
    [
        ("reliable", {"n_succeeded": 40, "n_health_failures": 0, "n_health": 40}),
        ("inconclusive", {"n_succeeded": 36, "n_health_failures": 4, "n_health": 40}),
        ("too_few_runs", {"n_invoked": 10, "n_health_failures": 5, "n_health": 10}),
        ("no_runs", {"n_invoked": 0, "n_succeeded": 0, "n_health_failures": 0, "n_health": 0}),
    ],
)
def test_flag_on_says_nothing_for_any_non_caveat_verdict(
    registry, monkeypatch, expected_verdict, over
):
    """Only `caveat` may speak. Every other verdict is absence of evidence or good news."""
    planner = _planner(registry)
    name = _tool_names(registry)[0]
    baseline = _baseline(planner._format_tools_for_prompt, monkeypatch, registry)
    verdicts = {name: _verdict_of(name, expected_verdict, **over)}

    monkeypatch.setenv(FLAG, "1")
    assert planner._format_tools_for_prompt(verdicts=verdicts) == baseline


def test_flag_on_with_no_verdicts_changes_nothing(registry, monkeypatch):
    """A failed reliability read is a fail-open empty dict; the prompt must not change."""
    planner = _planner(registry)
    baseline = _baseline(planner._format_tools_for_prompt, monkeypatch, registry)

    monkeypatch.setenv(FLAG, "1")
    assert planner._format_tools_for_prompt(verdicts={}) == baseline


def test_dspy_and_planner_render_the_same_block(registry, monkeypatch):
    """One shared formatter serves both paths (#1584); pin that they cannot drift apart.

    Comparing whole renders — not just the caveat prefix — is what catches a caveat whose CONTENT
    differs between the two paths.
    """
    name = _tool_names(registry)[0]

    monkeypatch.delenv(FLAG, raising=False)
    assert format_available_tools_for_planning() == _planner(registry)._format_tools_for_prompt()

    monkeypatch.setenv(FLAG, "1")
    verdicts = _caveated(name)
    dspy_block = format_available_tools_for_planning(verdicts=verdicts)
    planner_block = _planner(registry)._format_tools_for_prompt(verdicts=verdicts)
    assert dspy_block == planner_block
    assert len(_lines_with(dspy_block, CAVEAT_PREFIX)) == 1


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
    name = _tool_names(registry)[0]
    rewritten = "REWRITTEN BY ANOTHER LANE - describes the same tool in different words"

    schema = registry.get_schema(name)
    assert schema is not None, f"{name} must be registered for this test to mean anything"
    monkeypatch.setattr(schema, "description", rewritten)

    baseline = _baseline(planner._format_tools_for_prompt, monkeypatch, registry)
    # Positive control: the rewrite really did reach the rendered block, so a pass below is the
    # delta surviving a changed description rather than the mutation quietly not applying.
    assert f"Description: {rewritten}" in baseline.splitlines()

    monkeypatch.setenv(FLAG, "1")
    block = planner._format_tools_for_prompt(verdicts=_caveated(name))

    added = _added_lines(baseline, block)
    assert len(added) == 1 and added[0].startswith(CAVEAT_PREFIX)
    assert f"Description: {rewritten}" in block.splitlines()
