"""Both chat prompts must tell the model that forecasting now exists (#2115, demo 6.5).

A tool the model never selects is a tool that does not exist. Demo 6.5 has refused since
2026-07-29 partly because nothing in either system prompt named a forecaster, so the
model reached for `causal_analysis_tool` (16.9 s, wrong answer) instead.

The prompt text lives in ``src.kpi.capability_policy`` and is substituted into BOTH
surfaces by ``render_blocks`` — the entry point whose docstring already says it exists
"so a new generated block never costs another line in the two ratchet-pinned prompt
modules". That is exactly the constraint this lane is under, so it uses the seam rather
than widening the pin.
"""

from __future__ import annotations

import pytest

from src.kpi import capability_policy as cap

SURFACES = ("chatbot_graph", "copilotkit")


def _prompt(surface: str) -> str:
    if surface == "copilotkit":
        from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT as prompt
    else:
        from src.api.routes.chatbot_graph import E2I_CHATBOT_SYSTEM_PROMPT as prompt
    return prompt


@pytest.mark.parametrize("surface", SURFACES)
def test_the_rendered_prompt_names_the_forecast_tool(surface):
    assert "forecast_kpi_tool" in _prompt(surface)


@pytest.mark.parametrize("surface", SURFACES)
def test_no_placeholder_survives_into_the_prompt_the_model_sees(surface):
    """An unsubstituted '{capability_guidance}' would ship the brace to the model."""
    prompt = _prompt(surface)
    assert "{capability_guidance}" not in prompt
    assert "{breakdown_guidance}" not in prompt


@pytest.mark.parametrize("surface", SURFACES)
def test_the_breakdown_guidance_line_still_appears_exactly_once(surface):
    """The forecast block shares one placeholder with the breakdown block; the census
    in test_axis_capability_census_2150 counts this line, so it must stay one line."""
    lines = [ln for ln in _prompt(surface).splitlines() if ln.startswith("- BREAKDOWN GUIDANCE:")]
    assert len(lines) == 1


@pytest.mark.parametrize("surface", SURFACES)
def test_the_prompt_tells_the_model_the_forecast_cannot_see_outside_events(surface):
    """Demo 6.5 asks for the forecast AND its biggest risk in one breath. If the prompt
    does not separate those, the model will narrate a risk the forecast never modelled."""
    block = [ln for ln in _prompt(surface).splitlines() if "forecast_kpi_tool" in ln]
    assert block, "no forecast guidance line found"
    text = " ".join(block).lower()
    assert "univariate" in text
    assert "risk" in text
    assert "gap" in text or "causal" in text


@pytest.mark.parametrize("surface", SURFACES)
def test_the_prompt_separates_forecasting_from_the_tools_that_only_report_the_past(surface):
    """`kpi_calculate_tool` answers 'what was it'; the forecaster answers 'what will it
    be'. Naming the boundary is what stops the model answering a forecast ask with a
    current value, which is how 6.5 failed before."""
    block = " ".join(ln for ln in _prompt(surface).splitlines() if "forecast_kpi_tool" in ln)
    assert "kpi_calculate_tool" in block


def test_the_block_names_only_metrics_the_forecaster_can_actually_serve():
    """The prompt may not offer a forecast for a KPI that has no canonical series —
    that would instruct the model into a guaranteed refusal (the #2150 census rule)."""
    import re

    from src.kpi.canonical_volume_series import SUPPORTED_METRICS

    block = cap.forecast_guidance_block()
    # The offer is the parenthesised metric list, not every KPI name the block mentions
    # (it deliberately names kpi_calculate_tool's KPIs to fence them off).
    offer = re.search(r"volume question \(([^)]*?)only", block)
    assert offer, f"the block must state which metrics it forecasts: {block[:120]}"
    offered = {m.lower() for m in re.findall(r"[A-Za-z_]+", offer.group(1))}
    assert offered == set(SUPPORTED_METRICS), (
        f"the prompt offers {sorted(offered)} but the forecaster serves {sorted(SUPPORTED_METRICS)}"
    )


def test_render_blocks_substitutes_the_forecast_block():
    rendered = cap.render_blocks("A{capability_guidance}B")
    assert "forecast_kpi_tool" in rendered
    assert "BREAKDOWN GUIDANCE:" in rendered
    assert "{capability_guidance}" not in rendered
