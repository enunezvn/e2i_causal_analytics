"""Both chat prompts must tell the model that digital-twin simulation exists (#2211).

On 2026-09-22, with the cohort restored and ``/digital-twin/health`` reporting three brands
simulable, the AG-UI brain answered "Use the digital twin to simulate an email campaign
intervention for Kisqali HCPs" with *"The E2I platform doesn't include a 'digital twin'
simulation capability"* and called ``causal_analysis_tool``. Reproduced offline through
the real chat leg (same factory call, same prompt, same bound tools): 4/4 runs picked the
causal tool or the dark experiment_designer, 0/4 reached any simulation. Nothing in either
prompt or in any bound tool's description mentioned a twin, a simulation or a counterfactual
— a tool the model is never told about is a tool that does not exist (#2115's lesson).

The text lives in ``src.kpi.capability_policy`` and is substituted into BOTH surfaces by
``render_blocks``, the seam #2115 built so a new block costs nothing in the two
ratchet-pinned prompt modules.
"""

from __future__ import annotations

import re

import pytest

from src.kpi import capability_policy as cap

SURFACES = ("chatbot_graph", "copilotkit")


def _prompt(surface: str) -> str:
    if surface == "copilotkit":
        from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT as prompt
    else:
        from src.api.routes.chatbot_graph import E2I_CHATBOT_SYSTEM_PROMPT as prompt
    return prompt


def _block(surface: str) -> str:
    lines = [ln for ln in _prompt(surface).splitlines() if "digital_twin_simulate_tool" in ln]
    assert lines, "no digital-twin simulation guidance line found"
    return " ".join(lines)


@pytest.mark.parametrize("surface", SURFACES)
def test_the_rendered_prompt_names_the_twin_simulation_tool(surface):
    assert "digital_twin_simulate_tool" in _prompt(surface)


@pytest.mark.parametrize("surface", SURFACES)
def test_the_guidance_carries_the_vocabulary_of_the_two_failing_asks(surface):
    """TW.1 said 'use the digital twin … simulate'; TW.2 said 'run a counterfactual … what
    would happen … if we increased'. The line the model reads must carry those cues."""
    text = _block(surface).lower()
    for cue in ("digital twin", "simulat", "counterfactual", "what would happen"):
        assert cue in text, f"{cue!r} missing from the twin guidance"


@pytest.mark.parametrize("surface", SURFACES)
def test_the_guidance_forbids_denying_the_capability(surface):
    """The false sentence was a platform-level negative (prompt rule 10). The block must say
    the capability EXISTS and where it lives, so a refusal reports the tool's reason instead."""
    text = _block(surface)
    assert "never say" in text.lower()
    assert "/api/digital-twin/simulate" in text
    assert "Digital Twin page" in text


@pytest.mark.parametrize("surface", SURFACES)
def test_the_guidance_fences_the_causal_tool_off_a_simulation_ask(surface):
    """Both failing turns went to causal_analysis_tool. The boundary — drivers observed in
    the registry versus an intervention simulated forward — is what keeps an ordinary
    'what drives Kisqali conversion' ask on the causal tool and a simulation ask off it."""
    text = _block(surface)
    assert "causal_analysis_tool" in text
    assert "never simulates" in text


@pytest.mark.parametrize("surface", SURFACES)
def test_the_guidance_keeps_the_dark_agent_path_dark(surface):
    """experiment_designer's simulate_intervention is gated off by design (#705); the
    guidance must not send the ask there (the BEFORE probe did, via orchestrator_tool)."""
    text = _block(surface)
    assert "orchestrator_tool" in text
    assert "experiment_designer" in text


@pytest.mark.parametrize("surface", SURFACES)
def test_the_breakdown_guidance_line_still_appears_exactly_once(surface):
    """The new block shares the placeholder with the breakdown block; the #2150 census
    counts this line, so adding a block must not duplicate or drop it."""
    lines = [ln for ln in _prompt(surface).splitlines() if ln.startswith("- BREAKDOWN GUIDANCE:")]
    assert len(lines) == 1


def test_the_block_names_exactly_the_interventions_the_engine_serves():
    """The prompt may not offer an intervention the engine refuses (#2150 census rule), nor
    hide one it serves. The block derives its list from the light contract module; the
    engine's own catalog is the reference — if they drift, this fails."""
    from src.digital_twin.effect.provider import SUPPORTED_INTERVENTIONS

    block = cap.twin_simulation_guidance_block()
    offer = re.search(r"interventions: ([^;)]*)", block)
    assert offer, f"the block must list the interventions: {block[:160]}"
    offered = {m for m in re.findall(r"[a-z_]+", offer.group(1)) if "_" in m}
    assert offered == set(SUPPORTED_INTERVENTIONS), (
        f"prompt offers {sorted(offered)}, engine serves {sorted(SUPPORTED_INTERVENTIONS)}"
    )


def test_the_block_is_cheap_to_render():
    """The prompt is rendered at import of the two route modules. Importing any module of
    ``src.digital_twin`` costs 17 s and +548 MB (measured 2026-09-22 on this box), so the
    block must read the intervention names from the side-effect-free contract module."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from src.kpi import capability_policy as cap\n"
        "cap.twin_simulation_guidance_block()\n"
        "bad = [m for m in sys.modules if m.startswith('src.digital_twin')]\n"
        "assert not bad, bad\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-800:]


def test_render_blocks_substitutes_the_twin_block():
    rendered = cap.render_blocks("A{capability_guidance}B")
    assert "digital_twin_simulate_tool" in rendered
    assert "forecast_kpi_tool" in rendered
    assert "BREAKDOWN GUIDANCE:" in rendered
    assert "{capability_guidance}" not in rendered
