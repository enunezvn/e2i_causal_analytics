"""Both chat system prompts stop offering TRx Share as a patient-axis breakdown.

Chat session_1789548670222_fcscf3u (2026-09-16): the prompt's BREAKDOWN GUIDANCE
listed TRx-share among the axis-breakdown KPIs, so the model offered and served
"Remibrutinib TRx Share by severity tier and biologic status": a cross-indication
ratio per tier, and 100% for both biologic buckets. The tool now refuses it; the
prompts must not steer the model into it, and must name the real answer (TRx by the
axis, presented as the within-brand mix).
"""

import re

import pytest

from src.api.routes.chatbot_graph import E2I_CHATBOT_SYSTEM_PROMPT
from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT

_PROMPTS = {"copilotkit": E2I_COPILOT_SYSTEM_PROMPT, "chatbot_graph": E2I_CHATBOT_SYSTEM_PROMPT}


def _line(prompt: str, marker: str) -> str:
    lines = [ln for ln in prompt.splitlines() if ln.startswith(marker)]
    assert len(lines) == 1, (marker, len(lines))
    return lines[0]


@pytest.mark.unit
@pytest.mark.parametrize("surface", sorted(_PROMPTS))
def test_breakdown_guidance_does_not_offer_share_on_patient_axes(surface):
    guidance = _line(_PROMPTS[surface], "- BREAKDOWN GUIDANCE:")
    assert "TRx-share" not in guidance
    assert "volume KPIs and share" not in guidance
    assert "TRx share is NOT defined on a patient axis" in guidance
    assert "within-brand mix" in guidance


@pytest.mark.unit
@pytest.mark.parametrize("surface", sorted(_PROMPTS))
def test_honesty_guard_still_scopes_share_to_the_tracked_portfolio(surface):
    guard = _line(_PROMPTS[surface], "- HONESTY GUARD:")
    assert "TRACKED PORTFOLIO" in guard
    assert "never a share" in guard


@pytest.mark.unit
def test_both_surfaces_carry_the_same_breakdown_rules():
    for marker in ("- BREAKDOWN GUIDANCE:", "- HONESTY GUARD:"):
        assert _line(E2I_COPILOT_SYSTEM_PROMPT, marker) == _line(E2I_CHATBOT_SYSTEM_PROMPT, marker)


@pytest.mark.unit
@pytest.mark.parametrize("surface", sorted(_PROMPTS))
def test_breakdown_guidance_names_a_volume_kpi_that_actually_serves_the_axis(surface):
    """#2150: pin WHICH KPI the guidance tells the model to call, not just what it
    says about the share.

    The assertions above pin the SHARE half and pass today — but they never pin the
    volume half, so the clause sits NEXT TO the defect without covering it. The
    guidance says "For NRx/TRx/NBRx/conversion-rate patient-segment breakdowns, call
    kpi_calculate_tool", and measured at 0f4916916 ``recognize_kpi`` maps those three
    names to WS3-BI-005/006/007 — the CANONICAL ids, which refuse every patient axis
    since owner #11 moved the panel to WS3-BI-011..013. So the prompt steers the
    model into a call the tool refuses.

    Resolving the name through the production resolver rather than asserting a
    substring is what makes this a capability check: "TRx" is in the line either
    way, and only the resolved id tells you whether the instruction can be followed.
    """
    from src.api.routes.chatbot_tools import _PATIENT_AXIS_KPI_IDS
    from src.services.kpi_resolution import recognize_kpi

    guidance = _line(_PROMPTS[surface], "- BREAKDOWN GUIDANCE:")
    served = _PATIENT_AXIS_KPI_IDS["segment"]
    offenders = {}
    for term in ("TRx", "NRx", "NBRx"):
        if not re.search(rf"\b{term}\b(?! [Ss]hare)(?! Panel)", guidance):
            continue
        kpi = recognize_kpi(term)
        if kpi is None or kpi.id not in served:
            offenders[term] = "unresolvable" if kpi is None else kpi.id
    assert not offenders, (
        f"{surface}: the breakdown guidance names {offenders} for a patient-axis "
        f"breakdown, and those KPIs refuse it. Axis-serving ids: {sorted(served)}"
    )
