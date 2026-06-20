"""Unit coverage for the synthesize-node prompt builder.

``build_synthesis_prompt`` frames the user's question + the assistant's tool-call
args (brand/window) + the tool results so the synthesizer answers the ACTUAL
question and never re-asks for a brand/period the user already provided.
"""

import pytest

from src.api.routes.copilotkit import build_synthesis_prompt


@pytest.mark.unit
def test_includes_question_and_args():
    p = build_synthesis_prompt(
        "NRx for Kisqali past 3 months",
        [
            {
                "name": "kpi_calculate_tool",
                "args": {"kpi_name": "NRx", "brand": "Kisqali", "window": "past 3 months"},
            }
        ],
        [{"tool": "kpi_calculate_tool", "result": '{"value": 3394, "window_status": "applied"}'}],
    )
    assert "Kisqali" in p and "past 3 months" in p and "3394" in p
    assert "User question" in p


@pytest.mark.unit
def test_handles_empty():
    p = build_synthesis_prompt("", [], [])
    assert isinstance(p, str) and "User question" in p
