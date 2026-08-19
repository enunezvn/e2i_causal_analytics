"""#1713 (prompt half): per-KPI window attribution + polarity-aware status.

Eval 4.6 regressed in place: the shared header asserted "most recent 30
days ... data through 2026-08-17" for BOTH False Alert Rate and Override
Rate, but only WS2-TR-006 (Override Rate) carries `reporting_window` /
`data_through` — WS2-TR-005's payload has neither field. The baseline turn
passed specifically by refusing that borrowing. And "flagged warning
(below healthy threshold)" inverted direction on a lower-is-better metric
(false_positive_flag rate — warning means ABOVE threshold).

Prompt half only: rule 6 (Honest Windows) is tightened for the
shared-header case, and a new rule pins polarity-aware direction glosses.
The platform half (emitting the fields / polarity in payloads) is a
separate lane.
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT


def test_rule_6_shared_header_may_not_borrow_window_metadata():
    assert "a shared header or note" in E2I_COPILOT_SYSTEM_PROMPT
    # The honest alternative when sibling payloads differ.
    assert "name the asymmetry instead" in E2I_COPILOT_SYSTEM_PROMPT


def test_status_direction_needs_polarity_rule_present():
    assert "**Status Direction Needs Polarity**" in E2I_COPILOT_SYSTEM_PROMPT
    # The 4.6 shape: warning on a lower-is-better metric means ABOVE.
    assert "lower-is-better" in E2I_COPILOT_SYSTEM_PROMPT
    # The fallback when the payload doesn't state polarity.
    assert "without a direction gloss" in E2I_COPILOT_SYSTEM_PROMPT
