"""#1709: prompt rule 11 — verify directional premises before adopting them.

Eval A.10 (2026-08-19 full eval) headed its causal section "Kisqali TRx
Decline (Northeast) - Causal Drivers", adopting the user's unverified
"decline" premise as framing with no non-overlapping prior-period
comparison — while the kpi payload's own `window_coverage` pacing signal
(trailing_30d_share 0.3895 vs uniform_expected_share 0.3297) argued
AGAINST a late-quarter decline and went unused. The same answer refused
the parallel Remibrutinib "flat" premise correctly, so the discipline
exists but was applied to one premise and not the other in a single turn.

Rules 9/10 govern negatives (conversation / platform); neither governs
adopting a user's directional premise as framing. Rule 11 pins it.
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT


def test_rule_11_directional_premise_rule_present():
    assert "**Verify Directional Premises**" in E2I_COPILOT_SYSTEM_PROMPT
    # Premise adoption in framing is gated on a real comparison.
    assert "non-overlapping comparison" in E2I_COPILOT_SYSTEM_PROMPT
    # The honest fallback when no comparison exists.
    assert "the premise is unverified" in E2I_COPILOT_SYSTEM_PROMPT


def test_rule_11_uses_window_coverage_pacing_fields():
    # A.10's payload carried the disconfirming pacing signal unused.
    assert "`window_coverage`" in E2I_COPILOT_SYSTEM_PROMPT
    assert "`trailing_30d_share`" in E2I_COPILOT_SYSTEM_PROMPT
    assert "`uniform_expected_share`" in E2I_COPILOT_SYSTEM_PROMPT


def test_rule_11_one_standard_for_every_premise_in_turn():
    # A.10 verified the Remibrutinib premise while adopting the Kisqali
    # one — the same-turn double standard is named as the failure mode.
    assert "EVERY premise in the turn" in E2I_COPILOT_SYSTEM_PROMPT
