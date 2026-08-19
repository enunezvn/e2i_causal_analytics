"""#1712: rule 8 extension — no fabricated arithmetic bridges between fields.

Eval 3.4 wrote "Estimated duration: 196 days (12-week invitation window +
90-day outcome observation)". The payload carried `duration_estimate_days:
196` and the phase text as SEPARATE fields with no stated relationship, and
the asserted decomposition is arithmetically false: 12 weeks = 84 days,
84 + 90 = 174, not 196. This is the false-total family (baseline 3.6's 546
= 6mo + 3mo defect) moved to a new site; this run's own 3.6 kept the same
two fields correctly separate, so the composition is discretionary drift.

Rule 8 (Units Stay As Declared) already forbids unlicensed arithmetic on
relabeled units; it is extended rather than adding a new rule: joining
numeric payload fields with "+" / "=" / sum phrasing requires the payload
to assert the composition or the arithmetic to be verified first.
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT


def test_rule_8_forbids_unasserted_compositions():
    assert 'join numeric payload fields with "+", "="' in E2I_COPILOT_SYSTEM_PROMPT
    assert "unless the payload itself asserts that composition" in E2I_COPILOT_SYSTEM_PROMPT
    # The verified-arithmetic escape hatch, with the measured counterexample.
    assert "12 weeks = 84 days" in E2I_COPILOT_SYSTEM_PROMPT


def test_rule_8_unrelated_fields_stay_separate():
    # Coexisting duration/total fields with no stated relationship are
    # presented apart (as this run's own 3.6 correctly did).
    assert "no stated relationship are presented separately" in E2I_COPILOT_SYSTEM_PROMPT
