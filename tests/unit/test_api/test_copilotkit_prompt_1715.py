"""#1715: rule 7 synthetic-disclosure anchoring for multi-tool answers.

The 2026-08-19 full eval measured rule-7 drift: 10/10 disclosure in
sessions n1/n2 but 5/7 in n5n6 and 5/7 in the appendix — two regressions
(A.9-seed, 6.2) and two standing misses (A.9-followup, 6.5), all on turns
whose payloads carry `data_source: 'synthetic'` and/or
`evidence_is_synthetic: true`. The misses skew toward long multi-tool
answers (6.2 = six calls) where the disclosure competes with content.

This is reinforcement/placement of the EXISTING rule 7, not a new rule:
the obligation is restated as per-ANSWER (any one payload in the turn
triggers it), with an explicit multi-tool callout, and a second anchor in
Response Format where the final answer is composed.
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT


def test_rule_7_disclosure_attaches_per_answer():
    assert "attaches PER ANSWER" in E2I_COPILOT_SYSTEM_PROMPT
    # Any one synthetic-flagged payload among many binds the whole answer.
    assert "ANY payload in the turn" in E2I_COPILOT_SYSTEM_PROMPT
    # The measured drift site is named.
    assert "long multi-tool answers" in E2I_COPILOT_SYSTEM_PROMPT


def test_response_format_carries_second_anchor():
    # Placement nudge: a re-check where the final answer is composed.
    assert "re-check rule 7" in E2I_COPILOT_SYSTEM_PROMPT


def test_rule_7_original_trigger_fields_still_present():
    # The existing rule's trigger fields remain load-bearing.
    assert 'data_source: "synthetic"' in E2I_COPILOT_SYSTEM_PROMPT
    assert "evidence_is_synthetic: true" in E2I_COPILOT_SYSTEM_PROMPT
