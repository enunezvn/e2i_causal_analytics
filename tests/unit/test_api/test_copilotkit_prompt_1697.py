"""#1697: prompt rule 10 — negatives about the platform.

Three turns of the 2026-08-18 certification eval (session n5: 5.1, 5.3, 5.6 —
all three of its PARTIALs) each queried ONE narrow surface, got little or
nothing, and generalized the emptiness into a platform-level "does not exist"
claim. Each was refuted by the same run's own artifacts:

* 5.1 "no single blended health score exists" — health_score's own
  summary_template renders "Grade: {grade}, Score: {score:.1f}/100"
  (src/agents/health_score/dspy_integration.py), and the baseline run's 5.1
  payload carried "Grade: A, Score: 99.4/100".
* 5.3 "No ROC-AUC or calibration data is available for Kisqali" — turns 1.6
  and 6.1 of the SAME run served holdout_auc 0.7907 via
  predict_hcp_segment_likelihood_tool. 5.3 had routed to agent-activity logs
  (0 rows) instead of the champion surface.
* 5.6 "I don't have a direct query tool exposed for [the agent's] raw
  output" — e2i_data_query_tool(agent_analysis, agent_name=...) is exactly
  that tool, and the session had already used it three times.

Rule 9 governs negatives about the CONVERSATION; this pins the new rule for
negatives about the PLATFORM, plus the 5.3 routing guidance (model-quality
asks go to the champion surface, not activity logs).
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT


def test_rule_10_platform_negatives_present():
    assert "**Negatives About The Platform**" in E2I_COPILOT_SYSTEM_PROMPT
    # The scoping instruction: emptiness licenses a query-scoped negative only.
    assert "query-scoped negative" in E2I_COPILOT_SYSTEM_PROMPT
    # The verification instruction: check the toolbox and the session first.
    assert "check your own toolbox for a tool that serves it" in E2I_COPILOT_SYSTEM_PROMPT
    # The epistemic fallback: absence of evidence, not absence of the thing.
    assert "absence of evidence" in E2I_COPILOT_SYSTEM_PROMPT


def test_rule_10_names_the_measured_shapes():
    # The three measured false negatives ride as examples so the model
    # recognizes the exact recurring surfaces.
    assert "no blended health score exists" in E2I_COPILOT_SYSTEM_PROMPT
    assert "no ROC-AUC data is available" in E2I_COPILOT_SYSTEM_PROMPT
    assert "there is no direct query tool" in E2I_COPILOT_SYSTEM_PROMPT


def test_model_quality_asks_route_to_champion_surface():
    # 5.3 routing regression: ROC-AUC/calibration asks must go to
    # predict_hcp_segment_likelihood_tool (payload carries model_name /
    # holdout_auc / n_scored), never to agent-activity logs.
    assert "Also use it for model-quality asks" in E2I_COPILOT_SYSTEM_PROMPT
    assert "holdout_auc" in E2I_COPILOT_SYSTEM_PROMPT
    assert "a 0-row log query is not evidence that no metric exists" in E2I_COPILOT_SYSTEM_PROMPT


def test_rule_9_still_present():
    # Rule 10 supplements rule 9 (conversation negatives) — both load-bearing.
    assert "**Negatives About This Conversation**" in E2I_COPILOT_SYSTEM_PROMPT
