"""#1707: system-health asks route to the audit surface, not activity logs.

Measured flap across three consecutive live runs of turn 5.1 ("What is the
current system health score?"):

* post-#1690 run — routed to ``orchestrator_tool``: the health_score audit's
  payload carried the composite (``overall_health_score`` 99.42,
  ``health_grade`` "A", ``data_provenance`` measured) and the answer served
  "Grade A, 99.4/100".
* post-#1696 certification rerun — routed to ``e2i_data_query_tool``
  (agent_analysis = the agent's activity log): "there isn't a single
  aggregated figure" → the PARTIAL that seeded #1697.
* cert_post1706 — same log surface; honesty now contained (rule 10) but the
  answer served an average of per-run confidences (~0.871) as a proxy while
  the real composite sat one dispatch away.

``_classify_query_type`` is telemetry-only, so the prompt roster is the
routing seam. This pins the roster clause that claims system-health asks for
``orchestrator_tool`` and marks agent analyses as logs, not system health.
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT


def test_system_health_asks_route_to_orchestrator():
    assert "Also use it for SYSTEM-HEALTH asks" in E2I_COPILOT_SYSTEM_PROMPT
    # The serving surface and its composite payload fields, by name.
    assert "health_score audit" in E2I_COPILOT_SYSTEM_PROMPT
    assert "overall_health_score" in E2I_COPILOT_SYSTEM_PROMPT
    assert "health_grade" in E2I_COPILOT_SYSTEM_PROMPT


def test_log_confidence_proxy_forbidden():
    # The cert_post1706 failure shape: avg per-run confidence offered as a
    # stand-in for the composite.
    assert (
        "an average of per-run log confidences is NOT the health score" in E2I_COPILOT_SYSTEM_PROMPT
    )


def test_agent_analyses_marked_as_logs_not_health():
    # The e2i_data_query_tool roster clause that was attracting the ask.
    assert "NOT the system health score" in E2I_COPILOT_SYSTEM_PROMPT


def test_adjacent_rules_still_present():
    # #1697's two surfaces are neighbors of this edit — both load-bearing.
    assert "**Negatives About The Platform**" in E2I_COPILOT_SYSTEM_PROMPT
    assert "Also use it for model-quality asks" in E2I_COPILOT_SYSTEM_PROMPT
