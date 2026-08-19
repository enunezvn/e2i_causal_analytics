"""#1710: registry query phrasing, headline honesty, exclusivity negatives.

Eval 5.7 ("did the rep-visit effect pass refutation testing?") produced two
coupled defects:

* The bolded headline credited "the rep-detailing effect" with refutation
  results that belonged to `treatment_arm` — while the answer's OWN closing
  paragraph disclaimed exactly that ("I can't report refutation-test
  results for rep_detailing_high specifically, only for treatment_arm").
* "rep_detailing_high ... appears ONLY in the heterogeneous_optimizer CATE
  runs - it does not appear in this causal-paths registry" — false; the
  baseline run's `causal_analysis_tool(kpi_name='rep visit')` returned the
  registry row (0.043 / 0.930 / 5-of-5). The narrow query
  (kpi_name='treatment_initiated' + brand) missed it, and the narrow result
  was generalized into an unscoped registry-level exclusivity claim.

Three tightenings, no new rule: the `causal_analysis_tool` bullet gains
query-by-the-user's-phrasing + name-the-substituted-variable guidance;
rule 10 gains exclusivity negatives; the Response Format bold bullet gains
headline-must-not-contradict-body.
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT


def test_causal_registry_queried_by_users_phrasing():
    assert "by the USER'S variable phrasing" in E2I_COPILOT_SYSTEM_PROMPT
    # Substitution is allowed but must be named, headline included.
    assert '"closest match"' in E2I_COPILOT_SYSTEM_PROMPT
    assert "name the substituted variable" in E2I_COPILOT_SYSTEM_PROMPT


def test_exclusivity_negatives_are_rule_10_scoped():
    # "X appears ONLY in Y" claims get the same query-scoping as other
    # platform negatives — asserted only over sources actually queried.
    assert "Exclusivity negatives" in E2I_COPILOT_SYSTEM_PROMPT
    assert '"X appears ONLY in Y"' in E2I_COPILOT_SYSTEM_PROMPT


def test_bold_headline_may_not_assert_what_body_disclaims():
    assert "never assert what the same answer's body disclaims" in E2I_COPILOT_SYSTEM_PROMPT
    assert "the disclaimer wins" in E2I_COPILOT_SYSTEM_PROMPT
