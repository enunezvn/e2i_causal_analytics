"""#1711: prompt rule 12 — within-metric superlatives over mixed-metric tables.

Eval 1.7 asserted "Northeast has the lowest engagement achievement (63.0%
of target) among the four regions shown" when the engagement metric existed
for only THREE of the four regions (West had only trx / market_share rows).
The Achievement column mixed three metrics, and 63.0 WAS the minimum of the
whole column, so a value-column reader (the synthesis guard) finds no
contradiction — the defect is scope, not value. The claim was load-bearing:
it was the stated basis for the primary rep-capacity recommendation.

Rule 12 pins the domain discipline: a superlative over a metric quantifies
only over rows that carry that metric, and mixed-metric tables get an
explicit domain qualifier.
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT


def test_rule_12_superlative_domain_rule_present():
    assert "**Superlatives Quantify Over Metric Carriers**" in E2I_COPILOT_SYSTEM_PROMPT
    # The domain restriction: only rows that carry the metric count.
    assert "only over rows that actually carry metric X" in E2I_COPILOT_SYSTEM_PROMPT


def test_rule_12_mixed_metric_tables_need_explicit_qualifier():
    # The 1.7 shape: a mixed-metric value column silently widens the domain.
    assert "mixes different metrics" in E2I_COPILOT_SYSTEM_PROMPT
    assert '"of the three regions with engagement data"' in E2I_COPILOT_SYSTEM_PROMPT
