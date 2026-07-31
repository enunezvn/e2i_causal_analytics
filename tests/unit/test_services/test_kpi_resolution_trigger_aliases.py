"""KPI-vocabulary aliases for the trigger-effectiveness KPIs (#1360).

Before #1360, trigger-KPI phrasings leaned on the conservative name-token
fallback, which mis-resolved compound asks: any query containing "trigger"
matched Trigger Precision first (so "trigger acceptance rate" -> WS2-TR-001),
and "trigger funnel conversion" fell through to the WS3-BI-009 "conversion"
alias. Explicit aliases run BEFORE the fallback (longest alias wins), fixing
both without touching the fallback.
"""

import pytest

from src.services.kpi_resolution import recognize_kpi


@pytest.mark.unit
@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("trigger precision", "WS2-TR-001"),
        ("What is our trigger precision this month?", "WS2-TR-001"),
        ("acceptance rate", "WS2-TR-004"),
        # The pre-#1360 bug: the 'trigger' name-token matched Trigger Precision.
        ("trigger acceptance rate", "WS2-TR-004"),
        ("override rate", "WS2-TR-006"),
        ("trigger override rate", "WS2-TR-006"),
        ("trigger funnel conversion", "WS2-TR-009"),
        ("funnel conversion", "WS2-TR-009"),
        ("trigger funnel", "WS2-TR-009"),
        ("how is the trigger funnel looking", "WS2-TR-009"),
    ],
)
def test_trigger_effectiveness_aliases_resolve(query, expected_id):
    kpi = recognize_kpi(query)
    assert kpi is not None, f"{query!r} did not resolve"
    assert kpi.id == expected_id, f"{query!r} -> {kpi.id} (wanted {expected_id})"


@pytest.mark.unit
@pytest.mark.parametrize(
    "query,expected_id",
    [
        # WS3-BI-009 Conversion Rate must keep winning plain 'conversion' asks —
        # the funnel aliases are longer and only fire on funnel phrasings.
        ("conversion rate", "WS3-BI-009"),
        ("what is the conversion rate for Kisqali", "WS3-BI-009"),
        ("trx", "WS3-BI-005"),
        ("nbrx", "WS3-BI-007"),
    ],
)
def test_existing_alias_resolution_unchanged(query, expected_id):
    kpi = recognize_kpi(query)
    assert kpi is not None and kpi.id == expected_id
