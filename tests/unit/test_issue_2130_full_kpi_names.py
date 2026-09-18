"""Regression coverage for full-name KPI value lookups (#2130)."""

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    ("display_name", "stored_name"),
    [
        ("Total Prescriptions", "trx"),
        ("Total Prescriptions (TRx)", "trx"),
        ("total prescription", "trx"),
        ("New Prescriptions", "nrx"),
        ("New-to-Brand Prescriptions", "nbrx"),
        ("new to brand prescriptions", "nbrx"),
        ("TRx Share", "market_share"),
        ("share of total prescriptions", "market_share"),
    ],
)
def test_full_kpi_name_uses_the_stored_business_metric_key(display_name, stored_name):
    from src.api.routes.chatbot_tools import _normalize_metric_name

    assert _normalize_metric_name(display_name) == stored_name


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "Show me total prescriptions for Kisqali",
        "What are the new prescriptions for Fabhalta?",
        "How many new-to-brand prescriptions were there?",
        "Tell me about TRx share in the Northeast",
        "What is the share of total prescriptions for Kisqali?",
    ],
)
def test_full_kpi_name_routes_to_the_value_lookup_evidence_path(query):
    from src.agents.orchestrator.nodes.intent_classifier import KPI_VALUE_LOOKUP_RE

    assert KPI_VALUE_LOOKUP_RE.search(query), query


@pytest.mark.unit
def test_shared_business_metric_aliases_agree_with_kpi_recognition():
    from src.kpi.business_metric_vocabulary import (
        BUSINESS_METRIC_KPI_ALIASES,
        canonical_business_metric_name,
    )
    from src.services.kpi_resolution import recognize_kpi

    for phrase, expected_id in BUSINESS_METRIC_KPI_ALIASES.items():
        recognized = recognize_kpi(phrase)
        assert recognized is not None and recognized.id == expected_id, phrase
        assert canonical_business_metric_name(phrase) is not None, phrase


@pytest.mark.unit
@pytest.mark.parametrize(
    ("display_name", "stored_name"),
    [
        ("TRx", "trx"),
        ("NRx", "nrx"),
        ("NBRx", "nbrx"),
        ("Market Share", "market_share"),
        ("conversion-rate", "conversion_rate"),
        # Unsupported recognized KPIs must not be conflated with a different
        # stored metric merely because some words overlap.
        ("HCP Coverage", "hcp_coverage"),
        # Unknown values retain the existing transparent passthrough contract.
        ("Custom Metric", "custom_metric"),
    ],
)
def test_abbreviations_and_unsafe_mismatches_keep_their_existing_behavior(
    display_name, stored_name
):
    from src.api.routes.chatbot_tools import _normalize_metric_name

    assert _normalize_metric_name(display_name) == stored_name


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "Show me the prescription details for this patient",
        "Tell me about a new brand campaign",
        "How many total prescription errors occurred?",
        "How many total calls did the HCP receive?",
        "Show me the total prescriptions forecast for next month",
    ],
)
def test_full_name_routing_does_not_create_false_positive_value_lookups(query):
    from src.agents.orchestrator.nodes.intent_classifier import KPI_VALUE_LOOKUP_RE

    assert KPI_VALUE_LOOKUP_RE.search(query) is None, query
