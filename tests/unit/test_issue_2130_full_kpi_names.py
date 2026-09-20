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
        # Current main has no stored TRx-share key: ``market_share`` is a
        # different modeled quantity. Keep the transparent fallback until
        # #2159's canonical handler owns this phrase.
        ("TRx Share", "trx_share"),
        ("share of total prescriptions", "trx_share"),
    ],
)
def test_full_kpi_name_uses_a_safe_business_metric_filter_key(display_name, stored_name):
    from src.api.routes.chatbot_tools import _normalize_metric_name

    assert _normalize_metric_name(display_name) == stored_name


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "Show me total prescriptions for Kisqali",
        "Show me Total Prescriptions (TRx) for Kisqali",
        "What are the new prescriptions for Fabhalta?",
        "What are New Prescriptions (NRx) for Fabhalta?",
        "How many new-to-brand prescriptions were there?",
        "Show me new/to/brand prescriptions",
        "Tell me about TRx share in the Northeast",
        "What is the share of total prescriptions for Kisqali?",
    ],
)
def test_full_kpi_name_routes_to_the_value_lookup_evidence_path(query):
    from src.agents.orchestrator.nodes.intent_classifier import KPI_VALUE_LOOKUP_RE

    assert KPI_VALUE_LOOKUP_RE.search(query), query


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "What are the conversion rates for Kisqali?",
        "Show me the market shares for the three brands",
        "Tell me about conversion/rates in the Northeast",
    ],
)
def test_inflected_kpi_name_routes_with_the_shared_suffix_and_separator_rules(query):
    """Recognition, filtering, and direct routing must share one grammar."""
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
        ("conversion rates", "conversion_rate"),
        ("market shares", "market_share"),
        ("TRx's", "trx"),
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
    ("display_name", "unsafe_stored_name"),
    [
        ("Total Prescriptions (patients)", "trx"),
        ("New Prescriptions (patients)", "nrx"),
        ("TRx Share (market share)", "trx_share"),
    ],
)
def test_unknown_parenthetical_qualifier_does_not_collapse_to_the_base_metric(
    display_name, unsafe_stored_name
):
    """Only the registry abbreviation is presentation metadata.

    A quantity/axis in parentheses can change what the caller asked for.  It
    must not be discarded and silently filtered as the unqualified KPI.
    """
    from src.api.routes.chatbot_tools import _normalize_metric_name

    assert _normalize_metric_name(display_name) != unsafe_stored_name


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "Show me the prescription details for this patient",
        "Tell me about a new brand campaign",
        "How many total prescription errors occurred?",
        "How many patients received new prescriptions?",
        "How many HCPs wrote total prescriptions?",
        "Show me the patient count for new prescriptions",
        "What is the HCP count for total prescriptions?",
        "Give me prescriber counts for NRx",
        "Show me total prescriptions (patients) for Kisqali",
        "Tell me about TRx Share (market share)",
        "How many people received new prescriptions?",
        "How many individuals received new prescriptions?",
        "How many pharmacies filled new prescriptions?",
        "Tell me about how many people received new prescriptions?",
        "Show me patients receiving new prescriptions",
        "Give me pharmacies filling new prescriptions",
        "Tell me about people with new prescriptions",
        "How many total calls did the HCP receive?",
        "Show me the total prescriptions forecast for next month",
    ],
)
def test_full_name_routing_does_not_create_false_positive_value_lookups(query):
    from src.agents.orchestrator.nodes.intent_classifier import KPI_VALUE_LOOKUP_RE

    assert KPI_VALUE_LOOKUP_RE.search(query) is None, query
