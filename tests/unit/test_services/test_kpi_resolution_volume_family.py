"""Canonical names resolve to the canonical KPIs; panel names to the panel KPIs."""

import pytest

from src.kpi.registry import get_registry
from src.services.kpi_resolution import (
    _PANEL_MEMBER_ALIASES,
    KPI_SEMANTIC_NOTES,
    recognize_kpi,
    recognize_kpi_span,
)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is Kisqali TRx?", "WS3-BI-005"),
        ("What is the current TRx volume for Fabhalta?", "WS3-BI-005"),
        ("NRx for Remibrutinib", "WS3-BI-006"),
        ("NBRx for Kisqali", "WS3-BI-007"),
        ("TRx share for Kisqali", "WS3-BI-008"),
        ("TRx panel for Kisqali", "WS3-BI-011"),
        ("panel TRx by severity", "WS3-BI-011"),
        ("observed Rx events for Fabhalta", "WS3-BI-011"),
        ("NRx panel", "WS3-BI-012"),
        ("panel NBRx for Kisqali", "WS3-BI-013"),
        ("TRx share panel for Kisqali", "WS3-BI-014"),
        ("panel TRx share", "WS3-BI-014"),
    ],
)
def test_recognizes_the_right_family(query, expected):
    kpi = recognize_kpi(query)
    assert kpi is not None and kpi.id == expected, (query, kpi and kpi.id)


# --- codex r16-01: the cases the 12 above cannot see -------------------------------------
# All twelve pass under the shadowed option A, because they only ever use the short aliases.
# These ask the way a user naming the KPI would, which is what option A breaks.


@pytest.mark.parametrize("kpi_id", ["WS3-BI-011", "WS3-BI-012", "WS3-BI-013", "WS3-BI-014"])
def test_each_panel_kpi_resolves_from_its_own_registry_name(kpi_id):
    """The family name PREFIXES every member name, so a longest-alias-wins matcher sends
    012/013/014 to 011. Verified pre-fix: exactly those three are shadowed.

    SCOPE: this is a RESOLVER test and nothing more. It asserts which id the vocabulary
    binds — never that the chat consumer answers that string. A full registry name is not a
    value-lookup shape at all (see the note below Step 4); asserting an answer here, or
    reusing this string in a consumer test, would make a working #1475 guard look broken."""
    name = str(get_registry().get(kpi_id).name)
    kpi = recognize_kpi(name)
    assert kpi is not None and kpi.id == kpi_id, (name, kpi and kpi.id)


def test_the_family_phrase_is_an_explicit_decision_not_a_registry_accident():
    """`observed rx events` must resolve through an ALIAS, never through _best_name_match —
    where all four panel KPIs tie on matched tokens and registry ORDER decides. Reversing that
    order under the fallback flips this query to WS3-BI-014, a share: a different unit."""
    match = recognize_kpi_span("observed Rx events for Fabhalta")
    assert match is not None
    kpi, normalized, start, end = match
    assert kpi.id == "WS3-BI-011"
    assert normalized[start:end] == "observed rx events"


@pytest.mark.parametrize(
    "query,expected_id,expected_span_text",
    [
        ("NRx panel", "WS3-BI-012", "nrx panel"),
        ("cost of NRx panel", "WS3-BI-012", "nrx panel"),
        ("NBRx panel drivers", "WS3-BI-013", "nbrx panel"),
        ("TRx share panel for Kisqali", "WS3-BI-014", "trx share panel"),
        ("What is Kisqali TRx?", "WS3-BI-005", "trx"),
    ],
)
def test_the_span_points_at_the_matched_alias(query, expected_id, expected_span_text):
    """The #1475 governing-head guards slice this span: 'cost of X' makes X a modifier,
    'X drivers' makes X a causal outcome. A pre-pass that resolves the right id with the
    wrong span breaks those guards silently, so pin the span, not just the id."""
    match = recognize_kpi_span(query)
    assert match is not None, query
    kpi, normalized, start, end = match
    assert kpi.id == expected_id, (query, kpi.id)
    assert normalized[start:end] == expected_span_text, (query, normalized[start:end])


def test_member_aliases_are_matched_before_the_generic_alias_loop():
    """Structural, not incidental: the pre-pass exists because length order cannot express
    'specific before generic' when the generic string is the longer one.

    The two data assertions describe the dict. The BEHAVIOURAL one below is what actually
    witnesses the ordering: the plan asserted this test fails when the pre-pass is disabled,
    and with only the data assertions it did NOT -- they inspect `_PANEL_MEMBER_ALIASES`'s
    contents, which the control leaves defined, so the test could not fail for the reason it
    is named after. Measured under the option-A control: data-only version PASSED with the
    pre-pass commented out.
    """
    for alias, kpi_id in _PANEL_MEMBER_ALIASES.items():
        assert kpi_id.startswith("WS3-BI-01")
        # Every member alias is SHORTER than the family alias it must nonetheless beat.
        assert len(alias) < len("observed rx events") or "patient panel" in alias

    # A query carrying BOTH the family phrase and a member phrase must bind the MEMBER,
    # even though "observed rx events" (18) is longer than "nrx panel" (9) and appears
    # first. Under the generic length-ordered loop alone this resolves to WS3-BI-011.
    for phrase, expected in (
        ("observed rx events patient panel nrx", "WS3-BI-012"),
        ("observed rx events patient panel nbrx", "WS3-BI-013"),
        ("observed rx events patient panel trx share", "WS3-BI-014"),
        ("observed rx events patient panel trx", "WS3-BI-011"),
    ):
        kpi = recognize_kpi(phrase)
        assert kpi is not None and kpi.id == expected, (phrase, kpi and kpi.id)


def test_the_panel_share_carries_the_portfolio_note():
    assert KPI_SEMANTIC_NOTES["WS3-BI-014"] == KPI_SEMANTIC_NOTES["WS3-BI-008"]
