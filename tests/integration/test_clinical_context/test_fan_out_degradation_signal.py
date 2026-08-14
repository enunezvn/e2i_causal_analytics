"""Degradation signal for the clinical-context fan-out (#1612 AC4).

The four providers are individually fail-open, and ``_fan_out`` caches a
fully-live result *indefinitely* while caching a degraded one for only
``_FRAGMENT_TTL_DEGRADED_S``. That asymmetry is deliberate and correct, but it
means "every provider fell back" looks identical to "everything is fine" from
the outside: the payload still assembles, the endpoint still returns 200, and
the only trace is a ``logger.warning``.

This module asserts that ``fully_live`` is *reachable* for a known brand, so an
across-the-board outage is detectable rather than inferred from a log grep.

Measured 2026-08-14: Kisqali and Fabhalta both reach fully-live; Remibrutinib
legitimately does not (ChEMBL has no mechanism row for it and the PubMed
relevance search returns nothing for its term), so it degrades honestly to
``static_fallback`` / ``unavailable``. Anchoring on Kisqali therefore tests the
signal without encoding a false expectation about Remibrutinib.
"""

from __future__ import annotations

import pytest

from src.services.clinical_context.brand_map import resolve_brand_profile
from src.services.clinical_context.service import ClinicalContextService, reset_caches
from tests.integration.test_clinical_context._live_gate import requires_network

pytestmark = [pytest.mark.integration, pytest.mark.slow, requires_network]

# The four live-API sources that gate the fully-live decision. Competitors are
# curated by design and deliberately excluded from the check.
_LIVE_SOURCES = {
    "mechanism": "chembl",
    "endpoints": "clinicaltrials.gov",
    "citation": "pubmed",
    "indications": "openfda",
}


@pytest.fixture(autouse=True)
def _clear_fragment_cache() -> None:
    """Force a real fan-out; otherwise a cached entry would fake the result."""
    reset_caches()


def test_fully_live_fan_out_is_reachable_for_kisqali() -> None:
    """All four live providers must land for at least one known brand.

    A failure here means the fan-out is degraded across the board — the exact
    silent state #1612 was filed about. It names the offending provider(s) so
    the nightly failure is actionable without a log dive.
    """
    service = ClinicalContextService()
    profile = resolve_brand_profile("Kisqali")
    moa, eps, cite, indications, _competitors = service._fan_out(profile)

    actual = {
        "mechanism": moa.source,
        "endpoints": eps.source,
        "citation": cite.source,
        "indications": indications.source,
    }
    degraded = {k: v for k, v in actual.items() if v != _LIVE_SOURCES[k]}
    assert not degraded, (
        "clinical-context fan-out is degraded for Kisqali — these providers fell "
        f"back instead of returning live data: {degraded}. Expected {_LIVE_SOURCES}."
    )


def test_degraded_provider_is_labelled_not_silently_dropped() -> None:
    """Fail-open must stay *honest*: a fallback is labelled, never passed off as live.

    Remibrutinib is the measured natural example — ChEMBL carries no mechanism
    row for it, so the mechanism fragment falls back. The contract under test is
    not "Remibrutinib degrades" (that could change upstream at any time) but
    "whatever the source is, it is one of the declared values" — so a fallback
    can never masquerade as live.
    """
    service = ClinicalContextService()
    profile = resolve_brand_profile("Remibrutinib")
    moa, eps, cite, indications, competitors = service._fan_out(profile)

    assert moa.source in {"chembl", "static_fallback"}
    assert eps.source in {"clinicaltrials.gov", "static_fallback"}
    assert cite.source in {"pubmed", "pubmed_seed", "unavailable"}
    assert indications.source in {"openfda", "static_fallback"}
    assert competitors.source == "curated"

    # When the mechanism degrades, the payload must still carry the curated
    # fallback text rather than an empty/None value the UI would render blank.
    if moa.source == "static_fallback":
        assert moa.mechanism_of_action, "degraded mechanism fragment carries no fallback text"


def test_get_context_payload_carries_live_provenance() -> None:
    """The assembled payload must reach callers with LIVE provenance labels.

    codex review LOW (#1612): an earlier version asserted only that each
    ``source`` key was non-empty, which would stay green if all four APIs broke
    — every fragment still carries a ``static_fallback`` label. In the live lane
    a test without teeth is worse than no test, so this asserts the live source
    values end-to-end through ``get_context`` (the path the API routes and the
    Executive Brief actually call), not merely that a label exists.
    """
    service = ClinicalContextService()
    payload = service.get_context("Kisqali", "adherence")

    assert payload["mechanism"]["source"] == "chembl"
    assert payload["pivotal_endpoints"]["source"] == "clinicaltrials.gov"
    assert payload["approved_indications"]["source"] == "openfda"
    assert payload["mechanism"]["mechanism_of_action"], "live payload carries no MoA text"
    assert payload["approved_indications"]["indications"], "live payload carries no indications"

    # The honesty label distinguishes the SYNTHETIC effect estimate from the
    # REAL public-source clinical context; losing it would misrepresent both.
    assert "SYNTHETIC" in payload["honesty_label"]
    assert "REAL" in payload["honesty_label"]
