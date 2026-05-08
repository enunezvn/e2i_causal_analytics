"""Live integration test for CitationResolver (Phase 2.6).

Exercises Europe PMC against a real PMID and asserts the verification
pipeline produces a sensible CitationVerdict end-to-end.

Network targets:
    - Europe PMC (zero-auth, CC0).
    - UMLS UTS REST (per-developer key) for synonym expansion.

Skipped automatically when ``UMLS_UTS_API_KEY`` is absent (UMLS is the
synonym-expansion path that exercises the most code; without it the test
would only verify the Europe PMC happy-path which is already covered by
the mock-transport unit tests).

PMID choice:
    ``28846349`` — "Atopic Dermatitis" (StatPearls, 2025 revision). Chosen
    because it's a stable Europe PMC record with a 540-char abstract that
    repeatedly mentions atopic dermatitis. Initial v1 picked PMID
    ``20051597`` but Europe PMC's record for that PMID has no
    ``abstractText`` — fine for the prior auto-pass test, breaks the
    strengthened v2 hard-assert. The test asserts the abstract resolves
    and the entity is found; the causal-cue assertion is loose because
    review abstracts may not contain the curated cue verbs.
"""

from __future__ import annotations

import os

import pytest

from src.data.kg.citation_resolver import CitationResolver
from src.data.kg.europe_pmc import reset_caches as reset_europepmc_caches
from src.data.kg.umls_uts import reset_caches as reset_umls_caches

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_API_KEY = os.environ.get("UMLS_UTS_API_KEY")
_REASON = "UMLS_UTS_API_KEY not set; skipping live CitationResolver test."

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _API_KEY, reason=_REASON),
]


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_europepmc_caches()
    reset_umls_caches()


def test_resolve_pmid_returns_real_abstract() -> None:
    """A well-known atopic-dermatitis PMID must produce a non-empty abstract.

    Codex review LOW (2026-05-08): the previous version had an
    ``_allow_unresolved`` helper that always returned True, making the
    test pass even when Europe PMC silently broke. v2 hard-asserts the
    record is fetched and the abstract is non-empty; transient Europe
    PMC outages will surface as test failures rather than silently
    slipping through. The slow-tests workflow re-runs are the right
    place to absorb genuine transient unavailability.
    """
    with CitationResolver() as resolver:
        record = resolver.resolve_pmid("28846349")
    assert record is not None, (
        "Europe PMC returned no record for PMID 28846349 — either a "
        "regression in EuropePMCClient validation logic or a transient "
        "Europe PMC outage. Re-run on the slow-tests workflow before "
        "declaring a hard regression."
    )
    assert record.identifier == "28846349"
    assert record.identifier_kind == "pmid"
    assert record.source == "europe_pmc"
    assert len(record.abstract) > 0
    # The atopic-dermatitis review's abstract must mention the topic.
    assert "dermatitis" in record.abstract.lower()


def test_verify_citation_finds_atopic_dermatitis_entity() -> None:
    """Verify a known atopic-dermatitis review actually mentions the term."""
    with CitationResolver() as resolver:
        verdict = resolver.verify_citation(
            "28846349",
            identifier_kind="pmid",
            subject_name="atopic dermatitis",
            object_name="treatment",
            subject_cui="C0011615",  # UMLS preferred-name fan-out
        )
    assert verdict.identifier == "28846349"
    assert verdict.identifier_kind == "pmid"
    assert verdict.abstract_resolved, (
        "verify_citation must resolve the abstract for a well-known PMID; "
        "if this fails, EuropePMCClient is broken upstream."
    )
    # When the abstract resolves, the subject term must appear (it's the
    # paper's topic). The object term ("treatment") may or may not appear.
    assert any("dermatitis" in term.lower() for term in verdict.entities_found), (
        f"Expected an atopic-dermatitis match in entities_found; got {verdict.entities_found!r}"
    )
