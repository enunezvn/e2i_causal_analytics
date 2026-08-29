"""Live end-to-end coverage for the Phase 2.6 citation channel (#1608).

Covers the two acceptance criteria that need real network:

* **AC5** — a verified citation reaches the ``EnsembleVerdict`` evidence, and an
  unresolvable one degrades without failing the run.
* **AC6** — both sides of the resolver are exercised: the Europe PMC **PMID**
  path (already covered by ``test_citation_resolver_live.py``) and the Crossref
  **DOI** path (which had no live coverage at all).

Unlike ``test_citation_resolver_live.py`` this module is gated on **network
only**, not on ``UMLS_UTS_API_KEY``. Europe PMC and Crossref are zero-auth; UMLS
only widens the term list with synonyms. The UMLS-gated module skips in every CI
lane because no workflow sets that secret, so gating these on it would have made
them dead on arrival.

Marked ``slow`` so they run in ``slow-tests.yml`` Job A (``pytest tests/ -m
slow``) on the 05:00 UTC schedule rather than on every PR — third-party
endpoints do not belong on the PR-blocking lane.

Fixtures below were all measured live on 2026-08-14:

* PMID ``33730455`` (Pegcetacoplan vs Eculizumab in PNH) verifies for
  (pegcetacoplan, hemoglobinuria): both entities found, causal cue "inhibits",
  confidence 1.00.
* PMID ``27176981`` is "Enhanced absorption of graphene monolayer with a
  single-layer resonant grating" — a physics paper that the compiled Layer-4
  classifier nonetheless cites. It RESOLVES, so an existence-only check passes
  it; entity matching against pharma terms returns nothing and it is correctly
  NOT verified. This is the signal the channel adds.
* DOI ``10.1186/s13058-023-01623-6`` has a Crossref-deposited abstract (1781
  chars). Publisher coverage is uneven — NEJM deposits none, which is why
  ``CrossrefClient`` returning None for an NEJM DOI is correct behaviour and not
  a bug.
"""

from __future__ import annotations

import httpx
import pytest

from src.data.kg.citation_resolver import CitationResolver
from src.data.kg.crossref import reset_caches as reset_crossref_caches
from src.data.kg.ensemble_voter import EnsembleVoter, is_citation_verified
from src.data.kg.europe_pmc import reset_caches as reset_europepmc_caches
from src.data.kg.types import LLMVerdict

_GATE_URL = "https://connectivitycheck.gstatic.com/generate_204"


def _network_available() -> bool:
    try:
        return httpx.get(_GATE_URL, timeout=8.0, follow_redirects=True).status_code < 500
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not _network_available(), reason="No outbound network (#1608)."),
]

_VERIFIABLE_PMID = "33730455"
_VERIFIABLE_SUBJECT = "pegcetacoplan"
_VERIFIABLE_OBJECT = "hemoglobinuria"
# Real PMID, real abstract, completely off-topic for this classifier's domain.
_OFF_TOPIC_PMID = "27176981"
# A DOI whose publisher actually deposits an abstract to Crossref.
_CROSSREF_DOI = "10.1186/s13058-023-01623-6"

_ADVERSARIAL_MODERATE = {
    "layer": "3",
    "severity": "moderate",
    "remediation": "ambiguous",
    "evidence": "live citation-channel test",
    "z_score": 3.5,
    "actual_auc": 0.62,
    "null_mean": 0.50,
    "null_std": 0.035,
    "p_value": 0.001,
    "n_permutations": 200,
    "_hblp_classified": True,
}


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_europepmc_caches()
    reset_crossref_caches()


# ------------------------------------------------------------------ AC6: Crossref


def test_crossref_doi_path_resolves_a_real_abstract() -> None:
    """The Crossref DOI leg had NO live coverage before #1608."""
    with CitationResolver() as resolver:
        record = resolver.resolve_doi(_CROSSREF_DOI)

    assert record is not None, (
        f"Crossref returned no record for {_CROSSREF_DOI}. Note publisher coverage "
        "is uneven — the client correctly returns None when no abstract is "
        "deposited — but this DOI is known to carry one."
    )
    assert record.identifier_kind == "doi"
    assert record.source == "crossref"
    assert record.abstract.strip(), "Crossref record carried an empty abstract"
    # JATS/XHTML tags must be stripped so the substring matcher sees clean text.
    assert "<" not in record.abstract, f"markup leaked into the abstract: {record.abstract[:120]!r}"


def test_verify_citation_accepts_the_doi_identifier_kind() -> None:
    """``verify_citation(identifier_kind="doi")`` must route through Crossref."""
    with CitationResolver() as resolver:
        verdict = resolver.verify_citation(
            _CROSSREF_DOI,
            identifier_kind="doi",
            subject_name="breast cancer",
            object_name="return-to-work",
        )

    assert verdict.identifier == _CROSSREF_DOI
    assert verdict.identifier_kind == "doi"
    assert verdict.abstract_resolved, "the DOI path did not resolve an abstract"
    assert any("breast cancer" in term.lower() for term in verdict.entities_found), (
        f"expected a breast-cancer match in entities_found; got {verdict.entities_found!r}"
    )


# ----------------------------------------------------------- AC5: reaches evidence


def test_verified_citation_reaches_ensemble_verdict_evidence() -> None:
    """A genuinely verified citation must surface in the voter's audit trail."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )

    with CitationResolver() as resolver:
        verdict = resolver.verify_citation(
            _VERIFIABLE_PMID,
            identifier_kind="pmid",
            subject_name=_VERIFIABLE_SUBJECT,
            object_name=_VERIFIABLE_OBJECT,
        )

    assert is_citation_verified(verdict), (
        f"PMID {_VERIFIABLE_PMID} should verify for "
        f"({_VERIFIABLE_SUBJECT}, {_VERIFIABLE_OBJECT}); got entities="
        f"{verdict.entities_found!r} cue={verdict.causal_cue_found!r}"
    )

    legacy = _compose_legacy_verdict(
        _VERIFIABLE_SUBJECT,
        voter=EnsembleVoter(),
        adversarial_input=dict(_ADVERSARIAL_MODERATE),
        llm_verdict=LLMVerdict(
            causal_role="ancestor",
            mechanism=f"complement inhibition; PMID: {_VERIFIABLE_PMID}",
            recommended_remediation="keep",
            cited_pmids=(_VERIFIABLE_PMID,),
        ),
        citation_verdicts=(verdict,),
    )

    assert legacy["citations_checked"] == 1
    assert legacy["citations_verified"] == 1
    assert legacy["verified_citation_ids"] == [_VERIFIABLE_PMID]


def test_off_topic_citation_resolves_but_fails_verification() -> None:
    """The load-bearing case: resolvable is NOT the same as verified.

    PMID 27176981 is a graphene-physics paper cited by the compiled pharma
    causal-role classifier. Any existence-only check passes it. Only the entity
    co-mention + causal-cue check rejects it — which is the whole point of
    routing cited PMIDs through ``CitationResolver`` before they land in an
    audit trail looking like evidence.
    """
    with CitationResolver() as resolver:
        verdict = resolver.verify_citation(
            _OFF_TOPIC_PMID,
            identifier_kind="pmid",
            subject_name="ribociclib",
            object_name="breast cancer",
        )

    assert verdict.abstract_resolved, (
        "expected this PMID to RESOLVE — the test is meaningless otherwise, since "
        "its point is that resolution alone does not imply relevance"
    )
    assert not is_citation_verified(verdict), (
        "an off-topic abstract must NOT pass verification; got entities="
        f"{verdict.entities_found!r} cue={verdict.causal_cue_found!r}"
    )
    assert verdict.entities_found == (), (
        f"no pharma entity should match a graphene paper; got {verdict.entities_found!r}"
    )


def test_unresolvable_citation_degrades_without_failing() -> None:
    """#1608 AC3/AC5 — an unresolvable PMID yields an honest unverified verdict."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _CitationBudget,
        _resolve_citation_verdicts,
    )

    bogus = "99999999"
    with CitationResolver() as resolver:
        verdicts = _resolve_citation_verdicts(
            LLMVerdict(
                causal_role="ancestor",
                mechanism=f"fabricated citation; PMID: {bogus}",
                recommended_remediation="keep",
                cited_pmids=(bogus,),
            ),
            feature="serum_marker",
            contract=None,
            target="responder",
            resolver=resolver,
            budget=_CitationBudget(limit=5),
        )

    # The run continues; the citation is simply recorded as not verified.
    assert len(verdicts) == 1
    assert not is_citation_verified(verdicts[0])
    assert not verdicts[0].abstract_resolved


def test_resolver_is_usable_without_a_umls_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Europe PMC + Crossref are zero-auth; UMLS only adds synonyms.

    Constructing ``UMLSClient()`` unconditionally previously made the whole
    resolver unconstructible without ``UMLS_UTS_API_KEY``, which would have
    disabled this channel entirely in CI and in any deployment lacking the key.
    """
    monkeypatch.delenv("UMLS_UTS_API_KEY", raising=False)
    with CitationResolver() as resolver:
        assert resolver.umls is None
        record = resolver.resolve_pmid(_VERIFIABLE_PMID)
    assert record is not None, "Europe PMC must still work without a UMLS key"
    assert record.abstract.strip()
