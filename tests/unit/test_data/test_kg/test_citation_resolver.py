"""Unit tests for CitationResolver — the citation-verification layer.

Tests use stubbed clients so the interaction with Europe PMC + Crossref
+ UMLS is asserted without network calls.
"""

from __future__ import annotations

from typing import Optional

import pytest

from src.data.kg.citation_resolver import (
    CAUSAL_CUE_VERBS,
    WEIGHT_BOTH_ENTITIES,
    WEIGHT_CAUSAL_CUE,
    WEIGHT_COOCCURRENCE,
    CitationResolver,
    CitationResolverError,
    _find_causal_cue,
    _first_match,
)
from src.data.kg.types import AbstractRecord, KGConcept
from src.data.kg.umls_uts import UMLSAuthError, UMLSError


class _StubEuropePMC:
    def __init__(self, *, abstracts: Optional[dict[str, AbstractRecord]] = None) -> None:
        self._abstracts = abstracts or {}
        self.calls: list[tuple[str, ...]] = []

    def fetch_abstract(self, pmid: str) -> Optional[AbstractRecord]:
        self.calls.append(("fetch_abstract", pmid))
        return self._abstracts.get(pmid)

    def close(self) -> None:
        pass


class _StubCrossref:
    def __init__(self, *, doi_records: Optional[dict[str, AbstractRecord]] = None) -> None:
        self._doi_records = doi_records or {}
        self.calls: list[tuple[str, ...]] = []

    def fetch_doi_metadata(self, doi: str) -> Optional[AbstractRecord]:
        self.calls.append(("fetch_doi_metadata", doi))
        return self._doi_records.get(doi)

    def close(self) -> None:
        pass


class _StubUMLS:
    def __init__(
        self,
        *,
        concepts: Optional[dict[str, KGConcept]] = None,
        raise_auth: bool = False,
        raise_error: bool = False,
    ) -> None:
        self._concepts = concepts or {}
        self._raise_auth = raise_auth
        self._raise_error = raise_error

    def cui_lookup(self, cui: str) -> KGConcept:
        if self._raise_auth:
            raise UMLSAuthError("simulated auth fail")
        if self._raise_error:
            raise UMLSError("simulated transport fail")
        return self._concepts.get(cui, KGConcept(cui=cui, preferred_name=""))

    def close(self) -> None:
        pass


def _record(abstract: str, *, kind: str = "pmid", identifier: str = "12345678") -> AbstractRecord:
    return AbstractRecord(
        identifier=identifier,
        identifier_kind=kind,  # type: ignore[arg-type]
        title="Test",
        abstract=abstract,
        source="europe_pmc",
    )


def _resolver(
    *,
    europe_pmc: Optional[_StubEuropePMC] = None,
    crossref: Optional[_StubCrossref] = None,
    umls: Optional[_StubUMLS] = None,
) -> CitationResolver:
    return CitationResolver(
        europe_pmc=europe_pmc if europe_pmc is not None else _StubEuropePMC(),  # type: ignore[arg-type]
        crossref=crossref if crossref is not None else _StubCrossref(),  # type: ignore[arg-type]
        umls=umls if umls is not None else _StubUMLS(),  # type: ignore[arg-type]
    )


def test_first_match_returns_first_substring() -> None:
    haystack = "atopic dermatitis is treated by ibuprofen".lower()
    assert _first_match(["Ibuprofen", "naproxen"], haystack) == "Ibuprofen"
    assert _first_match(["aspirin"], haystack) is None


def test_first_match_skips_empty_terms() -> None:
    assert _first_match(["", " "], "abc") is None


def test_find_causal_cue_returns_first_present() -> None:
    """Whole-word match — 'reproduced' shouldn't false-positive 'induced'."""
    assert _find_causal_cue("the drug treats the disease".lower()) == "treats"
    # No causal cues, just associative language.
    assert _find_causal_cue("there is an association between x and y".lower()) is None


def test_find_causal_cue_uses_word_boundaries() -> None:
    """The literal substring 'induced' must NOT match inside 'reproduced'."""
    assert _find_causal_cue("the experiment was reproduced in 2018".lower()) is None


def test_verify_citation_full_match_max_confidence() -> None:
    """Both entities + causal cue + co-occurrence → max score."""

    europe_pmc = _StubEuropePMC(
        abstracts={"12345": _record("Ibuprofen treats atopic dermatitis effectively.")}
    )
    verdict = _resolver(europe_pmc=europe_pmc).verify_citation(
        "12345",
        identifier_kind="pmid",
        subject_name="Ibuprofen",
        object_name="atopic dermatitis",
    )
    assert verdict.abstract_resolved
    assert "Ibuprofen" in verdict.entities_found
    assert "atopic dermatitis" in verdict.entities_found
    assert verdict.causal_cue_found == "treats"
    assert verdict.overall_confidence == pytest.approx(
        WEIGHT_BOTH_ENTITIES + WEIGHT_CAUSAL_CUE + WEIGHT_COOCCURRENCE
    )


def test_verify_citation_entities_only_no_cue() -> None:
    """Both entities present, no causal cue → 0.5 score."""

    europe_pmc = _StubEuropePMC(
        abstracts={
            "12345": _record(
                "We observed atopic dermatitis in patients on ibuprofen with no significant relationship."
            )
        }
    )
    verdict = _resolver(europe_pmc=europe_pmc).verify_citation(
        "12345",
        identifier_kind="pmid",
        subject_name="ibuprofen",
        object_name="atopic dermatitis",
    )
    assert verdict.abstract_resolved
    assert verdict.causal_cue_found is None
    assert verdict.overall_confidence == pytest.approx(WEIGHT_BOTH_ENTITIES)


def test_verify_citation_cue_only_no_entities_yields_zero_confidence() -> None:
    """Codex review HIGH (2026-05-08): cue-only without entities → 0.0.

    A causal cue verb alone does not evidence a citation; the abstract
    could be about an unrelated relation. Without this guard, unrelated
    abstracts containing common cue verbs would silently rank above
    unresolved citations.
    """

    europe_pmc = _StubEuropePMC(
        abstracts={"12345": _record("Aspirin treats inflammation in many patients.")}
    )
    verdict = _resolver(europe_pmc=europe_pmc).verify_citation(
        "12345",
        identifier_kind="pmid",
        subject_name="ibuprofen",
        object_name="atopic dermatitis",
    )
    assert verdict.abstract_resolved
    assert verdict.entities_found == ()
    # Cue is still surfaced for diagnostics, but confidence is zero
    # because no entity matched.
    assert verdict.causal_cue_found == "treats"
    assert verdict.overall_confidence == 0.0


def test_verify_citation_one_entity_plus_cue_no_credit() -> None:
    """Only ONE of the two entities present + cue → still 0.0.

    Verifies the both-entities gate guards the cue-credit path even when
    one entity matches.
    """

    europe_pmc = _StubEuropePMC(
        abstracts={"12345": _record("Ibuprofen treats inflammation in many patients.")}
    )
    verdict = _resolver(europe_pmc=europe_pmc).verify_citation(
        "12345",
        identifier_kind="pmid",
        subject_name="ibuprofen",
        object_name="atopic dermatitis",  # not in the abstract
    )
    assert verdict.abstract_resolved
    # ``_first_match`` returns the candidate term as supplied (not the
    # cased form from the abstract); the match is case-insensitive.
    assert "ibuprofen" in verdict.entities_found
    # Subject match alone — both-entities gate fails → 0.0.
    assert verdict.overall_confidence == 0.0


def test_verify_citation_zero_when_unresolved() -> None:
    """Abstract not retrievable → confidence 0, abstract_resolved False."""

    europe_pmc = _StubEuropePMC()  # no abstracts registered
    verdict = _resolver(europe_pmc=europe_pmc).verify_citation(
        "missing",
        identifier_kind="pmid",
        subject_name="ibuprofen",
        object_name="atopic dermatitis",
    )
    assert not verdict.abstract_resolved
    assert verdict.overall_confidence == 0.0


def test_verify_citation_dispatches_doi_to_crossref() -> None:
    """identifier_kind=doi → uses crossref client."""

    crossref = _StubCrossref(
        doi_records={
            "10.1234/abc": _record(
                "Ibuprofen treats atopic dermatitis.",
                kind="doi",
                identifier="10.1234/abc",
            )
        }
    )
    verdict = _resolver(crossref=crossref).verify_citation(
        "10.1234/abc",
        identifier_kind="doi",
        subject_name="Ibuprofen",
        object_name="atopic dermatitis",
    )
    assert verdict.abstract_resolved
    assert verdict.identifier_kind == "doi"
    assert crossref.calls[0] == ("fetch_doi_metadata", "10.1234/abc")


def test_verify_citation_uses_umls_synonym_when_cui_provided() -> None:
    """If CUI is given, UMLS preferred name is added to the candidate terms."""

    europe_pmc = _StubEuropePMC(
        abstracts={
            # Abstract uses "Dermatitis, Atopic" (the UMLS preferred name)
            # rather than the user-supplied "atopic dermatitis".
            "12345": _record("Ibuprofen treats Dermatitis, Atopic.")
        }
    )
    umls = _StubUMLS(
        concepts={"C0011615": KGConcept(cui="C0011615", preferred_name="Dermatitis, Atopic")}
    )
    verdict = _resolver(europe_pmc=europe_pmc, umls=umls).verify_citation(
        "12345",
        identifier_kind="pmid",
        subject_name="Ibuprofen",
        object_name="rare-form-name",  # not in the abstract
        object_cui="C0011615",
    )
    # Object matched via the UMLS synonym.
    assert "Dermatitis, Atopic" in verdict.entities_found
    assert verdict.causal_cue_found == "treats"


def test_verify_citation_propagates_umls_auth_error() -> None:
    """UMLS auth dead → CitationResolverError surfaces, not silent failure."""

    europe_pmc = _StubEuropePMC(abstracts={"12345": _record("Ibuprofen treats atopic dermatitis.")})
    umls = _StubUMLS(raise_auth=True)
    with pytest.raises(CitationResolverError):
        _resolver(europe_pmc=europe_pmc, umls=umls).verify_citation(
            "12345",
            identifier_kind="pmid",
            subject_name="Ibuprofen",
            object_name="atopic dermatitis",
            object_cui="C0011615",
        )


def test_verify_citation_swallows_umls_transient_error() -> None:
    """Generic UMLS transport failure during synonym expansion: degrade,
    still attempt match against the user-supplied name only."""

    europe_pmc = _StubEuropePMC(abstracts={"12345": _record("Ibuprofen treats atopic dermatitis.")})
    umls = _StubUMLS(raise_error=True)
    verdict = _resolver(europe_pmc=europe_pmc, umls=umls).verify_citation(
        "12345",
        identifier_kind="pmid",
        subject_name="Ibuprofen",
        object_name="atopic dermatitis",
        object_cui="C0011615",
    )
    assert verdict.abstract_resolved
    assert "Ibuprofen" in verdict.entities_found
    assert "atopic dermatitis" in verdict.entities_found


def test_verify_citation_unsupported_identifier_kind() -> None:
    """Bogus identifier_kind → error CitationVerdict, not raise."""

    verdict = _resolver().verify_citation(
        "12345",
        identifier_kind="orcid",
        subject_name="X",
        object_name="Y",
    )
    assert not verdict.abstract_resolved
    assert verdict.error is not None
    assert "unsupported" in verdict.error


def test_resolve_pmid_swallows_europe_pmc_error() -> None:
    """Transport failures degrade gracefully — return None."""

    class _BrokenEuropePMC:
        def fetch_abstract(self, pmid: str) -> Optional[AbstractRecord]:
            from src.data.kg.europe_pmc import EuropePMCError

            raise EuropePMCError("simulated")

        def close(self) -> None:
            pass

    resolver = _resolver(europe_pmc=_BrokenEuropePMC())  # type: ignore[arg-type]
    assert resolver.resolve_pmid("anything") is None


def test_causal_cue_verbs_list_has_expected_terms() -> None:
    """Sanity: the curated cue list covers the high-frequency verbs."""
    for verb in ("treats", "causes", "induces", "inhibits", "prevents"):
        assert verb in CAUSAL_CUE_VERBS
