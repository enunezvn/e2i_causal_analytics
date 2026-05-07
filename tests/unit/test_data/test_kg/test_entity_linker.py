"""Unit tests for EntityLinker, using stubbed wrapper clients.

EntityLinker is composition; we stub its constituent clients (UMLS, RxNav,
Open Targets) rather than mocking httpx. Each test asserts only the wiring:
which client method is called for which input, and how their outputs collapse
into ``EntityLink`` records.
"""

from __future__ import annotations

from typing import Optional

import pytest

from src.data.kg.entity_linker import EntityLinker, EntityLinkerError
from src.data.kg.types import KGConcept
from src.data.kg.umls_uts import UMLSAuthError, UMLSError


class _StubUMLS:
    def __init__(
        self,
        *,
        cui_for_codes: Optional[dict[tuple[str, str], str]] = None,
        concepts: Optional[dict[str, KGConcept]] = None,
        search_results: Optional[list[dict[str, str]]] = None,
        raise_auth: bool = False,
        raise_error_on_lookup: bool = False,
    ) -> None:
        self._cui_for_codes = cui_for_codes or {}
        self._concepts = concepts or {}
        self._search_results = search_results or []
        self._raise_auth = raise_auth
        self._raise_error_on_lookup = raise_error_on_lookup
        self.calls: list[tuple[str, ...]] = []

    def code_to_cui(self, code: str, *, source: str) -> Optional[str]:
        self.calls.append(("code_to_cui", code, source))
        if self._raise_auth:
            raise UMLSAuthError("simulated auth fail")
        return self._cui_for_codes.get((code, source))

    def cui_lookup(self, cui: str) -> KGConcept:
        self.calls.append(("cui_lookup", cui))
        if self._raise_auth:
            raise UMLSAuthError("simulated auth fail")
        if self._raise_error_on_lookup:
            raise UMLSError("simulated lookup fail")
        return self._concepts.get(
            cui,
            KGConcept(cui=cui, preferred_name="<unknown>"),
        )

    def search(
        self, term: str, *, page_size: int = 5, search_type: str = "exact"
    ) -> list[dict[str, str]]:
        self.calls.append(("search", term))
        if self._raise_auth:
            raise UMLSAuthError("simulated auth fail")
        return self._search_results

    def close(self) -> None:
        pass


class _StubRxNav:
    def __init__(self, *, name_to_rxcui: Optional[dict[str, str]] = None) -> None:
        self._name_to_rxcui = name_to_rxcui or {}
        self.calls: list[tuple[str, ...]] = []

    def rxcui_for_name(self, name: str) -> Optional[str]:
        self.calls.append(("rxcui_for_name", name))
        return self._name_to_rxcui.get(name)

    def close(self) -> None:
        pass


class _StubOT:
    def close(self) -> None:
        pass


def _linker(
    *,
    umls: _StubUMLS,
    rxnav: Optional[_StubRxNav] = None,
    ot: Optional[_StubOT] = None,
) -> EntityLinker:
    return EntityLinker(
        umls=umls,  # type: ignore[arg-type]
        rxnav=rxnav if rxnav is not None else _StubRxNav(),  # type: ignore[arg-type]
        open_targets=ot if ot is not None else _StubOT(),  # type: ignore[arg-type]
    )


def test_resolve_icd10_happy_path() -> None:
    umls = _StubUMLS(
        cui_for_codes={("L20.9", "ICD10CM"): "C0011615"},
        concepts={
            "C0011615": KGConcept(
                cui="C0011615",
                preferred_name="Dermatitis, Atopic",
                semantic_types=("Disease or Syndrome",),
                atom_count=576,
            )
        },
    )
    link = _linker(umls=umls).resolve_icd10("L20.9")
    assert link.resolved
    assert link.concept is not None
    assert link.concept.cui == "C0011615"
    assert link.concept.preferred_name == "Dermatitis, Atopic"
    assert link.input_system == "ICD10CM"
    assert link.sources == ("ICD10CM",)


def test_resolve_loinc_uses_lnc_source() -> None:
    """LOINC must be mapped to UTS source 'LNC' under the hood."""
    umls = _StubUMLS(cui_for_codes={("12345-6", "LNC"): "C9999999"})
    link = _linker(umls=umls).resolve_loinc("12345-6")
    assert link.resolved
    # The first call to code_to_cui must have used 'LNC' not 'LOINC'.
    code_to_cui_calls = [c for c in umls.calls if c[0] == "code_to_cui"]
    assert code_to_cui_calls[0] == ("code_to_cui", "12345-6", "LNC")


def test_resolve_returns_unresolved_link_when_code_unknown() -> None:
    umls = _StubUMLS(cui_for_codes={})
    link = _linker(umls=umls).resolve_icd10("ZZZ")
    assert not link.resolved
    assert link.concept is None
    assert link.error is None  # absence is not an error


def test_resolve_empty_code_returns_link_without_calling_umls() -> None:
    umls = _StubUMLS()
    link = _linker(umls=umls).resolve_icd10("")
    assert not link.resolved
    assert link.error == "empty code"
    assert umls.calls == []


def test_resolve_drug_name_via_rxnav_then_umls() -> None:
    umls = _StubUMLS(
        cui_for_codes={("5640", "RXNORM"): "C0020740"},
        concepts={
            "C0020740": KGConcept(
                cui="C0020740",
                preferred_name="Ibuprofen",
                semantic_types=("Pharmacologic Substance",),
            )
        },
    )
    rxnav = _StubRxNav(name_to_rxcui={"ibuprofen": "5640"})
    link = _linker(umls=umls, rxnav=rxnav).resolve_drug_name("ibuprofen")
    assert link.resolved
    assert link.concept is not None
    assert link.concept.preferred_name == "Ibuprofen"
    # RxNav must have been called first.
    assert rxnav.calls[0] == ("rxcui_for_name", "ibuprofen")


def test_resolve_drug_name_falls_back_to_umls_search() -> None:
    """When RxNav has no match, EntityLinker falls back to UMLS free-text search."""
    umls = _StubUMLS(
        search_results=[{"ui": "C0011111", "name": "FreeText", "rootSource": "MTH"}],
        concepts={"C0011111": KGConcept(cui="C0011111", preferred_name="FreeText match")},
    )
    rxnav = _StubRxNav(name_to_rxcui={})
    link = _linker(umls=umls, rxnav=rxnav).resolve_drug_name("obscure-drug")
    assert link.resolved
    assert link.concept is not None
    assert link.concept.cui == "C0011111"
    # Should have tried RxNav, then fallen back to search.
    rxnav_calls = [c for c in rxnav.calls if c[0] == "rxcui_for_name"]
    search_calls = [c for c in umls.calls if c[0] == "search"]
    assert len(rxnav_calls) == 1
    assert len(search_calls) == 1


def test_resolve_drug_name_rxcui_not_in_umls_falls_back_to_search() -> None:
    """RxNav resolves the name but UMLS has no concept for that RxCUI.

    EntityLinker should fall through to UMLS free-text search rather than
    return an unresolved link, since the user asked for a drug name and we
    have a fallback path available.
    """
    umls = _StubUMLS(
        cui_for_codes={},  # RxCUI not in UMLS
        search_results=[{"ui": "C0011615", "name": "From search", "rootSource": "MTH"}],
        concepts={"C0011615": KGConcept(cui="C0011615", preferred_name="From search")},
    )
    rxnav = _StubRxNav(name_to_rxcui={"obscure-drug": "999999"})
    link = _linker(umls=umls, rxnav=rxnav).resolve_drug_name("obscure-drug")
    assert link.resolved
    assert link.concept is not None
    assert link.concept.cui == "C0011615"
    # Should have tried the RxCUI path first, then fallen back to search.
    assert ("code_to_cui", "999999", "RXNORM") in umls.calls
    assert any(c[0] == "search" for c in umls.calls)


def test_resolve_drug_name_handles_rxnav_exception_gracefully() -> None:
    class _BrokenRxNav(_StubRxNav):
        def rxcui_for_name(self, name: str) -> Optional[str]:
            raise RuntimeError("rxnav transient failure")

    umls = _StubUMLS(
        search_results=[{"ui": "C0011111", "name": "x", "rootSource": "MTH"}],
        concepts={"C0011111": KGConcept(cui="C0011111", preferred_name="x")},
    )
    rxnav = _BrokenRxNav()
    link = _linker(umls=umls, rxnav=rxnav).resolve_drug_name("x")
    assert link.resolved


def test_auth_failure_raises_entity_linker_error() -> None:
    umls = _StubUMLS(raise_auth=True)
    with pytest.raises(EntityLinkerError):
        _linker(umls=umls).resolve_icd10("L20.9")


def test_cui_lookup_failure_returns_partial_link() -> None:
    """If code_to_cui succeeds but cui_lookup fails, return the CUI with no name."""
    umls = _StubUMLS(
        cui_for_codes={("L20.9", "ICD10CM"): "C0011615"},
        raise_error_on_lookup=True,
    )
    link = _linker(umls=umls).resolve_icd10("L20.9")
    assert link.resolved  # we still have a CUI
    assert link.concept is not None
    assert link.concept.cui == "C0011615"
    assert link.concept.preferred_name == ""


def test_resolve_dispatches_by_system() -> None:
    umls = _StubUMLS(cui_for_codes={("99213", "CPT"): "C0011000"})
    link = _linker(umls=umls).resolve("99213", "CPT")
    assert link.resolved
    assert link.input_system == "CPT"


def test_resolve_drug_name_empty_returns_unresolved() -> None:
    umls = _StubUMLS()
    rxnav = _StubRxNav()
    link = _linker(umls=umls, rxnav=rxnav).resolve_drug_name("")
    assert not link.resolved
    assert link.error == "empty name"
    assert rxnav.calls == []
    assert umls.calls == []
