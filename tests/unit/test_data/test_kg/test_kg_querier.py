"""Unit tests for KnowledgeGraphQuerier (Phase 2.3).

Tests use stub UMLS + Open Targets clients to assert the wiring + KGEdge
shape contract; httpx-level behavior is covered by the per-client unit
tests already in this folder.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from src.data.kg.kg_querier import KnowledgeGraphQuerier, _extract_trailing_cui
from src.data.kg.open_targets import OpenTargetsError
from src.data.kg.types import KGEdge
from src.data.kg.umls_uts import UMLSAuthError, UMLSError


class _StubUMLS:
    def __init__(
        self,
        *,
        relations: Optional[list[dict[str, Any]]] = None,
        raise_auth: bool = False,
        raise_error: bool = False,
    ) -> None:
        self._relations = relations or []
        self._raise_auth = raise_auth
        self._raise_error = raise_error
        self.calls: list[tuple[str, ...]] = []

    def cui_relations(self, cui: str, *, page_size: int = 50) -> list[dict[str, Any]]:
        self.calls.append(("cui_relations", cui))
        if self._raise_auth:
            raise UMLSAuthError("simulated auth fail")
        if self._raise_error:
            raise UMLSError("simulated transport fail")
        return self._relations

    def close(self) -> None:
        pass


class _StubOT:
    def __init__(
        self,
        *,
        evidence: Optional[dict[str, Any]] = None,
        raise_error: bool = False,
    ) -> None:
        self._evidence = evidence or {"evidences": {"rows": []}}
        self._raise_error = raise_error
        self.calls: list[tuple[str, ...]] = []

    def drug_disease_evidence(
        self, drug_id: str, disease_id: str, *, size: int = 25
    ) -> dict[str, Any]:
        self.calls.append(("drug_disease_evidence", drug_id, disease_id))
        if self._raise_error:
            raise OpenTargetsError("simulated OT fail")
        return self._evidence

    def close(self) -> None:
        pass


def _querier(*, umls: _StubUMLS, ot: _StubOT) -> KnowledgeGraphQuerier:
    return KnowledgeGraphQuerier(umls=umls, open_targets=ot)  # type: ignore[arg-type]


def test_extract_trailing_cui_pulls_cui_from_url() -> None:
    url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011615"
    assert _extract_trailing_cui(url) == "C0011615"


def test_extract_trailing_cui_handles_trailing_slash() -> None:
    url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011615/"
    assert _extract_trailing_cui(url) == "C0011615"


def test_extract_trailing_cui_returns_empty_for_non_cui() -> None:
    assert _extract_trailing_cui("https://example.com/foo/bar") == ""
    assert _extract_trailing_cui("") == ""


def test_extract_trailing_cui_rejects_none_sentinel() -> None:
    url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/NONE"
    assert _extract_trailing_cui(url) == ""


def test_extract_trailing_cui_rejects_atom_ui() -> None:
    """Atom UIs (A...) are not CUIs; reject them so KGQuerier doesn't emit
    edges with non-CUI object_ids."""
    url = "https://uts-ws.nlm.nih.gov/rest/content/current/AUI/A12345"
    assert _extract_trailing_cui(url) == ""


def test_querier_close_only_closes_self_constructed_clients() -> None:
    """Borrowed clients must NOT be closed by KGQuerier.close()."""

    class _CloseTracking:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _BorrowedUMLS(_CloseTracking):
        def cui_relations(self, cui: str, *, page_size: int = 50) -> list[dict[str, Any]]:
            return []

    class _BorrowedOT(_CloseTracking):
        def drug_disease_evidence(
            self, drug_id: str, disease_id: str, *, size: int = 25
        ) -> dict[str, Any]:
            return {"evidences": {"rows": []}}

    borrowed_umls = _BorrowedUMLS()
    borrowed_ot = _BorrowedOT()
    querier = KnowledgeGraphQuerier(
        umls=borrowed_umls,  # type: ignore[arg-type]
        open_targets=borrowed_ot,  # type: ignore[arg-type]
    )
    querier.close()
    assert borrowed_umls.closed is False
    assert borrowed_ot.closed is False


def test_query_concept_relations_does_not_dedupe_cross_source_rows() -> None:
    """Multiple sources asserting the same relation produce multiple edges.

    UMLS often has both MSH and SNOMEDCT_US (or other vocabularies) asserting
    the same parent relation. v1 emits one edge per source-asserted row;
    callers can de-dup by (subject, predicate, object) if they want.
    """
    umls = _StubUMLS(
        relations=[
            {
                "relationLabel": "RB",
                "additionalRelationLabel": "isa",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011603",
                "rootSource": "MSH",
            },
            {
                "relationLabel": "RB",
                "additionalRelationLabel": "isa",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011603",
                "rootSource": "SNOMEDCT_US",
            },
        ]
    )
    ot = _StubOT()
    edges = _querier(umls=umls, ot=ot).query_disease_hierarchy("C0011615")
    assert len(edges) == 2
    sources = {e.datasource for e in edges}
    assert sources == {"MSH", "SNOMEDCT_US"}


def test_query_drug_disease_edges_happy_path() -> None:
    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "count": 2,
                "rows": [
                    {
                        "score": 0.91,
                        "datatypeId": "literature",
                        "datasourceId": "europepmc",
                        "literature": ["12345678", "87654321"],
                        "drug": {"id": "CHEMBL1234", "name": "DrugA"},
                        "disease": {"id": "EFO_0000270", "name": "atopic dermatitis"},
                    },
                    {
                        "score": 0.45,
                        "datatypeId": "clinical_trial",
                        "datasourceId": "clinical_trials",
                        "literature": [],
                        "drug": {"id": "CHEMBL1234", "name": "DrugA"},
                        "disease": {"id": "EFO_0000270", "name": "atopic dermatitis"},
                    },
                ],
            }
        }
    )
    edges = _querier(umls=umls, ot=ot).query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert len(edges) == 2
    e0 = edges[0]
    assert e0.subject_id == "CHEMBL1234"
    assert e0.subject_name == "DrugA"
    assert e0.object_id == "EFO_0000270"
    assert e0.predicate == "associated_with"
    assert e0.evidence_source == "open_targets"
    assert e0.score == pytest.approx(0.91)
    assert e0.pmids == ("12345678", "87654321")
    assert e0.datasource == "europepmc"
    assert edges[1].pmids == ()  # no literature on the second row
    assert edges[1].datasource == "clinical_trials"


def test_query_drug_disease_edges_empty_when_no_evidence() -> None:
    umls = _StubUMLS()
    ot = _StubOT(evidence={"evidences": {"rows": []}})
    assert _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y") == []


def test_query_drug_disease_edges_handles_null_rows() -> None:
    """GraphQL ``[Evidence!]`` (nullable list) returns null on resolver error;
    must collapse to [] without raising TypeError."""
    umls = _StubUMLS()
    ot = _StubOT(evidence={"evidences": {"rows": None}})
    assert _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y") == []


def test_query_drug_disease_edges_handles_null_evidences_object() -> None:
    """If the entire ``evidences`` block is null, still degrade to []."""
    umls = _StubUMLS()
    ot = _StubOT(evidence={"evidences": None})
    assert _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y") == []


def test_query_drug_disease_edges_swallows_open_targets_error() -> None:
    """Transport failures degrade gracefully — KGQuerier returns []."""
    umls = _StubUMLS()
    ot = _StubOT(raise_error=True)
    assert _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y") == []


def test_query_drug_disease_edges_rejects_nan_score() -> None:
    """NaN scores must collapse to None so ranking logic isn't poisoned."""
    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "rows": [
                    {
                        "score": float("nan"),
                        "datasourceId": "europepmc",
                        "literature": [],
                        "drug": {"id": "X", "name": "X"},
                        "disease": {"id": "Y", "name": "Y"},
                    }
                ]
            }
        }
    )
    edges = _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y")
    assert len(edges) == 1
    assert edges[0].score is None


def test_query_drug_disease_edges_rejects_inf_score() -> None:
    """+/-inf scores must also collapse to None."""
    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "rows": [
                    {
                        "score": float("inf"),
                        "datasourceId": "europepmc",
                        "literature": [],
                        "drug": {"id": "X"},
                        "disease": {"id": "Y"},
                    }
                ]
            }
        }
    )
    edges = _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y")
    assert len(edges) == 1
    assert edges[0].score is None


def test_query_drug_disease_edges_handles_missing_score_key() -> None:
    """If the row simply has no 'score' key, edge.score must be None."""
    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "rows": [
                    {
                        "datasourceId": "europepmc",
                        "literature": [],
                        "drug": {"id": "X"},
                        "disease": {"id": "Y"},
                    }
                ]
            }
        }
    )
    edges = _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y")
    assert len(edges) == 1
    assert edges[0].score is None


def test_query_disease_hierarchy_filters_to_taxonomic_predicates() -> None:
    umls = _StubUMLS(
        relations=[
            {
                "relationLabel": "RB",
                "additionalRelationLabel": "isa",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011603",
                "relatedIdName": "Dermatitis",
                "rootSource": "MSH",
            },
            {
                "relationLabel": "RO",
                "additionalRelationLabel": "may_be_treated_by",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0020740",
                "relatedIdName": "Ibuprofen",
                "rootSource": "MSH",
            },
            {
                "relationLabel": "RN",
                "additionalRelationLabel": "inverse_isa",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011620",
                "relatedIdName": "Atopic Dermatitis (variant)",
                "rootSource": "SNOMEDCT_US",
            },
        ]
    )
    ot = _StubOT()
    edges = _querier(umls=umls, ot=ot).query_disease_hierarchy("C0011615")
    # Only the two taxonomic edges (isa, inverse_isa) should pass; the
    # may_be_treated_by edge is filtered out.
    assert len(edges) == 2
    predicates = {e.predicate for e in edges}
    assert predicates == {"isa", "inverse_isa"}
    for e in edges:
        assert e.subject_id == "C0011615"
        assert e.evidence_source == "umls_relations"
        assert e.score is None
        assert e.pmids == ()


def test_query_disease_hierarchy_accepts_coarse_label_when_fine_empty() -> None:
    """If additionalRelationLabel is empty but relationLabel is PAR/CHD/RB/RN,
    treat as taxonomic."""
    umls = _StubUMLS(
        relations=[
            {
                "relationLabel": "PAR",
                "additionalRelationLabel": "",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011603",
                "relatedIdName": "Dermatitis",
                "rootSource": "MSH",
            }
        ]
    )
    ot = _StubOT()
    edges = _querier(umls=umls, ot=ot).query_disease_hierarchy("C0011615")
    assert len(edges) == 1
    assert edges[0].predicate == "par"


def test_query_concept_relations_no_filter_returns_all() -> None:
    umls = _StubUMLS(
        relations=[
            {
                "relationLabel": "RO",
                "additionalRelationLabel": "may_treat",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011615",
                "relatedIdName": "Atopic Dermatitis",
                "rootSource": "MSH",
            }
        ]
    )
    ot = _StubOT()
    edges = _querier(umls=umls, ot=ot).query_concept_relations("C0020740")
    assert len(edges) == 1
    assert edges[0].predicate == "may_treat"


def test_query_concept_relations_filters_by_predicate() -> None:
    umls = _StubUMLS(
        relations=[
            {
                "relationLabel": "RO",
                "additionalRelationLabel": "may_treat",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011615",
            },
            {
                "relationLabel": "RO",
                "additionalRelationLabel": "contraindicated_with",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0030193",
            },
        ]
    )
    ot = _StubOT()
    edges = _querier(umls=umls, ot=ot).query_concept_relations("C0020740", predicates={"may_treat"})
    assert len(edges) == 1
    assert edges[0].object_id == "C0011615"


def test_query_concept_relations_skips_rows_without_valid_cui() -> None:
    umls = _StubUMLS(
        relations=[
            {
                "relationLabel": "RO",
                "additionalRelationLabel": "may_treat",
                "relatedId": "not a url",
            },
            {
                "relationLabel": "RO",
                "additionalRelationLabel": "may_treat",
                # missing relatedId entirely
            },
        ]
    )
    ot = _StubOT()
    edges = _querier(umls=umls, ot=ot).query_concept_relations("C0020740")
    assert edges == []


def test_query_concept_relations_propagates_auth_error() -> None:
    """UMLS auth failures must surface (systemic), not silently degrade."""
    umls = _StubUMLS(raise_auth=True)
    ot = _StubOT()
    with pytest.raises(UMLSAuthError):
        _querier(umls=umls, ot=ot).query_concept_relations("C0011615")


def test_query_concept_relations_swallows_generic_umls_error() -> None:
    """Transport-level UMLS failures degrade gracefully."""
    umls = _StubUMLS(raise_error=True)
    ot = _StubOT()
    assert _querier(umls=umls, ot=ot).query_concept_relations("C0011615") == []


def test_querier_borrows_clients_from_entity_linker() -> None:
    """When constructed with an EntityLinker, KGQuerier reuses its clients."""
    from src.data.kg.entity_linker import EntityLinker

    class _StubRxNav:
        def close(self) -> None:
            pass

    umls = _StubUMLS()
    ot = _StubOT()
    linker = EntityLinker(
        umls=umls,  # type: ignore[arg-type]
        rxnav=_StubRxNav(),  # type: ignore[arg-type]
        open_targets=ot,  # type: ignore[arg-type]
    )
    querier = KnowledgeGraphQuerier(entity_linker=linker)
    assert querier.umls is umls
    assert querier.open_targets is ot


def test_kgedge_immutability() -> None:
    """KGEdge is frozen so callers can't mutate cached results."""
    edge = KGEdge(
        subject_id="X",
        predicate="treats",
        object_id="Y",
        evidence_source="open_targets",
    )
    with pytest.raises(Exception):  # FrozenInstanceError is a dataclass-specific subclass
        edge.score = 0.5  # type: ignore[misc]
