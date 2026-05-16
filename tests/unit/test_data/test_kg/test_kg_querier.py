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


def test_query_drug_disease_edges_propagates_open_targets_error() -> None:
    """Transport failures must surface so callers can distinguish "no
    edges" from "GraphQL/transport failure" (codex H1 from PR #102 review).
    Cache builders, EnsembleVoter, and operator-facing pipelines need
    typed errors to record ``status=source_error`` instead of
    ``status=queried_no_edges``."""
    umls = _StubUMLS()
    ot = _StubOT(raise_error=True)
    with pytest.raises(OpenTargetsError):
        _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y")


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


def test_query_concept_relations_propagates_generic_umls_error() -> None:
    """Transport-level UMLS failures must surface so callers can
    distinguish "no relations" from a transport failure (codex H1 from
    PR #102 review). Cache record producers downstream need this signal
    to set ``status=source_error`` instead of misclassifying as
    ``status=queried_no_edges``."""
    umls = _StubUMLS(raise_error=True)
    ot = _StubOT()
    with pytest.raises(UMLSError):
        _querier(umls=umls, ot=ot).query_concept_relations("C0011615")


def test_query_disease_hierarchy_propagates_generic_umls_error() -> None:
    """``query_disease_hierarchy`` delegates to ``query_concept_relations``;
    the same propagation contract must hold there so cache builders can
    surface transport failures even when the upstream method is the
    taxonomic one."""
    umls = _StubUMLS(raise_error=True)
    ot = _StubOT()
    with pytest.raises(UMLSError):
        _querier(umls=umls, ot=ot).query_disease_hierarchy("C0011615")


def test_query_disease_hierarchy_propagates_auth_error() -> None:
    """Auth failures must surface through the public hierarchy method too,
    even though it delegates to query_concept_relations internally."""
    umls = _StubUMLS(raise_auth=True)
    ot = _StubOT()
    with pytest.raises(UMLSAuthError):
        _querier(umls=umls, ot=ot).query_disease_hierarchy("C0011615")


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


# ---------------------------------------------------------------------------
# PR-0: predicate-by-datatypeId contract test (Phase 2.9 Stage 2 prerequisite)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "datatype_id, datasource_id, expected_predicate",
    [
        # known_drug (the ONLY treats-bearing datatype per Ochoa 2021):
        ("known_drug", "chembl", "treats"),
        ("known_drug", "clinical_trials", "treats"),
        # All six other canonical Open Targets datatypeIds (Ochoa 2021,
        # NAR Table 1) — every one must produce associated_with, not
        # treats. Codex PR-0 review L6: cover the full taxonomy so a
        # future code change that drifts the partition is caught.
        ("literature", "europepmc", "associated_with"),
        ("genetic_association", "eva", "associated_with"),
        ("affected_pathway", "progeny", "associated_with"),
        ("rna_expression", "expression_atlas", "associated_with"),
        ("somatic_mutation", "cancer_biomarkers", "associated_with"),
        ("animal_model", "phenodigm", "associated_with"),
    ],
)
def test_query_drug_disease_edges_predicate_by_datatype(
    datatype_id: str, datasource_id: str, expected_predicate: str
) -> None:
    """Each Open Targets datatypeId maps to one semantic predicate.

    The Open Targets data model (Ochoa 2021, NAR) defines seven canonical
    ``datatypeId`` values. Only ``known_drug`` carries drug-treats-disease
    semantics; all others are gene/target-disease association, literature,
    or pathway evidence. PR-0 maps ``datatypeId == "known_drug"`` →
    ``predicate="treats"`` at the querier boundary so the
    ``EnsembleVoter.classify_kg_signal`` treats path comes alive.

    Pre-fix the querier emitted ``predicate="associated_with"`` for ALL
    rows; the ``known_drug`` parametrize cases would fail under pre-fix
    code (proving the dead-signal bug existed). Post-fix the ``known_drug``
    rows produce ``"treats"``; the rest stay ``"associated_with"``.

    Reference: docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md
    """
    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "count": 1,
                "rows": [
                    {
                        "score": 0.85,
                        "datatypeId": datatype_id,
                        "datasourceId": datasource_id,
                        "literature": [],
                        "drug": {"id": "CHEMBL1234", "name": "drug-x"},
                        "disease": {"id": "EFO_0000270", "name": "disease-y"},
                    }
                ],
            }
        }
    )
    edges = _querier(umls=umls, ot=ot).query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert len(edges) == 1
    assert edges[0].predicate == expected_predicate
    assert edges[0].evidence_source == "open_targets"
    assert edges[0].datasource == datasource_id


# ---------------------------------------------------------------------------
# Issue #245: KGEdge.evidence + ChEMBL drug-disease evidence enrichment
# ---------------------------------------------------------------------------


class _StubChEMBL:
    """Stub ChEMBL client for KGQuerier wiring tests.

    Records every call so tests can assert call site invariants
    (cross-walk runs once per gene symbol, bioactivity is queried only
    when a target resolved, etc.). Does not exercise HTTP — that's
    covered by ``test_chembl.py``'s MockTransport tests.
    """

    def __init__(
        self,
        *,
        gene_to_target: dict[str, Optional[str]] | None = None,
        bioactivity: dict[str, list[Any]] | None = None,
    ) -> None:
        self._gene_to_target = gene_to_target or {}
        self._bioactivity = bioactivity or {}
        self.calls: list[tuple[str, ...]] = []

    def open_targets_target_to_chembl(self, gene_or_id: Optional[str]) -> Optional[str]:
        self.calls.append(("cross_walk", str(gene_or_id)))
        if not gene_or_id:
            return None
        return self._gene_to_target.get(gene_or_id)

    def get_bioactivity(self, target_chembl_id: str) -> list[Any]:
        self.calls.append(("get_bioactivity", target_chembl_id))
        return self._bioactivity.get(target_chembl_id, [])

    def close(self) -> None:
        pass


def test_kgedge_evidence_field_default_is_empty_tuple() -> None:
    """KGEdge.evidence is an additive optional field; default is an empty
    tuple so existing callers don't break."""
    edge = KGEdge(
        subject_id="X",
        predicate="treats",
        object_id="Y",
        evidence_source="open_targets",
    )
    assert edge.evidence == ()


def test_kgedge_evidence_carries_evidence_items() -> None:
    """KGEdge.evidence is an immutable tuple of EvidenceItem records."""
    from src.data.kg.types import EvidenceItem

    item = EvidenceItem(
        pmid="12345678",
        source="open_targets",
        chembl_target_id=None,
        datasource_score=0.91,
    )
    edge = KGEdge(
        subject_id="CHEMBL1234",
        predicate="treats",
        object_id="EFO_0000270",
        evidence_source="open_targets",
        evidence=(item,),
    )
    assert len(edge.evidence) == 1
    assert edge.evidence[0].pmid == "12345678"


def test_query_drug_disease_edges_populates_evidence_from_open_targets() -> None:
    """When Open Targets returns literature PMIDs + a datatype score, the
    resulting KGEdge.evidence carries one EvidenceItem per PMID. Each
    EvidenceItem records the datasource score so consumers can rank
    edges by evidence strength."""
    from src.data.kg.types import EvidenceItem  # noqa: F401 — referenced via attr

    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "count": 1,
                "rows": [
                    {
                        "score": 0.91,
                        "datatypeId": "known_drug",
                        "datasourceId": "chembl",
                        "literature": ["12345678", "87654321"],
                        "drug": {"id": "CHEMBL941", "name": "imatinib"},
                        "disease": {"id": "EFO_0000222", "name": "CML"},
                    }
                ],
            }
        }
    )
    edges = _querier(umls=umls, ot=ot).query_drug_disease_edges("CHEMBL941", "EFO_0000222")
    assert len(edges) == 1
    edge = edges[0]
    assert len(edge.evidence) == 2
    assert {ev.pmid for ev in edge.evidence} == {"12345678", "87654321"}
    for ev in edge.evidence:
        assert ev.source == "open_targets"
        assert ev.datasource_score == pytest.approx(0.91)
        # Without a ChEMBL client wired, no target ID is associated.
        assert ev.chembl_target_id is None


def test_query_drug_disease_edges_enriches_with_chembl_when_target_gene_present() -> None:
    """When (1) a ChEMBL client is attached AND (2) Open Targets exposes
    the drug's target gene, the querier cross-walks gene → ChEMBL target ID
    and tags every EvidenceItem with that ID.
    """
    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "count": 1,
                "rows": [
                    {
                        "score": 0.91,
                        "datatypeId": "known_drug",
                        "datasourceId": "chembl",
                        "literature": ["16480739"],
                        "drug": {"id": "CHEMBL941", "name": "imatinib"},
                        "disease": {"id": "EFO_0000222", "name": "CML"},
                        # Open Targets exposes the drug's primary target
                        # gene in the evidence row.
                        "target": {"id": "ENSG00000097007", "approvedSymbol": "ABL1"},
                    }
                ],
            }
        }
    )
    chembl = _StubChEMBL(gene_to_target={"ABL1": "CHEMBL1862"})
    querier = KnowledgeGraphQuerier(
        umls=umls,  # type: ignore[arg-type]
        open_targets=ot,  # type: ignore[arg-type]
        chembl=chembl,  # type: ignore[arg-type]
    )
    edges = querier.query_drug_disease_edges("CHEMBL941", "EFO_0000222")
    assert len(edges) == 1
    edge = edges[0]
    assert len(edge.evidence) == 1
    assert edge.evidence[0].chembl_target_id == "CHEMBL1862"
    # Cross-walk should have been called with the gene symbol.
    assert ("cross_walk", "ABL1") in chembl.calls


def test_query_drug_disease_edges_no_chembl_target_still_emits_evidence() -> None:
    """If ChEMBL cross-walk returns None (target not in ChEMBL), evidence
    items are still produced from Open Targets PMIDs — chembl_target_id is
    None. The path must not raise."""
    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "count": 1,
                "rows": [
                    {
                        "score": 0.5,
                        "datatypeId": "literature",
                        "datasourceId": "europepmc",
                        "literature": ["999"],
                        "drug": {"id": "CHEMBL1", "name": "x"},
                        "disease": {"id": "EFO_X", "name": "x"},
                        "target": {"id": "ENSG_unknown", "approvedSymbol": "UNKNOWN"},
                    }
                ],
            }
        }
    )
    chembl = _StubChEMBL(gene_to_target={"UNKNOWN": None})
    querier = KnowledgeGraphQuerier(
        umls=umls,  # type: ignore[arg-type]
        open_targets=ot,  # type: ignore[arg-type]
        chembl=chembl,  # type: ignore[arg-type]
    )
    edges = querier.query_drug_disease_edges("CHEMBL1", "EFO_X")
    assert len(edges) == 1
    assert edges[0].evidence[0].chembl_target_id is None


def test_query_drug_disease_edges_works_without_chembl_client() -> None:
    """Backwards-compatible path: existing callers that did not pass a
    ChEMBL client get the Open Targets PMID evidence threaded through
    KGEdge.evidence, and no ChEMBL HTTP is attempted."""
    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "count": 1,
                "rows": [
                    {
                        "score": 0.7,
                        "datatypeId": "known_drug",
                        "datasourceId": "chembl",
                        "literature": ["77"],
                        "drug": {"id": "CHEMBL1", "name": "x"},
                        "disease": {"id": "EFO_X", "name": "x"},
                    }
                ],
            }
        }
    )
    # No ``chembl=...`` kwarg.
    querier = KnowledgeGraphQuerier(
        umls=umls,  # type: ignore[arg-type]
        open_targets=ot,  # type: ignore[arg-type]
    )
    edges = querier.query_drug_disease_edges("CHEMBL1", "EFO_X")
    assert len(edges) == 1
    assert len(edges[0].evidence) == 1
    assert edges[0].evidence[0].chembl_target_id is None


def test_query_drug_disease_edges_skips_cross_walk_when_no_target_in_payload() -> None:
    """If the Open Targets row doesn't expose a target gene symbol, the
    ChEMBL cross-walk MUST NOT be attempted (avoid wasted HTTP)."""
    umls = _StubUMLS()
    ot = _StubOT(
        evidence={
            "evidences": {
                "count": 1,
                "rows": [
                    {
                        "score": 0.1,
                        "datatypeId": "literature",
                        "datasourceId": "europepmc",
                        "literature": ["1"],
                        "drug": {"id": "CHEMBL1", "name": "x"},
                        "disease": {"id": "EFO_X", "name": "x"},
                        # No "target" key.
                    }
                ],
            }
        }
    )
    chembl = _StubChEMBL()
    querier = KnowledgeGraphQuerier(
        umls=umls,  # type: ignore[arg-type]
        open_targets=ot,  # type: ignore[arg-type]
        chembl=chembl,  # type: ignore[arg-type]
    )
    edges = querier.query_drug_disease_edges("CHEMBL1", "EFO_X")
    assert len(edges) == 1
    # No cross-walk call.
    cross_walk_calls = [c for c in chembl.calls if c[0] == "cross_walk"]
    assert cross_walk_calls == []
