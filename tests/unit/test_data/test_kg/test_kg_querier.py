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
        self._evidence = evidence or {"drug": {"indications": {"rows": []}}}
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


# ---------------------------------------------------------------------------
# Drug -> disease edges, Open Targets v4 `drug.indications` schema (#1607)
#
# These replace the previous suite, which modelled a top-level
# `evidences(drugIds:, diseaseIds:)` Query field that Open Targets has REMOVED.
# Those tests were green while production returned HTTP 400 on every live call
# — they mocked a wire shape that no longer existed. Coverage dropped with the
# old schema, and why:
#
#   * score sanitisation (NaN / inf / missing) — `indications.rows` carries no
#     score field at all, so there is no number to sanitise.
#   * `EvidenceItem` / literature PMIDs and the ChEMBL target cross-walk
#     (issue #245) — both were sourced from evidence rows. Drug->disease
#     evidence with literature is not reachable in the v4 schema:
#     `Disease.evidences` REQUIRES a gene `ensemblIds` argument, and `Drug` no
#     longer exposes `linkedTargets`, so there is no drug->gene path. Verified
#     by live introspection 2026-08-14.
#   * datatypeId -> predicate mapping — replaced by `maxClinicalStage`, which
#     is a strictly better signal (it distinguishes an APPROVED indication from
#     a PHASE_1 exploratory one).
#
# The live counterpart is
# tests/integration/test_kg/test_kg_layer2_live_contracts.py.
# ---------------------------------------------------------------------------


def _indications(*rows: dict[str, Any], drug_name: str = "DrugA") -> dict[str, Any]:
    """Build a `drug.indications` payload in the current schema shape."""
    return {
        "drug": {
            "id": "CHEMBL1234",
            "name": drug_name,
            "maximumClinicalStage": "APPROVAL",
            "indications": {"count": len(rows), "rows": list(rows)},
        }
    }


def _row(disease_id: str, name: str, stage: str) -> dict[str, Any]:
    return {"disease": {"id": disease_id, "name": name}, "maxClinicalStage": stage}


def test_query_drug_disease_edges_emits_treats_for_approved_indication() -> None:
    umls = _StubUMLS()
    ot = _StubOT(evidence=_indications(_row("EFO_0000270", "atopic dermatitis", "APPROVAL")))
    edges = _querier(umls=umls, ot=ot).query_drug_disease_edges("CHEMBL1234", "EFO_0000270")

    assert len(edges) == 1
    edge = edges[0]
    assert edge.subject_id == "CHEMBL1234"
    assert edge.subject_name == "DrugA"
    assert edge.object_id == "EFO_0000270"
    assert edge.object_name == "atopic dermatitis"
    assert edge.predicate == "treats"
    assert edge.evidence_source == "open_targets"
    assert edge.datasource == "chembl_indications"
    # Honest emptiness: the indication list carries neither score nor PMIDs.
    assert edge.score is None
    assert edge.pmids == ()
    assert edge.evidence == ()


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("APPROVAL", "treats"),
        ("PHASE_3", "associated_with"),
        ("PHASE_2", "associated_with"),
        ("PHASE_1", "associated_with"),
        ("", "associated_with"),
        ("SOME_FUTURE_STAGE", "associated_with"),
    ],
)
def test_predicate_is_gated_on_clinical_stage(stage: str, expected: str) -> None:
    """Only a regulator-approved indication is a therapeutic claim.

    A deferred codex review (PR-0 M1) warned that emitting `treats` for ANY
    known-drug row lets a Phase I pairing produce a false-positive
    `leak_drug_treats_disease` verdict in the voter. `maxClinicalStage` makes
    that gate implementable; unknown/absent stages fall to the safe side.
    """
    umls = _StubUMLS()
    ot = _StubOT(evidence=_indications(_row("EFO_1", "d", stage)))
    edges = _querier(umls=umls, ot=ot).query_drug_disease_edges("CHEMBL1234", "EFO_1")
    assert len(edges) == 1
    assert edges[0].predicate == expected


def test_query_drug_disease_edges_filters_to_the_requested_disease() -> None:
    """The API returns the drug's WHOLE indication list; only the asked-for
    disease may become an edge, or the edge would encode a pair the caller
    never asked about."""
    umls = _StubUMLS()
    ot = _StubOT(
        evidence=_indications(
            _row("EFO_OTHER", "unrelated", "APPROVAL"),
            _row("EFO_0000270", "atopic dermatitis", "APPROVAL"),
            _row("EFO_ANOTHER", "also unrelated", "APPROVAL"),
        )
    )
    edges = _querier(umls=umls, ot=ot).query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert [e.object_id for e in edges] == ["EFO_0000270"]


def test_query_drug_disease_edges_empty_when_drug_has_no_indications() -> None:
    umls = _StubUMLS()
    ot = _StubOT(evidence=_indications())
    assert _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y") == []


def test_query_drug_disease_edges_handles_null_rows() -> None:
    """`indications.rows` is a nullable GraphQL list; null must collapse to []."""
    umls = _StubUMLS()
    ot = _StubOT(evidence={"drug": {"indications": {"rows": None}}})
    assert _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y") == []


def test_query_drug_disease_edges_handles_null_indications_object() -> None:
    umls = _StubUMLS()
    ot = _StubOT(evidence={"drug": {"indications": None}})
    assert _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y") == []


def test_query_drug_disease_edges_handles_null_drug_object() -> None:
    """A drug id Open Targets does not know resolves `drug` to null."""
    umls = _StubUMLS()
    ot = _StubOT(evidence={"drug": None})
    assert _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y") == []


def test_query_drug_disease_edges_skips_malformed_rows() -> None:
    umls = _StubUMLS()
    ot = _StubOT(evidence={"drug": {"indications": {"rows": ["not-a-dict", None]}}})
    assert _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y") == []


def test_query_drug_disease_edges_propagates_open_targets_error() -> None:
    """Transport failures must surface so callers can distinguish "no edges"
    from "GraphQL/transport failure" (codex H1 from PR #102 review). Cache
    builders need typed errors to record ``status=source_error`` instead of
    ``status=queried_no_edges`` — which is exactly how the dead query stayed
    invisible for so long."""
    umls = _StubUMLS()
    ot = _StubOT(raise_error=True)
    with pytest.raises(OpenTargetsError):
        _querier(umls=umls, ot=ot).query_drug_disease_edges("X", "Y")
