"""Integration test for ChEMBL drug-disease evidence enrichment (#245).

Pins the imatinib ↔ chronic myeloid leukemia (ABL1 target) end-to-end
flow through ``KnowledgeGraphQuerier.query_drug_disease_edges``:

  - Open Targets returns one ``known_drug`` evidence row with literature
    PMIDs and the drug's target gene symbol (ABL1).
  - The querier emits one ``KGEdge`` with ``predicate="treats"``.
  - The ChEMBL cross-walk resolves ABL1 → CHEMBL1862.
  - Every ``EvidenceItem`` on the edge carries (1) one of the PMIDs, (2)
    the Open Targets datasource score, and (3) ``chembl_target_id =
    "CHEMBL1862"``.

The HTTP layer is mocked via ``httpx.MockTransport`` so this test is
deterministic on CI (no live ChEMBL or Open Targets dependency).
Unit-level transport behaviour is covered in
``tests/unit/test_data/test_kg/test_chembl.py``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.data.kg.chembl import ChEMBLClient
from src.data.kg.chembl import reset_caches as chembl_reset_caches
from src.data.kg.kg_querier import KnowledgeGraphQuerier
from src.data.kg.open_targets import OpenTargetsClient
from src.data.kg.open_targets import reset_caches as ot_reset_caches
from src.data.kg.types import KGEdge

pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    chembl_reset_caches()
    ot_reset_caches()


def _ot_handler() -> Any:
    """Open Targets GraphQL mock — imatinib + CML known_drug evidence."""

    def handler(request: httpx.Request) -> httpx.Response:
        # The querier issues exactly one GraphQL POST for drug-disease.
        body = json.loads(request.content.decode("utf-8"))
        variables = body.get("variables", {})
        assert variables.get("drugId") == "CHEMBL941"
        assert variables.get("diseaseId") == "EFO_0000222"
        return httpx.Response(
            200,
            json={
                "data": {
                    "drug": {
                        "id": "CHEMBL941",
                        "name": "IMATINIB",
                        "indications": {
                            "rows": [
                                {
                                    "disease": {
                                        "id": "EFO_0000222",
                                        "name": "chronic myeloid leukemia",
                                    },
                                    "maxPhaseForIndication": 4,
                                }
                            ]
                        },
                    },
                    "evidences": {
                        "count": 1,
                        "rows": [
                            {
                                "score": 0.95,
                                "datatypeId": "known_drug",
                                "datasourceId": "chembl",
                                "literature": ["16480739", "11287973"],
                                "drug": {"id": "CHEMBL941", "name": "IMATINIB"},
                                "disease": {
                                    "id": "EFO_0000222",
                                    "name": "chronic myeloid leukemia",
                                },
                                "target": {
                                    "id": "ENSG00000097007",
                                    "approvedSymbol": "ABL1",
                                },
                            }
                        ],
                    },
                }
            },
        )

    return handler


def _chembl_handler() -> Any:
    """ChEMBL mock — target.json for ABL1 returns CHEMBL1862.

    Pins the live-API-valid ``target_synonym__iexact`` filter shape;
    if the production code regresses back to the nested
    ``target_components__component_synonyms__component_synonym__iexact``
    path the assertion trips and the cross-walk is exercised against
    the right URL.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/target.json" in request.url.path
        query = request.url.query.decode("utf-8")
        assert "target_synonym__iexact" in query, (
            "Cross-walk regressed to invalid nested filter path"
        )
        assert "ABL1" in query
        return httpx.Response(
            200,
            json={
                "targets": [
                    {
                        "target_chembl_id": "CHEMBL1862",
                        "pref_name": "Tyrosine-protein kinase ABL",
                        "target_component_synonyms": [
                            {"component_synonym": "ABL1", "syn_type": "GENE_SYMBOL"}
                        ],
                    }
                ]
            },
        )

    return handler


def test_imatinib_cml_evidence_enrichment_end_to_end() -> None:
    """End-to-end: Open Targets → KGEdge.evidence → ChEMBL cross-walk."""
    ot_client = OpenTargetsClient(client=httpx.Client(transport=httpx.MockTransport(_ot_handler())))
    chembl_client = ChEMBLClient(
        client=httpx.Client(transport=httpx.MockTransport(_chembl_handler()))
    )
    # KGQuerier construction requires a UMLS slot — give it a stub since
    # we are not exercising UMLS in this integration scenario.

    class _UMLSStub:
        def cui_relations(self, cui: str, *, page_size: int = 50) -> list[dict[str, Any]]:  # noqa: D401
            return []

        def close(self) -> None:
            return None

    querier = KnowledgeGraphQuerier(
        umls=_UMLSStub(),  # type: ignore[arg-type]
        open_targets=ot_client,
        chembl=chembl_client,
    )
    edges = querier.query_drug_disease_edges("CHEMBL941", "EFO_0000222")
    assert len(edges) == 1, "imatinib/CML must produce exactly one drug-disease edge"
    edge: KGEdge = edges[0]
    # 1. Edge shape contract.
    assert edge.subject_id == "CHEMBL941"
    assert edge.object_id == "EFO_0000222"
    assert edge.predicate == "treats"
    assert edge.evidence_source == "open_targets"
    assert edge.pmids == ("16480739", "11287973")
    # 2. Evidence threading.
    assert len(edge.evidence) == 2
    assert {ev.pmid for ev in edge.evidence} == {"16480739", "11287973"}
    for ev in edge.evidence:
        assert ev.source == "open_targets"
        assert ev.datasource_score == pytest.approx(0.95)
        # 3. Cross-walk ran and populated chembl_target_id.
        assert ev.chembl_target_id == "CHEMBL1862"
