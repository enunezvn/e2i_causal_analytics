"""Integration test for the imatinib <-> CML drug-disease edge (#245, #1607).

Pins the end-to-end flow through
``KnowledgeGraphQuerier.query_drug_disease_edges``: Open Targets returns the
drug's indication list, and the querier emits one ``KGEdge`` with
``predicate="treats"`` because the indication is approved.

**What this test used to pin, and why it no longer can.** Issue #245 threaded
per-PMID provenance onto each edge: literature PMIDs and a ChEMBL target id
cross-walked from the evidence row's target gene (ABL1 -> CHEMBL1862). Open
Targets has since REMOVED the top-level ``evidences`` field that supplied all
three, and ``Drug`` no longer exposes ``linkedTargets``, so there is no
drug->gene path left. The enrichment is not reachable and the assertions for it
have been dropped rather than mocked into looking alive — a mocked assertion for
an impossible code path is exactly what let the underlying query rot unnoticed
(it returned HTTP 400 on every live call while this suite stayed green).

The removal is pinned by
``test_kg_layer2_live_contracts.test_open_targets_graphql_rejects_the_removed_evidences_field``,
which fails if Open Targets restores the field — at which point the #245
enrichment becomes implementable again.

The HTTP layer is mocked via ``httpx.MockTransport`` so this test is
deterministic on CI. The live schema is exercised by
``tests/integration/test_kg/test_kg_layer2_live_contracts.py``.
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
        return httpx.Response(
            200,
            json={
                "data": {
                    "drug": {
                        "id": "CHEMBL941",
                        "name": "IMATINIB",
                        "maximumClinicalStage": "APPROVAL",
                        "indications": {
                            "count": 1,
                            "rows": [
                                {
                                    "disease": {
                                        "id": "EFO_0000222",
                                        "name": "chronic myeloid leukemia",
                                    },
                                    "maxClinicalStage": "APPROVAL",
                                }
                            ],
                        },
                    }
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
    edge = edges[0]
    assert isinstance(edge, KGEdge)
    assert edge.subject_id == "CHEMBL941"
    assert edge.subject_name == "IMATINIB"
    assert edge.object_id == "EFO_0000222"
    assert edge.object_name == "chronic myeloid leukemia"
    assert edge.predicate == "treats", "an APPROVAL-stage indication must read as 'treats'"
    assert edge.evidence_source == "open_targets"
    assert edge.datasource == "chembl_indications"
    # Honest emptiness. See the module docstring: the fields below had their
    # upstream source removed, and asserting fabricated values here would hide
    # that from the next reader.
    assert edge.pmids == ()
    assert edge.evidence == ()
    assert edge.score is None
