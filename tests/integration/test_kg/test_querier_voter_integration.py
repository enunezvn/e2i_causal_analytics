"""End-to-end: KGQuerier → EnsembleVoter integration.

Closes the fixture-realism gap that hid the predicate-mismatch bug
between PR #86 (querier) and PR #88 (voter). Each was unit-tested in
isolation: the querier with stub Open Targets responses asserting
``predicate=="associated_with"``, the voter with hand-crafted
``predicate="treats"`` edges. No test fed querier output through the
voter — so the bug (querier emits ``"associated_with"``; voter matches
only TREATS_PREDICATES) shipped silently.

These tests use realistic Open Targets responses (mixed datatypeId
values from the Ochoa 2021 NAR taxonomy) and assert the voter's
``classify_kg_signal`` produces the right signal class on the chained
output.

These are *integration* tests but do NOT make live HTTP calls — they
use the same ``_StubOT`` Python-class stub pattern as the unit tests
to keep the suite hermetic. Live Open Targets coverage is a future
follow-up.

Reference: docs/superpowers/specs/2026-05-08-kg-predicate-reconciliation-design.md
"""

from __future__ import annotations

from typing import Any, Optional

from src.data.kg.ensemble_voter import classify_kg_signal
from src.data.kg.kg_querier import KnowledgeGraphQuerier


class _StubUMLS:
    def cui_relations(self, cui: str, *, page_size: int = 50) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        pass


class _StubOT:
    def __init__(self, *, evidence: Optional[dict[str, Any]] = None) -> None:
        self._evidence = evidence or {"evidences": {"rows": []}}

    def drug_disease_evidence(
        self, drug_id: str, disease_id: str, *, size: int = 25
    ) -> dict[str, Any]:
        return self._evidence

    def close(self) -> None:
        pass


def _querier(*, evidence: dict[str, Any]) -> KnowledgeGraphQuerier:
    return KnowledgeGraphQuerier(
        umls=_StubUMLS(),  # type: ignore[arg-type]
        open_targets=_StubOT(evidence=evidence),  # type: ignore[arg-type]
    )


def test_known_drug_row_produces_leak_signal_through_voter() -> None:
    """Real-shaped Open Targets response with one known_drug + one literature
    row → voter classifies leak_drug_treats_disease.

    Pre-fix querier emitted "associated_with" for both rows; voter
    classified "no_signal". Post-fix the known_drug row emits "treats";
    voter classifies "leak_drug_treats_disease" using only that row.
    """
    querier = _querier(
        evidence={
            "evidences": {
                "rows": [
                    {
                        "datatypeId": "known_drug",
                        "datasourceId": "chembl",
                        "score": 0.95,
                        "literature": [],
                        "drug": {"id": "CHEMBL1234", "name": "drug-x"},
                        "disease": {"id": "EFO_0000270", "name": "disease-y"},
                    },
                    {
                        "datatypeId": "literature",
                        "datasourceId": "europepmc",
                        "score": 0.30,
                        "literature": ["12345"],
                        "drug": {"id": "CHEMBL1234", "name": "drug-x"},
                        "disease": {"id": "EFO_0000270", "name": "disease-y"},
                    },
                ]
            }
        }
    )
    edges = querier.query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert len(edges) == 2
    predicates = {e.predicate for e in edges}
    assert predicates == {"treats", "associated_with"}

    signal, considered = classify_kg_signal(
        edges,
        feature_entity_ids={"CHEMBL1234"},
        target_entity_ids={"EFO_0000270"},
    )
    assert signal == "leak_drug_treats_disease"
    # Only the known_drug edge counts toward classification.
    assert len(considered) == 1
    assert considered[0].predicate == "treats"
    assert considered[0].evidence_source == "open_targets"


def test_only_non_known_drug_rows_produce_no_signal_through_voter() -> None:
    """An Open Targets response with NO known_drug rows produces no
    treats signal. Confirms non-treats datatypes don't accidentally
    promote — addresses codex pressure-test concern about expanding
    TREATS_PREDICATES being semantically wrong.
    """
    querier = _querier(
        evidence={
            "evidences": {
                "rows": [
                    {
                        "datatypeId": "literature",
                        "datasourceId": "europepmc",
                        "score": 0.30,
                        "literature": ["12345"],
                        "drug": {"id": "CHEMBL1234", "name": "drug-x"},
                        "disease": {"id": "EFO_0000270", "name": "disease-y"},
                    },
                    {
                        "datatypeId": "genetic_association",
                        "datasourceId": "eva",
                        "score": 0.50,
                        "literature": [],
                        "drug": {"id": "CHEMBL1234", "name": "drug-x"},
                        "disease": {"id": "EFO_0000270", "name": "disease-y"},
                    },
                    {
                        "datatypeId": "affected_pathway",
                        "datasourceId": "progeny",
                        "score": 0.20,
                        "literature": [],
                        "drug": {"id": "CHEMBL1234", "name": "drug-x"},
                        "disease": {"id": "EFO_0000270", "name": "disease-y"},
                    },
                ]
            }
        }
    )
    edges = querier.query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert len(edges) == 3
    assert all(e.predicate == "associated_with" for e in edges)

    signal, considered = classify_kg_signal(
        edges,
        feature_entity_ids={"CHEMBL1234"},
        target_entity_ids={"EFO_0000270"},
    )
    assert signal == "no_signal"
    assert considered == ()


def test_empty_evidence_rows_produce_no_signal() -> None:
    """An Open Targets response with zero rows is the queried-no-edges
    case — distinct from the predicate-mismatch dead-signal bug. The
    PR-C cache builder will distinguish this from cache-missing via
    the CacheRecord.status enum.
    """
    querier = _querier(evidence={"evidences": {"rows": []}})
    edges = querier.query_drug_disease_edges("CHEMBL1234", "EFO_0000270")
    assert edges == []

    signal, considered = classify_kg_signal(
        edges,
        feature_entity_ids={"CHEMBL1234"},
        target_entity_ids={"EFO_0000270"},
    )
    assert signal == "no_signal"
    assert considered == ()
