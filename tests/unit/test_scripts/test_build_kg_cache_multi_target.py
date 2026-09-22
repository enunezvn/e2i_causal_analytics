"""Lane E — the drug-disease pass runs for EVERY target drug, not the first one.

The Lane A contrast is dupilumab vs omalizumab, so the causal cohorts' KG cache
is built against BOTH treatment concepts (``RXNORM:302379`` omalizumab and
``RXNORM:1876376`` dupilumab). Read from the source before this lane:
``_resolve_target_drug`` returned on the FIRST code that resolved to a ChEMBL id
(``scripts/build_kg_cache.py``), so a two-code target silently produced a
one-drug cache — the second drug never got a drug-disease pass, and nothing in
the artifact said so. These tests pin the per-drug pass.

All collaborators are in-memory stubs; no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.feature_contract import FeatureContract, KnowableAt
from src.data.kg.types import KGConcept, KGEdge

_CUI = "C0004096"  # asthma
_DISEASE_ID = "MONDO_0004979"


class _StubRxNav:
    def __init__(self, names: dict[str, str]) -> None:
        self._names = names

    def properties(self, rxcui: str) -> dict[str, Any] | None:
        name = self._names.get(rxcui)
        return {"rxcui": rxcui, "name": name} if name else None


class _StubOpenTargets:
    def __init__(self, chembl_by_name: dict[str, str]) -> None:
        self._chembl_by_name = chembl_by_name

    def search_drug(self, name: str) -> str | None:
        return self._chembl_by_name.get(name)

    def search_disease(self, name: str) -> str | None:
        return _DISEASE_ID


class _StubUMLS:
    def cui_lookup(self, cui: str) -> KGConcept:
        return KGConcept(
            cui=cui, preferred_name=f"name-{cui}", semantic_types=("Disease",), atom_count=1
        )


class _StubLinker:
    def __init__(self) -> None:
        self.rxnav = _StubRxNav({"302379": "omalizumab", "1876376": "dupilumab"})
        self.open_targets = _StubOpenTargets(
            {"omalizumab": "CHEMBL1201589", "dupilumab": "CHEMBL2364637"}
        )
        self.umls = _StubUMLS()

    def resolve(self, code: str, system: str):  # pragma: no cover - not used (UMLS-only feature)
        raise AssertionError("resolve must not be called for a UMLS-coded feature")


class _StubQuerier:
    def __init__(self) -> None:
        self.drug_disease_calls: list[tuple[str, str]] = []

    def query_disease_hierarchy(self, cui: str) -> list[KGEdge]:
        return []

    def query_drug_disease_edges(self, drug_id: str, disease_id: str) -> list[KGEdge]:
        self.drug_disease_calls.append((drug_id, disease_id))
        return [
            KGEdge(
                subject_id=drug_id,
                subject_name=drug_id,
                predicate="treats",
                object_id=disease_id,
                object_name="asthma",
                evidence_source="open_targets",
                datasource="chembl_indications",
            )
        ]


def _asthma_feature() -> FeatureContract:
    return FeatureContract(
        name="cci_chronic_pulmonary",
        knowable_at=KnowableAt(reference="index_date"),
        source="mart_comorbidity",
        kg_entity_codes=(("UMLS", _CUI),),
    )


def test_resolve_target_drugs_resolves_every_code() -> None:
    from scripts.build_kg_cache import _resolve_target_drugs

    drugs, errors = _resolve_target_drugs(
        [("RXNORM", "302379"), ("RXNORM", "1876376")],
        _StubLinker(),  # type: ignore[arg-type]
    )
    assert drugs == [("CHEMBL1201589", "302379"), ("CHEMBL2364637", "1876376")]
    assert errors == []


def test_resolve_target_drugs_keeps_going_past_an_unresolvable_code() -> None:
    """One bad code must not hide the drugs that DO resolve, and must be reported."""
    from scripts.build_kg_cache import _resolve_target_drugs

    drugs, errors = _resolve_target_drugs(
        [("RXNORM", "479158"), ("RXNORM", "1876376")],
        _StubLinker(),  # type: ignore[arg-type]
    )
    assert drugs == [("CHEMBL2364637", "1876376")]
    assert any("479158" in e for e in errors), errors


def test_single_drug_wrapper_still_returns_the_first_resolution() -> None:
    """``_resolve_target_drug`` keeps its (chembl_id, code, errors) contract."""
    from scripts.build_kg_cache import _resolve_target_drug

    chembl_id, code, errors = _resolve_target_drug(
        [("RXNORM", "302379"), ("RXNORM", "1876376")],
        _StubLinker(),  # type: ignore[arg-type]
    )
    assert (chembl_id, code) == ("CHEMBL1201589", "302379")
    assert errors == []


def test_build_cache_runs_the_drug_disease_pass_for_both_target_drugs(tmp_path: Path) -> None:
    """The load-bearing assertion: BOTH drugs' ``treats`` edges land in the record,
    each rewritten onto its OWN target code so ``classify_kg_signal._connects``
    can match either treatment concept, and ``source_subject_id`` keeps the
    ChEMBL id that produced it."""
    from scripts.build_kg_cache import build_cache_for_manifest

    querier = _StubQuerier()
    cache_path = build_cache_for_manifest(
        features=[_asthma_feature()],
        target_entity_codes=[("RXNORM", "302379"), ("RXNORM", "1876376")],
        out_dir=tmp_path,
        entity_linker=_StubLinker(),  # type: ignore[arg-type]
        kg_querier=querier,  # type: ignore[arg-type]
    )
    assert sorted(querier.drug_disease_calls) == [
        ("CHEMBL1201589", _DISEASE_ID),
        ("CHEMBL2364637", _DISEASE_ID),
    ]
    payload = json.loads(cache_path.read_text())
    assert len(payload) == 1
    record = payload[0]
    assert record["status"] == "ok"
    assert record["sources_attempted"] == ["umls_uts", "rxnav", "open_targets"]
    by_target = {e["subject_id"]: e for e in record["edges"] if e["predicate"] == "treats"}
    assert set(by_target) == {"302379", "1876376"}, record["edges"]
    assert by_target["302379"]["source_subject_id"] == "CHEMBL1201589"
    assert by_target["1876376"]["source_subject_id"] == "CHEMBL2364637"
    assert all(e["object_id"] == _CUI for e in by_target.values())
