"""Live contract tests for the KG Layer-2 clients: RxNav + Open Targets (#1607).

Before this, only UMLS and ``KnowledgeGraphQuerier``'s UMLS path had live
coverage; RxNav and Open Targets were exercised solely through
``httpx.MockTransport``. That gap hid a total outage: the Open Targets
drug-disease GraphQL query had drifted out of sync with the upstream schema and
returned **HTTP 400 for every call**, so ``query_drug_disease_edges`` could
never produce an edge. Mocked tests cannot catch a schema rename.

Measured against the live API on 2026-08-14:

* ``maxPhaseForIndication`` was renamed to ``maxClinicalStage`` on
  ``ClinicalIndicationFromDrug``.
* The top-level ``evidences`` Query field was REMOVED entirely; evidence now
  hangs off ``Disease.evidences`` and requires a gene ``ensemblIds`` argument,
  so it can no longer serve a drug->disease lookup.
* Drug->disease "treats" now lives in ``Drug.indications``, which additionally
  exposes ``maxClinicalStage`` (APPROVAL / PHASE_3 / ...) — the phase signal a
  deferred codex review asked for.

Gated on network only (Open Targets and RxNav are both zero-auth) and marked
``slow`` so they run on the ``slow-tests.yml`` schedule rather than the
PR-blocking lane.
"""

from __future__ import annotations

import httpx
import pytest

from src.data.kg.kg_querier import KnowledgeGraphQuerier
from src.data.kg.open_targets import OPEN_TARGETS_ENDPOINT, OpenTargetsClient
from src.data.kg.rxnav import RxNavClient

_GATE_URL = "https://connectivitycheck.gstatic.com/generate_204"


def _network_available() -> bool:
    try:
        return httpx.get(_GATE_URL, timeout=8.0, follow_redirects=True).status_code < 500
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not _network_available(), reason="No outbound network (#1607)."),
]


# ============================================================================ RxNav


def test_rxnav_resolves_exact_drug_names() -> None:
    """``rxcui_for_name`` backs EntityLinker's drug resolution."""
    client = RxNavClient()
    match = client.rxcui_for_name("ribociclib")
    assert match is not None, "RxNav returned no rxcui for ribociclib"
    assert match.rxcui == "1873916", f"ribociclib's stable RxCUI changed: {match.rxcui}"
    assert match.approximate is False, "an exact-name hit must not be flagged approximate"


def test_rxnav_properties_shape() -> None:
    """``properties`` must keep returning the keys EntityLinker reads."""
    client = RxNavClient()
    props = client.properties("1873916")
    assert isinstance(props, dict), "RxNav properties payload is no longer a dict"
    assert props.get("rxcui") == "1873916"
    assert props.get("name", "").lower() == "ribociclib"


def test_rxnav_unknown_code_returns_none_not_an_error() -> None:
    """An unknown RxCUI must degrade to None, which callers treat as unresolved.

    Measured: 479158 and 1011295 — the codes used in ``build_kg_cache.py``'s
    docstring example — are NOT valid RxCUIs and resolve to None here.
    """
    client = RxNavClient()
    assert client.properties("479158") is None


# ===================================================================== Open Targets


def test_open_targets_search_resolves_drug_and_disease_ids() -> None:
    """``search_drug`` / ``search_disease`` back the ChEMBL/MONDO cross-walk."""
    client = OpenTargetsClient()
    assert client.search_drug("ribociclib") == "CHEMBL3545110"
    assert client.search_disease("breast carcinoma") == "MONDO_0004989"


def test_open_targets_drug_disease_query_matches_the_live_schema() -> None:
    """The drug-disease GraphQL document must actually execute (#1607).

    This is the test whose absence let the query rot: every unit test mocks the
    transport, so a field rename upstream produced HTTP 400 on every live call
    with nothing turning red. Asserting on parsed edges rather than a 200 means
    a future rename fails here too.
    """
    client = OpenTargetsClient()
    payload = client.drug_disease_evidence("CHEMBL1201589", "MONDO_0005492")

    drug = payload.get("drug")
    assert isinstance(drug, dict), "Open Targets response lost its 'drug' object"
    assert drug.get("id") == "CHEMBL1201589"
    indications = (drug.get("indications") or {}).get("rows")
    assert isinstance(indications, list) and indications, (
        "drug.indications.rows is empty/absent — the drug->disease 'treats' "
        "relation is sourced from here"
    )
    row = indications[0]
    assert isinstance(row.get("disease"), dict), "indication row lost its disease object"
    assert "maxClinicalStage" in row, (
        "maxClinicalStage is absent — Open Targets renamed it again "
        "(it replaced maxPhaseForIndication)"
    )


def test_open_targets_graphql_rejects_the_removed_evidences_field() -> None:
    """Pin the schema change that broke us, so a revert is detectable.

    The old query used a top-level ``evidences(drugIds:, diseaseIds:)`` field.
    That field no longer exists. If Open Targets ever restores it, this test
    fails and the richer evidence path (which carried literature PMIDs and
    per-row scores) becomes available again — worth knowing.
    """
    response = httpx.post(
        OPEN_TARGETS_ENDPOINT,
        json={"query": 'query { evidences(drugIds: ["CHEMBL25"], size: 1) { count } }'},
        timeout=40.0,
    )
    body = response.json()
    assert "errors" in body, (
        "top-level 'evidences' now resolves again — the drug->disease evidence "
        "path with literature PMIDs may be restorable (see #1607)"
    )


def test_query_drug_disease_edges_produces_treats_edges() -> None:
    """End-to-end: a real approved indication becomes a `treats` KGEdge.

    ``classify_kg_signal`` only promotes ``leak_drug_treats_disease`` when the
    predicate is in ``TREATS_PREDICATES`` and ``evidence_source`` is
    ``open_targets``, so both are asserted here.
    """
    querier = KnowledgeGraphQuerier(open_targets=OpenTargetsClient())
    edges = querier.query_drug_disease_edges("CHEMBL1201589", "MONDO_0005492")

    assert edges, "omalizumab/urticaria is an APPROVED indication and must yield an edge"
    treats = [e for e in edges if e.predicate == "treats"]
    assert treats, f"no 'treats' edge among predicates {sorted({e.predicate for e in edges})}"

    edge = treats[0]
    assert edge.evidence_source == "open_targets"
    assert edge.subject_id == "CHEMBL1201589"
    assert edge.object_id == "MONDO_0005492"


def test_non_approved_indications_are_not_labelled_treats() -> None:
    """Phase gating: only an APPROVAL-stage indication earns `treats`.

    A deferred codex review (PR-0 M1) flagged that emitting `treats` for ANY
    known-drug row lets a Phase I exploratory pairing produce a
    false-positive ``leak_drug_treats_disease`` verdict. ``maxClinicalStage``
    makes that gate implementable; this pins it.

    Measured: omalizumab -> cold urticaria (MONDO_0022799) is PHASE_2.
    """
    querier = KnowledgeGraphQuerier(open_targets=OpenTargetsClient())
    edges = querier.query_drug_disease_edges("CHEMBL1201589", "MONDO_0022799")

    assert edges, "omalizumab/cold-urticaria is a known indication and must yield an edge"
    assert all(e.predicate != "treats" for e in edges), (
        "a PHASE_2 indication must NOT be labelled 'treats' — that would promote "
        "an exploratory pairing to leak_drug_treats_disease"
    )
