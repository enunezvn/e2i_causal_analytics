"""Live contract tests for ChEMBL / ClinicalTrials.gov v2 / PubMed (#1612).

These three feed narrative colour (mechanism of action, pivotal endpoints,
real-world-evidence citation) through ``ClinicalContextService._fan_out``. Every
provider is fail-open by design — a failure degrades to ``static_fallback`` or
``unavailable`` and logs a warning — which is correct behaviour and also exactly
what hides a permanent upstream break. Without a live assertion, a schema change
is indistinguishable from a transient blip and nothing ever turns red.

openFDA has its own module (``test_openfda_live_contract.py``) because it feeds
a *gate* rather than prose and is prioritised per #1612 AC5.

Two layers per API:

1. **Wire shape** — assert the nested JSON paths our parser actually walks.
2. **Parsed contract** — assert the client returns the typed shape callers read.

Every assertion below was confirmed against the real endpoints on 2026-08-14
before being written, so a failure means upstream drift rather than a guess.
"""

from __future__ import annotations

import re

import httpx
import pytest

from src.data.kg.chembl import CHEMBL_BASE, ChEMBLClient
from src.services.clinical_context.clients import (
    CLINICAL_TRIALS_BASE,
    PUBMED_BASE,
    ClinicalTrialsClient,
    PubMedClient,
)
from tests.integration.test_clinical_context._live_gate import requires_network

pytestmark = [pytest.mark.integration, pytest.mark.slow, requires_network]

_DRUG = "ribociclib"
_CONDITION = "breast cancer"
_NCT_RE = re.compile(r"^NCT\d{8}$")


# =========================================================================== ChEMBL


def test_chembl_wire_shape_molecule_and_mechanism() -> None:
    """``/molecule.json`` -> ``molecules[].molecule_chembl_id`` and
    ``/mechanism.json`` -> ``mechanisms[].mechanism_of_action`` must survive."""
    mol = httpx.get(
        f"{CHEMBL_BASE}/molecule.json",
        params={"pref_name__iexact": _DRUG, "limit": 5},
        timeout=30.0,
    )
    assert mol.status_code == 200, f"ChEMBL molecule search HTTP {mol.status_code}"
    molecules = mol.json().get("molecules")
    assert isinstance(molecules, list) and molecules, "ChEMBL payload lost 'molecules'"
    chembl_id = molecules[0].get("molecule_chembl_id")
    assert isinstance(chembl_id, str) and chembl_id.startswith("CHEMBL")

    mech = httpx.get(
        f"{CHEMBL_BASE}/mechanism.json",
        params={"molecule_chembl_id": chembl_id, "limit": 20},
        timeout=30.0,
    )
    assert mech.status_code == 200, f"ChEMBL mechanism HTTP {mech.status_code}"
    mechanisms = mech.json().get("mechanisms")
    assert isinstance(mechanisms, list) and mechanisms, "ChEMBL payload lost 'mechanisms'"
    assert isinstance(mechanisms[0].get("mechanism_of_action"), str), (
        "mechanisms[].mechanism_of_action is no longer a string"
    )


def test_chembl_mechanism_of_action_parsed_contract() -> None:
    """``ChEMBLMechanismProvider`` reads this exact return type."""
    client = ChEMBLClient()
    assert client.compound_search(_DRUG) == "CHEMBL3545110", (
        "ribociclib's stable ChEMBL id changed; the compound_search parse path "
        "or upstream identity has drifted"
    )
    moa = client.mechanism_of_action(_DRUG)
    assert isinstance(moa, str) and moa.strip(), f"ChEMBL returned no MoA for {_DRUG}"
    assert "kinase" in moa.lower(), f"ribociclib MoA no longer mentions a kinase: {moa!r}"


# ============================================================== ClinicalTrials.gov v2


def test_ctgov_wire_shape_protocol_section_paths() -> None:
    """Pin ``studies[].protocolSection.{identificationModule.nctId,
    outcomesModule.primaryOutcomes[].measure}`` — the exact walk in
    ``_primary_endpoints_uncached``."""
    response = httpx.get(
        f"{CLINICAL_TRIALS_BASE}/studies",
        params={
            "query.intr": _DRUG,
            "query.cond": _CONDITION,
            "fields": "NCTId,PrimaryOutcomeMeasure,PrimaryOutcomeTimeFrame",
            "pageSize": 8,
            "filter.overallStatus": "COMPLETED",
        },
        headers={"Accept": "application/json"},
        timeout=30.0,
    )
    assert response.status_code == 200, f"CT.gov v2 HTTP {response.status_code}"
    studies = response.json().get("studies")
    assert isinstance(studies, list) and studies, "CT.gov payload lost 'studies'"

    protocol = studies[0].get("protocolSection")
    assert isinstance(protocol, dict), "study lost 'protocolSection'"
    ident = protocol.get("identificationModule")
    assert isinstance(ident, dict) and _NCT_RE.match(str(ident.get("nctId")))

    # At least one study in the page must carry the outcomes path we read.
    assert any(
        isinstance(
            (s.get("protocolSection") or {}).get("outcomesModule", {}).get("primaryOutcomes"),
            list,
        )
        for s in studies
    ), "no study exposes protocolSection.outcomesModule.primaryOutcomes"


def test_ctgov_primary_endpoints_parsed_contract() -> None:
    """``ClinicalTrialsEndpointProvider`` reads measure / time_frame / nct_id."""
    client = ClinicalTrialsClient()
    endpoints = client.primary_endpoints(_DRUG, _CONDITION)
    assert endpoints, f"CT.gov returned no completed-study endpoints for {_DRUG}/{_CONDITION}"
    for endpoint in endpoints:
        assert isinstance(endpoint.measure, str) and endpoint.measure.strip()
        assert _NCT_RE.match(endpoint.nct_id), f"malformed NCT id {endpoint.nct_id!r}"


# =========================================================================== PubMed


def test_pubmed_wire_shape_esearch_and_esummary() -> None:
    """Pin ``esearchresult.idlist`` and ``result[pmid].{title,source,pubdate,
    articleids[].idtype}`` — the paths ``_esearch_top_pmid`` / ``_esummary`` walk."""
    search = httpx.get(
        f"{PUBMED_BASE}/esearch.fcgi",
        params={"db": "pubmed", "term": f"{_DRUG} {_CONDITION}", "retmode": "json", "retmax": 1},
        timeout=30.0,
    )
    assert search.status_code == 200, f"PubMed esearch HTTP {search.status_code}"
    idlist = search.json().get("esearchresult", {}).get("idlist")
    assert isinstance(idlist, list) and idlist, "esearchresult.idlist is empty/absent"
    pmid = idlist[0]

    summary = httpx.get(
        f"{PUBMED_BASE}/esummary.fcgi",
        params={"db": "pubmed", "id": pmid, "retmode": "json"},
        timeout=30.0,
    )
    assert summary.status_code == 200, f"PubMed esummary HTTP {summary.status_code}"
    record = summary.json().get("result", {}).get(str(pmid))
    assert isinstance(record, dict), "esummary result lost its per-PMID record"
    assert isinstance(record.get("title"), str) and record["title"].strip()
    # journal <- "source", pubdate <- "pubdate", doi <- articleids[idtype=="doi"]
    assert "source" in record and "pubdate" in record
    assert isinstance(record.get("articleids"), list)


def test_pubmed_top_article_parsed_contract() -> None:
    """``PubMedRWEProvider`` reads pmid / title / journal / url."""
    client = PubMedClient()
    article = client.top_article(f"{_DRUG} {_CONDITION}")
    assert article is not None, f"PubMed returned no article for {_DRUG} {_CONDITION}"
    assert article.pmid.isdigit(), f"non-numeric pmid {article.pmid!r}"
    assert article.title.strip()
    assert article.url == f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"


def test_pubmed_fetch_by_pmid_round_trips() -> None:
    """``fetch_by_pmid`` backs the curated seminal-RWE path in brand_map."""
    client = PubMedClient()
    article = client.fetch_by_pmid("38507751")
    assert article is not None, "PubMed could not resolve a known-stable PMID"
    assert article.pmid == "38507751"
    assert "ribociclib" in article.title.lower()


def test_pubmed_client_survives_real_rate_limiting() -> None:
    """The 429 retry must hold against *real* NCBI throttling, not just a mock.

    Measured 2026-08-14: 8 rapid unretried esearch calls returned
    ``[200, 200, 200, 429, 429, 429, 429, 200]`` — NCBI allows 3 req/s without
    an API key. This fires a burst wide enough to provoke that throttling
    through the client and asserts every call still resolves.

    Distinct PMIDs are used so the module-level ``lru_cache`` cannot satisfy the
    burst from cache and quietly turn this into a no-op.
    """
    pmids = ["38507751", "36097254", "39371251", "29320312", "35642282", "33301246"]
    client = PubMedClient()

    resolved = [client.fetch_by_pmid(pmid) for pmid in pmids]

    unresolved = [pmid for pmid, art in zip(pmids, resolved, strict=True) if art is None]
    assert not unresolved, (
        f"PubMed burst left {len(unresolved)}/{len(pmids)} PMIDs unresolved "
        f"({unresolved}); the HTTP-429 retry is not absorbing real throttling"
    )
