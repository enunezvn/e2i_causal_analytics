"""Contract tests for GET /causal/clinical-context: assembles real clinical
context per brand, maps our synthetic outcome, 404s an unknown brand, and stays
200 (degraded) when an upstream API is down. The ClinicalContextService is
patched so no live HTTP runs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.api.routes import causal as causal_routes
from src.api.schemas.causal import ClinicalContext


@pytest.mark.unit
def test_clinical_context_model_round_trips():
    ctx = ClinicalContext(
        brand="Kisqali",
        drug_name="ribociclib",
        disease="Malignant neoplasm of breast",
        our_outcome="persistent_180d",
        mapped_endpoint="Treatment persistence / duration of therapy",
        mechanism={"mechanism_of_action": "CDK4/6 inhibitor", "source": "chembl"},
        pivotal_endpoints={
            "endpoints": [{"measure": "Overall Survival (OS)"}],
            "source": "clinicaltrials.gov",
        },
        real_world_evidence={
            "pmid": "35642282",
            "title": "RWE",
            "journal": "J",
            "pubdate": "2023 Jul",
            "doi": "10.1/x",
            "url": "https://pubmed.ncbi.nlm.nih.gov/35642282/",
            "source": "pubmed",
        },
        honesty_label="estimate = synthetic; context = real, cited",
    )
    assert ctx.mechanism.mechanism_of_action == "CDK4/6 inhibitor"
    assert ctx.real_world_evidence is not None
    assert ctx.real_world_evidence.pmid == "35642282"


@pytest.mark.asyncio
async def test_endpoint_returns_assembled_context_for_known_brand():
    fake_ctx = {
        "brand": "Kisqali",
        "drug_name": "ribociclib",
        "disease": "Malignant neoplasm of breast",
        "our_outcome": "persistent_180d",
        "mapped_endpoint": "Treatment persistence / duration of therapy",
        "mechanism": {"mechanism_of_action": "CDK4/6 inhibitor", "source": "chembl"},
        "pivotal_endpoints": {
            "endpoints": [{"measure": "Overall Survival (OS)"}],
            "source": "clinicaltrials.gov",
        },
        "real_world_evidence": None,
        "honesty_label": "estimate = synthetic; context = real, cited",
    }
    with patch.object(
        causal_routes._clinical_context_service, "get_context", return_value=fake_ctx
    ):
        with patch.object(
            causal_routes,
            "_list_dataset_brands",
            return_value=["Kisqali", "Fabhalta", "Remibrutinib"],
        ):
            resp = await causal_routes.get_clinical_context(
                brand="Kisqali", outcome="persistent_180d", user={"sub": "t"}
            )
    assert resp.brand == "Kisqali"
    assert resp.drug_name == "ribociclib"
    assert resp.mechanism.source == "chembl"
    assert resp.real_world_evidence is None


@pytest.mark.asyncio
async def test_endpoint_surfaces_seminal_rwe_through_schema():
    """The curated brand-specific seminal RWE must survive the ClinicalContext
    schema round-trip (response_model=ClinicalContext). Regression guard: the
    field was originally plumbed through the service but NOT declared on the
    schema, so Pydantic's default extra='ignore' silently dropped it before it
    ever reached the frontend."""
    fake_ctx = {
        "brand": "Kisqali",
        "drug_name": "ribociclib",
        "disease": "Malignant neoplasm of breast",
        "our_outcome": "persistent_180d",
        "mapped_endpoint": "Treatment persistence / duration of therapy",
        "mechanism": {"mechanism_of_action": "CDK4/6 inhibitor", "source": "chembl"},
        "pivotal_endpoints": {
            "endpoints": [{"measure": "Overall Survival (OS)"}],
            "source": "clinicaltrials.gov",
        },
        "real_world_evidence": None,
        "seminal_real_world_evidence": {
            "pmid": "36135090",
            "title": "Real-World Clinical Outcomes of Ribociclib ...",
            "journal": "Current Oncology",
            "pubdate": "2022",
            "doi": "10.3390/curroncol29090521",
            "url": "https://pubmed.ncbi.nlm.nih.gov/36135090/",
            "source": "curated",
        },
        "honesty_label": "estimate = synthetic; context = real, cited",
    }
    with patch.object(
        causal_routes._clinical_context_service, "get_context", return_value=fake_ctx
    ):
        with patch.object(
            causal_routes,
            "_list_dataset_brands",
            return_value=["Kisqali", "Fabhalta", "Remibrutinib"],
        ):
            resp = await causal_routes.get_clinical_context(
                brand="Kisqali", outcome="persistent_180d", user={"sub": "t"}
            )
    assert resp.seminal_real_world_evidence is not None
    assert resp.seminal_real_world_evidence.pmid == "36135090"
    assert resp.seminal_real_world_evidence.source == "curated"
    # And the response_model serialization keeps it (not stripped on the way out).
    assert resp.model_dump()["seminal_real_world_evidence"]["pmid"] == "36135090"


@pytest.mark.asyncio
async def test_endpoint_404s_unknown_brand():
    from fastapi import HTTPException

    with patch.object(
        causal_routes, "_list_dataset_brands", return_value=["Kisqali", "Fabhalta", "Remibrutinib"]
    ):
        with pytest.raises(HTTPException) as ei:
            await causal_routes.get_clinical_context(
                brand="NotABrand", outcome="persistent_180d", user={"sub": "t"}
            )
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_endpoint_stays_200_when_service_degrades():
    # Even with everything on static fallback, the endpoint returns a 200 payload.
    degraded = {
        "brand": "Fabhalta",
        "drug_name": "iptacopan",
        "disease": "Paroxysmal nocturnal hemoglobinuria",
        "our_outcome": "treatment_initiated",
        "mapped_endpoint": "Treatment initiation (complement-inhibitor start/switch)",
        "mechanism": {
            "mechanism_of_action": "complement Factor B inhibitor",
            "source": "static_fallback",
        },
        "pivotal_endpoints": {
            "endpoints": [{"measure": "Transfusion avoidance"}],
            "source": "static_fallback",
        },
        "real_world_evidence": None,
        "honesty_label": "estimate = synthetic; context = real, cited",
    }
    with patch.object(
        causal_routes._clinical_context_service, "get_context", return_value=degraded
    ):
        with patch.object(causal_routes, "_list_dataset_brands", return_value=["Fabhalta"]):
            resp = await causal_routes.get_clinical_context(
                brand="Fabhalta", outcome="treatment_initiated", user={"sub": "t"}
            )
    assert resp.mechanism.source == "static_fallback"
    assert resp.pivotal_endpoints.source == "static_fallback"


# --- #1763: the endpoint carries the analysis, not just the brand ---

_ANALYSIS_CTX = {
    "brand": "Kisqali",
    "drug_name": "ribociclib",
    "disease": "Malignant neoplasm of breast",
    "our_outcome": "persistent_180d",
    "our_treatment": "copay_support",
    "mapped_endpoint": "Treatment persistence / duration of therapy",
    "treatment_context": {
        "column": "copay_support",
        "label": "Copay support",
        "framing": "receiving copay assistance",
        "kind": "commercial",
        "source": "curated",
    },
    "analysis_framing": (
        "This analysis estimates the effect of receiving copay assistance on 180-day "
        "treatment persistence for ribociclib in Malignant neoplasm of breast."
    ),
    "mechanism": {"mechanism_of_action": "CDK4/6 inhibitor", "source": "chembl"},
    "pivotal_endpoints": {
        "endpoints": [{"measure": "Overall Survival (OS)"}],
        "source": "clinicaltrials.gov",
    },
    "real_world_evidence": {
        "pmid": "1",
        "title": "Copay assistance and persistence",
        "journal": "J",
        "pubdate": "2024",
        "doi": None,
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "source": "pubmed",
        "search_term": "ribociclib breast cancer persistence copay assistance real-world",
    },
    "honesty_label": "estimate = synthetic; context = real, cited",
}


@pytest.mark.asyncio
async def test_endpoint_forwards_the_treatment_to_the_service():
    with (
        patch.object(
            causal_routes._clinical_context_service, "get_context"
        ) as mock_get,
        patch.object(causal_routes, "_list_dataset_brands", return_value=["Kisqali"]),
    ):
        mock_get.return_value = _ANALYSIS_CTX
        await causal_routes.get_clinical_context(
            brand="Kisqali",
            outcome="persistent_180d",
            treatment="copay_support",
            user={"role": "viewer"},
        )
    mock_get.assert_called_once_with(
        "Kisqali", "persistent_180d", treatment="copay_support", include_causal_evidence=True
    )


@pytest.mark.asyncio
async def test_endpoint_surfaces_the_analysis_framing_fields():
    """Pydantic v2 defaults to extra='ignore': a new service payload key that is not
    declared on the schema is silently DROPPED. Pin the whole analysis frame."""
    with (
        patch.object(
            causal_routes._clinical_context_service,
            "get_context",
            return_value=_ANALYSIS_CTX,
        ),
        patch.object(causal_routes, "_list_dataset_brands", return_value=["Kisqali"]),
    ):
        ctx = await causal_routes.get_clinical_context(
            brand="Kisqali",
            outcome="persistent_180d",
            treatment="copay_support",
            user={"role": "viewer"},
        )
    assert ctx.our_treatment == "copay_support"
    assert ctx.treatment_context is not None
    assert ctx.treatment_context.kind == "commercial"
    assert ctx.treatment_context.framing == "receiving copay assistance"
    assert ctx.analysis_framing is not None
    assert "copay assistance" in ctx.analysis_framing
    assert ctx.real_world_evidence is not None
    assert ctx.real_world_evidence.search_term == (
        "ribociclib breast cancer persistence copay assistance real-world"
    )


def _wire_client():
    """A real HTTP client over the causal router, viewer auth stubbed out. Direct
    function calls cannot exercise query-param defaults (an omitted param arrives as
    the Query(...) object), and they never check the param NAME on the wire."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.dependencies.auth import require_viewer

    app = FastAPI()
    app.include_router(causal_routes.router)
    app.dependency_overrides[require_viewer] = lambda: {"role": "viewer"}
    return TestClient(app)


@pytest.mark.unit
def test_treatment_is_a_real_query_param_on_the_wire():
    with (
        patch.object(
            causal_routes._clinical_context_service,
            "get_context",
            return_value=_ANALYSIS_CTX,
        ) as mock_get,
        patch.object(causal_routes, "_list_dataset_brands", return_value=["Kisqali"]),
    ):
        response = _wire_client().get(
            "/causal/clinical-context",
            params={
                "brand": "Kisqali",
                "outcome": "persistent_180d",
                "treatment": "copay_support",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["our_treatment"] == "copay_support"
    assert body["treatment_context"]["kind"] == "commercial"
    assert body["analysis_framing"].startswith("This analysis estimates the effect of ")
    assert body["real_world_evidence"]["search_term"]
    mock_get.assert_called_once_with(
        "Kisqali", "persistent_180d", treatment="copay_support", include_causal_evidence=True
    )


@pytest.mark.unit
def test_endpoint_treatment_is_optional_brand_level_view_still_works():
    """The leaderboard MoA chip asks brand+outcome only; that must keep working and
    must not invent an analysis frame."""
    payload = dict(_ANALYSIS_CTX)
    payload.update({"our_treatment": None, "treatment_context": None, "analysis_framing": None})
    with (
        patch.object(
            causal_routes._clinical_context_service, "get_context", return_value=payload
        ) as mock_get,
        patch.object(causal_routes, "_list_dataset_brands", return_value=["Kisqali"]),
    ):
        response = _wire_client().get(
            "/causal/clinical-context",
            params={"brand": "Kisqali", "outcome": "persistent_180d"},
        )
    assert response.status_code == 200
    mock_get.assert_called_once_with(
        "Kisqali", "persistent_180d", treatment=None, include_causal_evidence=True
    )
    body = response.json()
    assert body["our_treatment"] is None
    assert body["treatment_context"] is None
    assert body["analysis_framing"] is None


# --- #1763 Phase 2: the evidence block reaches the client -----------------------

_EVIDENCE_CTX = dict(_ANALYSIS_CTX)
_EVIDENCE_CTX.update(
    {
        "our_treatment": "treatment_arm",
        "treatment_context": {
            "column": "treatment_arm",
            "label": "Treatment arm",
            "framing": "being on a ribociclib-containing regimen",
            "kind": "drug_therapy",
            "source": "curated",
        },
        "causal_evidence": {
            "status": "evidence",
            "indication_edge": {
                "predicate": "associated_with",
                "disease_id": "MONDO_0007254",
                "disease_name": "breast cancer",
                "max_clinical_stage": "PHASE_3",
                "source": "open_targets",
            },
            "citations": [
                {
                    "pmid": "40896422",
                    "title": "Real-world effectiveness of CDK4/6i",
                    "journal": "Front Oncol",
                    "pubdate": "2025",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/40896422/",
                    "entities_found": ["ribociclib", "breast cancer"],
                    "confidence": 0.5,
                    "source": "pubmed+europepmc",
                }
            ],
            "note": "Open Targets lags the FDA label.",
        },
    }
)


@pytest.mark.asyncio
async def test_endpoint_asks_for_the_causal_evidence_block():
    """The panel the user opened is exactly where the live evidence lookup belongs
    (the leaderboard fan-out deliberately does not pay for it)."""
    with (
        patch.object(
            causal_routes._clinical_context_service, "get_context", return_value=_EVIDENCE_CTX
        ) as mock_get,
        patch.object(causal_routes, "_list_dataset_brands", return_value=["Kisqali"]),
    ):
        await causal_routes.get_clinical_context(
            brand="Kisqali",
            outcome="persistent_180d",
            treatment="treatment_arm",
            user={"role": "viewer"},
        )
    mock_get.assert_called_once_with(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )


@pytest.mark.asyncio
async def test_causal_evidence_survives_the_schema():
    with (
        patch.object(
            causal_routes._clinical_context_service, "get_context", return_value=_EVIDENCE_CTX
        ),
        patch.object(causal_routes, "_list_dataset_brands", return_value=["Kisqali"]),
    ):
        ctx = await causal_routes.get_clinical_context(
            brand="Kisqali",
            outcome="persistent_180d",
            treatment="treatment_arm",
            user={"role": "viewer"},
        )
    assert ctx.causal_evidence is not None
    assert ctx.causal_evidence.status == "evidence"
    assert ctx.causal_evidence.indication_edge is not None
    assert ctx.causal_evidence.indication_edge.predicate == "associated_with"
    assert ctx.causal_evidence.indication_edge.max_clinical_stage == "PHASE_3"
    assert ctx.causal_evidence.citations[0].pmid == "40896422"
    assert ctx.causal_evidence.citations[0].entities_found == ["ribociclib", "breast cancer"]
    assert ctx.causal_evidence.note


@pytest.mark.asyncio
async def test_commercial_lever_evidence_state_reaches_the_client_verbatim():
    payload = dict(_EVIDENCE_CTX)
    payload["causal_evidence"] = {
        "status": "commercial_lever",
        "indication_edge": None,
        "citations": [],
        "note": "Copay support is a commercial access/promotion lever.",
    }
    with (
        patch.object(
            causal_routes._clinical_context_service, "get_context", return_value=payload
        ),
        patch.object(causal_routes, "_list_dataset_brands", return_value=["Kisqali"]),
    ):
        ctx = await causal_routes.get_clinical_context(
            brand="Kisqali",
            outcome="persistent_180d",
            treatment="copay_support",
            user={"role": "viewer"},
        )
    assert ctx.causal_evidence is not None
    assert ctx.causal_evidence.status == "commercial_lever"
    assert ctx.causal_evidence.indication_edge is None
    assert ctx.causal_evidence.citations == []
