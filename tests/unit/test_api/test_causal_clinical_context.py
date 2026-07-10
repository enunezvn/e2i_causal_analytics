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
        pivotal_endpoints={"endpoints": ["Overall Survival (OS)"], "source": "clinicaltrials.gov"},
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
            "endpoints": ["Overall Survival (OS)"],
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
            "endpoints": ["Overall Survival (OS)"],
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
        "pivotal_endpoints": {"endpoints": ["Transfusion avoidance"], "source": "static_fallback"},
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
