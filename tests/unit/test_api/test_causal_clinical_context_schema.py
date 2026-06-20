"""Schema contract for the OpenFDA clinical-context additions: ApprovedIndications +
CompetitorLandscape on ClinicalContext, and DiscoveredEffect.clinical_context (a
forward reference resolved via model_rebuild)."""

from __future__ import annotations

import pytest

from src.api.schemas.causal import (
    ApprovedIndications,
    ClinicalContext,
    CompetitorLandscape,
    DiscoveredEffect,
    MechanismOfAction,
    PivotalEndpoint,
)


def _clinical_context() -> ClinicalContext:
    return ClinicalContext(
        brand="Kisqali",
        drug_name="ribociclib",
        disease="Malignant neoplasm of breast",
        our_outcome="persistent_180d",
        mechanism=MechanismOfAction(mechanism_of_action="CDK4/6 inhibitor", source="chembl"),
        pivotal_endpoints=PivotalEndpoint(
            endpoints=["Overall Survival (OS)"], source="clinicaltrials.gov"
        ),
        approved_indications=ApprovedIndications(
            indications=["HR+/HER2- breast cancer"],
            limitations_of_use=None,
            boxed_warning=None,
            source="openfda",
        ),
        competitor_landscape=CompetitorLandscape(
            competitors=["Ibrance (palbociclib)", "Verzenio (abemaciclib)"],
            count=2,
            source="curated",
        ),
        honesty_label="synthetic estimate / real context",
    )


@pytest.mark.unit
def test_discovered_effect_clinical_context_defaults_none():
    e = DiscoveredEffect(treatment="treatment_arm", outcome="persistent_180d", status="completed")
    assert e.clinical_context is None


@pytest.mark.unit
def test_clinical_context_carries_indications_and_competitors():
    cc = _clinical_context()
    assert cc.approved_indications is not None
    assert cc.approved_indications.source == "openfda"
    assert cc.competitor_landscape is not None
    assert cc.competitor_landscape.count == 2
    assert cc.competitor_landscape.source == "curated"


@pytest.mark.unit
def test_discovered_effect_embeds_clinical_context_forward_ref():
    """DiscoveredEffect.clinical_context forward-references ClinicalContext (defined
    later in the module); model_rebuild() must have resolved it so this round-trips
    through serialization (the FastAPI response path)."""
    e = DiscoveredEffect(
        treatment="treatment_arm",
        outcome="persistent_180d",
        status="completed",
        clinical_context=_clinical_context(),
    )
    assert e.clinical_context is not None
    assert e.clinical_context.competitor_landscape is not None
    assert e.clinical_context.competitor_landscape.count == 2
    dumped = e.model_dump()
    assert dumped["clinical_context"]["approved_indications"]["source"] == "openfda"
    assert dumped["clinical_context"]["competitor_landscape"]["source"] == "curated"
