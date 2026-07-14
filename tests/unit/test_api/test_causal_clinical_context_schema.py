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
    PivotalEndpointItem,
)


def _clinical_context() -> ClinicalContext:
    return ClinicalContext(
        brand="Kisqali",
        drug_name="ribociclib",
        disease="Malignant neoplasm of breast",
        our_outcome="persistent_180d",
        mechanism=MechanismOfAction(mechanism_of_action="CDK4/6 inhibitor", source="chembl"),
        pivotal_endpoints=PivotalEndpoint(
            endpoints=[
                PivotalEndpointItem(
                    measure="Overall Survival (OS)",
                    time_frame="Up to 5 years",
                    nct_id="NCT01958021",
                )
            ],
            source="clinicaltrials.gov",
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
def test_pivotal_endpoint_item_carries_measure_time_frame_and_nct():
    """A pivotal endpoint is now structured: verbatim measure + time frame + NCT id,
    and it round-trips through the FastAPI response serialization."""
    item = PivotalEndpointItem(
        measure="Change From Baseline in Weekly Urticaria Score (UAS7) at Week 12",
        time_frame="Baseline, Week 12",
        nct_id="NCT05030311",
    )
    assert item.time_frame == "Baseline, Week 12"
    assert item.nct_id == "NCT05030311"
    # time_frame / nct_id are optional (curated fallback has neither).
    bare = PivotalEndpointItem(measure="UCT7 (Urticaria Control Test)")
    assert bare.time_frame is None and bare.nct_id is None

    cc = _clinical_context()
    dumped = cc.model_dump()
    ep0 = dumped["pivotal_endpoints"]["endpoints"][0]
    assert ep0["measure"] == "Overall Survival (OS)"
    assert ep0["time_frame"] == "Up to 5 years"
    assert ep0["nct_id"] == "NCT01958021"


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
