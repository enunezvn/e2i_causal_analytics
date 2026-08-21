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


# --- #1763: the analysis frame is part of the wire contract ---


@pytest.mark.unit
def test_treatment_context_model_round_trips_on_clinical_context():
    from src.api.schemas.causal import TreatmentContext

    ctx = _clinical_context().model_copy(
        update={
            "our_treatment": "treatment_arm",
            "treatment_context": TreatmentContext(
                column="treatment_arm",
                label="Treatment arm",
                framing="being on a ribociclib-containing regimen",
                kind="drug_therapy",
                source="curated",
            ),
            "analysis_framing": (
                "This analysis estimates the effect of being on a ribociclib-containing "
                "regimen on 180-day treatment persistence for ribociclib in Malignant "
                "neoplasm of breast."
            ),
        }
    )
    dumped = ctx.model_dump()
    assert dumped["our_treatment"] == "treatment_arm"
    assert dumped["treatment_context"]["kind"] == "drug_therapy"
    assert dumped["analysis_framing"].startswith("This analysis estimates the effect of ")


@pytest.mark.unit
def test_analysis_fields_are_optional_and_default_to_none():
    """A brand-level payload (no treatment) must still validate — the panel then
    renders exactly as it did before #1763."""
    ctx = _clinical_context()
    assert ctx.our_treatment is None
    assert ctx.treatment_context is None
    assert ctx.analysis_framing is None


@pytest.mark.unit
def test_clinical_context_validates_a_raw_service_payload_with_the_new_keys():
    """Pydantic v2 defaults to extra='ignore' — an undeclared key is dropped
    silently. Validate from a RAW dict (as the route does) so a missing field
    declaration fails here instead of vanishing on the wire."""
    payload = _clinical_context().model_dump()
    payload["our_treatment"] = "copay_support"
    payload["treatment_context"] = {
        "column": "copay_support",
        "label": "Copay support",
        "framing": "receiving copay assistance",
        "kind": "commercial",
        "source": "curated",
    }
    payload["analysis_framing"] = "This analysis estimates the effect of X on Y."
    payload["real_world_evidence"] = {
        "pmid": "1",
        "title": "t",
        "journal": "j",
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "source": "pubmed",
        "search_term": "ribociclib breast cancer copay assistance",
    }
    revalidated = ClinicalContext.model_validate(payload)
    assert revalidated.our_treatment == "copay_support"
    assert revalidated.treatment_context is not None
    assert revalidated.treatment_context.label == "Copay support"
    assert revalidated.analysis_framing == "This analysis estimates the effect of X on Y."
    assert revalidated.real_world_evidence is not None
    assert (
        revalidated.real_world_evidence.search_term == "ribociclib breast cancer copay assistance"
    )
