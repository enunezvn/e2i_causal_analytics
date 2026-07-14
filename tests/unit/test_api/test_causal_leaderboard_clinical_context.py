"""Task 6: the discover-effects leaderboard attaches brand+outcome clinical context
to each completed row, FAIL-OPEN — a context failure never disrupts the row or job,
and rows without a brand / estimate are skipped (no fetch)."""

from __future__ import annotations

import pytest

from src.api.routes import causal as causal_routes
from src.api.schemas.causal import DiscoveredEffect

_PAYLOAD = {
    "brand": "Kisqali",
    "drug_name": "ribociclib",
    "disease": "Malignant neoplasm of breast",
    "our_outcome": "persistent_180d",
    "mapped_endpoint": None,
    "mechanism": {"mechanism_of_action": "CDK4/6 inhibitor", "source": "chembl"},
    "pivotal_endpoints": {
        "endpoints": [{"measure": "Overall Survival (OS)"}],
        "source": "clinicaltrials.gov",
    },
    "real_world_evidence": None,
    "approved_indications": {
        "indications": ["HR+/HER2- breast cancer"],
        "limitations_of_use": None,
        "boxed_warning": None,
        "source": "openfda",
    },
    "competitor_landscape": {
        "competitors": ["Ibrance (palbociclib)"],
        "count": 1,
        "source": "curated",
    },
    "honesty_label": "synthetic estimate / real context",
}


@pytest.mark.asyncio
async def test_attach_clinical_context_happy_path(monkeypatch):
    monkeypatch.setattr(
        causal_routes._clinical_context_service, "get_context", lambda b, o: _PAYLOAD
    )
    eff = DiscoveredEffect(
        treatment="treatment_arm",
        outcome="persistent_180d",
        brand="Kisqali",
        status="completed",
        ate=0.12,
    )
    await causal_routes._attach_clinical_context(eff)
    assert eff.clinical_context is not None
    assert eff.clinical_context.competitor_landscape is not None
    assert eff.clinical_context.competitor_landscape.count == 1
    assert eff.clinical_context.approved_indications is not None
    assert eff.clinical_context.approved_indications.source == "openfda"


@pytest.mark.asyncio
async def test_attach_clinical_context_fail_open(monkeypatch):
    def _boom(brand, outcome):
        raise RuntimeError("clinical-context API unavailable")

    monkeypatch.setattr(causal_routes._clinical_context_service, "get_context", _boom)
    eff = DiscoveredEffect(
        treatment="treatment_arm",
        outcome="persistent_180d",
        brand="Kisqali",
        status="completed",
        ate=0.12,
    )
    # Must NOT raise; the row survives with no context.
    await causal_routes._attach_clinical_context(eff)
    assert eff.clinical_context is None


@pytest.mark.asyncio
async def test_attach_clinical_context_skips_without_brand_or_estimate(monkeypatch):
    calls = {"n": 0}

    def _track(brand, outcome):
        calls["n"] += 1
        return _PAYLOAD

    monkeypatch.setattr(causal_routes._clinical_context_service, "get_context", _track)
    no_brand = DiscoveredEffect(treatment="t", outcome="o", status="pending")
    no_estimate = DiscoveredEffect(treatment="t", outcome="o", brand="Kisqali", status="running")
    await causal_routes._attach_clinical_context(no_brand)
    await causal_routes._attach_clinical_context(no_estimate)
    assert calls["n"] == 0  # neither triggered a fetch
    assert no_brand.clinical_context is None
    assert no_estimate.clinical_context is None
