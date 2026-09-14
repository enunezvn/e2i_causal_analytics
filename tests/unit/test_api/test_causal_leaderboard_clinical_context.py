"""Task 6: the discover-effects leaderboard attaches brand+outcome clinical context
to each completed row, FAIL-OPEN — a context failure never disrupts the row or job,
and rows without a brand / estimate are skipped (no fetch)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.routes.causal import catalog as causal_catalog
from src.api.routes.causal import discovery as causal_routes
from src.api.schemas.causal import DiscoveredEffect

# #1991 debt 4: the route no longer builds a ClinicalContextService at import —
# ``catalog._get_clinical_context_service()`` builds it on first use and caches it
# in ``catalog._clinical_context_service``. Seeding that cache with a stub is the
# ONE patch point that serves every reader, whichever module's binding of the
# accessor it calls, because the accessor resolves the global in ``catalog``.
_clinical_context_service = SimpleNamespace(get_context=lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _stub_clinical_context_service(monkeypatch):
    monkeypatch.setattr(causal_catalog, "_clinical_context_service", _clinical_context_service)


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
        _clinical_context_service,
        "get_context",
        lambda b, o, treatment=None: _PAYLOAD,
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
    def _boom(brand, outcome, treatment=None):
        raise RuntimeError("clinical-context API unavailable")

    monkeypatch.setattr(_clinical_context_service, "get_context", _boom)
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

    def _track(brand, outcome, treatment=None):
        calls["n"] += 1
        return _PAYLOAD

    monkeypatch.setattr(_clinical_context_service, "get_context", _track)
    no_brand = DiscoveredEffect(treatment="t", outcome="o", status="pending")
    no_estimate = DiscoveredEffect(treatment="t", outcome="o", brand="Kisqali", status="running")
    await causal_routes._attach_clinical_context(no_brand)
    await causal_routes._attach_clinical_context(no_estimate)
    assert calls["n"] == 0  # neither triggered a fetch
    assert no_brand.clinical_context is None
    assert no_estimate.clinical_context is None


# --- #1763: the leaderboard row already knows its treatment — pass it through ---


@pytest.mark.asyncio
async def test_attach_clinical_context_passes_the_row_treatment(monkeypatch):
    """Every leaderboard row is a (treatment -> outcome) analysis. Fetching context
    for the brand+outcome only is what made the panel read as 'accurate but
    unrelated' to the analysis being interrogated (#1763)."""
    seen = {}

    def _capture(brand, outcome, treatment=None):
        seen["args"] = (brand, outcome, treatment)
        return _PAYLOAD

    monkeypatch.setattr(_clinical_context_service, "get_context", _capture)
    eff = DiscoveredEffect(
        treatment="copay_support",
        outcome="persistent_180d",
        brand="Kisqali",
        status="completed",
        ate=0.12,
    )
    await causal_routes._attach_clinical_context(eff)
    assert seen["args"] == ("Kisqali", "persistent_180d", "copay_support")
