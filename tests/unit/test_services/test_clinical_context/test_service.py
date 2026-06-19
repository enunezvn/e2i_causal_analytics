"""The service fans out across providers, maps the synthetic outcome to the real
endpoint, caches per (brand,disease), degrades gracefully, and always labels the
synthetic/real boundary."""

from __future__ import annotations

import pytest

from src.services.clinical_context.clients import PubMedArticle
from src.services.clinical_context.providers import (
    CitationFragment,
    EndpointsFragment,
    MechanismFragment,
)
from src.services.clinical_context.service import ClinicalContextService, reset_caches


class _StubProvider:
    def __init__(self, fragment, counter=None, name="stub"):
        self._fragment = fragment
        self._counter = counter
        self.provider_name = name

    def enrich(self, profile):
        if self._counter is not None:
            self._counter["n"] += 1
        return self._fragment


@pytest.fixture(autouse=True)
def _clear() -> None:
    reset_caches()


def _service(moa_frag, ep_frag, cite_frag, counters=None):
    counters = counters or {}
    return ClinicalContextService(
        mechanism_provider=_StubProvider(moa_frag, counters.get("moa")),
        endpoints_provider=_StubProvider(ep_frag, counters.get("ep")),
        citation_provider=_StubProvider(cite_frag, counters.get("cite")),
    )


def test_get_context_assembles_all_three_sources():
    art = PubMedArticle(pmid="35642282", title="RWE", journal="J", doi="10.1/x")
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        EndpointsFragment(["Overall Survival (OS)", "PFS"], "clinicaltrials.gov"),
        CitationFragment(art, "pubmed"),
    )
    ctx = svc.get_context("Kisqali", "persistent_180d")
    assert ctx["brand"] == "Kisqali"
    assert ctx["drug_name"] == "ribociclib"
    assert ctx["disease"] == "Malignant neoplasm of breast"
    assert ctx["mechanism"]["mechanism_of_action"] == "CDK4/6 inhibitor"
    assert ctx["mechanism"]["source"] == "chembl"
    assert ctx["pivotal_endpoints"]["endpoints"][0] == "Overall Survival (OS)"
    assert ctx["pivotal_endpoints"]["source"] == "clinicaltrials.gov"
    # The synthetic outcome is mapped to the real endpoint framing.
    assert "persist" in ctx["mapped_endpoint"].lower()
    assert ctx["our_outcome"] == "persistent_180d"
    # The real-world-evidence citation round-trips.
    assert ctx["real_world_evidence"]["pmid"] == "35642282"
    assert ctx["real_world_evidence"]["url"] == "https://pubmed.ncbi.nlm.nih.gov/35642282/"
    # The honesty label is ALWAYS present and names the boundary.
    assert "synthetic" in ctx["honesty_label"].lower()
    assert "real" in ctx["honesty_label"].lower()


def test_degrades_when_all_providers_fall_back():
    svc = _service(
        MechanismFragment("complement Factor B inhibitor", "static_fallback"),
        EndpointsFragment(["Transfusion avoidance"], "static_fallback"),
        CitationFragment(None, "unavailable"),
    )
    ctx = svc.get_context("Fabhalta", "treatment_initiated")
    assert ctx["mechanism"]["source"] == "static_fallback"
    assert ctx["pivotal_endpoints"]["source"] == "static_fallback"
    assert ctx["real_world_evidence"] is None
    assert ctx["honesty_label"]  # still present


def test_cache_is_per_brand_disease_not_per_outcome():
    counters = {"moa": {"n": 0}, "ep": {"n": 0}, "cite": {"n": 0}}
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        EndpointsFragment(["OS"], "clinicaltrials.gov"),
        CitationFragment(None, "unavailable"),
        counters,
    )
    a = svc.get_context("Kisqali", "persistent_180d")
    b = svc.get_context("Kisqali", "treatment_initiated")  # same brand, diff outcome
    # The expensive provider fan-out ran ONCE (cached per brand/disease)...
    assert counters["moa"]["n"] == 1
    assert counters["ep"]["n"] == 1
    # ...but the outcome->endpoint mapping differs per call.
    assert a["mapped_endpoint"] != b["mapped_endpoint"]


def test_unknown_brand_raises_keyerror():
    svc = _service(
        MechanismFragment("x", "static_fallback"),
        EndpointsFragment([], "static_fallback"),
        CitationFragment(None, "unavailable"),
    )
    with pytest.raises(KeyError):
        svc.get_context("NotABrand", "persistent_180d")
