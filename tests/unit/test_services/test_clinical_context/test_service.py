"""The service fans out across providers, maps the synthetic outcome to the real
endpoint, caches per (brand,disease), degrades gracefully, and always labels the
synthetic/real boundary."""

from __future__ import annotations

import pytest

from src.services.clinical_context.clients import PubMedArticle
from src.services.clinical_context.providers import (
    CitationFragment,
    CompetitorFragment,
    EndpointsFragment,
    IndicationsFragment,
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


def _service(moa_frag, ep_frag, cite_frag, counters=None, ind_frag=None, comp_frag=None):
    counters = counters or {}
    # Default the two new fragments to their live/intended sources so the existing
    # cache / fully-live tests behave; pass ind_frag / comp_frag to vary them.
    ind_frag = ind_frag or IndicationsFragment(["Indication"], None, None, "openfda")
    comp_frag = comp_frag or CompetitorFragment(["Rival (generic)"], 1, "curated")
    return ClinicalContextService(
        mechanism_provider=_StubProvider(moa_frag, counters.get("moa")),
        endpoints_provider=_StubProvider(ep_frag, counters.get("ep")),
        citation_provider=_StubProvider(cite_frag, counters.get("cite")),
        indications_provider=_StubProvider(ind_frag, counters.get("ind")),
        competitor_provider=_StubProvider(comp_frag, counters.get("comp")),
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
    # FDA-label indications + curated competitor landscape are attached + sourced.
    assert ctx["approved_indications"]["indications"] == ["Indication"]
    assert ctx["approved_indications"]["source"] == "openfda"
    assert ctx["competitor_landscape"]["count"] == 1
    assert ctx["competitor_landscape"]["source"] == "curated"


def test_degrades_when_all_providers_fall_back():
    svc = _service(
        MechanismFragment("complement Factor B inhibitor", "static_fallback"),
        EndpointsFragment(["Transfusion avoidance"], "static_fallback"),
        CitationFragment(None, "unavailable"),
        ind_frag=IndicationsFragment(["PNH"], None, "boxed warning", "static_fallback"),
    )
    ctx = svc.get_context("Fabhalta", "treatment_initiated")
    assert ctx["mechanism"]["source"] == "static_fallback"
    assert ctx["pivotal_endpoints"]["source"] == "static_fallback"
    assert ctx["real_world_evidence"] is None
    assert ctx["approved_indications"]["source"] == "static_fallback"
    # Competitors stay curated even in a degraded fan-out (curated is the SSOT).
    assert ctx["competitor_landscape"]["source"] == "curated"
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


def test_degraded_result_self_heals_not_cached_permanently(monkeypatch):
    """A transient failure (degraded fan-out) must self-heal: after the degraded
    window the next request re-attempts the live APIs instead of returning the
    cached fallback for the whole process lifetime."""
    import src.services.clinical_context.service as svc_mod

    # Make a degraded entry immediately stale so we don't wait the real TTL.
    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 0.0)

    calls = {"n": 0}

    class _HealingMechProvider:
        provider_name = "healing"

        def enrich(self, profile):
            calls["n"] += 1
            if calls["n"] == 1:
                return MechanismFragment("BTK inhibitor", "static_fallback")  # transient failure
            return MechanismFragment("Bruton tyrosine kinase inhibitor", "chembl")  # recovered

    svc = ClinicalContextService(
        mechanism_provider=_HealingMechProvider(),
        endpoints_provider=_StubProvider(EndpointsFragment(["UAS7"], "clinicaltrials.gov")),
        citation_provider=_StubProvider(CitationFragment(None, "unavailable")),
        indications_provider=_StubProvider(
            IndicationsFragment(["CSU"], "Not indicated for other forms", None, "openfda")
        ),
        competitor_provider=_StubProvider(
            CompetitorFragment(["Xolair (omalizumab)"], 1, "curated")
        ),
    )
    first = svc.get_context("Remibrutinib", "persistent_180d")
    assert first["mechanism"]["source"] == "static_fallback"
    second = svc.get_context("Remibrutinib", "persistent_180d")
    # The degraded entry was not frozen -> the live API was retried and recovered.
    assert second["mechanism"]["source"] == "chembl"
    assert second["mechanism"]["mechanism_of_action"] == "Bruton tyrosine kinase inhibitor"


def test_fully_live_result_is_cached_indefinitely(monkeypatch):
    """A fully-live fan-out is cached and reused even with the degraded TTL at zero
    (only degraded entries are re-attempted, not fully-live ones)."""
    import src.services.clinical_context.service as svc_mod

    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 0.0)
    counters = {"moa": {"n": 0}, "ep": {"n": 0}, "cite": {"n": 0}}
    art = PubMedArticle(pmid="35642282", title="RWE", journal="J", doi="10.1/x")
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        EndpointsFragment(["OS"], "clinicaltrials.gov"),
        CitationFragment(art, "pubmed"),
        counters,
    )
    svc.get_context("Kisqali", "persistent_180d")
    svc.get_context("Kisqali", "persistent_180d")
    assert counters["moa"]["n"] == 1
    assert counters["ep"]["n"] == 1
    assert counters["cite"]["n"] == 1


def test_competitors_curated_does_not_block_fully_live_cache(monkeypatch):
    """A curated competitor source is the intended SSOT, not a degradation — with the
    four live-API providers live, the result is fully-live and cached even at
    degraded-TTL=0 (competitors being 'curated' must not force a re-fetch)."""
    import src.services.clinical_context.service as svc_mod

    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 0.0)
    counters = {"moa": {"n": 0}}
    art = PubMedArticle(pmid="1", title="t", journal="j", doi="10.1/z")
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        EndpointsFragment(["OS"], "clinicaltrials.gov"),
        CitationFragment(art, "pubmed"),
        counters,
        ind_frag=IndicationsFragment(["BC"], None, None, "openfda"),
        comp_frag=CompetitorFragment(["Ibrance (palbociclib)"], 1, "curated"),
    )
    svc.get_context("Kisqali", "persistent_180d")
    svc.get_context("Kisqali", "persistent_180d")
    assert counters["moa"]["n"] == 1  # cached indefinitely despite curated competitors


def test_openfda_down_degrades_and_self_heals(monkeypatch):
    """When OpenFDA indications fall to static_fallback the result is degraded and
    self-heals (re-fetched after the degraded window) — the OpenFDA-down path."""
    import src.services.clinical_context.service as svc_mod

    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 0.0)
    calls = {"n": 0}

    class _HealingIndProvider:
        provider_name = "healing-ind"

        def enrich(self, profile):
            calls["n"] += 1
            src = "static_fallback" if calls["n"] == 1 else "openfda"
            return IndicationsFragment(["CSU"], None, None, src)

    art = PubMedArticle(pmid="1", title="t", journal="j", doi="10.1/z")
    svc = ClinicalContextService(
        mechanism_provider=_StubProvider(MechanismFragment("BTK inhibitor", "chembl")),
        endpoints_provider=_StubProvider(EndpointsFragment(["UAS7"], "clinicaltrials.gov")),
        citation_provider=_StubProvider(CitationFragment(art, "pubmed")),
        indications_provider=_HealingIndProvider(),
        competitor_provider=_StubProvider(CompetitorFragment([], 0, "curated")),
    )
    first = svc.get_context("Remibrutinib", "persistent_180d")
    assert first["approved_indications"]["source"] == "static_fallback"
    second = svc.get_context("Remibrutinib", "persistent_180d")
    assert second["approved_indications"]["source"] == "openfda"  # self-healed
