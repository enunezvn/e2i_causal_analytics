"""The service fans out across providers, maps the synthetic outcome to the real
endpoint, caches per (brand,disease), degrades gracefully, and always labels the
synthetic/real boundary."""

from __future__ import annotations

import pytest

from src.services.clinical_context.clients import CTGovEndpoint, PubMedArticle
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


def _eps(measures, source="clinicaltrials.gov"):
    """Build an EndpointsFragment from bare measure strings (the real client returns
    CTGovEndpoint; time_frame / nct_id default to None here)."""
    return EndpointsFragment([CTGovEndpoint(m) for m in measures], source)


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
        _eps(["Overall Survival (OS)", "PFS"], "clinicaltrials.gov"),
        CitationFragment(art, "pubmed"),
    )
    ctx = svc.get_context("Kisqali", "persistent_180d")
    assert ctx["brand"] == "Kisqali"
    assert ctx["drug_name"] == "ribociclib"
    assert ctx["disease"] == "Malignant neoplasm of breast"
    assert ctx["mechanism"]["mechanism_of_action"] == "CDK4/6 inhibitor"
    assert ctx["mechanism"]["source"] == "chembl"
    assert ctx["pivotal_endpoints"]["endpoints"][0]["measure"] == "Overall Survival (OS)"
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


def test_get_context_carries_endpoint_time_frame_and_nct():
    """Structured endpoint provenance (time_frame + nct_id) must survive the service
    payload assembly as a dict, not just the measure text."""
    svc = _service(
        MechanismFragment("BTK inhibitor", "chembl"),
        EndpointsFragment(
            [
                CTGovEndpoint(
                    "Change From Baseline in Weekly Urticaria Score (UAS7) at Week 12",
                    "Baseline, Week 12",
                    "NCT05030311",
                )
            ],
            "clinicaltrials.gov",
        ),
        CitationFragment(None, "unavailable"),
    )
    ctx = svc.get_context("Remibrutinib", "treatment_initiated")
    ep = ctx["pivotal_endpoints"]["endpoints"][0]
    assert ep["measure"].startswith("Change From Baseline in Weekly Urticaria Score (UAS7)")
    assert ep["time_frame"] == "Baseline, Week 12"
    assert ep["nct_id"] == "NCT05030311"


def test_degrades_when_all_providers_fall_back():
    svc = _service(
        MechanismFragment("complement Factor B inhibitor", "static_fallback"),
        _eps(["Transfusion avoidance"], "static_fallback"),
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
        _eps(["OS"], "clinicaltrials.gov"),
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
        _eps([], "static_fallback"),
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
        endpoints_provider=_StubProvider(_eps(["UAS7"], "clinicaltrials.gov")),
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
        _eps(["OS"], "clinicaltrials.gov"),
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
        _eps(["OS"], "clinicaltrials.gov"),
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
        endpoints_provider=_StubProvider(_eps(["UAS7"], "clinicaltrials.gov")),
        citation_provider=_StubProvider(CitationFragment(art, "pubmed")),
        indications_provider=_HealingIndProvider(),
        competitor_provider=_StubProvider(CompetitorFragment([], 0, "curated")),
    )
    first = svc.get_context("Remibrutinib", "persistent_180d")
    assert first["approved_indications"]["source"] == "static_fallback"
    second = svc.get_context("Remibrutinib", "persistent_180d")
    assert second["approved_indications"]["source"] == "openfda"  # self-healed


# --- #1763: the context must be about the ANALYSIS, not just the brand ---


def test_treatment_threads_into_the_payload():
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        _eps(["OS"], "clinicaltrials.gov"),
        CitationFragment(None, "unavailable"),
    )
    ctx = svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm")
    assert ctx["our_treatment"] == "treatment_arm"
    assert ctx["treatment_context"] is not None
    assert ctx["treatment_context"]["column"] == "treatment_arm"
    assert ctx["treatment_context"]["kind"] == "drug_therapy"
    assert ctx["treatment_context"]["source"] == "curated"
    assert "ribociclib" in ctx["treatment_context"]["framing"].lower()
    assert ctx["analysis_framing"].startswith("This analysis estimates the effect of ")


def test_no_treatment_yields_an_honest_empty_analysis_frame():
    """The brand-level view (leaderboard MoA chip) passes no treatment: the payload
    must say so rather than invent an analysis it does not have."""
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        _eps(["OS"], "clinicaltrials.gov"),
        CitationFragment(None, "unavailable"),
    )
    ctx = svc.get_context("Kisqali", "persistent_180d")
    assert ctx["our_treatment"] is None
    assert ctx["treatment_context"] is None
    assert ctx["analysis_framing"] is None


def test_unmapped_treatment_is_reported_but_not_framed():
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        _eps(["OS"], "clinicaltrials.gov"),
        CitationFragment(None, "unavailable"),
    )
    ctx = svc.get_context("Kisqali", "persistent_180d", treatment="made_up_treatment")
    assert ctx["our_treatment"] == "made_up_treatment"
    assert ctx["treatment_context"] is None
    assert ctx["analysis_framing"] is None


class _CapturingCitationProvider:
    """Records the profile each citation lookup was made with."""

    provider_name = "capture"

    def __init__(self, fragment):
        self._fragment = fragment
        self.terms = []

    def enrich(self, profile):
        self.terms.append(profile.analysis_rwe_search_term)
        return self._fragment


def test_citation_provider_receives_the_analysis_specific_search_term():
    cap = _CapturingCitationProvider(CitationFragment(None, "unavailable"))
    svc = ClinicalContextService(
        mechanism_provider=_StubProvider(MechanismFragment("CDK4/6 inhibitor", "chembl")),
        endpoints_provider=_StubProvider(_eps(["OS"], "clinicaltrials.gov")),
        citation_provider=cap,
        indications_provider=_StubProvider(IndicationsFragment(["BC"], None, None, "openfda")),
        competitor_provider=_StubProvider(CompetitorFragment(["Ibrance (palbociclib)"], 1)),
    )
    svc.get_context("Kisqali", "persistent_180d", treatment="copay_support")
    assert len(cap.terms) == 1
    term = cap.terms[0].lower()
    assert "ribociclib" in term
    assert "copay" in term
    assert "persistence" in term


def test_citation_refetches_per_analysis_while_brand_fragments_stay_cached(monkeypatch):
    """The brand-level fan-out (MoA / endpoints / label / competitors) does NOT vary
    by analysis and must not be re-fetched; the literature search DOES vary and must
    be re-run per analysis. Splitting these is the whole point of the per-term
    citation cache."""
    import src.services.clinical_context.service as svc_mod

    # Prove the citation re-fetch is driven by the analysis, not by TTL expiry.
    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 10_000.0)
    art = PubMedArticle(pmid="1", title="t", journal="j", doi="10.1/z")
    counters = {"moa": {"n": 0}, "ep": {"n": 0}, "cite": {"n": 0}}
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        _eps(["OS"], "clinicaltrials.gov"),
        CitationFragment(art, "pubmed"),
        counters,
    )
    svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm")
    svc.get_context("Kisqali", "persistent_180d", treatment="copay_support")
    assert counters["moa"]["n"] == 1
    assert counters["ep"]["n"] == 1
    assert counters["cite"]["n"] == 2


def test_same_analysis_reuses_the_cached_citation(monkeypatch):
    import src.services.clinical_context.service as svc_mod

    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 0.0)
    art = PubMedArticle(pmid="1", title="t", journal="j", doi="10.1/z")
    counters = {"cite": {"n": 0}}
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        _eps(["OS"], "clinicaltrials.gov"),
        CitationFragment(art, "pubmed"),
        counters,
    )
    svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm")
    svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm")
    assert counters["cite"]["n"] == 1


def test_unavailable_citation_self_heals_per_analysis(monkeypatch):
    """An unavailable/seed citation is degraded and must be re-attempted after the
    self-heal window — the same guarantee the brand-level fan-out has."""
    import src.services.clinical_context.service as svc_mod

    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 0.0)
    counters = {"cite": {"n": 0}}
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        _eps(["OS"], "clinicaltrials.gov"),
        CitationFragment(None, "unavailable"),
        counters,
    )
    svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm")
    svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm")
    assert counters["cite"]["n"] == 2


def test_citation_payload_discloses_the_term_that_was_searched():
    art = PubMedArticle(pmid="35642282", title="RWE", journal="J", doi="10.1/x")
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        _eps(["OS"], "clinicaltrials.gov"),
        CitationFragment(art, "pubmed", "ribociclib breast cancer persistence real-world"),
    )
    ctx = svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm")
    assert (
        ctx["real_world_evidence"]["search_term"]
        == "ribociclib breast cancer persistence real-world"
    )
    # The curated seminal citation was never "searched" — it must not claim a term.
    assert ctx["seminal_real_world_evidence"]["search_term"] is None


# --- #1763 Phase 2: the evidence block is opt-in per call ----------------------


class _StubEvidenceProvider:
    def __init__(self, fragment):
        self._fragment = fragment
        self.calls = []

    def evidence(self, profile, *, outcome, treatment_context, search_term):
        self.calls.append(
            (profile.brand, outcome, getattr(treatment_context, "column", None), search_term)
        )
        return self._fragment


def _evidence_fragment():
    from src.services.clinical_context.causal_evidence import (
        CausalEvidenceFragment,
        IndicationEdge,
        VerifiedCitation,
    )

    return CausalEvidenceFragment(
        status="evidence",
        indication_edge=IndicationEdge(
            predicate="associated_with",
            drug_id="CHEMBL3545110",
            drug_name="RIBOCICLIB",
            disease_id="MONDO_0007254",
            disease_name="breast cancer",
            max_clinical_stage="PHASE_3",
            source="open_targets",
        ),
        citations=[
            VerifiedCitation(
                pmid="1",
                title="Ribociclib persistence in breast cancer",
                journal="J",
                pubdate="2024",
                url="https://pubmed.ncbi.nlm.nih.gov/1/",
                entities_found=("ribociclib", "breast cancer"),
                confidence=0.5,
                source="pubmed+europepmc",
            )
        ],
        note="Open Targets stages lag the FDA label.",
    )


def _service_with_evidence(provider):
    return ClinicalContextService(
        mechanism_provider=_StubProvider(MechanismFragment("CDK4/6 inhibitor", "chembl")),
        endpoints_provider=_StubProvider(_eps(["OS"], "clinicaltrials.gov")),
        citation_provider=_StubProvider(CitationFragment(None, "unavailable")),
        indications_provider=_StubProvider(IndicationsFragment(["BC"], None, None, "openfda")),
        competitor_provider=_StubProvider(CompetitorFragment(["Ibrance (palbociclib)"], 1)),
        causal_evidence_provider=provider,
    )


def test_causal_evidence_is_attached_when_requested():
    provider = _StubEvidenceProvider(_evidence_fragment())
    ctx = _service_with_evidence(provider).get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    ev = ctx["causal_evidence"]
    assert ev["status"] == "evidence"
    assert ev["indication_edge"]["disease_name"] == "breast cancer"
    assert ev["indication_edge"]["max_clinical_stage"] == "PHASE_3"
    assert ev["citations"][0]["pmid"] == "1"
    assert ev["citations"][0]["confidence"] == 0.5
    assert ev["note"]
    # It was asked about THIS analysis, with the analysis-composed query.
    assert provider.calls[0][0] == "Kisqali"
    assert provider.calls[0][2] == "treatment_arm"
    assert "persistence" in provider.calls[0][3]


def test_causal_evidence_is_not_fetched_by_default():
    """The leaderboard fan-out attaches context to every row; the evidence lookup is
    several live calls per analysis and nothing in the leaderboard renders it, so it
    stays opt-in and says so rather than looking unavailable."""
    provider = _StubEvidenceProvider(_evidence_fragment())
    ctx = _service_with_evidence(provider).get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm"
    )
    assert provider.calls == []
    assert ctx["causal_evidence"]["status"] == "not_requested"
    assert ctx["causal_evidence"]["citations"] == []


def test_causal_evidence_is_absent_without_a_treatment():
    """No treatment means no analysis to gather evidence for."""
    provider = _StubEvidenceProvider(_evidence_fragment())
    ctx = _service_with_evidence(provider).get_context(
        "Kisqali", "persistent_180d", include_causal_evidence=True
    )
    assert provider.calls == []
    assert ctx["causal_evidence"] is None


def test_causal_evidence_never_breaks_the_payload():
    class _BoomProvider:
        def evidence(self, profile, *, outcome, treatment_context, search_term):
            raise RuntimeError("open targets down")

    ctx = _service_with_evidence(_BoomProvider()).get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    assert ctx["causal_evidence"]["status"] == "unavailable"
    assert ctx["mechanism"]["source"] == "chembl"  # the rest of the payload survives


def test_causal_evidence_is_cached_per_analysis():
    provider = _StubEvidenceProvider(_evidence_fragment())
    svc = _service_with_evidence(provider)
    svc.get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    svc.get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    svc.get_context(
        "Kisqali", "persistent_180d", treatment="copay_support", include_causal_evidence=True
    )
    assert len(provider.calls) == 2


def test_evidence_cache_key_is_bounded_by_the_curated_universe():
    """`outcome` and `treatment` arrive from query params. Keying the evidence cache
    on the RAW outcome would let any authenticated caller grow this module-level dict
    without bound — and re-hit Open Targets / PubMed / Europe PMC for every novel
    string. The key is the analysis as the curated maps define it, which collapses
    unmapped outcomes onto the brand-level query they already fall back to."""
    import src.services.clinical_context.service as svc_mod

    provider = _StubEvidenceProvider(_evidence_fragment())
    svc = _service_with_evidence(provider)
    for i in range(20):
        svc.get_context(
            "Kisqali",
            f"made_up_outcome_{i}",
            treatment="treatment_arm",
            include_causal_evidence=True,
        )
    assert len(svc_mod._EVIDENCE_CACHE) == 1
    assert len(provider.calls) == 1


def test_an_uncurated_treatment_is_never_cached():
    """An unmapped treatment yields an immediate honest 'unavailable' with no live
    call, so caching it buys nothing and would be the same unbounded-key hazard."""
    import src.services.clinical_context.service as svc_mod

    provider = _StubEvidenceProvider(_evidence_fragment())
    svc = _service_with_evidence(provider)
    for i in range(20):
        svc.get_context(
            "Kisqali", "persistent_180d", treatment=f"junk_{i}", include_causal_evidence=True
        )
    assert svc_mod._EVIDENCE_CACHE == {}


def test_a_half_degraded_evidence_fragment_self_heals(monkeypatch):
    """The evidence block must not freeze an upstream outage into the process for
    good: a fragment whose sources partly failed is degraded, not settled."""
    import src.services.clinical_context.service as svc_mod
    from src.services.clinical_context.causal_evidence import CausalEvidenceFragment

    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 0.0)
    degraded = CausalEvidenceFragment(
        status="evidence",
        indication_edge=None,
        citations=[],
        note="Open Targets was unreachable.",
        sources_unavailable=("open_targets",),
    )
    provider = _StubEvidenceProvider(degraded)
    svc = _service_with_evidence(provider)
    svc.get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    svc.get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    assert len(provider.calls) == 2


def test_a_brand_level_citation_fallback_is_retried_not_frozen(monkeypatch):
    """`pubmed_brand` means the ANALYSIS query returned nothing — and the provider
    cannot tell a genuine zero-hit from a swallowed 429. Caching it forever under the
    analysis key would freeze a transient failure for the process lifetime."""
    import src.services.clinical_context.service as svc_mod

    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 0.0)
    art = PubMedArticle(pmid="1", title="t", journal="j", doi="10.1/z")
    counters = {"cite": {"n": 0}}
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        _eps(["OS"], "clinicaltrials.gov"),
        CitationFragment(
            art, "pubmed_brand", "ribociclib persistence adherence breast cancer real-world"
        ),
        counters,
    )
    svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm")
    svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm")
    assert counters["cite"]["n"] == 2


def test_evidence_payload_discloses_unavailable_sources():
    from src.services.clinical_context.causal_evidence import CausalEvidenceFragment

    provider = _StubEvidenceProvider(
        CausalEvidenceFragment(
            status="evidence",
            indication_edge=None,
            citations=[],
            note="Open Targets was unreachable for this analysis.",
            sources_unavailable=("open_targets",),
        )
    )
    ctx = _service_with_evidence(provider).get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    assert ctx["causal_evidence"]["sources_unavailable"] == ["open_targets"]


def test_a_degraded_evidence_result_never_overwrites_a_complete_one():
    """codex iter-2 MEDIUM. Two requests can miss the cache together. The slower one
    must not replace a complete fragment with the degraded one it happened to get
    from a transient upstream failure — that would serve the outage to everyone for
    the whole self-heal window even though a good answer already existed."""
    import time as _time

    import src.services.clinical_context.service as svc_mod
    from src.services.clinical_context.brand_map import (
        compose_rwe_search_term,
        resolve_brand_profile,
    )
    from src.services.clinical_context.causal_evidence import CausalEvidenceFragment

    profile = resolve_brand_profile("Kisqali")
    key = (
        "Kisqali",
        "treatment_arm",
        compose_rwe_search_term(profile, "persistent_180d", "treatment_arm"),
    )
    complete = _evidence_fragment()
    degraded = CausalEvidenceFragment(
        status="evidence",
        indication_edge=None,
        citations=[],
        note="Open Targets was unreachable.",
        sources_unavailable=("open_targets",),
    )

    class _RacingProvider:
        """Passes the cache check, then the OTHER request finishes first."""

        def evidence(self, profile, *, outcome, treatment_context, search_term):
            svc_mod._EVIDENCE_CACHE[key] = (complete, _time.monotonic(), True)
            return degraded

    ctx = _service_with_evidence(_RacingProvider()).get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    kept, _stored_at, complete_flag = svc_mod._EVIDENCE_CACHE[key]
    assert complete_flag is True
    assert kept.sources_unavailable == ()
    # The racer still returns what IT measured — it does not lie about its own call.
    assert ctx["causal_evidence"]["sources_unavailable"] == ["open_targets"]


# --- #1767: an unchecked literature result must not be cached as settled ---------


class _OutageResolver:
    """What a total Europe PMC outage produces: every abstract unresolved, with the
    verdict carrying the transport error."""

    def verify_citation(self, identifier, **kw):
        from src.data.kg.types import CitationVerdict

        return CitationVerdict(
            identifier=identifier,
            identifier_kind="pmid",
            abstract_resolved=False,
            entities_found=(),
            causal_cue_found=None,
            overall_confidence=0.0,
            error="Europe PMC transport error: simulated",
        )


class _OkOpenTargets:
    def search_drug(self, name):
        return "CHEMBL3545110"

    def search_disease(self, name):
        return "MONDO_0007254"

    def drug_disease_evidence(self, drug_chembl_id, disease_efo_id):
        return {
            "drug": {
                "id": "CHEMBL3545110",
                "name": "RIBOCICLIB",
                "indications": {
                    "rows": [
                        {
                            "disease": {"id": "MONDO_0007254", "name": "breast cancer"},
                            "maxClinicalStage": "PHASE_3",
                        }
                    ]
                },
            }
        }


class _OkPubMed:
    def search_pmids(self, term, *, retmax=5):
        return ["1", "2", "3"]

    def fetch_by_pmid(self, pmid):
        return None


def test_a_europe_pmc_outage_is_not_cached_as_a_settled_no_literature(monkeypatch):
    """THE #1767 REGRESSION, end to end through the real provider.

    Open Targets answers, Europe PMC is down, so no candidate can be verified. The
    fragment used to come back status='evidence' with citations=[] and an EMPTY
    sources_unavailable, which makes ``complete`` True — pinning "there is no
    literature for this analysis" for the life of the worker process. It must be
    stored DEGRADED so it self-heals through the 600s window instead.
    """
    import src.services.clinical_context.service as svc_mod
    from src.services.clinical_context.causal_evidence import CausalEvidenceProvider

    monkeypatch.setattr(svc_mod, "_FRAGMENT_TTL_DEGRADED_S", 0.0)
    provider = CausalEvidenceProvider(
        open_targets=_OkOpenTargets(),
        pubmed=_OkPubMed(),
        resolver=_OutageResolver(),
    )
    svc = _service_with_evidence(provider)
    ctx = svc.get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    evidence = ctx["causal_evidence"]
    assert evidence["citations"] == []
    assert "europe_pmc" in evidence["sources_unavailable"]

    assert len(svc_mod._EVIDENCE_CACHE) == 1
    _frag, _stored_at, complete = next(iter(svc_mod._EVIDENCE_CACHE.values()))
    assert complete is False, "an unchecked literature result must never be cached as complete"


def test_a_budget_truncated_literature_check_is_degraded_without_blaming_a_source(monkeypatch):
    """codex iter-1 HIGH (#1767). Stopping early under our OWN wall-clock budget
    leaves the literature question unfinished, so the fragment must not be cached as
    settled — but Europe PMC must not be named either. Naming a healthy source is
    the same dishonesty inverted, and it would re-hit three upstreams every 600s for
    the life of the process."""
    import src.services.clinical_context.causal_evidence as ev_mod
    import src.services.clinical_context.service as svc_mod
    from src.services.clinical_context.causal_evidence import CausalEvidenceProvider

    monkeypatch.setattr(ev_mod, "_VERIFICATION_BUDGET_S", 0.0)

    class _WeakThenUnreached:
        """Candidate 1 resolves and is genuinely weak; 2 and 3 are never reached."""

        def verify_citation(self, identifier, **kw):
            from src.data.kg.types import CitationVerdict

            return CitationVerdict(
                identifier=identifier,
                identifier_kind="pmid",
                abstract_resolved=True,
                entities_found=("ribociclib",),
                causal_cue_found=None,
                overall_confidence=0.1,
                error=None,
            )

    provider = CausalEvidenceProvider(
        open_targets=_OkOpenTargets(), pubmed=_OkPubMed(), resolver=_WeakThenUnreached()
    )
    ctx = _service_with_evidence(provider).get_context(
        "Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True
    )
    assert ctx["causal_evidence"]["sources_unavailable"] == []
    assert "europe pmc" not in ctx["causal_evidence"]["note"].lower()

    _frag, _stored_at, complete = next(iter(svc_mod._EVIDENCE_CACHE.values()))
    assert complete is False, "an unfinished literature check must not be cached as settled"


# --- #1775: the payload must GROUND the scenario -------------------------------


def _svc_with_label_considerations():
    """A service whose openFDA fragment carries label considerations, as the real
    provider now does."""
    from src.services.clinical_context.label_considerations import (
        DOSAGE_SECTION,
        WARNINGS_SECTION,
        LabelConsideration,
    )

    considerations = (
        LabelConsideration(
            title="QT Interval Prolongation",
            detail="Monitor electrocardiograms (ECGs) and electrolytes prior to initiation.",
            section=WARNINGS_SECTION,
            references="2.2 , 5.3",
        ),
        LabelConsideration(
            title="Dosage and administration",
            detail="Dose interruption, reduction, and/or discontinuation may be required "
            "based on individual safety and tolerability.",
            section=DOSAGE_SECTION,
            references="2.2",
        ),
    )
    return ClinicalContextService(
        mechanism_provider=_StubProvider(MechanismFragment("CDK4/6 inhibitor", "chembl")),
        endpoints_provider=_StubProvider(_eps(["OS"], "clinicaltrials.gov")),
        citation_provider=_StubProvider(CitationFragment(None, "unavailable")),
        indications_provider=_StubProvider(
            IndicationsFragment(["BC"], None, None, "openfda", considerations)
        ),
        competitor_provider=_StubProvider(CompetitorFragment(["Ibrance (palbociclib)"], 1)),
    )


def test_a_commercial_analysis_is_grounded_in_the_payload():
    """THE #1775 REGRESSION at the payload boundary. copay_support used to receive
    no clinical grounding at all."""
    ctx = _svc_with_label_considerations().get_context(
        "Kisqali", "persistent_180d", treatment="copay_support"
    )
    grounding = ctx["analysis_grounding"]
    # POSITIVE CONTROL: assert something is actually there before asserting about it.
    assert len(grounding["label_considerations"]) >= 1, grounding
    first = grounding["label_considerations"][0]
    assert first["source"] == "openfda"
    assert first["references"], "a consideration must cite the label section it came from"
    assert grounding["competitive_context"]
    assert grounding["outcome_theme"] == "persistence"


def test_grounding_is_absent_for_the_brand_level_view():
    """No treatment means no scenario to ground; inventing one is the #1763 defect."""
    ctx = _svc_with_label_considerations().get_context("Kisqali", "persistent_180d")
    assert ctx["analysis_grounding"] is None


def test_grounding_never_asserts_the_label_speaks_to_the_lever():
    ctx = _svc_with_label_considerations().get_context(
        "Kisqali", "persistent_180d", treatment="copay_support"
    )
    note = ctx["analysis_grounding"]["note"].lower()
    assert "says nothing about" in note
    assert "not the complete" in note


def test_grounding_survives_the_api_response_model():
    """#1775 wire guard. `response_model=ClinicalContext` DROPS any key the Pydantic
    schema does not declare, so the whole grounding feature shipped invisibly to the
    panel until the schema knew about it — a green backend and an unchanged UI.
    Verified by construction: this test failed before AnalysisGrounding existed."""
    from src.api.schemas.causal import ClinicalContext

    payload = _svc_with_label_considerations().get_context(
        "Kisqali", "persistent_180d", treatment="copay_support"
    )
    dumped = ClinicalContext.model_validate(payload).model_dump()
    grounding = dumped.get("analysis_grounding")
    assert grounding is not None, "response_model stripped analysis_grounding"
    # POSITIVE CONTROL: a present-but-empty block would satisfy `is not None`.
    assert len(grounding["label_considerations"]) >= 1
    first = grounding["label_considerations"][0]
    assert first["references"] and first["detail"] and first["source"] == "openfda"
    assert grounding["competitive_context"]


@pytest.mark.unit
def test_grounding_is_null_not_an_empty_object_when_there_is_nothing_to_ground(monkeypatch):
    """codex iter-12 LOW. `treatment is not None` gated the payload, so a grounding
    with no considerations, no competitive context and no note shipped as an empty
    OBJECT while the schema and the TS type both document `null` for "no scenario to
    ground". Not user-visible — the panel declines to render it — but a wire contract
    that disagrees with its own documentation is the next defect waiting for a
    consumer who believes the documentation.

    Exercised through the SERVICE, not by asserting a dataclass default. The first
    version of this test did the latter and proved nothing, which is the precise
    failure shape this round was about.
    """
    import src.services.clinical_context.service as svc
    from src.services.clinical_context.analysis_grounding import AnalysisGrounding

    service = _svc_with_label_considerations()
    # POSITIVE CONTROL: the same call yields a populated object before we empty it,
    # so a later `is None` cannot pass just because the path went missing.
    populated = service.get_context("Kisqali", "persistent_180d", treatment="copay_support")
    assert populated["analysis_grounding"] is not None
    assert populated["analysis_grounding"]["label_considerations"]

    monkeypatch.setattr(svc, "ground_analysis", lambda *a, **k: AnalysisGrounding())
    ctx = _svc_with_label_considerations().get_context(
        "Kisqali", "persistent_180d", treatment="copay_support"
    )
    assert ctx["analysis_grounding"] is None, ctx["analysis_grounding"]


def test_one_process_cold_fills_a_repeated_request_exactly_once():
    """Every cache this service owns fills on the FIRST call for a given request
    shape -- no dict fills later than the others.

    This pins the invariant that licenses the #1768 measurement. That measurement
    counted SLOW responses to identical sequential requests and read the count as
    the number of worker processes reached, on the reasoning that one process
    cannot cold-fill the same key twice. The codex audit of #1786 was right that
    the reasoning has a second exit: if `_FRAGMENT_CACHE`, `_CITATION_CACHE` and
    `_EVIDENCE_CACHE` could fill on DIFFERENT calls, a single worker would also
    produce two slow responses and the count would prove nothing about workers.

    So assert it here, in-process, where worker count is fixed at one: after the
    first call every provider is already at its final call count, and it stays
    there. A future change that defers any fragment to a later request breaks
    this test rather than silently invalidating the measurement in the comment
    above the caches in service.py.
    """
    counters = {
        "moa": {"n": 0},
        "ep": {"n": 0},
        "cite": {"n": 0},
        "ind": {"n": 0},
        "comp": {"n": 0},
    }
    art = PubMedArticle(pmid="35642282", title="RWE", journal="J", doi="10.1/x")
    evidence = _StubEvidenceProvider(_evidence_fragment())
    svc = ClinicalContextService(
        mechanism_provider=_StubProvider(
            MechanismFragment("CDK4/6 inhibitor", "chembl"), counters["moa"]
        ),
        endpoints_provider=_StubProvider(_eps(["OS"], "clinicaltrials.gov"), counters["ep"]),
        citation_provider=_StubProvider(CitationFragment(art, "pubmed"), counters["cite"]),
        indications_provider=_StubProvider(
            IndicationsFragment(["BC"], None, None, "openfda"), counters["ind"]
        ),
        competitor_provider=_StubProvider(
            CompetitorFragment(["Ibrance (palbociclib)"], 1, "curated"), counters["comp"]
        ),
        causal_evidence_provider=evidence,
    )
    call = {
        "brand": "Kisqali",
        "outcome": "persistent_180d",
        "treatment": "treatment_arm",
        "include_causal_evidence": True,
    }

    svc.get_context(**call)
    after_first = {k: v["n"] for k, v in counters.items()}
    after_first["evidence"] = len(evidence.calls)

    # Positive control for THIS test: the first call must actually have exercised
    # every provider, otherwise "nothing grew afterwards" is vacuous.
    assert all(n == 1 for n in after_first.values()), after_first

    for _ in range(5):
        svc.get_context(**call)

    after_all = {k: v["n"] for k, v in counters.items()}
    after_all["evidence"] = len(evidence.calls)
    # Nothing cold-filled after the first call, so a second slow response in one
    # process is not reachable by this route.
    assert after_all == after_first, after_all
