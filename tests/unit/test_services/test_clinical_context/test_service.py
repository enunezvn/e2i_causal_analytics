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
        self.calls.append((profile.brand, outcome, getattr(treatment_context, "column", None), search_term))
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
    svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True)
    svc.get_context("Kisqali", "persistent_180d", treatment="treatment_arm", include_causal_evidence=True)
    svc.get_context("Kisqali", "persistent_180d", treatment="copay_support", include_causal_evidence=True)
    assert len(provider.calls) == 2
