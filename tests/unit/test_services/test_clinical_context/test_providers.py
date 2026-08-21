"""Each provider is best-effort: live value preferred, static fallback when the
API is down/empty, with an honest source label. No live HTTP (clients injected)."""

from __future__ import annotations

import pytest

from src.services.clinical_context.brand_map import resolve_brand_profile
from src.services.clinical_context.clients import CTGovEndpoint
from src.services.clinical_context.providers import (
    ChEMBLMechanismProvider,
    ClinicalTrialsEndpointProvider,
    CuratedCompetitorProvider,
    OpenFDAIndicationsProvider,
    PubMedRWEProvider,
)


class _FakeChEMBL:
    def __init__(self, moa):
        self._moa = moa

    def mechanism_of_action(self, drug_name):  # noqa: D401
        if isinstance(self._moa, Exception):
            raise self._moa
        return self._moa


class _FakeCTGov:
    def __init__(self, eps):
        self._eps = eps

    def primary_endpoints(self, intervention, condition, *, limit=8):
        if isinstance(self._eps, Exception):
            raise self._eps
        # Accept bare measure strings for brevity; the real client returns CTGovEndpoint.
        return [e if isinstance(e, CTGovEndpoint) else CTGovEndpoint(measure=e) for e in self._eps]


class _FakePubMed:
    def __init__(self, art=None, by_pmid=None):
        self._art = art
        self._by_pmid = by_pmid

    def top_article(self, term):
        if isinstance(self._art, Exception):
            raise self._art
        return self._art

    def fetch_by_pmid(self, pmid):
        return self._by_pmid


@pytest.mark.unit
def test_chembl_provider_prefers_live_moa():
    profile = resolve_brand_profile("Kisqali")
    frag = ChEMBLMechanismProvider(
        client=_FakeChEMBL("Cyclin-dependent kinase 4 inhibitor")
    ).enrich(profile)
    assert frag.mechanism_of_action == "Cyclin-dependent kinase 4 inhibitor"
    assert frag.source == "chembl"


@pytest.mark.unit
def test_chembl_provider_falls_back_on_error():
    from src.data.kg.chembl import ChEMBLError

    profile = resolve_brand_profile("Kisqali")
    frag = ChEMBLMechanismProvider(client=_FakeChEMBL(ChEMBLError("boom"))).enrich(profile)
    # Static spec-pinned fallback, honestly labelled.
    assert frag.mechanism_of_action == "CDK4/6 inhibitor"
    assert frag.source == "static_fallback"


@pytest.mark.unit
def test_ctgov_provider_prefers_live_then_falls_back():
    profile = resolve_brand_profile("Fabhalta")
    live = ClinicalTrialsEndpointProvider(
        client=_FakeCTGov(["Transfusion avoidance", "LDH normalization"])
    ).enrich(profile)
    assert [e.measure for e in live.endpoints] == ["Transfusion avoidance", "LDH normalization"]
    assert live.source == "clinicaltrials.gov"

    from src.services.clinical_context.clients import ClinicalTrialsError

    down = ClinicalTrialsEndpointProvider(client=_FakeCTGov(ClinicalTrialsError("503"))).enrich(
        profile
    )
    assert [e.measure for e in down.endpoints] == list(profile.pivotal_endpoints_fallback)
    # Curated fallback endpoints have no source trial.
    assert all(e.time_frame is None and e.nct_id is None for e in down.endpoints)
    assert down.source == "static_fallback"


@pytest.mark.unit
def test_ctgov_provider_empty_live_uses_fallback():
    profile = resolve_brand_profile("Remibrutinib")
    frag = ClinicalTrialsEndpointProvider(client=_FakeCTGov([])).enrich(profile)
    assert [e.measure for e in frag.endpoints] == list(profile.pivotal_endpoints_fallback)
    assert frag.source == "static_fallback"


@pytest.mark.unit
def test_pubmed_provider_prefers_search_then_seed_pmid():
    from src.services.clinical_context.clients import PubMedArticle

    profile = resolve_brand_profile("Kisqali")
    art = PubMedArticle(pmid="36097254", title="Live hit", journal="J", doi="10.1/y")
    frag = PubMedRWEProvider(client=_FakePubMed(art=art)).enrich(profile)
    assert frag.citation is not None
    assert frag.citation.pmid == "36097254"
    assert frag.source == "pubmed"


@pytest.mark.unit
def test_pubmed_provider_falls_back_to_seed_pmid_on_no_hits():
    from src.services.clinical_context.clients import PubMedArticle

    profile = resolve_brand_profile("Kisqali")  # rwe_seed_pmid = 35642282
    seed = PubMedArticle(pmid="35642282", title="Seed RWE", journal="J Oncol Pharm Pract")
    frag = PubMedRWEProvider(client=_FakePubMed(art=None, by_pmid=seed)).enrich(profile)
    assert frag.citation is not None and frag.citation.pmid == "35642282"
    assert frag.source == "pubmed_seed"


@pytest.mark.unit
def test_pubmed_provider_none_when_no_hit_and_no_seed():
    profile = resolve_brand_profile("Fabhalta")  # rwe_seed_pmid = None
    frag = PubMedRWEProvider(client=_FakePubMed(art=None, by_pmid=None)).enrich(profile)
    assert frag.citation is None
    assert frag.source == "unavailable"


@pytest.mark.unit
def test_ctgov_provider_drops_safety_endpoints_keeps_efficacy():
    """CT.gov primary outcomes often mix safety/AE measures with efficacy ones; the
    provider drops the safety measures and surfaces only the efficacy endpoints
    under the (now-accurate) 'pivotal endpoints' framing."""
    profile = resolve_brand_profile("Remibrutinib")
    frag = ClinicalTrialsEndpointProvider(
        client=_FakeCTGov(
            [
                "Number of Participants With Treatment-emergent Adverse Events (AEs)",
                "Mean Change From Baseline in Weekly Urticaria Activity Score (UAS7) at Week 12",
                "Safety and tolerability of remibrutinib",
            ]
        )
    ).enrich(profile)
    assert frag.source == "clinicaltrials.gov"
    assert [e.measure for e in frag.endpoints] == [
        "Mean Change From Baseline in Weekly Urticaria Activity Score (UAS7) at Week 12"
    ]


@pytest.mark.unit
def test_ctgov_provider_carries_time_frame_and_nct_through_fragment():
    """The provider must preserve each surviving endpoint's time_frame + nct_id (not
    just the measure), and drop safety endpoints by their .measure text."""
    profile = resolve_brand_profile("Remibrutinib")
    frag = ClinicalTrialsEndpointProvider(
        client=_FakeCTGov(
            [
                CTGovEndpoint(
                    "Number of Participants With Treatment-emergent Adverse Events (AEs)",
                    "Baseline up to 56 weeks",
                    "NCT05048342",
                ),
                CTGovEndpoint(
                    "Change From Baseline in Weekly Urticaria Score (UAS7) at Week 12",
                    "Baseline, Week 12",
                    "NCT05030311",
                ),
            ]
        )
    ).enrich(profile)
    assert frag.source == "clinicaltrials.gov"
    # Safety endpoint dropped by .measure; the efficacy one survives WITH provenance.
    assert [e.measure for e in frag.endpoints] == [
        "Change From Baseline in Weekly Urticaria Score (UAS7) at Week 12"
    ]
    assert frag.endpoints[0].time_frame == "Baseline, Week 12"
    assert frag.endpoints[0].nct_id == "NCT05030311"


@pytest.mark.unit
def test_ctgov_provider_all_safety_falls_back_to_curated_efficacy():
    """When CT.gov returns ONLY safety/PK measures, the provider prefers the curated
    efficacy fallback (the documented 'only safety endpoints' path) rather than
    surfacing safety measures as pivotal efficacy endpoints."""
    profile = resolve_brand_profile("Remibrutinib")
    frag = ClinicalTrialsEndpointProvider(
        client=_FakeCTGov(
            [
                "Number of Participants With Adverse Events",
                "Pharmacokinetics of remibrutinib",
            ]
        )
    ).enrich(profile)
    assert [e.measure for e in frag.endpoints] == list(profile.pivotal_endpoints_fallback)
    assert frag.source == "static_fallback"


# --- Task 3: OpenFDA indications + curated competitor providers ---


class _FakeOpenFDA:
    """Fake _OpenFDAClient: fetch_label returns a canned label (or raises); the
    extraction helpers return canned values regardless of the label payload."""

    def __init__(self, label=None, indications=None, lou=None, boxed=None, raise_on_fetch=None):
        self._label = label
        self._indications = indications if indications is not None else []
        self._lou = lou
        self._boxed = boxed
        self._raise = raise_on_fetch

    def fetch_label(self, drug_name):
        if self._raise is not None:
            raise self._raise
        return self._label

    def approved_indications(self, label):
        return list(self._indications)

    def limitations_of_use(self, label):
        return self._lou

    def boxed_warning(self, label):
        return self._boxed


@pytest.mark.unit
def test_openfda_indications_prefers_live_label():
    profile = resolve_brand_profile("Remibrutinib")
    frag = OpenFDAIndicationsProvider(
        client=_FakeOpenFDA(
            label={"openfda": {"generic_name": ["remibrutinib"]}},
            indications=["Chronic spontaneous urticaria (CSU) in adults"],
            lou="Not indicated for other forms of urticaria.",
            boxed=None,
        )
    ).enrich(profile)
    assert frag.source == "openfda"
    assert frag.approved_indications == ["Chronic spontaneous urticaria (CSU) in adults"]
    assert frag.limitations_of_use == "Not indicated for other forms of urticaria."
    assert frag.boxed_warning is None


@pytest.mark.unit
def test_openfda_indications_falls_back_on_error():
    from src.services.clinical_context.clients import OpenFDAError

    profile = resolve_brand_profile("Fabhalta")
    frag = OpenFDAIndicationsProvider(
        client=_FakeOpenFDA(raise_on_fetch=OpenFDAError("503"))
    ).enrich(profile)
    assert frag.source == "static_fallback"
    assert frag.approved_indications == list(profile.indications_fallback)
    # Fabhalta carries a curated boxed-warning fallback.
    assert frag.boxed_warning == profile.boxed_warning_fallback
    assert frag.boxed_warning


@pytest.mark.unit
def test_openfda_indications_falls_back_when_no_label():
    profile = resolve_brand_profile("Kisqali")
    frag = OpenFDAIndicationsProvider(client=_FakeOpenFDA(label=None)).enrich(profile)
    assert frag.source == "static_fallback"
    assert frag.approved_indications == list(profile.indications_fallback)


@pytest.mark.unit
def test_openfda_indications_falls_back_when_live_indications_empty():
    profile = resolve_brand_profile("Kisqali")
    frag = OpenFDAIndicationsProvider(client=_FakeOpenFDA(label={"x": 1}, indications=[])).enrich(
        profile
    )
    assert frag.source == "static_fallback"
    assert frag.approved_indications == list(profile.indications_fallback)


@pytest.mark.unit
@pytest.mark.parametrize(
    "brand,expected_member",
    [
        ("Kisqali", "Verzenio (abemaciclib)"),
        ("Fabhalta", "Soliris (eculizumab)"),
        ("Remibrutinib", "Xolair (omalizumab)"),
    ],
)
def test_curated_competitor_resolves_by_disease(brand, expected_member):
    profile = resolve_brand_profile(brand)
    frag = CuratedCompetitorProvider().enrich(profile)
    assert frag.source == "curated"
    assert frag.count == len(frag.competitors)
    assert frag.count > 0
    assert expected_member in frag.competitors


@pytest.mark.unit
def test_curated_competitor_unknown_disease_is_empty():
    from types import SimpleNamespace

    fake_profile = SimpleNamespace(disease="Nonexistent Disease", competitor_map={})
    frag = CuratedCompetitorProvider().enrich(fake_profile)  # type: ignore[arg-type]
    assert frag.competitors == []
    assert frag.count == 0
    assert frag.source == "curated"


# --- #1763: the RWE search follows the ANALYSIS, with an honest fallback ladder ---


class _TermAwarePubMed:
    """Returns a different article per search term (and records what was searched)."""

    def __init__(self, by_term=None, by_pmid=None):
        self._by_term = by_term or {}
        self._by_pmid = by_pmid
        self.terms = []

    def top_article(self, term):
        self.terms.append(term)
        return self._by_term.get(term)

    def fetch_by_pmid(self, pmid):
        return self._by_pmid


@pytest.mark.unit
def test_pubmed_provider_searches_the_analysis_term_when_present():
    from dataclasses import replace

    from src.services.clinical_context.clients import PubMedArticle

    profile = replace(
        resolve_brand_profile("Kisqali"),
        analysis_rwe_search_term="ribociclib breast cancer copay assistance persistence",
    )
    hit = PubMedArticle(pmid="111", title="Copay + persistence", journal="J")
    client = _TermAwarePubMed(by_term={profile.analysis_rwe_search_term: hit})
    frag = PubMedRWEProvider(client=client).enrich(profile)
    assert frag.citation is not None and frag.citation.pmid == "111"
    assert frag.source == "pubmed"
    # The payload discloses WHAT was searched, so an analyst can judge relevance.
    assert frag.search_term == profile.analysis_rwe_search_term
    assert client.terms == [profile.analysis_rwe_search_term]


@pytest.mark.unit
def test_pubmed_provider_falls_back_to_the_brand_term_and_says_so():
    """A narrow analysis-specific query can legitimately return nothing. Falling back
    to the brand-level query is right — but the source must NOT still claim the
    citation is analysis-specific."""
    from dataclasses import replace

    from src.services.clinical_context.clients import PubMedArticle

    base = resolve_brand_profile("Kisqali")
    profile = replace(base, analysis_rwe_search_term="ribociclib breast cancer no such thing")
    brand_hit = PubMedArticle(pmid="222", title="Brand-level RWE", journal="J")
    client = _TermAwarePubMed(by_term={base.rwe_search_term: brand_hit})
    frag = PubMedRWEProvider(client=client).enrich(profile)
    assert frag.citation is not None and frag.citation.pmid == "222"
    assert frag.source == "pubmed_brand"
    assert frag.search_term == base.rwe_search_term
    assert client.terms == [profile.analysis_rwe_search_term, base.rwe_search_term]


@pytest.mark.unit
def test_pubmed_provider_ladder_ends_at_the_seed_then_unavailable():
    from dataclasses import replace

    from src.services.clinical_context.clients import PubMedArticle

    base = resolve_brand_profile("Kisqali")  # rwe_seed_pmid = 35642282
    profile = replace(base, analysis_rwe_search_term="nothing matches this")
    seed = PubMedArticle(pmid="35642282", title="Seed RWE", journal="J")
    frag = PubMedRWEProvider(client=_TermAwarePubMed(by_pmid=seed)).enrich(profile)
    assert frag.source == "pubmed_seed"
    assert frag.search_term is None  # a curated seed was not found by searching

    fab = replace(resolve_brand_profile("Fabhalta"), analysis_rwe_search_term="nothing")
    empty = PubMedRWEProvider(client=_TermAwarePubMed()).enrich(fab)
    assert empty.citation is None and empty.source == "unavailable"


@pytest.mark.unit
def test_pubmed_provider_without_an_analysis_term_stays_brand_level_pubmed():
    """No analysis term (the brand-level view) => one search, labelled plain
    'pubmed' — the pre-#1763 behaviour, unchanged."""
    from src.services.clinical_context.clients import PubMedArticle

    profile = resolve_brand_profile("Kisqali")
    hit = PubMedArticle(pmid="333", title="Brand RWE", journal="J")
    client = _TermAwarePubMed(by_term={profile.rwe_search_term: hit})
    frag = PubMedRWEProvider(client=client).enrich(profile)
    assert frag.source == "pubmed"
    assert frag.search_term == profile.rwe_search_term
    assert client.terms == [profile.rwe_search_term]
