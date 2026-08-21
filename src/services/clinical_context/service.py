"""ClinicalContextService — fan out the providers into one payload.

Caches the BRAND-level live provider fan-out (mechanism / endpoints / label /
competitors) per (brand, disease) — none of it varies by the analysis. The
literature citation DOES vary by the analysis (#1763), so it is cached separately
per (brand, composed search term). The outcome -> real-endpoint mapping and the
treatment framing are applied per call from the local brand_map. Always attaches
the synthetic/real honesty label. Builds default real REST clients; injectable
for tests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from src.services.clinical_context.brand_map import (
    BrandClinicalProfile,
    analysis_framing_sentence,
    compose_rwe_search_term,
    endpoint_mapping_for_outcome,
    resolve_brand_profile,
    treatment_context_for,
)
from src.services.clinical_context.clients import (
    ClinicalTrialsClient,
    PubMedClient,
    _OpenFDAClient,
)
from src.services.clinical_context.providers import (
    ChEMBLMechanismProvider,
    CitationFragment,
    ClinicalContextProvider,
    ClinicalTrialsEndpointProvider,
    CompetitorFragment,
    CuratedCompetitorProvider,
    EndpointsFragment,
    IndicationsFragment,
    MechanismFragment,
    OpenFDAIndicationsProvider,
    PubMedRWEProvider,
)

logger = logging.getLogger(__name__)

HONESTY_LABEL = (
    "Effect estimate = a SYNTHETIC patient cohort (gold-standard demo data). "
    "Clinical context below (mechanism of action, pivotal endpoints, FDA-label "
    "indications, real-world evidence) is REAL from public biomedical/regulatory "
    "sources; the competitor landscape is a curated reference. Each item is "
    "labelled with its source."
)

_BrandFragmentTuple = Tuple[
    MechanismFragment,
    EndpointsFragment,
    IndicationsFragment,
    CompetitorFragment,
]

# A DEGRADED result (any provider on a static_fallback / unavailable source, e.g.
# from a transient PubMed 429 or CT.gov timeout) is reused only briefly so the
# layer self-heals — the next request after this window re-attempts the live APIs
# instead of caching a transient failure for the whole process lifetime. A
# FULLY-LIVE result is cached indefinitely (biomedical facts change slowly).
_FRAGMENT_TTL_DEGRADED_S = 600.0

# Per-(brand,disease) cache of the BRAND-level fragments + the monotonic time the
# entry was stored + whether it is fully live. Keyed by a tuple so every analysis
# of one brand reuses the single fan-out. Bounded by the 3-brand universe.
_FRAGMENT_CACHE: Dict[Tuple[str, str], Tuple[_BrandFragmentTuple, float, bool]] = {}

# Per-(brand, search term) cache of the literature citation. Split out of the
# brand-level cache because the citation is the ONE fragment that must follow the
# analysis (#1763): caching it per brand is exactly what made the panel show a
# citation unrelated to the treatment -> outcome pair being interrogated. Bounded
# by 3 brands x the curated (outcome, treatment) universe.
_CITATION_CACHE: Dict[Tuple[str, str], Tuple[CitationFragment, float, bool]] = {}


class ClinicalContextService:
    """Assemble a brand's clinical context from the providers."""

    def __init__(
        self,
        *,
        mechanism_provider: Optional[ClinicalContextProvider] = None,
        endpoints_provider: Optional[ClinicalContextProvider] = None,
        citation_provider: Optional[ClinicalContextProvider] = None,
        indications_provider: Optional[ClinicalContextProvider] = None,
        competitor_provider: Optional[ClinicalContextProvider] = None,
    ) -> None:
        # Default real providers wire the public-REST clients; tests inject stubs.
        # Q1: _default_chembl() already returns a ChEMBLMechanismProvider — single-wrap.
        self._mechanism = mechanism_provider or _default_chembl()
        self._endpoints = endpoints_provider or ClinicalTrialsEndpointProvider(
            client=ClinicalTrialsClient()
        )
        self._citation = citation_provider or PubMedRWEProvider(client=PubMedClient())
        self._indications = indications_provider or OpenFDAIndicationsProvider(
            client=_OpenFDAClient()
        )
        self._competitor = competitor_provider or CuratedCompetitorProvider()

    def _fan_out(self, profile: BrandClinicalProfile) -> _BrandFragmentTuple:
        """The brand-level fragments — none of these vary by the analysis."""
        key = (profile.brand, profile.disease)
        cached = _FRAGMENT_CACHE.get(key)
        if cached is not None:
            frags, stored_at, fully_live = cached
            # Reuse a fully-live result indefinitely; reuse a degraded result only
            # within the self-heal window, else fall through and retry the live APIs.
            if fully_live or (time.monotonic() - stored_at) < _FRAGMENT_TTL_DEGRADED_S:
                return frags
        moa = self._mechanism.enrich(profile)
        eps = self._endpoints.enrich(profile)
        indications = self._indications.enrich(profile)
        competitors = self._competitor.enrich(profile)
        assert isinstance(moa, MechanismFragment)
        assert isinstance(eps, EndpointsFragment)
        assert isinstance(indications, IndicationsFragment)
        assert isinstance(competitors, CompetitorFragment)
        # Competitors are curated by design (the chosen SSOT), so "curated" is the
        # intended live state — it does NOT make the result degraded. Only the
        # live-API providers gate the fully-live (cache-indefinitely) decision.
        fully_live = (
            moa.source == "chembl"
            and eps.source == "clinicaltrials.gov"
            and indications.source == "openfda"
        )
        _FRAGMENT_CACHE[key] = (
            (moa, eps, indications, competitors),
            time.monotonic(),
            fully_live,
        )
        return moa, eps, indications, competitors

    def _citation_for(self, profile: BrandClinicalProfile, search_term: str) -> CitationFragment:
        """The literature citation for ONE analysis, cached per (brand, query).

        The provider receives a copy of the profile carrying the analysis-specific
        query, so the provider contract (``enrich(profile)``) is unchanged. When the
        composed query IS the curated brand query (nothing analysis-specific to
        compose), no analysis term is attached and the provider behaves exactly as
        it did pre-#1763 — one search, labelled plain ``pubmed``.
        """
        key = (profile.brand, search_term)
        cached = _CITATION_CACHE.get(key)
        if cached is not None:
            frag, stored_at, fully_live = cached
            if fully_live or (time.monotonic() - stored_at) < _FRAGMENT_TTL_DEGRADED_S:
                return frag
        analysis_profile = profile
        if search_term != profile.rwe_search_term:
            analysis_profile = replace(profile, analysis_rwe_search_term=search_term)
        cite = self._citation.enrich(analysis_profile)
        assert isinstance(cite, CitationFragment)
        # A live PubMed answer (analysis-level or brand-level) is a real result and is
        # cached; a seed / unavailable citation is degraded and self-heals.
        fully_live = cite.source in ("pubmed", "pubmed_brand")
        _CITATION_CACHE[key] = (cite, time.monotonic(), fully_live)
        return cite

    def get_context(
        self, brand: str, outcome: str, treatment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return the assembled clinical-context payload for one analysis.

        ``treatment`` is optional: with it, the payload frames the specific
        (treatment -> outcome) analysis and the literature search follows that
        analysis; without it (the brand-level view) the analysis frame is omitted
        entirely rather than guessed.

        Raises ``KeyError`` on an unknown brand (the endpoint maps it to 404).
        Never raises on an API failure — providers degrade to static fallbacks.
        """
        profile = resolve_brand_profile(brand)
        moa, eps, indications, competitors = self._fan_out(profile)
        cite = self._citation_for(profile, compose_rwe_search_term(profile, outcome, treatment))
        citation_payload: Optional[Dict[str, Any]] = None
        if cite.citation is not None:
            citation_payload = {
                "pmid": cite.citation.pmid,
                "title": cite.citation.title,
                "journal": cite.citation.journal,
                "pubdate": cite.citation.pubdate,
                "doi": cite.citation.doi,
                "url": cite.citation.url,
                "source": cite.source,
                "search_term": cite.search_term,
            }
        # Curated brand-SPECIFIC seminal RWE (from the brand map). Deterministic and
        # always present for brands that have one, so the brand of interest gets a
        # brand-faithful reference regardless of what the live relevance search above
        # returned. The URL is built from the PMID; source is honestly "curated".
        seminal_payload: Optional[Dict[str, Any]] = None
        if profile.seminal_rwe:
            s = profile.seminal_rwe
            pmid = s.get("pmid")
            seminal_payload = {
                "pmid": pmid,
                "title": s.get("title"),
                "journal": s.get("journal"),
                "pubdate": s.get("year"),
                "doi": s.get("doi"),
                "url": (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None),
                "source": "curated",
                # Curated, not found by searching — it must not claim a query.
                "search_term": None,
            }
        # The treatment side of the analysis (#1763). None when no treatment was
        # supplied or the column has no curated framing — an unframed analysis is
        # reported honestly rather than described with an invented treatment.
        treatment_ctx = treatment_context_for(brand, treatment)
        treatment_payload: Optional[Dict[str, Any]] = None
        if treatment_ctx is not None:
            treatment_payload = {
                "column": treatment_ctx.column,
                "label": treatment_ctx.label,
                "framing": treatment_ctx.framing,
                "kind": treatment_ctx.kind,
                "source": "curated",
            }
        return {
            "brand": profile.brand,
            "drug_name": profile.drug_name,
            "disease": profile.disease,
            "our_outcome": outcome,
            "our_treatment": treatment,
            "mapped_endpoint": endpoint_mapping_for_outcome(brand, outcome),
            "treatment_context": treatment_payload,
            "analysis_framing": analysis_framing_sentence(profile, outcome, treatment),
            "mechanism": {
                "mechanism_of_action": moa.mechanism_of_action,
                "source": moa.source,
            },
            "pivotal_endpoints": {
                "endpoints": [
                    {"measure": e.measure, "time_frame": e.time_frame, "nct_id": e.nct_id}
                    for e in eps.endpoints
                ],
                "source": eps.source,
            },
            "real_world_evidence": citation_payload,
            "seminal_real_world_evidence": seminal_payload,
            "approved_indications": {
                "indications": list(indications.approved_indications),
                "limitations_of_use": indications.limitations_of_use,
                "boxed_warning": indications.boxed_warning,
                "source": indications.source,
            },
            "competitor_landscape": {
                "competitors": list(competitors.competitors),
                "count": competitors.count,
                "source": competitors.source,
            },
            "honesty_label": HONESTY_LABEL,
        }


def _default_chembl() -> ChEMBLMechanismProvider:
    """Build the default real ChEMBL provider (lazy import of the kg client keeps
    the import graph cheap and avoids a hard dependency at module import)."""
    from src.data.kg.chembl import ChEMBLClient

    return ChEMBLMechanismProvider(client=ChEMBLClient())


def reset_caches() -> None:
    """Clear the brand-level fragment cache + the per-analysis citation cache + the
    underlying REST client caches (useful in tests)."""
    _FRAGMENT_CACHE.clear()
    _CITATION_CACHE.clear()
    from src.data.kg.chembl import reset_caches as chembl_reset
    from src.services.clinical_context.clients import reset_caches as clients_reset

    chembl_reset()
    clients_reset()
