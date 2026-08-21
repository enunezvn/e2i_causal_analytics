"""Clinical-context providers — the extensible enrichment core.

A ``ClinicalContextProvider`` enriches a ``BrandClinicalProfile`` from ONE
source, best-effort, with a static fallback and an honest source label. The
service fans out across providers and assembles the payload.

Adding a source (the DEFERRED openFDA / UMLS work) = add a subclass here; the
service and endpoint need no change. NO openFDA/UMLS code lives here yet.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

from src.services.clinical_context.brand_map import BrandClinicalProfile
from src.services.clinical_context.clients import (
    CTGovEndpoint,
    PubMedArticle,
)

# Best-effort contract: the providers' broad `except Exception` deliberately also
# swallows ClinicalTrialsError / PubMedError from clients.py (degrade to fallback).

logger = logging.getLogger(__name__)


# --- Typed fragments each provider returns -------------------------------------


@dataclass(frozen=True)
class MechanismFragment:
    mechanism_of_action: str
    source: str  # "chembl" | "static_fallback"


@dataclass(frozen=True)
class EndpointsFragment:
    endpoints: List[CTGovEndpoint] = field(default_factory=list)
    source: str = "static_fallback"  # "clinicaltrials.gov" | "static_fallback"


@dataclass(frozen=True)
class CitationFragment:
    citation: Optional[PubMedArticle]
    # "pubmed" (the analysis-specific search, or the brand search on the brand-level
    # view) | "pubmed_brand" (the analysis-specific search found nothing and the
    # brand-level search answered instead) | "pubmed_seed" | "unavailable"
    source: str
    # The query that produced this citation, so the panel can disclose WHAT was
    # searched. None for a curated seed (it was not found by searching).
    search_term: Optional[str] = None


@dataclass(frozen=True)
class IndicationsFragment:
    approved_indications: List[str] = field(default_factory=list)
    limitations_of_use: Optional[str] = None
    boxed_warning: Optional[str] = None
    source: str = "static_fallback"  # "openfda" | "static_fallback"


@dataclass(frozen=True)
class CompetitorFragment:
    competitors: List[str] = field(default_factory=list)
    count: int = 0
    # Always "curated" — the chosen SSOT. OpenFDA/ATC auto-derivation was disproved
    # as clinically misleading for our brands (e.g. a urticaria drug landing in a
    # broad transplant-immunosuppressant ATC bucket), so competitors are curated.
    source: str = "curated"


# --- Minimal structural protocols so tests can inject fakes --------------------


class _ChEMBLLike(Protocol):
    def mechanism_of_action(self, drug_name: str) -> Optional[str]: ...


class _CTGovLike(Protocol):
    def primary_endpoints(
        self, intervention: str, condition: str, *, limit: int = 8
    ) -> List[CTGovEndpoint]: ...


class _PubMedLike(Protocol):
    def top_article(self, term: str) -> Optional[PubMedArticle]: ...

    def fetch_by_pmid(self, pmid: str) -> Optional[PubMedArticle]: ...


class _OpenFDALike(Protocol):
    def fetch_label(self, drug_name: str) -> Optional[dict[str, Any]]: ...

    def approved_indications(self, label: dict[str, Any]) -> List[str]: ...

    def limitations_of_use(self, label: dict[str, Any]) -> Optional[str]: ...

    def boxed_warning(self, label: dict[str, Any]) -> Optional[str]: ...


# --- The provider interface ----------------------------------------------------


class ClinicalContextProvider(ABC):
    """Enriches a brand's clinical profile from one source, best-effort."""

    provider_name: str = "provider"

    @abstractmethod
    def enrich(self, profile: BrandClinicalProfile) -> object:
        """Return this provider's typed fragment. MUST NOT raise on an API
        failure — degrade to the static fallback and label the source."""
        raise NotImplementedError


class ChEMBLMechanismProvider(ClinicalContextProvider):
    """Drug -> mechanism of action via ChEMBL, with the static MoA fallback."""

    provider_name = "chembl_mechanism"

    def __init__(self, client: _ChEMBLLike) -> None:
        self._client = client

    def enrich(self, profile: BrandClinicalProfile) -> MechanismFragment:
        try:
            moa = self._client.mechanism_of_action(profile.drug_name)
        except Exception as exc:  # noqa: BLE001 — best-effort; any failure => fallback
            logger.warning(
                "clinical-context: ChEMBL MoA lookup failed for %s: %s", profile.drug_name, exc
            )
            moa = None
        if moa:
            return MechanismFragment(mechanism_of_action=moa, source="chembl")
        return MechanismFragment(mechanism_of_action=profile.moa_fallback, source="static_fallback")


# A registered CT.gov "primary outcome" is frequently a safety / tolerability /
# PK measure, not the pivotal EFFICACY endpoint a clinician means by "pivotal
# endpoint". Drop the obvious safety measures so they are not surfaced under the
# "pivotal endpoints" framing; when nothing efficacy-like remains, the provider
# prefers the curated efficacy fallback.
_SAFETY_ENDPOINT_PATTERNS = (
    "adverse event",
    "treatment-emergent",
    "treatment emergent",
    "serious adverse",
    "safety",
    "tolerability",
    "pharmacokinetic",
    "pharmacodynamic",
)


def _is_safety_endpoint(measure: str) -> bool:
    """True if a CT.gov outcome ``measure`` is an obvious safety / tolerability /
    PK measure rather than an efficacy endpoint."""
    m = measure.lower()
    return any(pattern in m for pattern in _SAFETY_ENDPOINT_PATTERNS)


class ClinicalTrialsEndpointProvider(ClinicalContextProvider):
    """Disease -> real pivotal endpoints via ClinicalTrials.gov, with the static
    endpoint fallback. CT.gov primary outcomes often include safety / PK measures,
    so this provider drops the obvious safety endpoints and prefers the curated
    efficacy fallback when the live result has no efficacy endpoint left."""

    provider_name = "clinicaltrials_endpoints"

    def __init__(self, client: _CTGovLike) -> None:
        self._client = client

    def enrich(self, profile: BrandClinicalProfile) -> EndpointsFragment:
        try:
            endpoints = self._client.primary_endpoints(profile.drug_name, profile.disease)
        except Exception as exc:  # noqa: BLE001 — best-effort; any failure => fallback
            logger.warning(
                "clinical-context: ClinicalTrials lookup failed for %s/%s: %s",
                profile.drug_name,
                profile.disease,
                exc,
            )
            endpoints = []
        # Keep only efficacy endpoints; a live result that is all-safety degrades to
        # the curated efficacy fallback (the documented "only safety endpoints" path).
        efficacy = [e for e in endpoints if not _is_safety_endpoint(e.measure)]
        if efficacy:
            return EndpointsFragment(endpoints=efficacy, source="clinicaltrials.gov")
        # Curated fallback strings have no source trial, so time_frame / nct_id are None.
        return EndpointsFragment(
            endpoints=[CTGovEndpoint(measure=m) for m in profile.pivotal_endpoints_fallback],
            source="static_fallback",
        )


class PubMedRWEProvider(ClinicalContextProvider):
    """Real-world-evidence citation via PubMed, down an honest ladder:

    1. the ANALYSIS-specific query (drug + disease + this analysis's outcome and
       treatment themes) when the service composed one (#1763);
    2. the curated BRAND-level query — labelled ``pubmed_brand`` so the panel never
       claims a brand-level paper is about this particular analysis;
    3. the curated seed PMID;
    4. unavailable (honest — never a fabricated citation).

    MEASURED 2026-08-21 against live PubMed: 6/10 analysis-composed queries returned
    a real analysis-relevant article (e.g. a ribociclib patient-access-programme
    study for psp_enrolled, an iptacopan C5-switch study for
    complement_inhibitor_status); the other 4 fell through this ladder as designed.
    """

    provider_name = "pubmed_rwe"

    def __init__(self, client: _PubMedLike) -> None:
        self._client = client

    def _search(self, term: str) -> Optional[PubMedArticle]:
        """One best-effort relevance search; any failure reads as no hit."""
        try:
            return self._client.top_article(term)
        except Exception as exc:  # noqa: BLE001 — best-effort; any failure => next rung
            logger.warning("clinical-context: PubMed search failed for %r: %s", term, exc)
            return None

    def enrich(self, profile: BrandClinicalProfile) -> CitationFragment:
        analysis_term = profile.analysis_rwe_search_term
        if analysis_term:
            article = self._search(analysis_term)
            if article is not None:
                return CitationFragment(
                    citation=article, source="pubmed", search_term=analysis_term
                )
        article = self._search(profile.rwe_search_term)
        if article is not None:
            # A brand-level answer to an analysis-specific question is still useful,
            # but it must be labelled as brand-level, not passed off as analysis-level.
            source = "pubmed_brand" if analysis_term else "pubmed"
            return CitationFragment(
                citation=article, source=source, search_term=profile.rwe_search_term
            )
        if profile.rwe_seed_pmid:
            try:
                seed = self._client.fetch_by_pmid(profile.rwe_seed_pmid)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "clinical-context: PubMed seed fetch failed for %s: %s",
                    profile.rwe_seed_pmid,
                    exc,
                )
                seed = None
            if seed is not None:
                return CitationFragment(citation=seed, source="pubmed_seed")
        return CitationFragment(citation=None, source="unavailable")


class OpenFDAIndicationsProvider(ClinicalContextProvider):
    """Drug -> FDA-label approved indications + limitations of use + boxed warning
    via OpenFDA, with the curated static fallback. Real-first: when a live label is
    found WITH indications, the live indications/LoU/boxed-warning are authoritative
    (even if LoU/boxed are absent for that drug). Any failure / no label / no live
    indication degrades to the curated brand-map fallback, honestly labelled."""

    provider_name = "openfda_indications"

    def __init__(self, client: _OpenFDALike) -> None:
        self._client = client

    def enrich(self, profile: BrandClinicalProfile) -> IndicationsFragment:
        try:
            label = self._client.fetch_label(profile.drug_name)
        except Exception as exc:  # noqa: BLE001 — best-effort; any failure => fallback
            logger.warning(
                "clinical-context: OpenFDA label lookup failed for %s: %s",
                profile.drug_name,
                exc,
            )
            label = None
        if label:
            indications = self._client.approved_indications(label)
            if indications:
                return IndicationsFragment(
                    approved_indications=indications,
                    limitations_of_use=self._client.limitations_of_use(label),
                    boxed_warning=self._client.boxed_warning(label),
                    source="openfda",
                )
        return IndicationsFragment(
            approved_indications=list(profile.indications_fallback),
            limitations_of_use=profile.limitations_fallback,
            boxed_warning=profile.boxed_warning_fallback,
            source="static_fallback",
        )


class CuratedCompetitorProvider(ClinicalContextProvider):
    """Therapeutic competitors from the curated, evidence-grounded brand map, keyed
    by the brand's disease. ``source`` is always ``"curated"`` — the chosen single
    source of truth (OpenFDA/ATC auto-derivation was disproved as clinically
    misleading for our brands), never fabricated and never auto-derived. An unknown
    disease yields an empty, honest result (count 0)."""

    provider_name = "curated_competitor"

    def enrich(self, profile: BrandClinicalProfile) -> CompetitorFragment:
        competitors = list(profile.competitor_map.get(profile.disease.lower(), []))
        return CompetitorFragment(competitors=competitors, count=len(competitors), source="curated")
