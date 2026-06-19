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
from typing import List, Optional, Protocol

from src.services.clinical_context.brand_map import BrandClinicalProfile
from src.services.clinical_context.clients import (
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
    endpoints: List[str] = field(default_factory=list)
    source: str = "static_fallback"  # "clinicaltrials.gov" | "static_fallback"


@dataclass(frozen=True)
class CitationFragment:
    citation: Optional[PubMedArticle]
    source: str  # "pubmed" | "pubmed_seed" | "unavailable"


# --- Minimal structural protocols so tests can inject fakes --------------------


class _ChEMBLLike(Protocol):
    def mechanism_of_action(self, drug_name: str) -> Optional[str]: ...


class _CTGovLike(Protocol):
    def primary_endpoints(
        self, intervention: str, condition: str, *, limit: int = 8
    ) -> List[str]: ...


class _PubMedLike(Protocol):
    def top_article(self, term: str) -> Optional[PubMedArticle]: ...

    def fetch_by_pmid(self, pmid: str) -> Optional[PubMedArticle]: ...


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
        efficacy = [e for e in endpoints if not _is_safety_endpoint(e)]
        if efficacy:
            return EndpointsFragment(endpoints=efficacy, source="clinicaltrials.gov")
        return EndpointsFragment(
            endpoints=list(profile.pivotal_endpoints_fallback), source="static_fallback"
        )


class PubMedRWEProvider(ClinicalContextProvider):
    """Real-world-evidence citation via PubMed: relevance search, then the
    curated seed PMID, then unavailable (honest — never a fabricated citation)."""

    provider_name = "pubmed_rwe"

    def __init__(self, client: _PubMedLike) -> None:
        self._client = client

    def enrich(self, profile: BrandClinicalProfile) -> CitationFragment:
        try:
            article = self._client.top_article(profile.rwe_search_term)
        except Exception as exc:  # noqa: BLE001 — best-effort; any failure => try seed
            logger.warning(
                "clinical-context: PubMed search failed for %r: %s",
                profile.rwe_search_term,
                exc,
            )
            article = None
        if article is not None:
            return CitationFragment(citation=article, source="pubmed")
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
