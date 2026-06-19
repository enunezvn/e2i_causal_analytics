"""ClinicalContextService — fan out the providers into one payload.

Caches the live provider fan-out per (brand, disease) (the live lookups do not
vary by outcome); the outcome -> real-endpoint mapping is applied per call from
the local brand_map. Always attaches the synthetic/real honesty label. Builds
default real REST clients; injectable for tests.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.services.clinical_context.brand_map import (
    BrandClinicalProfile,
    endpoint_mapping_for_outcome,
    resolve_brand_profile,
)
from src.services.clinical_context.clients import (
    ClinicalTrialsClient,
    PubMedClient,
)
from src.services.clinical_context.providers import (
    ChEMBLMechanismProvider,
    CitationFragment,
    ClinicalContextProvider,
    ClinicalTrialsEndpointProvider,
    EndpointsFragment,
    MechanismFragment,
    PubMedRWEProvider,
)

logger = logging.getLogger(__name__)

HONESTY_LABEL = (
    "Effect estimate = a SYNTHETIC patient cohort (gold-standard demo data). "
    "Clinical context below (mechanism of action, pivotal endpoints, real-world "
    "evidence) is REAL and cited from public biomedical sources."
)

_FragmentTuple = Tuple[MechanismFragment, EndpointsFragment, CitationFragment]

# A DEGRADED fan-out (any provider on a static_fallback / unavailable source, e.g.
# from a transient PubMed 429 or CT.gov timeout) is reused only briefly so the
# layer self-heals — the next request after this window re-attempts the live APIs
# instead of caching a transient failure for the whole process lifetime. A
# FULLY-LIVE result is cached indefinitely (biomedical facts change slowly).
_FRAGMENT_TTL_DEGRADED_S = 600.0

# Per-(brand,disease) cache of the assembled fragments + the monotonic time the
# entry was stored + whether it is fully live. Keyed by a tuple so two outcomes
# for one brand reuse the single fan-out. Bounded by the 3-brand universe.
_FRAGMENT_CACHE: Dict[Tuple[str, str], Tuple[_FragmentTuple, float, bool]] = {}


class ClinicalContextService:
    """Assemble a brand's clinical context from the providers."""

    def __init__(
        self,
        *,
        mechanism_provider: Optional[ClinicalContextProvider] = None,
        endpoints_provider: Optional[ClinicalContextProvider] = None,
        citation_provider: Optional[ClinicalContextProvider] = None,
    ) -> None:
        # Default real providers wire the public-REST clients; tests inject stubs.
        # Q1: _default_chembl() already returns a ChEMBLMechanismProvider — single-wrap.
        self._mechanism = mechanism_provider or _default_chembl()
        self._endpoints = endpoints_provider or ClinicalTrialsEndpointProvider(
            client=ClinicalTrialsClient()
        )
        self._citation = citation_provider or PubMedRWEProvider(client=PubMedClient())

    def _fan_out(self, profile: BrandClinicalProfile) -> _FragmentTuple:
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
        cite = self._citation.enrich(profile)
        assert isinstance(moa, MechanismFragment)
        assert isinstance(eps, EndpointsFragment)
        assert isinstance(cite, CitationFragment)
        fully_live = (
            moa.source == "chembl"
            and eps.source == "clinicaltrials.gov"
            and cite.source == "pubmed"
        )
        _FRAGMENT_CACHE[key] = ((moa, eps, cite), time.monotonic(), fully_live)
        return moa, eps, cite

    def get_context(self, brand: str, outcome: str) -> Dict[str, Any]:
        """Return the assembled clinical-context payload for (brand, outcome).

        Raises ``KeyError`` on an unknown brand (the endpoint maps it to 404).
        Never raises on an API failure — providers degrade to static fallbacks.
        """
        profile = resolve_brand_profile(brand)
        moa, eps, cite = self._fan_out(profile)
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
            }
        return {
            "brand": profile.brand,
            "drug_name": profile.drug_name,
            "disease": profile.disease,
            "our_outcome": outcome,
            "mapped_endpoint": endpoint_mapping_for_outcome(brand, outcome),
            "mechanism": {
                "mechanism_of_action": moa.mechanism_of_action,
                "source": moa.source,
            },
            "pivotal_endpoints": {
                "endpoints": list(eps.endpoints),
                "source": eps.source,
            },
            "real_world_evidence": citation_payload,
            "honesty_label": HONESTY_LABEL,
        }


def _default_chembl() -> ChEMBLMechanismProvider:
    """Build the default real ChEMBL provider (lazy import of the kg client keeps
    the import graph cheap and avoids a hard dependency at module import)."""
    from src.data.kg.chembl import ChEMBLClient

    return ChEMBLMechanismProvider(client=ChEMBLClient())


def reset_caches() -> None:
    """Clear the per-(brand,disease) fragment cache + the underlying REST client
    caches (useful in tests)."""
    _FRAGMENT_CACHE.clear()
    from src.data.kg.chembl import reset_caches as chembl_reset
    from src.services.clinical_context.clients import reset_caches as clients_reset

    chembl_reset()
    clients_reset()
