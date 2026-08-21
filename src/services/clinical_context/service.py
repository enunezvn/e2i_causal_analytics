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

from src.services.clinical_context.analysis_grounding import ground_analysis
from src.services.clinical_context.brand_map import (
    BrandClinicalProfile,
    TreatmentContext,
    analysis_framing_sentence,
    compose_rwe_search_term,
    endpoint_mapping_for_outcome,
    resolve_brand_profile,
    treatment_context_for,
)
from src.services.clinical_context.causal_evidence import (
    NOT_REQUESTED,
    CausalEvidenceFragment,
    CausalEvidenceProviderLike,
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
# FULLY-LIVE result is reused for the rest of the worker's life (biomedical facts
# change slowly); that is NOT unbounded — gunicorn runs with `--max-requests 1000
# --max-requests-jitter 50`, and gunicorn ADDS the jitter (`max_requests +
# randint(0, jitter)`, workers/base.py) rather than spreading it either side, so a
# worker recycles after 1000-1050 requests and takes these dicts with it.
_FRAGMENT_TTL_DEGRADED_S = 600.0

# THESE CACHES ARE PER-WORKER, AND THAT MAKES LATENCY HERE EASY TO MISREAD (#1768).
# The API runs `--workers 2`, and every cache below is a plain module-level dict, so
# each worker holds its own copy. Consequences, in order of how often they bite:
#
#   1. A "cold vs warm" latency figure taken against this endpoint really measures
#      WHICH WORKER ANSWERED, not whether the cache is warm. During #1763
#      certification a 21.4s call was quoted as a slow warm hit; it was a cold miss
#      on the other worker. Do not quote timings from this path without accounting
#      for that — issue enough identical requests to fill every worker first.
#   2. Two workers can in principle serve different answers for identical requests.
#      Measured on 2026-08-21 after #1767 landed: 96 identical requests over three
#      (brand, outcome, treatment) cases, on both warm and freshly-cold caches,
#      produced ZERO divergence. The cold-fill COUNT was the control — two per
#      case, read as "both workers were reached".
#
#      THAT READING HAS THREE PRECONDITIONS, all of which held for the recorded
#      run, and all of which a re-run must reproduce or the control is void:
#        - Requests must be SEQUENTIAL. None of these dicts is singleflighted, so
#          two CONCURRENT requests for one key can both miss before either stores,
#          and one worker would then produce two cold fills on its own.
#        - The run must stay inside one cache generation: shorter than
#          _FRAGMENT_TTL_DEGRADED_S if any fragment came back degraded, and well
#          short of the ~1000 requests that recycle a worker. Either boundary
#          crossed mid-run lets ONE worker legitimately cold-fill twice.
#        - Every request must have the same shape, `include_causal_evidence`
#          included, since that gates whether the evidence fan-out runs at all.
#      The remaining exit — separate dicts filling on different calls, which would
#      also give one worker two cold fills — is closed in-tree by
#      test_one_process_cold_fills_a_repeated_request_exactly_once, which pins
#      that every provider is exhausted by the FIRST call. Break that test and
#      this measurement stops meaning what it says.
#
#      A shared/Redis cache was considered and rejected on that measurement — it
#      would add serialization, a version-namespaced key and a cross-process
#      downgrade guard to a fail-open path, to fix something not currently
#      observable. If divergence ever does resurface, re-measure it the same way
#      before building it.

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

# Per-analysis cache of the public-KG evidence block. The key is (brand, curated
# treatment column, composed query) — NOT the raw outcome/treatment strings, which
# arrive from query params: keying on those would let any caller grow this dict
# without bound and re-hit Open Targets / PubMed / Europe PMC for each novel string.
# Every component is drawn from the curated maps, so the key space is bounded by the
# 3-brand universe. Several live calls back this fragment, so it is gathered only
# when asked for and reused for the self-heal window when it came back degraded.
_EVIDENCE_CACHE: Dict[Tuple[str, str, str], Tuple[CausalEvidenceFragment, float, bool]] = {}


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
        causal_evidence_provider: Optional[CausalEvidenceProviderLike] = None,
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
        # Built lazily on first use: constructing it opens the Open Targets / Europe
        # PMC / PubMed clients, and most calls (the whole leaderboard fan-out) never
        # ask for the evidence block.
        self._causal_evidence: Optional[CausalEvidenceProviderLike] = causal_evidence_provider

    def _fan_out(self, profile: BrandClinicalProfile) -> _BrandFragmentTuple:
        """The brand-level fragments — none of these vary by the analysis."""
        key = (profile.brand, profile.disease)
        cached = _FRAGMENT_CACHE.get(key)
        if cached is not None:
            frags, stored_at, fully_live = cached
            # Reuse a fully-live result for the worker's life; reuse a degraded one only
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
        # live-API providers gate the fully-live (reuse-for-worker-life) decision.
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
        # Only an answer to the query we asked is settled. `pubmed_brand` means the
        # ANALYSIS query returned nothing — and the provider cannot tell a genuine
        # zero-hit from a swallowed 429 or timeout, so caching it forever under the
        # analysis key would freeze a transient failure for the process lifetime.
        # It self-heals through the degraded window instead. Seed / unavailable
        # likewise.
        fully_live = cite.source == "pubmed"
        _CITATION_CACHE[key] = (cite, time.monotonic(), fully_live)
        return cite

    def _evidence_provider(self) -> CausalEvidenceProviderLike:
        if self._causal_evidence is None:
            from src.services.clinical_context.causal_evidence import (
                default_causal_evidence_provider,
            )

            self._causal_evidence = default_causal_evidence_provider()
        return self._causal_evidence

    def _causal_evidence_for(
        self,
        profile: BrandClinicalProfile,
        outcome: str,
        treatment: str,
        treatment_ctx: Optional[TreatmentContext],
        search_term: str,
    ) -> CausalEvidenceFragment:
        """The public-KG evidence for ONE analysis, cached per (brand, outcome,
        treatment). FAIL-OPEN: any failure degrades to an honest ``unavailable``
        fragment rather than taking the whole payload down."""
        # An uncurated treatment resolves to an immediate honest "unavailable" with
        # no live call, so it is neither worth caching nor safe to key on.
        key = (
            (profile.brand, treatment_ctx.column, search_term)
            if treatment_ctx is not None
            else None
        )
        if key is not None:
            cached = _EVIDENCE_CACHE.get(key)
            if cached is not None:
                frag, stored_at, complete = cached
                if complete or (time.monotonic() - stored_at) < _FRAGMENT_TTL_DEGRADED_S:
                    return frag
        try:
            evidence = self._evidence_provider().evidence(
                profile,
                outcome=outcome,
                treatment_context=treatment_ctx,
                search_term=search_term,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; never fails the payload
            logger.warning(
                "clinical-context: causal evidence unavailable for %s/%s: %s",
                profile.brand,
                treatment,
                exc,
            )
            evidence = CausalEvidenceFragment(
                status="unavailable",
                note="The public evidence sources could not be reached for this analysis.",
            )
        # Worth retrying: an outright "unavailable", and any fragment where a source
        # was asked and failed (its absence is unknown, not settled). A commercial
        # lever has nothing to re-fetch and a fully-answered result is stable.
        if key is not None:
            complete = (
                evidence.status != "unavailable"
                and not evidence.sources_unavailable
                # We stopped early under our own budget: unfinished, so not settled.
                # Kept apart from sources_unavailable so a healthy Europe PMC is
                # never named as the reason (codex iter-1 HIGH, #1767).
                and not evidence.checks_incomplete
            )
            # Two requests can miss the cache together; the slower one must not
            # replace a complete fragment with the degraded one it happened to get
            # from a transient upstream failure.
            existing = _EVIDENCE_CACHE.get(key)
            if not (existing is not None and existing[2] and not complete):
                _EVIDENCE_CACHE[key] = (evidence, time.monotonic(), complete)
        return evidence

    def get_context(
        self,
        brand: str,
        outcome: str,
        treatment: Optional[str] = None,
        *,
        include_causal_evidence: bool = False,
    ) -> Dict[str, Any]:
        """Return the assembled clinical-context payload for one analysis.

        ``treatment`` is optional: with it, the payload frames the specific
        (treatment -> outcome) analysis and the literature search follows that
        analysis; without it (the brand-level view) the analysis frame is omitted
        entirely rather than guessed.

        ``include_causal_evidence`` gates the public-KG evidence block (Open Targets
        indication edge + abstract-verified literature). It is several live calls per
        analysis, so the leaderboard fan-out — which renders none of it — leaves it
        off and the payload says ``not_requested`` rather than looking unavailable.

        Raises ``KeyError`` on an unknown brand (the endpoint maps it to 404).
        Never raises on an API failure — providers degrade to static fallbacks.
        """
        profile = resolve_brand_profile(brand)
        moa, eps, indications, competitors = self._fan_out(profile)
        search_term = compose_rwe_search_term(profile, outcome, treatment)
        cite = self._citation_for(profile, search_term)
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
        # #1775: ground the SCENARIO. The label considerations ride along on the
        # brand-level indications fragment (already cached), and the selection by
        # outcome costs no I/O. Commercial levers are grounded like any other
        # analysis — declining to claim the label speaks to a lever is right,
        # declining to ground the analysis at all was not.
        grounding = ground_analysis(
            profile,
            outcome=outcome,
            treatment_context=treatment_context_for(profile.brand, treatment),
            label_considerations=indications.label_considerations,
            # Provenance, not decoration: an empty list means "the label was read and
            # carries none" under openfda and "we could not read the label" under the
            # curated fallback. Those are different claims (#1767).
            label_source=indications.source,
        )
        grounding_payload: Optional[Dict[str, Any]] = None
        # `treatment is not None` is necessary but not sufficient. An uncurated
        # treatment yields a grounding with no considerations, no competitive context
        # and no note, and we were shipping that as an EMPTY OBJECT while the schema
        # and the TS type both document `null` for "no scenario to ground". The panel
        # happened not to render it, so nothing user-visible was wrong — but a wire
        # contract that disagrees with its own documentation is the next defect
        # waiting for a consumer who trusts the docs (codex iter-12 LOW).
        has_grounding = bool(
            grounding.label_considerations or grounding.competitive_context or grounding.note
        )
        if treatment is not None and has_grounding:
            grounding_payload = {
                "label_considerations": [
                    {
                        "title": c.title,
                        "detail": c.detail,
                        "section": c.section,
                        "references": c.references,
                        "source": c.source,
                    }
                    for c in grounding.label_considerations
                ],
                "competitive_context": grounding.competitive_context,
                "note": grounding.note,
                "outcome_theme": grounding.outcome_theme,
            }
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
        # Public-KG evidence for THIS analysis. Absent entirely without a treatment
        # (there is no analysis to gather evidence for).
        evidence_payload: Optional[Dict[str, Any]] = None
        if treatment:
            evidence = (
                self._causal_evidence_for(profile, outcome, treatment, treatment_ctx, search_term)
                if include_causal_evidence
                else NOT_REQUESTED
            )
            evidence_payload = {
                "status": evidence.status,
                "indication_edge": (
                    {
                        "predicate": evidence.indication_edge.predicate,
                        "drug_id": evidence.indication_edge.drug_id,
                        "drug_name": evidence.indication_edge.drug_name,
                        "disease_id": evidence.indication_edge.disease_id,
                        "disease_name": evidence.indication_edge.disease_name,
                        "max_clinical_stage": evidence.indication_edge.max_clinical_stage,
                        "source": evidence.indication_edge.source,
                    }
                    if evidence.indication_edge is not None
                    else None
                ),
                "sources_unavailable": list(evidence.sources_unavailable),
                "citations": [
                    {
                        "pmid": c.pmid,
                        "title": c.title,
                        "journal": c.journal,
                        "pubdate": c.pubdate,
                        "url": c.url,
                        "entities_found": list(c.entities_found),
                        "confidence": c.confidence,
                        "source": c.source,
                    }
                    for c in evidence.citations
                ],
                "note": evidence.note,
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
            "analysis_grounding": grounding_payload,
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
            "causal_evidence": evidence_payload,
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
    _EVIDENCE_CACHE.clear()
    from src.data.kg.chembl import reset_caches as chembl_reset
    from src.services.clinical_context.clients import reset_caches as clients_reset

    chembl_reset()
    clients_reset()
