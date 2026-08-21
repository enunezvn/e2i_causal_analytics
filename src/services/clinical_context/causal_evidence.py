"""Causal-evidence provider (#1763 Phase 2) — public-KG evidence for ONE analysis.

The clinical-context panel used to answer "what is this drug"; this provider
answers the question the analyst actually has on screen: *what does public
evidence say about this treatment -> outcome analysis?* It reuses the existing
knowledge-graph stack rather than adding a new source:

- ``src.data.kg.open_targets`` — the drug -> indication edge, with the clinical
  stage Open Targets records for THAT indication node.
- ``src.services.clinical_context.clients.PubMedClient`` — the analysis-composed
  relevance search (the same query the RWE citation uses).
- ``src.data.kg.citation_resolver.CitationResolver`` — verification: a citation is
  surfaced only when the abstract actually names both entities.

Three honesty rules are load-bearing here:

1. **The indication stage is read off the matched disease NODE, never off the
   drug.** Open Targets reports ``maximumClinicalStage: APPROVAL`` for ribociclib
   (it is approved for something) while its breast-cancer node is ``PHASE_3``.
   Reading the drug-wide stage would assert an approval Open Targets never made
   for this indication.
2. **Open Targets staging lags the FDA label**, so a sub-APPROVAL edge carries a
   note saying the label section is the approval authority. The edge is a
   development-stage signal, not an approval statement.
3. **A commercial lever gets no clinical evidence at all.** copay support, PSP
   enrolment, detailing, sampling and NBA triggers have no biomedical literature
   about the lever; attaching the drug's evidence under an "evidence for this
   analysis" heading would be exactly the confusion #1763 is about.
4. **Only a drug-therapy treatment gets the drug -> indication edge.** When the
   treatment is a patient-state contrast (advanced-line disease, UAS7 severity)
   the therapy's indication says nothing about the contrast under study, so the
   edge is not fetched at all — the literature (retrieved with the covariate's own
   theme in the query) stands alone, with a note saying what was verified.

MEASURED live 2026-08-21 (all three brands): every brand's indication node
matched by EXACT id; ribociclib/breast cancer came back PHASE_3 under a drug-wide
APPROVAL (rule 1 above is not hypothetical), iptacopan/PNH came back APPROVAL;
citation verification cleared 10/11 candidate PMIDs at >= 0.5 with both entities
found, the one failure being a Europe PMC read timeout (rule: unresolved abstract
= "could not check", so it is dropped, not shown as weak).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence

from src.data.kg.types import CitationVerdict
from src.services.clinical_context.brand_map import BrandClinicalProfile, TreatmentContext

logger = logging.getLogger(__name__)

# Open Targets stage vocabulary, weakest -> strongest. Used ONLY to pick the
# strongest stage when several rows match the same indication; an unknown stage
# sorts lowest rather than being dropped (we still show it, honestly labelled).
_STAGE_ORDER: dict[str, int] = {
    "PRECLINICAL": 0,
    "EARLY_PHASE_1": 1,
    "PHASE_1": 2,
    "PHASE_1_2": 3,
    "PHASE_2": 4,
    "PHASE_2_3": 5,
    "PHASE_3": 6,
    "PHASE_4": 7,
    "APPROVED": 8,
    "APPROVAL": 8,
}

# Same vocabulary kg_querier gates ``treats`` on — kept identical so the two
# consumers of Open Targets cannot drift apart.
_APPROVED_CLINICAL_STAGES = frozenset({"APPROVAL", "APPROVED"})

# How many PMIDs to consider, and how many verified citations to surface. Each
# candidate costs one Europe PMC fetch, so the candidate list is small and
# verification stops as soon as the cap is met.
_MAX_CANDIDATE_PMIDS = 5
_MAX_CITATIONS = 3

# Every candidate costs one Europe PMC round trip, and Europe PMC DOES time out
# (measured on 2026-08-21). This whole block runs inside one request while a user
# waits on a panel, so each upstream gets a shorter timeout than the KG defaults and
# verification stops on a wall-clock budget, returning what it has. MEASURED
# end-to-end on the full stack the same day: typically 0.3-3s warm/cold, but one
# slow-upstream window hit 31.8s — which is why these are tighter than the defaults.
_EUROPE_PMC_TIMEOUT_S = 6.0
_OPEN_TARGETS_TIMEOUT_S = 8.0
_VERIFICATION_BUDGET_S = 12.0

# A citation clears the bar only when the abstract names BOTH entities — that is
# exactly what CitationResolver scores 0.5 for (a causal cue adds more on top).
_MIN_CITATION_CONFIDENCE = 0.5

_FDA_LABEL_NOTE = (
    "Open Targets records the clinical stage per indication and lags the FDA "
    "label; approval status for this brand comes from the approved-use section of "
    "this panel, not from this edge."
)


def _same_molecule(resolved: str, expected: str) -> bool:
    """True when the Open Targets record we read is the molecule we asked about.

    ``search_drug`` is a relevance-ranked search with no exact-match guarantee, and
    the panel attributes the edge to a drug name — so an unverified record would
    render a regulatory-sounding claim sourced from a different molecule, invisibly.
    Salt / ester forms ("RIBOCICLIB SUCCINATE") are the same molecule and match; an
    empty resolved name is unverifiable and therefore does NOT match.
    """
    r = resolved.strip().casefold()
    e = expected.strip().casefold()
    if not r or not e:
        return False
    return r == e or e in r.replace("-", " ").split()


@dataclass(frozen=True)
class IndicationEdge:
    """The drug -> indication edge for the analysis's own disease node."""

    predicate: str  # "treats" (approved) | "associated_with" (in development)
    # The molecule Open Targets actually answered about — verified against the
    # brand's INN before the edge is emitted, and surfaced so the panel names what
    # matched rather than the curated name it assumed.
    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    max_clinical_stage: str
    source: str = "open_targets"


@dataclass(frozen=True)
class VerifiedCitation:
    """A citation whose abstract was fetched and checked, not merely retrieved."""

    pmid: str
    title: str
    journal: Optional[str]
    pubdate: Optional[str]
    url: str
    entities_found: tuple[str, ...]
    confidence: float
    source: str = "pubmed+europepmc"


@dataclass(frozen=True)
class CausalEvidenceFragment:
    """What public evidence says about ONE analysis.

    ``status``: ``evidence`` (something was found) | ``commercial_lever`` (the
    clinical sources do not speak to this treatment side) | ``unavailable``
    (asked, nothing usable came back) | ``not_requested`` (the caller did not ask
    for the live lookup — the leaderboard fan-out does not).
    """

    status: str
    indication_edge: Optional[IndicationEdge] = None
    citations: List[VerifiedCitation] = field(default_factory=list)
    note: str = ""
    # Sources that were ASKED and failed. Without this, an Open Targets outage is
    # indistinguishable from "no indication edge exists" — an absence of evidence
    # read as evidence of absence — and the service would cache it as settled.
    sources_unavailable: tuple[str, ...] = ()


NOT_REQUESTED = CausalEvidenceFragment(
    status="not_requested",
    note=(
        "Analysis-specific evidence is gathered when the analysis is opened, not "
        "for every leaderboard row."
    ),
)


# --- Minimal structural protocols so tests can inject fakes --------------------


class _OpenTargetsLike(Protocol):
    def search_drug(self, name: str) -> Optional[str]: ...

    def search_disease(self, name: str) -> Optional[str]: ...

    def drug_disease_evidence(
        self, drug_chembl_id: str, disease_efo_id: str
    ) -> dict[str, Any]: ...


class _PubMedSearchLike(Protocol):
    def search_pmids(self, term: str, *, retmax: int = 5) -> List[str]: ...

    def fetch_by_pmid(self, pmid: str) -> Any: ...


class _CitationResolverLike(Protocol):
    def verify_citation(
        self,
        identifier: str,
        *,
        identifier_kind: str = "pmid",
        subject_name: str,
        object_name: str,
    ) -> CitationVerdict: ...


class CausalEvidenceProviderLike(Protocol):
    """What the service needs from an evidence provider (tests inject their own)."""

    def evidence(
        self,
        profile: BrandClinicalProfile,
        *,
        outcome: str,
        treatment_context: Optional[TreatmentContext],
        search_term: str,
    ) -> CausalEvidenceFragment: ...


class CausalEvidenceProvider:
    """Assemble the public-KG evidence for one analysis. Best-effort throughout:
    any upstream failure degrades to an honest ``unavailable``, never to a
    fabricated or borrowed claim."""

    provider_name = "causal_evidence"

    def __init__(
        self,
        *,
        open_targets: _OpenTargetsLike,
        pubmed: _PubMedSearchLike,
        resolver: _CitationResolverLike,
    ) -> None:
        self._open_targets = open_targets
        self._pubmed = pubmed
        self._resolver = resolver

    # -- indication edge --------------------------------------------------------

    def _indication_edge(self, profile: BrandClinicalProfile) -> Optional[IndicationEdge]:
        drug_id = self._open_targets.search_drug(profile.drug_name)
        if not drug_id:
            return None
        disease_id = self._open_targets.search_disease(profile.disease_search_term)
        payload = self._open_targets.drug_disease_evidence(drug_id, disease_id or "")
        drug = (payload or {}).get("drug") or {}
        resolved_name = str(drug.get("name") or "")
        if not _same_molecule(resolved_name, profile.drug_name):
            logger.warning(
                "causal-evidence: Open Targets resolved %r to %r — refusing to claim "
                "an indication edge for another molecule",
                profile.drug_name,
                resolved_name,
            )
            return None
        rows: Sequence[dict[str, Any]] = ((drug.get("indications") or {}).get("rows")) or []
        # EXACT disease-node match first. A loose name match ("cancer") would let
        # prostate / endometrial rows speak for a breast-cancer analysis, which is
        # the same borrowed-relevance failure this whole issue is about.
        matches = (
            [r for r in rows if ((r.get("disease") or {}).get("id")) == disease_id]
            if disease_id
            else []
        )
        if not matches and not disease_id:
            # No id resolved at all: fall back to rows whose name IS the disease
            # term in FULL (not merely a shared word).
            term = profile.disease_search_term.lower()
            matches = [
                r for r in rows if term and term == ((r.get("disease") or {}).get("name") or "").lower()
            ]
        if not matches:
            return None
        best = max(matches, key=lambda r: _STAGE_ORDER.get(str(r.get("maxClinicalStage") or ""), -1))
        disease = best.get("disease") or {}
        stage = str(best.get("maxClinicalStage") or "UNKNOWN")
        return IndicationEdge(
            predicate=("treats" if stage in _APPROVED_CLINICAL_STAGES else "associated_with"),
            drug_id=str(drug.get("id") or ""),
            drug_name=resolved_name,
            disease_id=str(disease.get("id") or ""),
            disease_name=str(disease.get("name") or ""),
            max_clinical_stage=stage,
            source="open_targets",
        )

    # -- verified literature ----------------------------------------------------

    def _citations(self, profile: BrandClinicalProfile, search_term: str) -> List[VerifiedCitation]:
        pmids = self._pubmed.search_pmids(search_term, retmax=_MAX_CANDIDATE_PMIDS)
        out: List[VerifiedCitation] = []
        started = time.monotonic()
        for pmid in pmids:
            if len(out) >= _MAX_CITATIONS:
                break
            # Reserve room for a WHOLE client timeout: a candidate started at
            # budget-minus-epsilon can still burn a full Europe PMC timeout on top,
            # so "elapsed < budget" would not bound the wait it claims to bound.
            if time.monotonic() - started >= max(
                0.0, _VERIFICATION_BUDGET_S - _EUROPE_PMC_TIMEOUT_S
            ):
                # Out of budget: surface what verified rather than making the user
                # wait on a slow upstream. Nothing unverified is shown either way.
                logger.info(
                    "causal-evidence: verification budget exhausted after %d citation(s)", len(out)
                )
                break
            try:
                verdict = self._resolver.verify_citation(
                    pmid,
                    identifier_kind="pmid",
                    subject_name=profile.drug_name,
                    # The plain-language disease term: the SSOT coding string
                    # ("Malignant neoplasm of breast") never appears in an abstract.
                    object_name=profile.disease_search_term,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort; skip this candidate
                logger.warning("causal-evidence: verification failed for PMID %s: %s", pmid, exc)
                continue
            # An unresolved abstract means "could not check", NOT "checked and weak" —
            # it must never reach the panel as a weak-but-shown citation.
            if not verdict.abstract_resolved:
                continue
            if verdict.overall_confidence < _MIN_CITATION_CONFIDENCE:
                continue
            try:
                article = self._pubmed.fetch_by_pmid(pmid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("causal-evidence: summary fetch failed for PMID %s: %s", pmid, exc)
                article = None
            out.append(
                VerifiedCitation(
                    pmid=pmid,
                    title=(getattr(article, "title", None) or f"PMID {pmid}"),
                    journal=getattr(article, "journal", None),
                    pubdate=getattr(article, "pubdate", None),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    entities_found=tuple(verdict.entities_found),
                    confidence=verdict.overall_confidence,
                    source="pubmed+europepmc",
                )
            )
        return out

    # -- the public entry point -------------------------------------------------

    def evidence(
        self,
        profile: BrandClinicalProfile,
        *,
        outcome: str,
        treatment_context: Optional[TreatmentContext],
        search_term: str,
    ) -> CausalEvidenceFragment:
        """Evidence for ``profile``'s (treatment -> outcome) analysis."""
        if treatment_context is None:
            return CausalEvidenceFragment(
                status="unavailable",
                note=(
                    "No curated clinical framing for this treatment column, so there "
                    "is no analysis to gather evidence for."
                ),
            )
        if treatment_context.kind == "commercial":
            return CausalEvidenceFragment(
                status="commercial_lever",
                note=(
                    f"{treatment_context.label} is a commercial access/promotion lever. "
                    "Biomedical and regulatory sources describe the therapy and its "
                    "indication, not this lever, so no clinical evidence is claimed for "
                    "the treatment side of this analysis."
                ),
            )
        # The indication edge is a claim about the THERAPY. It belongs to an analysis
        # whose treatment IS the therapy; for a patient-state contrast it would be the
        # drug's evidence rendered as evidence about the contrast.
        unavailable: List[str] = []
        edge = None
        if treatment_context.kind == "drug_therapy":
            try:
                edge = self._indication_edge(profile)
            except Exception as exc:  # noqa: BLE001 — best-effort; the edge is optional
                logger.warning(
                    "causal-evidence: Open Targets lookup failed for %s: %s",
                    profile.drug_name,
                    exc,
                )
                edge = None
                unavailable.append("open_targets")
        try:
            citations = self._citations(profile, search_term)
        except Exception as exc:  # noqa: BLE001 — best-effort; literature is optional
            logger.warning("causal-evidence: literature search failed for %r: %s", search_term, exc)
            citations = []
            unavailable.append("pubmed")
        if edge is None and not citations:
            return CausalEvidenceFragment(
                status="unavailable",
                note=(
                    "No indication edge or verifiable literature came back for this "
                    "analysis from the public sources."
                ),
                sources_unavailable=tuple(unavailable),
            )
        notes = [
            f"Literature searched as: {search_term!r}; a citation is shown only when its "
            f"abstract names both {profile.drug_name} and {profile.disease_search_term}."
        ]
        if treatment_context.kind == "clinical_covariate":
            notes.insert(
                0,
                f"{treatment_context.label} is a patient-state variable used as an "
                f"observational treatment, not a therapy, so no drug-indication claim is "
                f"made for it; the literature below was retrieved with this contrast in "
                f"the query and verified on {profile.drug_name} + "
                f"{profile.disease_search_term}.",
            )
        if edge is not None and edge.predicate != "treats":
            notes.insert(0, _FDA_LABEL_NOTE)
        if unavailable:
            # An outage must not read as a settled absence.
            notes.insert(
                0,
                f"{' and '.join(unavailable)} was unreachable for this analysis, so what "
                f"is missing below is unknown, not absent.",
            )
        return CausalEvidenceFragment(
            status="evidence",
            indication_edge=edge,
            citations=citations,
            note=" ".join(notes),
            sources_unavailable=tuple(unavailable),
        )


def default_causal_evidence_provider() -> CausalEvidenceProvider:
    """Build the real provider (lazy imports keep the module graph cheap and let the
    service import without the KG stack's optional auth)."""
    from src.data.kg.citation_resolver import CitationResolver
    from src.data.kg.europe_pmc import EuropePMCClient
    from src.data.kg.open_targets import OpenTargetsClient
    from src.services.clinical_context.clients import PubMedClient

    return CausalEvidenceProvider(
        open_targets=OpenTargetsClient(timeout=_OPEN_TARGETS_TIMEOUT_S),
        pubmed=PubMedClient(),
        # A tighter Europe PMC timeout than the KG default: this path runs while a
        # user waits on a panel, not in a batch job.
        resolver=CitationResolver(europe_pmc=EuropePMCClient(timeout=_EUROPE_PMC_TIMEOUT_S)),
    )
