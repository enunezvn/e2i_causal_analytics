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
_MAX_CANDIDATE_PMIDS = 3
_MAX_CITATIONS = 2

# Europe PMC is the slow upstream here, and it is slow BY ITSELF, not because of
# the network: MEASURED 2026-08-21 from the API container, resultType=core answered
# in 8.3 / 8.6 / 10.0 / 16.1s with DNS 0.09s, TCP 0.08s and TLS 1.4s — and the host
# was equally slow (11.7s) for the same query. A 6s timeout (calibrated on a lucky
# fast window) therefore dropped EVERY abstract in production: the live panel showed
# an evidence block with no literature at all while every unit test stayed green.
# The timeout now covers the observed worst case; the wall-clock budget bounds how
# many of these a user can wait through, and the candidate list is short because
# each candidate is expensive.
_EUROPE_PMC_TIMEOUT_S = 20.0
_OPEN_TARGETS_TIMEOUT_S = 8.0
_VERIFICATION_BUDGET_S = 22.0
# Each ACCEPTED citation costs one more PubMed summary call, which lands after the
# verification check — so three of them could add three client timeouts to a path
# that looks bounded. Past this, the citation is still surfaced (it was verified),
# just without its title/journal.
_SUMMARY_BUDGET_S = 16.0

# A citation clears the bar only when the abstract names BOTH entities — that is
# exactly what CitationResolver scores 0.5 for (a causal cue adds more on top).
_MIN_CITATION_CONFIDENCE = 0.5

# Human-readable names for the machine-readable ``sources_unavailable`` keys. The
# panel renders the note verbatim, so "europe_pmc was unreachable" would leak an
# internal identifier at the exact moment we are asking the analyst to trust us.
_SOURCE_DISPLAY = {
    "open_targets": "Open Targets",
    "pubmed": "PubMed",
    "europe_pmc": "Europe PMC",
}


def _incomplete_note(budget: int, local: int) -> str:
    """Said when WE stopped early rather than when an upstream failed.

    It must not name a source — accusing a healthy Europe PMC is the same dishonesty
    inverted (codex iter-1 HIGH) — and it must not assert a reason we did not
    observe: a local verification bug is not a budget timeout (codex iter-2 HIGH).
    """
    if budget and local:
        why = "the verification budget ran out and a verification step failed"
    elif budget:
        why = "the verification budget ran out before every candidate was checked"
    else:
        why = "a verification step failed before it could return"
    return (
        f"The literature check did not finish for this analysis — {why} — so what "
        f"is missing below is unknown, not absent."
    )


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
    # We stopped early under OUR OWN budget, or a check blew up locally — so the
    # literature question is unfinished, but NO upstream is being accused (codex
    # iter-1 HIGH, #1767). Naming Europe PMC here would be the inverse dishonesty:
    # a healthy source reported as unreachable, re-fetched every 600s forever.
    #
    # Deliberately NOT serialized onto the wire: the payload is assembled key by
    # key in service.py, the analyst-facing truth is carried by ``note``, and the
    # only machine consumer is the service's cache-completeness decision.
    checks_incomplete: bool = False


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

    def drug_disease_evidence(self, drug_chembl_id: str, disease_efo_id: str) -> dict[str, Any]: ...


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
                r
                for r in rows
                if term and term == ((r.get("disease") or {}).get("name") or "").lower()
            ]
        if not matches:
            return None
        best = max(
            matches, key=lambda r: _STAGE_ORDER.get(str(r.get("maxClinicalStage") or ""), -1)
        )
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

    def _citations(
        self, profile: BrandClinicalProfile, search_term: str
    ) -> tuple[List[VerifiedCitation], int, int, int]:
        """Verified literature for this analysis, plus WHY anything is missing.

        Returns ``(citations, unreachable, unchecked_budget, unchecked_local)``:

        - ``unreachable``      — Europe PMC RAISED for this candidate. A real outage.
        - ``unchecked_budget`` — our own wall-clock budget stopped us before this
          candidate was examined.
        - ``unchecked_local``  — the verification call blew up locally. The resolver
          swallows ``EuropePMCError`` itself, so what escapes is not evidence that
          the upstream is down.

        All three mean "not a settled absence", and they must stay apart. Reporting a
        healthy Europe PMC as unreachable is the same dishonesty inverted, and so is
        telling the analyst the budget ran out when a local bug is what actually
        stopped us.
        """
        pmids = self._pubmed.search_pmids(search_term, retmax=_MAX_CANDIDATE_PMIDS)
        out: List[VerifiedCitation] = []
        unreachable = 0
        unchecked_budget = 0
        unchecked_local = 0
        started = time.monotonic()
        for attempt, pmid in enumerate(pmids):
            if len(out) >= _MAX_CITATIONS:
                break
            # Reserve room for a WHOLE client timeout before starting ANOTHER
            # candidate: one started at budget-minus-epsilon can still burn a full
            # Europe PMC timeout on top, so "elapsed < budget" would not bound the
            # wait it claims to bound. The FIRST candidate always runs, though —
            # "the budget cannot fit a call" must mean "verify one and stop", never
            # "verify nothing", which is what silently emptied this block in prod.
            if attempt > 0 and time.monotonic() - started >= max(
                0.0, _VERIFICATION_BUDGET_S - _EUROPE_PMC_TIMEOUT_S
            ):
                # Out of budget: surface what verified rather than making the user
                # wait on a slow upstream. Nothing unverified is shown either way.
                logger.info(
                    "causal-evidence: verification budget exhausted after %d citation(s)", len(out)
                )
                # Everything from this candidate on was never examined. Truncating
                # the search is not the same as searching and finding nothing — but
                # it is OUR budget that stopped, so Europe PMC is not accused.
                unchecked_budget += len(pmids) - attempt
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
                # The resolver swallows EuropePMCError itself, so what escapes here
                # is not evidence that the upstream is down. Unfinished, not accused —
                # and not a budget timeout either.
                unchecked_local += 1
                continue
            # An unresolved abstract means "could not check", NOT "checked and weak" —
            # it must never reach the panel as a weak-but-shown citation. The verdict's
            # ``error`` is set only when the source RAISED; unset means the source
            # answered and simply holds no abstract, which IS a settled negative.
            if not verdict.abstract_resolved:
                if verdict.error:
                    unreachable += 1
                continue
            if verdict.overall_confidence < _MIN_CITATION_CONFIDENCE:
                continue
            article = None
            if time.monotonic() - started < _SUMMARY_BUDGET_S:
                try:
                    article = self._pubmed.fetch_by_pmid(pmid)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "causal-evidence: summary fetch failed for PMID %s: %s", pmid, exc
                    )
                    article = None
            else:
                logger.info("causal-evidence: summary budget exhausted; PMID %s unadorned", pmid)
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
        return out, unreachable, unchecked_budget, unchecked_local

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
                    "Open Targets and the FDA label describe the therapy and its "
                    "indication, not this lever, so no indication or approval claim is "
                    "made for the treatment side of this analysis. The real-world-"
                    "evidence citation above is a different matter: its search carries "
                    "this lever's own health-services theme when one exists (copay "
                    "assistance, patient support programmes), so it can legitimately "
                    "speak to it."
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
        checks_incomplete = False
        incomplete_note = ""
        try:
            citations, lit_unreachable, lit_budget, lit_local = self._citations(
                profile, search_term
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; literature is optional
            logger.warning("causal-evidence: literature search failed for %r: %s", search_term, exc)
            citations = []
            unavailable.append("pubmed")
        else:
            # Nothing verified, and we did not get a real answer for at least one
            # candidate: the literature question is UNANSWERED, not answered "none".
            # Saying so is also what keeps the service from caching this as settled
            # for the worker's process lifetime (#1767).
            if not citations:
                if lit_unreachable:
                    # Europe PMC actually failed. Name it.
                    unavailable.append("europe_pmc")
                elif lit_budget or lit_local:
                    # We stopped early ourselves. Unfinished, but nobody is at fault.
                    checks_incomplete = True
                    incomplete_note = _incomplete_note(lit_budget, lit_local)

        # WHY anything is missing, composed ONCE and used by both returns. Building
        # it per-return is how the "unavailable" path came to omit the outage
        # disclosure entirely while still reporting sources_unavailable — a note
        # contradicting the payload beside it (codex iter-2 HIGH).
        disclosure: List[str] = []
        if unavailable:
            # An outage must not read as a settled absence.
            names = [_SOURCE_DISPLAY.get(src, src) for src in unavailable]
            verb = "was" if len(names) == 1 else "were"
            disclosure.append(
                f"{' and '.join(names)} {verb} unreachable for this analysis, so what "
                f"is missing below is unknown, not absent."
            )
        if checks_incomplete:
            disclosure.append(incomplete_note)
        # A patient-state treatment never asks Open Targets at all, so the absence of
        # an edge is BY DESIGN. Saying "no indication edge came back" would report a
        # question we deliberately never asked as one that returned empty.
        is_covariate = treatment_context.kind == "clinical_covariate"

        if edge is None and not citations:
            parts = list(disclosure)
            if not parts:
                # Nothing failed and nothing was truncated: this really is a settled
                # absence, and we can say so.
                parts.append(
                    "No verifiable literature came back for this analysis from the public sources."
                    if is_covariate
                    else (
                        "No indication edge or verifiable literature came back for this "
                        "analysis from the public sources."
                    )
                )
            if is_covariate:
                parts.append(
                    f"{treatment_context.label} is a patient-state variable used as an "
                    f"observational treatment, not a therapy, so no drug-indication "
                    f"claim was sought for it."
                )
            return CausalEvidenceFragment(
                status="unavailable",
                note=" ".join(parts),
                sources_unavailable=tuple(unavailable),
                checks_incomplete=checks_incomplete,
            )
        notes = [
            f"Literature searched as: {search_term!r}; a citation is shown only when its "
            f"abstract names both {profile.drug_name} and {profile.disease_search_term}."
        ]
        if is_covariate:
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
        notes = disclosure + notes
        return CausalEvidenceFragment(
            status="evidence",
            indication_edge=edge,
            citations=citations,
            note=" ".join(notes),
            sources_unavailable=tuple(unavailable),
            checks_incomplete=checks_incomplete,
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
