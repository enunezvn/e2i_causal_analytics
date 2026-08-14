"""Phase 2.6 — CitationResolver.

Verifies that a PMID/DOI cited as evidence for a subject-object relation
actually contains both entities and a causal cue verb in its abstract.
This is the verification rail of Layer 2's ensemble: Open Targets supplies
PMIDs in evidence rows, but those PMIDs are uncurated — many cite
correlation studies, comorbidity registries, or population-level
associations that say "X is associated with Y" without claiming a causal
mechanism. CitationResolver filters those out before Phase 2.5
(CausalRoleClassifier) consumes the evidence.

Verification logic (per ``verify_citation``):
    1. Fetch the abstract from Europe PMC (PMID) or Crossref (DOI).
    2. Look for either entity name OR any of its UMLS synonyms in the
       abstract text (case-insensitive substring match). Both subject and
       object must appear.
    3. Look for any causal cue verb in ``CAUSAL_CUE_VERBS`` in the
       abstract.
    4. Score: 0.5 if both entities found, +0.3 if a causal cue is found,
       +0.2 if both found AND the cue and entities co-occur within a
       short window (currently document-level, sentence-level co-occurrence
       deferred to v2). Returned as ``CitationVerdict.overall_confidence``.

Synonym lookup uses ``UMLSClient.cui_lookup`` to fetch the preferred name
and (when available) the atom list. v1 uses the preferred name only;
the atom-list fan-out is a v2 punt because each atom-list call is an
extra UTS round trip per CUI.

Reference: ``.claude/plans/adaptive_temporal_validity_redesign.md`` Phase 2.6.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from src.data.kg.crossref import CrossrefClient, CrossrefError
from src.data.kg.europe_pmc import EuropePMCClient, EuropePMCError
from src.data.kg.types import AbstractRecord, CitationVerdict, KGConcept
from src.data.kg.umls_uts import UMLSAuthError, UMLSClient, UMLSError

logger = logging.getLogger(__name__)


# Causal cue list: verbs and stock phrases that claim a causal mechanism
# in scientific abstracts. The list is intentionally narrow — false
# positives (e.g., "X is associated with Y", "patients treated with X")
# would defeat the whole point of this filter.
#
# Codex review MEDIUM (2026-05-08) pruned ambiguous verbs:
#   - "treated" — non-causal in "patients treated with X" passive observational shape.
#   - "improved" — "patient improved" can mean "got better" with no causal attribution.
#   - "reduced" — "reduced model" / "reduced sample" are common non-causal usages.
#   - "blocks", "blocked" — high false-positive in non-pharma contexts.
#   - "prevents", "prevented" — "prevented from enrolling" is non-causal.
# Their active-voice counterparts ("treats", "improves", "reduces",
# "prevents") are kept but only when they appear with both entities AND
# co-occurrence is enforced upstream by the both-entities gate.
#
# Multi-word phrases added per codex flag — these were called out as
# common in causal abstracts but absent from v1. Multi-word phrases are
# matched as literal substrings (with word boundaries) so they don't
# false-positive against partial matches.
CAUSAL_CUE_VERBS: tuple[str, ...] = (
    # Single-word verbs (active voice; passive forms removed by codex review).
    "treats",
    "causes",
    "caused",
    "induces",
    "induced",
    "ameliorates",
    "ameliorated",
    "improves",
    "reduces",
    "inhibits",
    "inhibited",
    "alleviates",
    "alleviated",
    "mediates",
    "mediated",
    "triggers",
    "triggered",
    # Multi-word causal phrases (codex review addition).
    "leads to",
    "led to",
    "results in",
    "resulted in",
    "responsible for",
    "mechanism of action",
    "due to",
)


# Confidence weights for the score aggregation. Keep these as named
# constants so future calibration (Phase 4 active learning) can tune
# them against labeled data.
WEIGHT_BOTH_ENTITIES = 0.5
WEIGHT_CAUSAL_CUE = 0.3
WEIGHT_COOCCURRENCE = 0.2


class CitationResolverError(Exception):
    """Raised on CitationResolver-fatal failures (e.g., UMLS auth dead)."""


class CitationResolver:
    """Compose Europe PMC + Crossref + UMLS clients into citation verification.

    Args:
        europe_pmc: Optional pre-constructed Europe PMC client.
        crossref: Optional pre-constructed Crossref client.
        umls: Optional pre-constructed UMLS client (used for synonym
            expansion). If None, one is built when ``UMLS_UTS_API_KEY`` is
            available; otherwise ``self.umls`` is None and citation
            verification only matches the ``preferred_name`` of each entity
            (no synonyms), which is a weaker but still useful check.
    """

    def __init__(
        self,
        *,
        europe_pmc: Optional[EuropePMCClient] = None,
        crossref: Optional[CrossrefClient] = None,
        umls: Optional[UMLSClient] = None,
    ) -> None:
        self._owns_europe_pmc = europe_pmc is None
        self._owns_crossref = crossref is None
        self._owns_umls = umls is None
        self.europe_pmc = europe_pmc if europe_pmc is not None else EuropePMCClient()
        self.crossref = crossref if crossref is not None else CrossrefClient()
        # UMLS is OPTIONAL — it only widens the term list with synonyms.
        # ``UMLSClient()`` raises ``UMLSAuthError`` when no key is present, so
        # constructing it unconditionally made this whole resolver
        # unconstructible in any environment without ``UMLS_UTS_API_KEY``
        # (CI included) and left the degraded mode this docstring promises
        # unreachable. Europe PMC and Crossref are zero-auth and must stay
        # usable on their own (#1608).
        self.umls: Optional[UMLSClient]
        if umls is not None:
            self.umls = umls
        else:
            try:
                self.umls = UMLSClient()
            except UMLSAuthError:
                logger.info(
                    "CitationResolver: no UMLS_UTS_API_KEY — synonym expansion "
                    "disabled; matching preferred names only."
                )
                self.umls = None

    def __enter__(self) -> "CitationResolver":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_europe_pmc:
            self.europe_pmc.close()
        if self._owns_crossref:
            self.crossref.close()
        if self._owns_umls and self.umls is not None:
            self.umls.close()

    def resolve_pmid(self, pmid: str) -> Optional[AbstractRecord]:
        """Fetch the abstract for a PMID. Returns None when unavailable."""
        try:
            return self.europe_pmc.fetch_abstract(pmid)
        except EuropePMCError as exc:
            logger.warning("Europe PMC fetch failed for PMID %s: %s", pmid, exc)
            return None

    def resolve_doi(self, doi: str) -> Optional[AbstractRecord]:
        """Fetch metadata + abstract for a DOI. Returns None when unavailable."""
        try:
            return self.crossref.fetch_doi_metadata(doi)
        except CrossrefError as exc:
            logger.warning("Crossref fetch failed for DOI %s: %s", doi, exc)
            return None

    def verify_citation(
        self,
        identifier: str,
        *,
        identifier_kind: str = "pmid",
        subject_name: str,
        object_name: str,
        subject_cui: Optional[str] = None,
        object_cui: Optional[str] = None,
    ) -> CitationVerdict:
        """Verify that a PMID/DOI's abstract co-mentions the entities + a causal cue.

        Args:
            identifier: PMID (e.g., ``"12345678"``) or DOI (e.g.,
                ``"10.1234/abc.2024.001"``).
            identifier_kind: ``"pmid"`` or ``"doi"``. Selects the resolution
                client.
            subject_name: Preferred name of the subject entity.
            object_name: Preferred name of the object entity.
            subject_cui: Optional UMLS CUI for the subject; when provided, the
                preferred name from UMLS is used in addition to ``subject_name``.
            object_cui: Optional UMLS CUI for the object; same role.

        Returns:
            ``CitationVerdict`` with the abstract-resolved flag, the
            entities found, the causal cue found (if any), and the
            aggregated confidence.
        """
        if identifier_kind not in ("pmid", "doi"):
            # Codex review MEDIUM (2026-05-08): preserve the original input
            # in the error verdict rather than hard-coding "pmid". A verdict
            # that pretends an invalid input was a PMID would mislead
            # Phase 2.7 EnsembleVoter diagnostics and aggregation buckets.
            return CitationVerdict(
                identifier=identifier,
                identifier_kind=identifier_kind,
                abstract_resolved=False,
                error=f"unsupported identifier_kind: {identifier_kind}",
            )
        record = (
            self.resolve_pmid(identifier)
            if identifier_kind == "pmid"
            else self.resolve_doi(identifier)
        )
        if record is None:
            return CitationVerdict(
                identifier=identifier,
                identifier_kind=identifier_kind,  # type: ignore[arg-type]
                abstract_resolved=False,
                overall_confidence=0.0,
            )
        # Build the candidate term lists for each entity. The preferred
        # name supplied by the caller is always included; UMLS preferred
        # names (when CUI given) are added as synonyms.
        subject_terms = self._candidate_terms(subject_name, subject_cui)
        object_terms = self._candidate_terms(object_name, object_cui)
        haystack_lower = record.abstract.lower()
        subject_match = _first_match(subject_terms, haystack_lower)
        object_match = _first_match(object_terms, haystack_lower)
        causal_cue = _find_causal_cue(haystack_lower)
        entities_found: list[str] = []
        if subject_match:
            entities_found.append(subject_match)
        if object_match:
            entities_found.append(object_match)
        # Codex review HIGH (2026-05-08): cue-verb credit is only awarded
        # when BOTH entities are also present. A causal cue alone, without
        # the subject/object pair, doesn't constitute evidence — the
        # abstract could be about an entirely different relation (e.g.,
        # "ibuprofen treats inflammation" doesn't verify a citation for
        # "ibuprofen treats atopic dermatitis"). Without this guard,
        # unrelated abstracts containing common cue verbs would silently
        # rank above unresolved citations in Phase 2.7's aggregation.
        confidence = 0.0
        if subject_match and object_match:
            confidence += WEIGHT_BOTH_ENTITIES
            if causal_cue:
                confidence += WEIGHT_CAUSAL_CUE
                confidence += WEIGHT_COOCCURRENCE
        return CitationVerdict(
            identifier=identifier,
            identifier_kind=identifier_kind,  # type: ignore[arg-type]
            abstract_resolved=True,
            entities_found=tuple(entities_found),
            causal_cue_found=causal_cue,
            overall_confidence=confidence,
        )

    def _candidate_terms(
        self,
        primary_name: str,
        cui: Optional[str],
    ) -> list[str]:
        """Build the list of names to match in an abstract.

        Includes ``primary_name`` always, plus the UMLS preferred name when
        a CUI is supplied (and UMLS auth doesn't fail). v1 stops there;
        atom-list synonym fanout is a v2 enhancement.
        """
        terms: list[str] = []
        if primary_name:
            terms.append(primary_name)
        if cui and self.umls is not None:
            try:
                concept: KGConcept = self.umls.cui_lookup(cui)
            except UMLSAuthError as exc:
                raise CitationResolverError(f"UMLS auth failed: {exc}") from exc
            except UMLSError as exc:
                logger.warning("UMLS cui_lookup failed for synonym expansion of %s: %s", cui, exc)
            else:
                if concept.preferred_name and concept.preferred_name not in terms:
                    terms.append(concept.preferred_name)
        return terms


def _first_match(terms: Iterable[str], haystack_lower: str) -> Optional[str]:
    """Return the first term that appears in ``haystack_lower`` as a
    whole-word match.

    Codex review HIGH (2026-05-08): naive substring matching produced
    false positives like "asthma" matching inside "asthmatic", "RA"
    matching ordinary text, and short drug names matching unrelated
    fragments. Boundary handling:

    - When the term's first/last character is alphanumeric, use ``\\b``
      to enforce word boundaries. This catches the asthma/asthmatic case.
    - When the first/last character is NON-alphanumeric (e.g., the
      trailing ``)`` in "C-reactive protein (CRP)"), skip the boundary
      anchor on that side — ``\\b`` is defined as a transition between
      word and non-word, so two non-word chars in a row don't form one,
      and including the anchor would silently miss the match.

    Returns the original-cased term so the caller can surface it back
    (case-insensitive matching against the pre-lowered haystack).
    """
    for term in terms:
        if not term:
            continue
        term_lower = term.lower()
        left = r"\b" if term_lower[0].isalnum() else ""
        right = r"\b" if term_lower[-1].isalnum() else ""
        pattern = f"{left}{re.escape(term_lower)}{right}"
        if re.search(pattern, haystack_lower):
            return term
    return None


def _find_causal_cue(haystack_lower: str) -> Optional[str]:
    """Return the first causal cue verb that appears in ``haystack_lower``.

    Match is whole-word (``\\b`` boundaries) so substrings of unrelated
    words don't false-positive (e.g., "induced" inside "reproduced").
    """
    for cue in CAUSAL_CUE_VERBS:
        if re.search(rf"\b{re.escape(cue)}\b", haystack_lower):
            return cue
    return None
