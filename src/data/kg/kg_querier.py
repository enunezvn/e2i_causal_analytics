"""Phase 2.3 — KnowledgeGraphQuerier.

Wraps the EntityLinker-composed clients (UMLS, Open Targets, RxNav) and
returns structured ``KGEdge`` triples that the rest of Layer 2
(``CausalRoleClassifier``, ``CitationResolver``, ``EnsembleVoter``) consumes.

Design:
    KGQuerier does NOT own its own client connections — it borrows them from
    an ``EntityLinker`` (or accepts the constituent clients directly) so the
    same connection pool and LRU cache backs the whole Layer 2 pipeline. A
    caller who passes an already-constructed EntityLinker gets transitive
    cache hits for free.

Surface:
    - ``query_drug_disease_edges(drug_id, disease_id)`` — the drug's Open
      Targets INDICATION list, filtered to ``disease_id`` and mapped to
      ``KGEdge`` records, with the predicate gated on clinical stage so only
      an approved indication reads as ``treats``. ``drug_id`` is a ChEMBL ID;
      ``disease_id`` an EFO/MONDO ID (resolve via ``OpenTargetsClient
      .search_drug`` / ``.search_disease``). See #1607 for the schema
      migration that replaced the removed ``evidences`` field.
    - ``query_disease_hierarchy(cui)`` — UMLS relations endpoint filtered to
      ``isa``/``parent``/``child``-shaped relations. Returns subclass and
      superclass edges so the LLM can reason about taxonomic relationships
      ("L20.9 atopic dermatitis is_a inflammatory skin condition").
    - ``query_concept_relations(cui, predicates=None)`` — generic UMLS
      relations call; ``predicates`` filters by ``additionalRelationLabel``
      (e.g., ``["may_treat", "may_be_treated_by"]`` for drug-disease assoc).

Phase 2.5 (CausalRoleClassifier) will compose these calls per-feature: given
an EntityLink for a feature, fan out to ``query_disease_hierarchy`` for any
disease CUI and ``query_drug_disease_edges`` for any drug-disease pair, and
hand the resulting ``KGEdge`` list to the DSPy program.

Reference: `.claude/plans/adaptive_temporal_validity_redesign.md` Phase 2.3.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from src.data.kg.chembl import ChEMBLClient
from src.data.kg.entity_linker import EntityLinker
from src.data.kg.open_targets import OpenTargetsClient, OpenTargetsError
from src.data.kg.types import KGEdge
from src.data.kg.umls_uts import UMLSAuthError, UMLSClient, UMLSError

logger = logging.getLogger(__name__)

# ``maxClinicalStage`` values that constitute a regulator-approved therapeutic
# claim. Only these earn ``predicate="treats"`` — everything else (PHASE_1 ..
# PHASE_3, and any future stage token) degrades to ``associated_with`` so an
# investigational pairing cannot promote a feature to
# ``leak_drug_treats_disease`` in the voter. Unknown/absent stages fall to the
# conservative side by construction (#1607).
_APPROVED_CLINICAL_STAGES: frozenset[str] = frozenset({"APPROVAL", "APPROVED"})


class KnowledgeGraphQuerier:
    """Structured KGEdge extraction over UMLS + Open Targets + ChEMBL.

    Args:
        umls: A constructed UMLSClient. If None and ``entity_linker`` is also
            None, a default client is constructed (which reads the
            ``UMLS_UTS_API_KEY`` env var).
        open_targets: A constructed OpenTargetsClient. Same fallback rules.
        chembl: Optional ChEMBLClient. Accepted and closed like the other
            clients, but no longer consulted by ``query_drug_disease_edges``:
            the #245 enrichment it fed (``KGEdge.evidence[i].chembl_target_id``,
            cross-walked from an Open Targets evidence row's target gene) lost
            its input when Open Targets removed the ``evidences`` field. Kept in
            the signature so existing callers are unaffected (#1607).
        entity_linker: Optional pre-constructed ``EntityLinker``; if
            provided, its UMLS + Open Targets clients are reused so caching
            and connection pooling are shared with the linker.
    """

    def __init__(
        self,
        *,
        umls: Optional[UMLSClient] = None,
        open_targets: Optional[OpenTargetsClient] = None,
        chembl: Optional[ChEMBLClient] = None,
        entity_linker: Optional[EntityLinker] = None,
    ) -> None:
        # Track which clients we constructed so close() only closes those —
        # never the ones borrowed from a caller-supplied EntityLinker.
        self._owns_umls = False
        self._owns_open_targets = False
        # ChEMBL is never auto-constructed; the path is opt-in.
        self._owns_chembl = False
        # UMLS is held as Optional and built LAZILY by the ``umls`` property
        # (#1629). Building it here made a MISSING CREDENTIAL fatal to the whole
        # querier: ``UMLSClient()`` raises ``UMLSAuthError`` without a key, so
        # ``query_drug_disease_edges`` — pure Open Targets, zero-auth, never
        # touches UMLS — could not run without one. That is what turned the two
        # #1607 live contracts red in the nightly (#1627), where no UMLS secret
        # exists.
        #
        # Deferring to first use preserves ``query_concept_relations``'
        # documented contract exactly: ``UMLSAuthError`` still reaches the
        # caller, just at the point of use. It deliberately does NOT follow
        # ``CitationResolver``'s degrade-to-None pattern — there UMLS is an
        # optional enrichment, here it is load-bearing (84 of 99 cached edges
        # come from ``umls_relations``), so a missing key must never be
        # indistinguishable from "this concept has no relations".
        self._umls: Optional[UMLSClient] = umls
        if entity_linker is not None:
            if self._umls is None:
                self._umls = entity_linker.umls
            self.open_targets = (
                open_targets if open_targets is not None else entity_linker.open_targets
            )
        else:
            if open_targets is None:
                self.open_targets = OpenTargetsClient()
                self._owns_open_targets = True
            else:
                self.open_targets = open_targets
        # Optional ChEMBL client. Borrowed only — KGQuerier never
        # auto-constructs ChEMBL (would surprise existing callers).
        self.chembl: Optional[ChEMBLClient] = chembl

    @property
    def umls(self) -> UMLSClient:
        """The UMLS client, constructed on first use (#1629).

        Raises ``UMLSAuthError`` here rather than in ``__init__`` when no
        credential is available, so zero-auth paths (Open Targets drug-disease
        edges) stay usable without one while the UMLS paths still fail loudly.
        Ownership is recorded at construction time, so a querier that never
        touches UMLS neither builds nor closes a client.
        """
        if self._umls is None:
            self._umls = UMLSClient()
            self._owns_umls = True
        return self._umls

    def __enter__(self) -> "KnowledgeGraphQuerier":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close any clients KGQuerier itself constructed.

        Borrowed clients (passed in by the caller or via ``entity_linker``)
        are NOT closed here — their lifetime is the caller's responsibility.
        """
        # Guarded on ``_umls`` rather than the property: closing must never be
        # what triggers lazy construction (#1629).
        if self._owns_umls and self._umls is not None:
            self._umls.close()
        if self._owns_open_targets:
            self.open_targets.close()
        if self._owns_chembl and self.chembl is not None:
            self.chembl.close()

    def query_drug_disease_edges(
        self,
        drug_id: str,
        disease_id: str,
    ) -> list[KGEdge]:
        """Open Targets drug → disease evidence as ``KGEdge`` records.

        ``drug_id`` should be a ChEMBL ID (e.g., ``"CHEMBL1234"``) and
        ``disease_id`` should be an EFO/MONDO ID (e.g., ``"EFO_0000270"``).
        Caller is responsible for resolving CUIs/RxCUIs to ChEMBL/EFO via
        ``EntityLinker`` + ``OpenTargetsClient.search_drug``/``search_disease``
        before invoking this method.

        Sourced from the drug's INDICATION list, filtered to ``disease_id``.
        Each matching indication row produces ONE edge whose:
            - subject_id   = drug ChEMBL ID
            - object_id    = disease EFO/MONDO ID
            - predicate    = ``"treats"`` when the row's ``maxClinicalStage``
                             is an approved stage, else ``"associated_with"``.
                             The voter's ``classify_kg_signal`` consumes the
                             predicate to drive the
                             ``leak_drug_treats_disease`` classification — see
                             docs/superpowers/specs/2026-05-08-kg-predicate-
                             reconciliation-design.md.
            - evidence_source = ``"open_targets"``
            - datasource   = ``"chembl_indications"``
            - score        = ``None``
            - pmids        = ``()``

        Schema migration (#1607): this previously read a top-level
        ``evidences(drugIds:, diseaseIds:)`` field that supplied a per-row
        ``score``, ``literature`` PMIDs and a ``target`` gene for ChEMBL
        cross-walk enrichment. Open Targets REMOVED that field — evidence now
        requires a gene ``ensemblIds`` argument and ``Drug`` no longer exposes
        ``linkedTargets``, so no drug->gene path remains. The query returned
        HTTP 400 on every live call until this migration, so ``score``/``pmids``
        are not a regression against a working baseline: they were unreachable.
        In exchange, ``maxClinicalStage`` supplies the phase gate that a
        deferred codex review (PR-0 M1) asked for, so an investigational
        pairing can no longer masquerade as a therapeutic claim.

        Returns an empty list if the drug has no matching indication.
        ``OpenTargetsError`` is logged and re-raised so callers can distinguish
        "no evidence" from a transport / GraphQL failure (codex H1 from PR #102
        review) — the distinction that would have surfaced the dead query
        sooner. Drug and disease names are populated from the response.
        """
        try:
            data = self.open_targets.drug_disease_evidence(drug_id, disease_id)
        except OpenTargetsError as exc:
            logger.warning(
                "Open Targets drug-disease query failed for %s/%s: %s",
                drug_id,
                disease_id,
                exc,
            )
            raise
        drug = data.get("drug") or {}
        indications = drug.get("indications") or {}
        # ``indications.rows`` is a nullable GraphQL list, so ``or []``
        # collapses both the absent-key and explicit-null cases; without it
        # ``for row in rows`` would raise TypeError on a partial response.
        rows = (indications.get("rows") or []) if isinstance(indications, dict) else []

        edges: list[KGEdge] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            disease = row.get("disease") or {}
            row_disease_id = str(disease.get("id") or "")
            # The API returns the drug's WHOLE indication list; keep only the
            # disease the caller asked about, so an edge always encodes the
            # (drug, disease) pair the voter is reasoning about.
            if row_disease_id != disease_id:
                continue

            # Phase gating (deferred codex PR-0 review M1, now implementable).
            # ``maxClinicalStage`` is APPROVAL for a regulator-approved
            # indication and PHASE_1/2/3 for one still under investigation.
            # Only an approved indication is a therapeutic CLAIM; emitting
            # ``treats`` for a Phase I pairing would let an exploratory trial
            # promote a feature to ``leak_drug_treats_disease`` in the voter
            # and produce a false-positive leak verdict.
            stage = str(row.get("maxClinicalStage") or "").upper()
            predicate = "treats" if stage in _APPROVED_CLINICAL_STAGES else "associated_with"

            edges.append(
                KGEdge(
                    subject_id=str(drug.get("id") or drug_id),
                    subject_name=str(drug.get("name") or ""),
                    predicate=predicate,
                    object_id=row_disease_id or disease_id,
                    object_name=str(disease.get("name") or ""),
                    evidence_source="open_targets",
                    # The removed ``evidences`` field was the only source of
                    # per-row scores and literature PMIDs; the indication list
                    # carries neither. Left explicitly empty rather than
                    # fabricated (see the schema note in open_targets.py).
                    score=None,
                    pmids=(),
                    datasource="chembl_indications",
                    evidence=(),
                    raw=row,
                )
            )
        return edges

    # ``_resolve_chembl_target`` lived here until #1607. It cross-walked an Open
    # Targets evidence row's target gene to a ChEMBL target id for the per-PMID
    # provenance of issue #245. Its input — the top-level ``evidences`` Query
    # field carrying ``target.approvedSymbol`` — was REMOVED from the Open
    # Targets v4 schema, and ``Drug`` no longer exposes ``linkedTargets``, so
    # there is no drug->gene path left to feed it. Removed rather than left
    # unreachable; `test_open_targets_graphql_rejects_the_removed_evidences_field`
    # fails if Open Targets restores the field, at which point the
    # implementation is recoverable from git history.

    def query_disease_hierarchy(self, cui: str) -> list[KGEdge]:
        """UMLS taxonomic relations for a disease CUI.

        Filters to relations whose ``additionalRelationLabel`` is one of
        ``"isa"``, ``"inverse_isa"``, ``"is_a"``, ``"subclass_of"``,
        ``"superclass_of"`` — i.e., the parent/child taxonomic shape.
        ``RB``/``RN`` (related-broader/narrower) are also accepted as edges
        when no ``additionalRelationLabel`` clarifies them.
        """
        return self.query_concept_relations(
            cui,
            predicates={
                "isa",
                "inverse_isa",
                "is_a",
                "subclass_of",
                "superclass_of",
            },
            include_coarse_labels={"PAR", "CHD", "RB", "RN"},
        )

    def query_concept_relations(
        self,
        cui: str,
        *,
        predicates: Optional[Iterable[str]] = None,
        include_coarse_labels: Optional[Iterable[str]] = None,
    ) -> list[KGEdge]:
        """Generic UMLS relations → KGEdge transformer.

        Args:
            cui: Subject UMLS CUI.
            predicates: If provided, keep only edges whose
                ``additionalRelationLabel`` is in this set.
            include_coarse_labels: If provided, also keep edges whose coarse
                ``relationLabel`` is in this set even when their fine-grained
                label is empty.

        Returns:
            ``KGEdge`` records with subject = ``cui``, object = the related
            CUI extracted from ``relatedId``, predicate = the
            ``additionalRelationLabel`` (or coarse ``relationLabel`` when
            the additional one is empty). UMLS does not score relations, so
            ``score`` and ``pmids`` are always None / empty here. ``UMLSError``
            (any UMLS subclass — auth, transport, request) is logged and
            re-raised so callers can distinguish "no relations" from a
            transport failure (codex H1 from PR #102 review).
        """
        try:
            rows = self.umls.cui_relations(cui)
        except UMLSError as exc:
            # ``UMLSAuthError`` is a UMLSError subclass and stays in this
            # branch; both surface to the caller. Logging at warning level
            # preserves the prior observability contract.
            if not isinstance(exc, UMLSAuthError):
                logger.warning("UMLS cui_relations failed for %s: %s", cui, exc)
            raise
        predicate_set = {p.lower() for p in predicates} if predicates else None
        coarse_set = {c.upper() for c in include_coarse_labels} if include_coarse_labels else None
        edges: list[KGEdge] = []
        for row in rows:
            additional = (row.get("additionalRelationLabel") or "").lower()
            coarse = (row.get("relationLabel") or "").upper()
            related_url = row.get("relatedId") or ""
            related_cui = _extract_trailing_cui(related_url)
            if not related_cui:
                continue
            # Three accepted shapes: fine-grained predicate match,
            # coarse-label match when fine is empty, or no filter at all.
            match_fine = predicate_set is not None and additional in predicate_set
            match_coarse = coarse_set is not None and coarse in coarse_set and not additional
            no_filter = predicate_set is None and coarse_set is None
            if not (match_fine or match_coarse or no_filter):
                continue
            edges.append(
                KGEdge(
                    subject_id=cui,
                    predicate=additional or coarse.lower() or "related_to",
                    object_id=related_cui,
                    object_name=str(row.get("relatedIdName") or ""),
                    evidence_source="umls_relations",
                    datasource=row.get("rootSource"),
                    raw=row,
                )
            )
        return edges


def _extract_trailing_cui(related_url: str) -> str:
    """Pull the trailing CUI from a UTS ``relatedId`` URL.

    UTS ``relatedId`` looks like ``https://uts-ws.nlm.nih.gov/rest/content/2026AA/CUI/C0011615``.
    Returns the trailing segment if it begins with ``C`` and is otherwise
    plausibly a CUI; empty string otherwise.
    """
    if not isinstance(related_url, str) or not related_url:
        return ""
    tail = related_url.rstrip("/").split("/")[-1]
    if tail.startswith("C") and tail != "NONE":
        return tail
    return ""
