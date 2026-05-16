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
    - ``query_drug_disease_edges(drug_id, disease_id)`` — Open Targets
      evidence rows mapped to ``KGEdge`` records with PMID provenance and
      evidence score. ``drug_id`` accepts ChEMBL IDs (preferred) or RxCUIs;
      ``disease_id`` accepts EFO/MONDO IDs (preferred) or UMLS CUIs.
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
import math
from typing import Iterable, Optional

from src.data.kg.chembl import ChEMBLClient
from src.data.kg.entity_linker import EntityLinker
from src.data.kg.open_targets import OpenTargetsClient, OpenTargetsError
from src.data.kg.types import EvidenceItem, KGEdge
from src.data.kg.umls_uts import UMLSAuthError, UMLSClient, UMLSError

logger = logging.getLogger(__name__)


class KnowledgeGraphQuerier:
    """Structured KGEdge extraction over UMLS + Open Targets + ChEMBL.

    Args:
        umls: A constructed UMLSClient. If None and ``entity_linker`` is also
            None, a default client is constructed (which reads the
            ``UMLS_UTS_API_KEY`` env var).
        open_targets: A constructed OpenTargetsClient. Same fallback rules.
        chembl: Optional ChEMBLClient. When provided, drug-disease edges
            populate ``KGEdge.evidence[i].chembl_target_id`` by
            cross-walking the Open Targets target gene symbol → ChEMBL
            target ID. When ``None``, evidence is still threaded from
            Open Targets but ``chembl_target_id`` is left ``None`` on
            every item. v1 callers that pre-date #245 keep working
            unchanged.
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
        if entity_linker is not None:
            self.umls = umls if umls is not None else entity_linker.umls
            self.open_targets = (
                open_targets if open_targets is not None else entity_linker.open_targets
            )
        else:
            if umls is None:
                self.umls = UMLSClient()
                self._owns_umls = True
            else:
                self.umls = umls
            if open_targets is None:
                self.open_targets = OpenTargetsClient()
                self._owns_open_targets = True
            else:
                self.open_targets = open_targets
        # Optional ChEMBL client. Borrowed only — KGQuerier never
        # auto-constructs ChEMBL (would surprise existing callers).
        self.chembl: Optional[ChEMBLClient] = chembl

    def __enter__(self) -> "KnowledgeGraphQuerier":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close any clients KGQuerier itself constructed.

        Borrowed clients (passed in by the caller or via ``entity_linker``)
        are NOT closed here — their lifetime is the caller's responsibility.
        """
        if self._owns_umls:
            self.umls.close()
        if self._owns_open_targets:
            self.open_targets.close()
        if self._owns_chembl and self.chembl is not None:
            self.chembl.close()

    def query_drug_disease_edges(
        self,
        drug_id: str,
        disease_id: str,
        *,
        size: int = 25,
    ) -> list[KGEdge]:
        """Open Targets drug → disease evidence as ``KGEdge`` records.

        ``drug_id`` should be a ChEMBL ID (e.g., ``"CHEMBL1234"``) and
        ``disease_id`` should be an EFO/MONDO ID (e.g., ``"EFO_0000270"``).
        Caller is responsible for resolving CUIs/RxCUIs to ChEMBL/EFO via
        ``EntityLinker`` + ``OpenTargetsClient.search_drug``/``search_disease``
        before invoking this method.

        Each evidence row produces ONE edge whose:
            - subject_id   = drug ChEMBL ID
            - object_id    = disease EFO/MONDO ID
            - predicate    = ``"treats"`` when the row's
                             ``datatypeId == "known_drug"`` (Open Targets'
                             unique drug-indication datatype, Ochoa 2021
                             NAR), else ``"associated_with"``. The voter's
                             ``classify_kg_signal`` consumes the predicate
                             to drive the ``leak_drug_treats_disease``
                             classification — see PR-0 reconciliation
                             design (docs/superpowers/specs/2026-05-08-
                             kg-predicate-reconciliation-design.md).
            - evidence_source = ``"open_targets"``
            - score        = the evidence row's ``score`` (0–1)
            - pmids        = literature list (Europe PMC IDs)
            - datasource   = the row's ``datasourceId``

        Returns an empty list if Open Targets has no evidence. ``OpenTargetsError``
        is logged and re-raised so callers can distinguish "no evidence" from
        a transport / GraphQL failure (codex H1 from PR #102 review). Drug and
        disease names are populated from the response when available.
        """
        try:
            data = self.open_targets.drug_disease_evidence(drug_id, disease_id, size=size)
        except OpenTargetsError as exc:
            logger.warning(
                "Open Targets drug-disease query failed for %s/%s: %s",
                drug_id,
                disease_id,
                exc,
            )
            raise
        evidences = data.get("evidences", {})
        # ``evidences.rows`` is a GraphQL nullable-list field (`[Evidence!]`,
        # not `[Evidence!]!`), so the resolver may legitimately return null
        # on partial failure. ``dict.get("rows", [])`` returns the explicit
        # null value rather than the default ``[]``; ``or []`` collapses both
        # the absent-key AND null-value cases to an empty list. Without
        # this, ``for row in rows`` would raise ``TypeError: 'NoneType' is
        # not iterable``, propagating to downstream Phase 2.5 callers.
        rows = (evidences.get("rows") or []) if isinstance(evidences, dict) else []
        edges: list[KGEdge] = []
        for row in rows:
            drug = row.get("drug") or {}
            disease = row.get("disease") or {}
            literature = row.get("literature") or []
            pmids = tuple(str(p) for p in literature if p)
            score_raw = row.get("score")
            # ``isinstance(float('nan'), (int, float))`` is True, so a NaN
            # score (possible from numpy-backed JSON parsers or upstream
            # numeric corruption) would silently propagate into KGEdge.score.
            # NaN comparisons return False for all orderings, which would
            # poison Phase 2.5/2.6 selection logic (max/sort/threshold) by
            # making the broken edge invisible to ranking. ``math.isfinite``
            # rejects NaN and ±inf; both belong as ``None`` in the public
            # KGEdge contract.
            score = (
                float(score_raw)
                if isinstance(score_raw, (int, float)) and math.isfinite(score_raw)
                else None
            )
            # Open Targets datatypeId taxonomy (Ochoa 2021, NAR): the
            # ONLY datatype carrying drug-treats-disease semantics is
            # ``known_drug``. All other datatypes (literature,
            # genetic_association, affected_pathway, rna_expression,
            # somatic_mutation, animal_model) are gene/target-disease
            # association, not therapeutic claim. Keying on
            # ``datatypeId`` (the data-model invariant) rather than
            # ``datasourceId`` (a contributing-pipeline detail) is
            # future-proof: new sources Open Targets adds with
            # ``datatypeId="known_drug"`` (e.g., ``fda_label``,
            # ``clinical_trials_v2``) are picked up automatically.
            #
            # DEFERRED — phase-gating refinement (codex PR-0 review M1,
            # 2026-05-08). A ``known_drug`` row's
            # ``drug.indications.maxPhaseForIndication`` (already pulled
            # by the GraphQL query at ``open_targets.py:53``) ranges from
            # 0 (preclinical) to 4 (regulatory-approved indication). This
            # implementation emits ``predicate="treats"`` for ANY
            # known_drug row regardless of phase — so a Phase I
            # exploratory trial is treated identically to an FDA-
            # approved label. The voter's ``classify_kg_signal`` will
            # therefore promote a Phase I row to
            # ``leak_drug_treats_disease`` if it connects feature/target,
            # which can produce false-positive leak verdicts for
            # exploratory drug-disease pairings. A future PR should gate
            # ``predicate="treats"`` on
            # ``maxPhaseForIndication >= 4`` (or surface phase as a
            # KGEdge field for the voter to weight). Tracked in spec
            # ``docs/superpowers/specs/2026-05-08-kg-predicate-
            # reconciliation-design.md`` §"Out of scope (future work)".
            datatype_id = str(row.get("datatypeId") or "")
            predicate = "treats" if datatype_id == "known_drug" else "associated_with"
            # ----------------------------------------------------------
            # Issue #245: per-evidence-item provenance threading.
            # The KGEdge.pmids tuple remains a coarse list of PMIDs for
            # backwards compat; KGEdge.evidence carries the structured
            # per-PMID provenance with optional ChEMBL target cross-walk.
            # ----------------------------------------------------------
            chembl_target_id = self._resolve_chembl_target(row)
            evidence_items = tuple(
                EvidenceItem(
                    pmid=pmid,
                    source="open_targets",
                    chembl_target_id=chembl_target_id,
                    datasource_score=score,
                )
                for pmid in pmids
            )
            edges.append(
                KGEdge(
                    subject_id=str(drug.get("id") or drug_id),
                    subject_name=str(drug.get("name") or ""),
                    predicate=predicate,
                    object_id=str(disease.get("id") or disease_id),
                    object_name=str(disease.get("name") or ""),
                    evidence_source="open_targets",
                    score=score,
                    pmids=pmids,
                    datasource=row.get("datasourceId"),
                    evidence=evidence_items,
                    raw=row,
                )
            )
        return edges

    def _resolve_chembl_target(self, row: dict[str, object]) -> Optional[str]:
        """Resolve an Open Targets evidence row's target → ChEMBL target ID.

        Returns ``None`` when (a) no ChEMBL client is attached, (b) the
        row exposes no ``target.approvedSymbol``, or (c) the ChEMBL
        cross-walk returns no match. Errors from the ChEMBL client are
        logged at warning level and swallowed to ``None`` so a transient
        ChEMBL outage does not break the Open Targets evidence path
        (the v1 contract is "ChEMBL enrichment is best-effort, not a
        gating dependency").
        """
        if self.chembl is None:
            return None
        target = row.get("target")
        if not isinstance(target, dict):
            return None
        gene_symbol = target.get("approvedSymbol")
        if not isinstance(gene_symbol, str) or not gene_symbol:
            return None
        try:
            return self.chembl.open_targets_target_to_chembl(gene_symbol)
        except Exception as exc:  # noqa: BLE001 — best-effort enrichment
            logger.warning("ChEMBL cross-walk failed for gene %s: %s", gene_symbol, exc)
            return None

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
