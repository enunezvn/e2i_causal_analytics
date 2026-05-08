"""Shared dataclasses for Layer 2 KG clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

EvidenceSource = Literal[
    "open_targets",
    "umls_relations",
    "rxnav",
    "europe_pmc",
    "manual",
]

CodeSystem = Literal[
    "ICD10CM",
    "ICD10",
    "RXNORM",
    "LOINC",
    "CPT",
    "HCPCS",
    "SNOMEDCT_US",
    "MESH",
]


@dataclass(frozen=True)
class KGConcept:
    """A single UMLS concept after cross-walk.

    Returned by both ``UMLSClient.cui_lookup`` and as the canonical payload
    inside ``EntityLink.concept``. ``semantic_types`` and ``atom_count`` come
    from the UTS ``content/CUI`` endpoint; ``preferred_name`` is the canonical
    English label.
    """

    cui: str
    preferred_name: str
    semantic_types: tuple[str, ...] = ()
    atom_count: Optional[int] = None


@dataclass(frozen=True)
class EntityLink:
    """The result of resolving a single code → UMLS concept.

    ``input_code`` and ``input_system`` are the caller's inputs. ``concept`` is
    None when no UMLS concept maps from the code; ``error`` captures why.
    ``sources`` lists which UTS source vocabularies the cross-walk traversed
    (helps audit "this CSU ICD-10 code resolved via SNOMEDCT_US"). The
    distinction between "no result" (``concept is None`` and ``error is None``)
    and "API error" (``error`` populated) lets the caller decide whether to
    retry or accept the absence.
    """

    input_code: str
    input_system: CodeSystem
    concept: Optional[KGConcept] = None
    sources: tuple[str, ...] = ()
    error: Optional[str] = None
    raw: Optional[dict] = field(default=None, repr=False)

    @property
    def resolved(self) -> bool:
        return self.concept is not None


@dataclass(frozen=True)
class KGEdge:
    """A single Subject–Predicate–Object triple with provenance.

    The output of Phase 2.3 ``KnowledgeGraphQuerier``. Every edge is
    grounded to specific UMLS CUIs (or external IDs like ChEMBL/EFO that
    can be cross-walked back to a CUI) and carries the evidence trail that
    Phase 2.6 ``CitationResolver`` will verify.

    Attributes:
        subject_id: The subject of the triple (UMLS CUI or external ID).
        subject_name: Human-readable label, populated when known.
        predicate: The relation type. Open Targets evidence rows produce
            edges with predicates like ``"treats"`` or ``"indicated_for"``;
            UMLS relations produce predicates like ``"is_a"``,
            ``"has_finding_site"``, ``"part_of"``.
        object_id: The object of the triple.
        object_name: Human-readable label, populated when known.
        evidence_source: Which client produced this edge.
        score: Optional 0–1 confidence/evidence score (Open Targets supplies
            one per evidence row; UMLS relations don't).
        pmids: Tuple of PubMed IDs that document the relation. Empty when
            the source doesn't carry literature provenance.
        datasource: Sub-source identifier (e.g., Open Targets'
            ``datasourceId``: "europepmc", "chembl", "clinical_trials").
    """

    subject_id: str
    predicate: str
    object_id: str
    evidence_source: EvidenceSource
    subject_name: str = ""
    object_name: str = ""
    score: Optional[float] = None
    pmids: tuple[str, ...] = ()
    datasource: Optional[str] = None
    raw: Optional[dict] = field(default=None, repr=False)
