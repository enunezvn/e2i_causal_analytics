"""Shared dataclasses for Layer 2 KG clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

CausalRole = Literal[
    "ancestor",
    "confounder",
    "instrument",
    "mediator",
    "collider",
    "descendant",
]

Remediation = Literal[
    "drop",
    "window",
    "transform",
    "keep_with_caveat",
    "keep",
    "review",
]

EnsembleSeverity = Literal["high", "moderate", "info", "abstain"]

EnsembleDecidedBy = Literal["layer_1", "adversarial", "kg", "llm", "abstain"]

KGSignal = Literal[
    "leak_drug_treats_disease",
    "taxonomic_descendant",
    "no_signal",
    "contradictory",
]

EvidenceSource = Literal[
    "open_targets",
    "umls_relations",
    "rxnav",
    "europe_pmc",
    "crossref",
    "manual",
]


@dataclass(frozen=True)
class AbstractRecord:
    """A retrieved scientific publication abstract.

    Returned by ``EuropePMCClient.fetch_abstract`` and
    ``CrossrefClient.fetch_doi_metadata``. ``CitationResolver`` then runs
    entity-presence and causal-cue verification over the ``abstract``
    text.

    The ``identifier`` field is whichever of (PMID, DOI) the caller used to
    fetch the record; ``identifier_kind`` records which.
    """

    identifier: str
    identifier_kind: Literal["pmid", "doi"]
    title: str
    abstract: str
    source: Literal["europe_pmc", "crossref"]
    journal: Optional[str] = None
    year: Optional[int] = None
    raw: Optional[dict] = field(default=None, repr=False)


@dataclass(frozen=True)
class CitationVerdict:
    """Verification record for a single PMID/DOI cited as evidence for a
    subject-object relation.

    A citation passes when:
        1. The abstract was successfully resolved (``abstract_resolved``).
        2. Both the subject and object entities (or any of their UMLS
           synonyms) appear in the abstract text (``entities_found`` carries
           the matched terms).
        3. At least one causal cue verb from ``CAUSAL_CUE_VERBS`` appears in
           the abstract (``causal_cue_found`` is the first matched verb).

    ``overall_confidence`` is a 0-1 score that aggregates the three factors;
    callers should treat it as a relative ranking signal, not an absolute
    threshold.

    ``identifier_kind`` widened to ``str`` (was ``Literal["pmid", "doi"]``)
    so that error verdicts produced when the caller supplies an unsupported
    kind preserve the original input rather than masquerading as a PMID.
    Successful verdicts still always carry ``"pmid"`` or ``"doi"``;
    consumers should check ``error is None`` before trusting the kind.
    """

    identifier: str
    identifier_kind: str
    abstract_resolved: bool
    entities_found: tuple[str, ...] = ()
    causal_cue_found: Optional[str] = None
    overall_confidence: float = 0.0
    error: Optional[str] = None


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

    ``confidence`` is a 0-1 score capturing how much we trust the resolution.
    Sources of uncertainty:
        - RxNav approximate-match fallback for drug names (e.g., typos
          getting silently corrected) → confidence < 1.0.
        - UMLS free-text search results past the first hit → confidence < 1.0.
        - Direct source-code → CUI cross-walks via UTS exact match → 1.0.
    Phase 2.6 ``CitationResolver`` consumes this when ranking competing
    EntityLinks; values of None mean "exact match" (full confidence).
    """

    input_code: str
    input_system: CodeSystem
    concept: Optional[KGConcept] = None
    sources: tuple[str, ...] = ()
    error: Optional[str] = None
    confidence: Optional[float] = None
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


@dataclass(frozen=True)
class LLMVerdict:
    """Minimal mirror of `CausalRoleClassifier`'s prediction.

    The DSPy-produced `dspy.Prediction` object carries `causal_role`,
    `mechanism`, and `recommended_remediation` fields. Phase 2.7
    `EnsembleVoter` accepts this lightweight dataclass instead so unit
    tests can construct verdicts without instantiating an LM, and so the
    voter has no `dspy` import.

    Callers that already have a `dspy.Prediction` should adapt at the
    call site:

        verdict = LLMVerdict(
            causal_role=prediction.causal_role,
            mechanism=prediction.mechanism,
            recommended_remediation=prediction.recommended_remediation,
            cited_pmids=tuple(extract_pmids(prediction.mechanism)),
        )

    `cited_pmids` is whichever PMIDs the caller chose to extract from
    `mechanism` for citation verification by `CitationResolver`. The
    voter does not parse `mechanism` itself; it consumes the
    pre-extracted tuple alongside the matching `CitationVerdict`s.
    """

    causal_role: CausalRole
    mechanism: str
    recommended_remediation: Remediation
    cited_pmids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnsembleVerdict:
    """Phase 2.7 final verdict for one feature.

    Composed by `EnsembleVoter.vote` from up to four upstream verdicts
    (Layer 1 manifest contract, Layer 3 adversarial probe, Layer 2 KG
    edges, Layer 4 LLM classification) plus citation verification.
    Carries the full audit trail required by acceptance criterion #4 of
    the adaptive temporal-validity redesign: every feature decision must
    have a structured record naming the deciding layer, the evidence,
    and the upstream verdicts considered.

    Attributes:
        feature_name: The feature this verdict pertains to.
        severity: ``"high"`` / ``"moderate"`` / ``"info"`` / ``"abstain"``.
            ``"high"`` means the feature must not enter the model.
            ``"abstain"`` means the voter could not reach a verdict and
            the feature is queued for human-in-the-loop adjudication.
        remediation: Recommended action: ``"drop"``, ``"window"``,
            ``"transform"``, ``"keep_with_caveat"``, ``"keep"``, or
            ``"review"`` (used with ``severity="abstain"``).
        decided_by: Which upstream layer drove the verdict. ``"abstain"``
            when no layer produced a confident signal.
        final_role: The causal role this feature played (only when an
            upstream layer determined one — Layer 1/Adversarial vetoes
            do not assign a role beyond "this is a leak"; the voter
            populates ``"descendant"`` as the conventional role for
            those cases). ``None`` when abstaining.
        confidence: Aggregate 0–1 confidence in the verdict. Layer 1
            deterministic vetoes carry confidence 1.0; adversarial
            vetoes 0.95; LLM-driven verdicts modulated by KG agreement
            and citation verification (see module docstring for the
            exact weights). 0.0 on abstain.
        kg_signal: Coarse classification of the KG edges considered.
        kg_edges_considered: The subset of KG edges relevant to the
            feature/target pair (filtered by `_classify_kg_signal`).
        verified_citations: Citation verdicts whose abstract resolved
            AND found both entities AND found a causal cue (i.e.,
            `overall_confidence` >= the both-entities + cue threshold).
        unverified_citations: Citation verdicts that failed any of the
            above checks (resolution failure, missing entities, or
            missing causal cue).
        disagreements: Free-text log of contradictions across upstream
            verdicts. Empty tuple when sources agreed (or when only one
            source spoke).
        evidence: Free-text log of corroborating evidence. Always
            populated — even abstain verdicts record why the voter
            couldn't decide.
        layer_1_input: Raw Layer 1 verdict dict (or None).
        adversarial_input: Raw adversarial verdict dict (or None).
        llm_input: Raw LLM verdict (or None).
    """

    feature_name: str
    severity: EnsembleSeverity
    remediation: Remediation
    decided_by: EnsembleDecidedBy
    confidence: float
    final_role: Optional[CausalRole] = None
    kg_signal: KGSignal = "no_signal"
    kg_edges_considered: tuple[KGEdge, ...] = ()
    verified_citations: tuple[CitationVerdict, ...] = ()
    unverified_citations: tuple[CitationVerdict, ...] = ()
    disagreements: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    layer_1_input: Optional[dict] = field(default=None, repr=False)
    adversarial_input: Optional[dict] = field(default=None, repr=False)
    llm_input: Optional[LLMVerdict] = field(default=None, repr=False)
