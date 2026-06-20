"""Layer 2 — Knowledge-Graph clients and EntityLinker.

This package implements Phase 2.1 of the adaptive temporal-validity redesign
(`.claude/plans/adaptive_temporal_validity_redesign.md`). The v1 ontology stack
was researched and recorded in
``~/.claude/projects/-home-enunez-Projects-e2i-causal-analytics/memory/layer2_kg_ontology_recommendation_20260507.md``:

- ``umls_uts``     — canonical cross-walk for ICD-10/RxCUI/LOINC/CPT/HCPCS ↔ UMLS CUI
- ``open_targets`` — drug-disease evidence with Europe PMC PMID provenance
- ``rxnav``        — drug-name normalization to RxCUI

``EntityLinker`` composes these clients and exposes the single public surface
the rest of Layer 2 (``CausalRoleClassifier``, ``CitationResolver``,
``EnsembleVoter``) consumes: code → ``EntityLink`` records.
"""

from src.data.kg.adversarial_probe import (
    AdversarialProbe,
    AdversarialProbeError,
)
from src.data.kg.chembl import (
    Activity,
    ChEMBLClient,
    ChEMBLError,
    Mechanism,
)
from src.data.kg.citation_resolver import (
    CAUSAL_CUE_VERBS,
    CitationResolver,
    CitationResolverError,
)
from src.data.kg.crossref import CrossrefClient, CrossrefError
from src.data.kg.ensemble_voter import (
    EnsembleVoter,
    classify_kg_signal,
    is_citation_verified,
)
from src.data.kg.entity_linker import EntityLinker, EntityLinkerError
from src.data.kg.europe_pmc import EuropePMCClient, EuropePMCError
from src.data.kg.kg_querier import KnowledgeGraphQuerier
from src.data.kg.open_targets import OpenTargetsClient, OpenTargetsError
from src.data.kg.rxnav import RxCUIMatch, RxNavClient, RxNavError
from src.data.kg.types import (
    AbstractRecord,
    AdversarialProbeResult,
    CausalRole,
    CitationVerdict,
    EnsembleDecidedBy,
    EnsembleSeverity,
    EnsembleVerdict,
    EntityLink,
    EvidenceItem,
    EvidenceSource,
    KGConcept,
    KGEdge,
    KGSignal,
    LLMVerdict,
    ProbeOutcome,
    Remediation,
)
from src.data.kg.umls_uts import (
    UMLSAuthError,
    UMLSClient,
    UMLSError,
    UMLSNotFoundError,
)

__all__ = [
    "AbstractRecord",
    "Activity",
    "AdversarialProbe",
    "AdversarialProbeError",
    "AdversarialProbeResult",
    "CAUSAL_CUE_VERBS",
    "CausalRole",
    "ChEMBLClient",
    "ChEMBLError",
    "Mechanism",
    "CitationResolver",
    "CitationResolverError",
    "CitationVerdict",
    "CrossrefClient",
    "CrossrefError",
    "EnsembleDecidedBy",
    "EnsembleSeverity",
    "EnsembleVerdict",
    "EnsembleVoter",
    "EntityLink",
    "EntityLinker",
    "EntityLinkerError",
    "EuropePMCClient",
    "EuropePMCError",
    "EvidenceItem",
    "EvidenceSource",
    "KGConcept",
    "KGEdge",
    "KGSignal",
    "KnowledgeGraphQuerier",
    "LLMVerdict",
    "OpenTargetsClient",
    "OpenTargetsError",
    "ProbeOutcome",
    "Remediation",
    "RxCUIMatch",
    "RxNavClient",
    "RxNavError",
    "UMLSAuthError",
    "UMLSClient",
    "UMLSError",
    "UMLSNotFoundError",
    "classify_kg_signal",
    "is_citation_verified",
]
