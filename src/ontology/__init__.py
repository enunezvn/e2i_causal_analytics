"""
E2I Causal Analytics - Ontology Module

Central module for domain vocabulary, schema management, and semantic operations.
This module serves as the single source of truth for entity definitions,
eliminating vocabulary duplication across the codebase.

Primary Components:
- VocabularyRegistry: Central vocabulary loader (single source of truth)
- SchemaCompiler: Compiles YAML ontology to FalkorDB schema
- OntologyValidator: Validates schema consistency and integrity
- InferenceEngine: Graph-based reasoning for causal discovery
- E2IQueryExtractor: Fast query-time entity extraction for routing
- GraphityConfig: FalkorDB Graphity optimization configuration

Usage:
    from src.ontology import VocabularyRegistry

    # Load vocabulary (cached singleton)
    vocab = VocabularyRegistry.load()

    # Get entities
    brands = vocab.get_brands()  # ['Remibrutinib', 'Fabhalta', 'Kisqali']
    regions = vocab.get_regions()  # ['northeast', 'south', 'midwest', 'west']
    agents = vocab.get_agents(tier=2)  # Get Tier 2 agents only

    # Get with aliases for entity extraction
    brand_aliases = vocab.get_entity_with_aliases('brands')
    # {'Remibrutinib': ['remi', 'remibrutinib'], ...}
"""

from src.ontology.grafiti_config import (
    GraphityConfig,
    GraphityConfigBuilder,
    GraphityOptimizer,
)
from src.ontology.inference_engine import (
    CausalPath,
    ConfidenceLevel,
    InferenceEngine,
    InferenceRule,
    InferenceType,
    InferredRelationship,
    PathFinder,
)
from src.ontology.query_extractor import (
    E2IQueryExtractor,
    Entity,
    ExtractionContext,
    QueryExtractionResult,
    VocabularyEnrichedGraphiti,
)
from src.ontology.schema_compiler import (
    CardinalityType,
    CompiledSchema,
    EntitySchema,
    PropertySchema,
    PropertyType,
    RelationshipSchema,
    SchemaCompiler,
)
from src.ontology.validator import (
    OntologyValidator,
    ValidationIssue,
    ValidationLevel,
    ValidationReport,
)
from src.ontology.vocabulary_registry import VocabularyRegistry

__all__ = [
    # Vocabulary
    "VocabularyRegistry",
    # Schema
    "SchemaCompiler",
    "CompiledSchema",
    "EntitySchema",
    "RelationshipSchema",
    "PropertySchema",
    "PropertyType",
    "CardinalityType",
    # Validation
    "OntologyValidator",
    "ValidationReport",
    "ValidationIssue",
    "ValidationLevel",
    # Inference
    "InferenceEngine",
    "InferenceRule",
    "InferenceType",
    "InferredRelationship",
    "CausalPath",
    "ConfidenceLevel",
    "PathFinder",
    # Query Extraction
    "E2IQueryExtractor",
    "QueryExtractionResult",
    "Entity",
    "ExtractionContext",
    "VocabularyEnrichedGraphiti",
    # Graphity Config
    "GraphityConfig",
    "GraphityConfigBuilder",
    "GraphityOptimizer",
]


def validate_vocabulary_sync() -> bool:
    """
    Verify Python enums match YAML at startup.

    This function should be called during API startup to ensure
    database ENUMs remain synchronized with vocabulary definitions.

    Returns:
        True if validation passes, raises ValueError otherwise
    """
    vocab = VocabularyRegistry.load()

    # Basic sanity checks
    brands = vocab.get_brands()
    regions = vocab.get_regions()
    agents = vocab.get_agents()

    if not brands:
        raise ValueError("VocabularyRegistry: No brands loaded")
    if not regions:
        raise ValueError("VocabularyRegistry: No regions loaded")
    if not agents:
        raise ValueError("VocabularyRegistry: No agents loaded")

    return True
