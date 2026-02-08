"""
E2I Graphiti Configuration
Entity type mappings and configuration loader for Graphiti integration.

Usage:
    from src.memory.graphiti_config import (
        E2IEntityType,
        E2IRelationshipType,
        get_graphiti_config,
    )

    config = get_graphiti_config()
    entity_types = config.entity_types
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "005_memory_config.yaml"


class E2IEntityType(str, Enum):
    """E2I entity types for knowledge graph."""

    PATIENT = "Patient"
    HCP = "HCP"
    BRAND = "Brand"
    REGION = "Region"
    KPI = "KPI"
    CAUSAL_PATH = "CausalPath"
    TRIGGER = "Trigger"
    AGENT = "Agent"
    # Graphiti-specific types
    EPISODE = "Episode"
    COMMUNITY = "Community"


class E2IRelationshipType(str, Enum):
    """E2I relationship types for knowledge graph."""

    TREATED_BY = "TREATED_BY"
    PRESCRIBED = "PRESCRIBED"
    PRESCRIBES = "PRESCRIBES"
    CAUSES = "CAUSES"
    IMPACTS = "IMPACTS"
    INFLUENCES = "INFLUENCES"
    DISCOVERED = "DISCOVERED"
    GENERATED = "GENERATED"
    # Graphiti-specific relationships
    MENTIONS = "MENTIONS"
    MEMBER_OF = "MEMBER_OF"
    RELATES_TO = "RELATES_TO"


@dataclass
class GraphitiEntityConfig:
    """Configuration for a single entity type."""

    name: str
    label: str
    description: str = ""
    properties: List[str] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)


@dataclass
class GraphitiRelationshipConfig:
    """Configuration for a single relationship type."""

    name: str
    label: str
    description: str = ""
    properties: List[str] = field(default_factory=list)


@dataclass
class GraphitiConfig:
    """
    Main Graphiti configuration container.
    Provides typed access to entity and relationship configurations.
    """

    enabled: bool = True
    model: str = "claude-3-5-sonnet-latest"  # Sonnet for Graphiti operations
    graph_name: str = "e2i_semantic"

    # Entity configurations
    entity_types: List[E2IEntityType] = field(default_factory=lambda: list(E2IEntityType))
    entity_configs: Dict[str, GraphitiEntityConfig] = field(default_factory=dict)

    # Relationship configurations
    relationship_types: List[E2IRelationshipType] = field(
        default_factory=lambda: list(E2IRelationshipType)
    )
    relationship_configs: Dict[str, GraphitiRelationshipConfig] = field(default_factory=dict)

    # Episode settings
    episode_batch_size: int = 5
    min_confidence_for_extraction: float = 0.6

    # Connection settings
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379  # 6379 internal (docker), 6381 external (host)
    falkordb_password: Optional[str] = None

    # Cache settings
    cache_enabled: bool = True
    cache_ttl_minutes: int = 5

    def get_entity_label(self, entity_type: E2IEntityType) -> str:
        """Get the graph label for an entity type."""
        if entity_type.value in self.entity_configs:
            return self.entity_configs[entity_type.value].label
        return entity_type.value

    def get_relationship_label(self, rel_type: E2IRelationshipType) -> str:
        """Get the graph label for a relationship type."""
        if rel_type.value in self.relationship_configs:
            return self.relationship_configs[rel_type.value].label
        return rel_type.value


# Default entity configurations with E2I-specific properties
DEFAULT_ENTITY_CONFIGS = {
    E2IEntityType.PATIENT.value: GraphitiEntityConfig(
        name="Patient",
        label="Patient",
        description="Patient in the healthcare system",
        properties=["patient_id", "condition", "treatment_status", "journey_stage"],
        indexes=["patient_id"],
    ),
    E2IEntityType.HCP.value: GraphitiEntityConfig(
        name="HCP",
        label="HCP",
        description="Healthcare Provider (physician, specialist)",
        properties=["hcp_id", "npi", "specialty", "region", "tier"],
        indexes=["hcp_id", "npi"],
    ),
    E2IEntityType.BRAND.value: GraphitiEntityConfig(
        name="Brand",
        label="Brand",
        description="Pharmaceutical brand/product",
        properties=["brand_name", "indication", "therapeutic_area", "launch_date"],
        indexes=["brand_name"],
    ),
    E2IEntityType.REGION.value: GraphitiEntityConfig(
        name="Region",
        label="Region",
        description="Geographic region or territory",
        properties=["region_id", "region_name", "parent_region"],
        indexes=["region_id"],
    ),
    E2IEntityType.KPI.value: GraphitiEntityConfig(
        name="KPI",
        label="KPI",
        description="Key Performance Indicator",
        properties=["kpi_name", "kpi_type", "unit", "target_value"],
        indexes=["kpi_name"],
    ),
    E2IEntityType.CAUSAL_PATH.value: GraphitiEntityConfig(
        name="CausalPath",
        label="CausalPath",
        description="Discovered causal relationship chain",
        properties=["path_id", "source", "target", "effect_size", "confidence"],
        indexes=["path_id"],
    ),
    E2IEntityType.TRIGGER.value: GraphitiEntityConfig(
        name="Trigger",
        label="Trigger",
        description="Event or condition that triggers an action",
        properties=["trigger_id", "trigger_type", "condition", "action"],
        indexes=["trigger_id"],
    ),
    E2IEntityType.AGENT.value: GraphitiEntityConfig(
        name="Agent",
        label="Agent",
        description="E2I AI agent",
        properties=["agent_name", "tier", "capabilities"],
        indexes=["agent_name"],
    ),
    E2IEntityType.EPISODE.value: GraphitiEntityConfig(
        name="Episode",
        label="Episode",
        description="Graphiti temporal episode",
        properties=["episode_id", "content", "source", "session_id", "valid_at", "invalid_at"],
        indexes=["episode_id", "session_id"],
    ),
    E2IEntityType.COMMUNITY.value: GraphitiEntityConfig(
        name="Community",
        label="Community",
        description="Graphiti entity community/cluster",
        properties=["community_id", "name", "summary", "member_count"],
        indexes=["community_id"],
    ),
}


# Default relationship configurations
DEFAULT_RELATIONSHIP_CONFIGS = {
    E2IRelationshipType.TREATED_BY.value: GraphitiRelationshipConfig(
        name="TREATED_BY",
        label="TREATED_BY",
        description="Patient treated by HCP",
        properties=["start_date", "end_date", "treatment_type"],
    ),
    E2IRelationshipType.PRESCRIBED.value: GraphitiRelationshipConfig(
        name="PRESCRIBED",
        label="PRESCRIBED",
        description="Patient prescribed a brand",
        properties=["prescription_date", "dosage", "duration"],
    ),
    E2IRelationshipType.PRESCRIBES.value: GraphitiRelationshipConfig(
        name="PRESCRIBES",
        label="PRESCRIBES",
        description="HCP prescribes a brand",
        properties=["frequency", "preference_score"],
    ),
    E2IRelationshipType.CAUSES.value: GraphitiRelationshipConfig(
        name="CAUSES",
        label="CAUSES",
        description="Causal relationship between entities",
        properties=["effect_size", "confidence", "p_value", "mechanism"],
    ),
    E2IRelationshipType.IMPACTS.value: GraphitiRelationshipConfig(
        name="IMPACTS",
        label="IMPACTS",
        description="Entity impacts a KPI or outcome",
        properties=["impact_direction", "impact_magnitude", "lag_days"],
    ),
    E2IRelationshipType.INFLUENCES.value: GraphitiRelationshipConfig(
        name="INFLUENCES",
        label="INFLUENCES",
        description="Entity influences another entity",
        properties=["influence_type", "strength", "context"],
    ),
    E2IRelationshipType.DISCOVERED.value: GraphitiRelationshipConfig(
        name="DISCOVERED",
        label="DISCOVERED",
        description="Agent discovered a relationship or insight",
        properties=["discovery_date", "method", "confidence"],
    ),
    E2IRelationshipType.GENERATED.value: GraphitiRelationshipConfig(
        name="GENERATED",
        label="GENERATED",
        description="Agent generated an output or analysis",
        properties=["generation_date", "output_type"],
    ),
    E2IRelationshipType.MENTIONS.value: GraphitiRelationshipConfig(
        name="MENTIONS",
        label="MENTIONS",
        description="Episode mentions an entity (Graphiti)",
        properties=["confidence", "context", "position"],
    ),
    E2IRelationshipType.MEMBER_OF.value: GraphitiRelationshipConfig(
        name="MEMBER_OF",
        label="MEMBER_OF",
        description="Entity is member of a community (Graphiti)",
        properties=["membership_score", "role"],
    ),
    E2IRelationshipType.RELATES_TO.value: GraphitiRelationshipConfig(
        name="RELATES_TO",
        label="RELATES_TO",
        description="General relationship between entities",
        properties=["relationship_type", "weight", "context"],
    ),
}


def load_graphiti_config(config_path: Optional[Path] = None) -> GraphitiConfig:
    """
    Load Graphiti configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config/005_memory_config.yaml

    Returns:
        GraphitiConfig with all parsed settings
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Load raw YAML config
    if config_path.exists():
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        raw_config = {}

    # Get environment
    environment = os.environ.get("E2I_ENVIRONMENT", raw_config.get("environment", "local_pilot"))

    # Extract semantic/graphity section
    semantic_config = raw_config.get("memory_backends", {}).get("semantic", {}).get(environment, {})
    graphity_config = semantic_config.get("graphity", {})
    cache_config = semantic_config.get("cache", {})

    # Parse entity types from config
    entity_type_names = graphity_config.get("entity_types", [e.value for e in E2IEntityType])
    entity_types = []
    for name in entity_type_names:
        try:
            entity_types.append(E2IEntityType(name))
        except ValueError:
            logger.warning(f"Unknown entity type in config: {name}")

    # Parse relationship types from config
    rel_type_names = graphity_config.get(
        "relationship_types", [r.value for r in E2IRelationshipType]
    )
    relationship_types = []
    for name in rel_type_names:
        try:
            relationship_types.append(E2IRelationshipType(name))
        except ValueError:
            logger.warning(f"Unknown relationship type in config: {name}")

    # Get reflector settings for extraction confidence
    reflector_config = raw_config.get("cognitive_workflow", {}).get("reflector", {})

    # Build config
    config = GraphitiConfig(
        enabled=graphity_config.get("enabled", True),
        model=graphity_config.get("model", "claude-3-5-sonnet-20241022"),
        graph_name=semantic_config.get("graph_name", "e2i_semantic"),
        entity_types=entity_types or list(E2IEntityType),
        entity_configs=DEFAULT_ENTITY_CONFIGS.copy(),
        relationship_types=relationship_types or list(E2IRelationshipType),
        relationship_configs=DEFAULT_RELATIONSHIP_CONFIGS.copy(),
        episode_batch_size=reflector_config.get("graphity_batch_size", 5),
        min_confidence_for_extraction=reflector_config.get("min_confidence_for_learning", 0.6),
        falkordb_host=os.environ.get("FALKORDB_HOST", "localhost"),
        falkordb_port=int(os.environ.get("FALKORDB_PORT", "6379")),
        falkordb_password=os.environ.get("FALKORDB_PASSWORD"),
        cache_enabled=cache_config.get("enabled", True),
        cache_ttl_minutes=raw_config.get("performance", {})
        .get("cache", {})
        .get("semantic_cache_ttl_minutes", 5),
    )

    logger.info(
        f"Loaded Graphiti config: {len(config.entity_types)} entity types, {len(config.relationship_types)} relationship types"
    )
    return config


_graphiti_config: Optional[GraphitiConfig] = None


def get_graphiti_config() -> GraphitiConfig:
    """
    Get the cached Graphiti configuration singleton.

    Returns:
        GraphitiConfig singleton
    """
    global _graphiti_config
    if _graphiti_config is None:
        _graphiti_config = load_graphiti_config()
    return _graphiti_config


def clear_graphiti_config_cache() -> None:
    """Clear the config cache to force a reload."""
    global _graphiti_config
    _graphiti_config = None
    logger.info("Graphiti config cache cleared")
