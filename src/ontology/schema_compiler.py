"""
E2I Causal Analytics - Ontology Schema Compiler
Compiles YAML ontology definitions into FalkorDB schema with Graphity optimizations
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class CardinalityType(Enum):
    """Relationship cardinality types"""

    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_MANY = "M:N"


class PropertyType(Enum):
    """Supported property types in FalkorDB"""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    JSON = "json"
    ARRAY = "array"


@dataclass
class PropertySchema:
    """Schema definition for entity/relationship properties"""

    name: str
    property_type: PropertyType
    required: bool = False
    indexed: bool = False
    unique: bool = False
    default: Optional[Any] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class EntitySchema:
    """Schema definition for graph entities"""

    label: str
    properties: List[PropertySchema]
    primary_key: str
    indexes: List[str] = field(default_factory=list)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipSchema:
    """Schema definition for graph relationships"""

    type: str
    from_label: str
    to_label: str
    properties: List[PropertySchema]
    cardinality: CardinalityType
    bidirectional: bool = False
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledSchema:
    """Complete compiled ontology schema"""

    entities: Dict[str, EntitySchema]
    relationships: Dict[str, RelationshipSchema]
    constraints: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    graphity_config: Dict[str, Any]
    version: str
    metadata: Dict[str, Any]


class SchemaCompiler:
    """
    Compiles YAML ontology definitions into FalkorDB-compatible schema
    with Graphity optimizations for the E2I semantic memory layer.
    """

    def __init__(self, ontology_dir: Path):
        """
        Initialize compiler with ontology directory

        Args:
            ontology_dir: Directory containing YAML ontology files
        """
        self.ontology_dir = Path(ontology_dir)
        self.entities: Dict[str, EntitySchema] = {}
        self.relationships: Dict[str, RelationshipSchema] = {}
        self.constraints: List[Dict[str, Any]] = []
        self.indexes: List[Dict[str, Any]] = []

    def compile(self) -> CompiledSchema:
        """
        Compile all ontology YAML files into unified schema

        Returns:
            CompiledSchema with all entities, relationships, and constraints
        """
        logger.info(f"Compiling ontology from {self.ontology_dir}")

        # Load all YAML files
        schema_files = list(self.ontology_dir.glob("*.yaml"))
        logger.info(f"Found {len(schema_files)} schema files")

        for schema_file in schema_files:
            self._load_schema_file(schema_file)

        # Validate cross-references
        self._validate_references()

        # Generate constraints
        self._generate_constraints()

        # Generate indexes
        self._generate_indexes()

        # Generate Graphity optimization config
        graphity_config = self._generate_graphity_config()

        compiled = CompiledSchema(
            entities=self.entities,
            relationships=self.relationships,
            constraints=self.constraints,
            indexes=self.indexes,
            graphity_config=graphity_config,
            version="1.0",
            metadata={"compiled_files": [f.name for f in schema_files]},
        )

        logger.info(
            f"Compilation complete: {len(self.entities)} entities, "
            f"{len(self.relationships)} relationships"
        )

        return compiled

    def _load_schema_file(self, filepath: Path) -> None:
        """Load and parse a single YAML schema file"""
        logger.debug(f"Loading {filepath.name}")

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        # Load entities
        if "entities" in data:
            for entity_def in data["entities"]:
                entity = self._parse_entity(entity_def)
                self.entities[entity.label] = entity

        # Load relationships
        if "relationships" in data:
            for rel_def in data["relationships"]:
                relationship = self._parse_relationship(rel_def)
                self.relationships[relationship.type] = relationship

    def _parse_entity(self, entity_def: Dict[str, Any]) -> EntitySchema:
        """Parse entity definition from YAML"""
        properties = []
        for prop_def in entity_def.get("properties", []):
            prop = PropertySchema(
                name=prop_def["name"],
                property_type=PropertyType(prop_def["type"]),
                required=prop_def.get("required", False),
                indexed=prop_def.get("indexed", False),
                unique=prop_def.get("unique", False),
                default=prop_def.get("default"),
                constraints=prop_def.get("constraints", {}),
                description=prop_def.get("description"),
            )
            properties.append(prop)

        return EntitySchema(
            label=entity_def["label"],
            properties=properties,
            primary_key=entity_def["primary_key"],
            indexes=entity_def.get("indexes", []),
            description=entity_def.get("description"),
            metadata=entity_def.get("metadata", {}),
        )

    def _parse_relationship(self, rel_def: Dict[str, Any]) -> RelationshipSchema:
        """Parse relationship definition from YAML"""
        properties = []
        for prop_def in rel_def.get("properties", []):
            prop = PropertySchema(
                name=prop_def["name"],
                property_type=PropertyType(prop_def["type"]),
                required=prop_def.get("required", False),
                indexed=prop_def.get("indexed", False),
                description=prop_def.get("description"),
            )
            properties.append(prop)

        return RelationshipSchema(
            type=rel_def["type"],
            from_label=rel_def["from"],
            to_label=rel_def["to"],
            properties=properties,
            cardinality=CardinalityType(rel_def.get("cardinality", "1:N")),
            bidirectional=rel_def.get("bidirectional", False),
            description=rel_def.get("description"),
            metadata=rel_def.get("metadata", {}),
        )

    def _validate_references(self) -> None:
        """Validate that all relationship endpoints reference existing entities"""
        errors = []

        for rel_type, rel in self.relationships.items():
            if rel.from_label not in self.entities:
                errors.append(
                    f"Relationship '{rel_type}' references unknown entity '{rel.from_label}'"
                )

            if rel.to_label not in self.entities:
                errors.append(
                    f"Relationship '{rel_type}' references unknown entity '{rel.to_label}'"
                )

        if errors:
            raise ValueError("Schema validation failed:\n" + "\n".join(errors))

    def _generate_constraints(self) -> None:
        """Generate FalkorDB constraints from schema"""
        # Unique constraints
        for entity in self.entities.values():
            for prop in entity.properties:
                if prop.unique:
                    self.constraints.append(
                        {"type": "unique", "entity": entity.label, "property": prop.name}
                    )

        # Primary key constraints
        for entity in self.entities.values():
            self.constraints.append(
                {"type": "primary_key", "entity": entity.label, "property": entity.primary_key}
            )

        # Cardinality constraints (for enforcement)
        for rel in self.relationships.values():
            if rel.cardinality == CardinalityType.ONE_TO_ONE:
                self.constraints.append(
                    {
                        "type": "cardinality",
                        "relationship": rel.type,
                        "max_outgoing": 1,
                        "max_incoming": 1,
                    }
                )

    def _generate_indexes(self) -> None:
        """Generate FalkorDB indexes from schema"""
        # Property indexes
        for entity in self.entities.values():
            for prop in entity.properties:
                if prop.indexed:
                    self.indexes.append(
                        {
                            "entity": entity.label,
                            "property": prop.name,
                            "type": "exact"
                            if prop.property_type in [PropertyType.STRING, PropertyType.INTEGER]
                            else "range",
                        }
                    )

            # Explicit indexes
            for index_prop in entity.indexes:
                if index_prop != entity.primary_key:  # PK auto-indexed
                    self.indexes.append(
                        {"entity": entity.label, "property": index_prop, "type": "exact"}
                    )

    def _generate_graphity_config(self) -> Dict[str, Any]:
        """
        Generate Graphity optimization configuration
        Graphity improves traversal performance through edge grouping
        """
        # Identify high-degree nodes (hubs) that benefit from Graphity
        hub_entities = []
        for entity in self.entities.values():
            # Count incoming relationships
            incoming = sum(1 for r in self.relationships.values() if r.to_label == entity.label)
            if incoming >= 3:  # Hub threshold
                hub_entities.append(entity.label)

        # Identify frequently traversed paths
        traversal_patterns = []
        for rel in self.relationships.values():
            traversal_patterns.append(
                {
                    "pattern": f"({rel.from_label})-[:{rel.type}]->({rel.to_label})",
                    "edge_type": rel.type,
                    "estimated_frequency": "high" if rel.from_label in hub_entities else "medium",
                }
            )

        return {
            "enabled": True,
            "hub_entities": hub_entities,
            "edge_grouping": {
                "strategy": "by_type",  # Group edges by relationship type
                "chunk_size": 1000,  # Edges per chunk
            },
            "traversal_patterns": traversal_patterns,
            "cache_policy": {
                "hot_paths": True,  # Cache frequently traversed paths
                "ttl_seconds": 3600,
            },
        }

    def export_cypher_ddl(self, compiled: CompiledSchema) -> str:
        """
        Export compiled schema as Cypher DDL statements

        Args:
            compiled: Compiled schema to export

        Returns:
            String containing Cypher DDL statements
        """
        statements = []
        statements.append("// E2I Ontology Schema - Generated DDL")
        statements.append(f"// Version: {compiled.version}")
        statements.append("")

        # Create indexes
        statements.append("// Indexes")
        for idx in compiled.indexes:
            stmt = f"CREATE INDEX FOR (n:{idx['entity']}) ON (n.{idx['property']})"
            statements.append(stmt)
        statements.append("")

        # Create constraints
        statements.append("// Constraints")
        for constraint in compiled.constraints:
            if constraint["type"] == "unique":
                stmt = (
                    f"CREATE CONSTRAINT FOR (n:{constraint['entity']}) "
                    f"REQUIRE n.{constraint['property']} IS UNIQUE"
                )
                statements.append(stmt)

        return "\n".join(statements)

    def export_json_schema(self, compiled: CompiledSchema) -> Dict[str, Any]:
        """
        Export compiled schema as JSON Schema format

        Args:
            compiled: Compiled schema to export

        Returns:
            JSON Schema dictionary
        """
        schema: Dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "E2I Ontology Schema",
            "version": compiled.version,
            "entities": {},
            "relationships": {},
        }

        # Export entities
        for label, entity in compiled.entities.items():
            schema["entities"][label] = {
                "type": "object",
                "description": entity.description,
                "primaryKey": entity.primary_key,
                "properties": {},
                "required": [],
            }

            for prop in entity.properties:
                schema["entities"][label]["properties"][prop.name] = {
                    "type": prop.property_type.value,
                    "description": prop.description,
                }
                if prop.required:
                    schema["entities"][label]["required"].append(prop.name)

        # Export relationships
        for rel_type, rel in compiled.relationships.items():
            schema["relationships"][rel_type] = {
                "from": rel.from_label,
                "to": rel.to_label,
                "cardinality": rel.cardinality.value,
                "bidirectional": rel.bidirectional,
                "properties": {
                    prop.name: {"type": prop.property_type.value} for prop in rel.properties
                },
            }

        return schema


# CLI interface
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: schema_compiler.py <ontology_dir> [--output-format cypher|json]")
        sys.exit(1)

    ontology_dir = Path(sys.argv[1])
    output_format = sys.argv[2] if len(sys.argv) > 2 else "json"

    compiler = SchemaCompiler(ontology_dir)
    compiled = compiler.compile()

    if output_format == "cypher":
        print(compiler.export_cypher_ddl(compiled))
    else:
        print(json.dumps(compiler.export_json_schema(compiled), indent=2))
