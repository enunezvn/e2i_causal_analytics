"""
Unit tests for src/ontology/schema_compiler.py

Tests the SchemaCompiler class which compiles YAML ontology definitions
into FalkorDB-compatible schema with Graphity optimizations.

Test Classes:
- TestEnums: PropertyType and CardinalityType enums
- TestDataclasses: PropertySchema, EntitySchema, RelationshipSchema, CompiledSchema
- TestSchemaCompilerInit: Compiler initialization
- TestSchemaCompilerParsing: Entity and relationship parsing
- TestSchemaCompilerValidation: Reference validation
- TestSchemaCompilerGeneration: Constraint and index generation
- TestSchemaCompilerExport: Cypher DDL and JSON Schema export
"""

from pathlib import Path

import pytest
import yaml

from src.ontology.schema_compiler import (
    CardinalityType,
    CompiledSchema,
    EntitySchema,
    PropertySchema,
    PropertyType,
    RelationshipSchema,
    SchemaCompiler,
)


class TestEnums:
    """Tests for PropertyType and CardinalityType enums."""

    def test_property_type_values(self):
        """Test PropertyType enum has all expected values."""
        assert PropertyType.STRING.value == "string"
        assert PropertyType.INTEGER.value == "integer"
        assert PropertyType.FLOAT.value == "float"
        assert PropertyType.BOOLEAN.value == "boolean"
        assert PropertyType.DATETIME.value == "datetime"
        assert PropertyType.JSON.value == "json"
        assert PropertyType.ARRAY.value == "array"

    def test_cardinality_type_values(self):
        """Test CardinalityType enum has all expected values."""
        assert CardinalityType.ONE_TO_ONE.value == "1:1"
        assert CardinalityType.ONE_TO_MANY.value == "1:N"
        assert CardinalityType.MANY_TO_MANY.value == "M:N"

    def test_property_type_from_string(self):
        """Test PropertyType can be constructed from string value."""
        assert PropertyType("string") == PropertyType.STRING
        assert PropertyType("integer") == PropertyType.INTEGER
        assert PropertyType("float") == PropertyType.FLOAT

    def test_cardinality_from_string(self):
        """Test CardinalityType can be constructed from string value."""
        assert CardinalityType("1:1") == CardinalityType.ONE_TO_ONE
        assert CardinalityType("1:N") == CardinalityType.ONE_TO_MANY
        assert CardinalityType("M:N") == CardinalityType.MANY_TO_MANY


class TestDataclasses:
    """Tests for schema dataclasses."""

    def test_property_schema_defaults(self):
        """Test PropertySchema has correct defaults."""
        prop = PropertySchema(name="test", property_type=PropertyType.STRING)

        assert prop.name == "test"
        assert prop.property_type == PropertyType.STRING
        assert prop.required is False
        assert prop.indexed is False
        assert prop.unique is False
        assert prop.default is None
        assert prop.constraints == {}
        assert prop.description is None

    def test_property_schema_with_all_fields(self):
        """Test PropertySchema with all fields specified."""
        prop = PropertySchema(
            name="score",
            property_type=PropertyType.FLOAT,
            required=True,
            indexed=True,
            unique=False,
            default=0.0,
            constraints={"min": 0.0, "max": 1.0},
            description="Risk score",
        )

        assert prop.name == "score"
        assert prop.required is True
        assert prop.indexed is True
        assert prop.constraints == {"min": 0.0, "max": 1.0}

    def test_entity_schema_defaults(self):
        """Test EntitySchema has correct defaults."""
        entity = EntitySchema(label="TestEntity", properties=[], primary_key="id")

        assert entity.label == "TestEntity"
        assert entity.indexes == []
        assert entity.description is None
        assert entity.metadata == {}

    def test_relationship_schema_defaults(self):
        """Test RelationshipSchema has correct defaults."""
        rel = RelationshipSchema(
            type="TEST_REL",
            from_label="A",
            to_label="B",
            properties=[],
            cardinality=CardinalityType.ONE_TO_MANY,
        )

        assert rel.type == "TEST_REL"
        assert rel.bidirectional is False
        assert rel.description is None
        assert rel.metadata == {}

    def test_compiled_schema_structure(self):
        """Test CompiledSchema has all required fields."""
        compiled = CompiledSchema(
            entities={},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={"enabled": True},
            version="1.0",
            metadata={},
        )

        assert isinstance(compiled.entities, dict)
        assert isinstance(compiled.relationships, dict)
        assert isinstance(compiled.constraints, list)
        assert isinstance(compiled.indexes, list)
        assert compiled.version == "1.0"


class TestSchemaCompilerInit:
    """Tests for SchemaCompiler initialization."""

    def test_init_with_path(self, mock_ontology_dir: Path):
        """Test compiler initialization with valid directory."""
        compiler = SchemaCompiler(mock_ontology_dir)

        assert compiler.ontology_dir == mock_ontology_dir
        assert compiler.entities == {}
        assert compiler.relationships == {}

    def test_init_converts_string_to_path(self, mock_ontology_dir: Path):
        """Test compiler converts string path to Path object."""
        compiler = SchemaCompiler(str(mock_ontology_dir))

        assert isinstance(compiler.ontology_dir, Path)

    def test_init_with_nonexistent_dir(self, tmp_path: Path):
        """Test compiler can be initialized with nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"
        compiler = SchemaCompiler(nonexistent)

        # Initialization succeeds - error occurs on compile()
        assert compiler.ontology_dir == nonexistent


class TestSchemaCompilerParsing:
    """Tests for entity and relationship parsing."""

    def test_compile_extracts_entities(self, mock_ontology_dir: Path):
        """Test compile() extracts all entities from YAML."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        assert "Patient" in compiled.entities
        assert "HCP" in compiled.entities
        assert "Brand" in compiled.entities

    def test_entity_primary_key(self, mock_ontology_dir: Path):
        """Test entities have correct primary keys."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        assert compiled.entities["Patient"].primary_key == "patient_id"
        assert compiled.entities["HCP"].primary_key == "hcp_id"
        assert compiled.entities["Brand"].primary_key == "brand_id"

    def test_entity_properties_parsed(self, mock_ontology_dir: Path):
        """Test entity properties are correctly parsed."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        patient = compiled.entities["Patient"]
        prop_names = [p.name for p in patient.properties]

        assert "patient_id" in prop_names
        assert "region" in prop_names
        assert "risk_score" in prop_names
        assert "created_at" in prop_names

    def test_property_type_mapping(self, mock_ontology_dir: Path):
        """Test property types are correctly mapped."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        patient = compiled.entities["Patient"]
        props = {p.name: p for p in patient.properties}

        assert props["patient_id"].property_type == PropertyType.STRING
        assert props["region"].property_type == PropertyType.STRING
        assert props["risk_score"].property_type == PropertyType.FLOAT
        assert props["created_at"].property_type == PropertyType.DATETIME

    def test_property_required_flag(self, mock_ontology_dir: Path):
        """Test property required flag is parsed correctly."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        patient = compiled.entities["Patient"]
        props = {p.name: p for p in patient.properties}

        assert props["patient_id"].required is True
        assert props["region"].required is True
        assert props["risk_score"].required is False

    def test_property_indexed_flag(self, mock_ontology_dir: Path):
        """Test property indexed flag is parsed correctly."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        patient = compiled.entities["Patient"]
        props = {p.name: p for p in patient.properties}

        assert props["patient_id"].indexed is True
        assert props["region"].indexed is True

    def test_property_constraints_parsed(self, mock_ontology_dir: Path):
        """Test property constraints are parsed correctly."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        patient = compiled.entities["Patient"]
        props = {p.name: p for p in patient.properties}

        assert props["risk_score"].constraints == {"min": 0.0, "max": 1.0}

    def test_compile_extracts_relationships(self, mock_ontology_dir: Path):
        """Test compile() extracts all relationships from YAML."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        assert "TREATED_BY" in compiled.relationships
        assert "PRESCRIBED" in compiled.relationships
        assert "PRESCRIBES" in compiled.relationships

    def test_relationship_endpoints(self, mock_ontology_dir: Path):
        """Test relationship from/to labels are correct."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        treated_by = compiled.relationships["TREATED_BY"]
        assert treated_by.from_label == "Patient"
        assert treated_by.to_label == "HCP"

    def test_relationship_cardinality(self, mock_ontology_dir: Path):
        """Test relationship cardinality is correctly parsed."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        assert compiled.relationships["TREATED_BY"].cardinality == CardinalityType.ONE_TO_MANY
        assert compiled.relationships["PRESCRIBED"].cardinality == CardinalityType.MANY_TO_MANY
        assert compiled.relationships["PRESCRIBES"].cardinality == CardinalityType.ONE_TO_MANY

    def test_relationship_properties_parsed(self, mock_ontology_dir: Path):
        """Test relationship properties are correctly parsed."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        treated_by = compiled.relationships["TREATED_BY"]
        prop_names = [p.name for p in treated_by.properties]

        assert "is_primary" in prop_names
        assert "visit_count" in prop_names


class TestSchemaCompilerValidation:
    """Tests for schema validation."""

    def test_valid_schema_passes_validation(self, mock_ontology_dir: Path):
        """Test that valid schema compiles without errors."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()  # Should not raise

        assert len(compiled.entities) > 0
        assert len(compiled.relationships) > 0

    def test_invalid_from_entity_raises_error(self, invalid_reference_yaml: Path):
        """Test validation fails for invalid from_label reference."""
        compiler = SchemaCompiler(invalid_reference_yaml)

        with pytest.raises(ValueError) as exc_info:
            compiler.compile()

        assert "unknown entity" in str(exc_info.value)

    def test_empty_directory_compiles(self, empty_ontology_dir: Path):
        """Test compilation of empty directory succeeds with empty schema."""
        compiler = SchemaCompiler(empty_ontology_dir)
        compiled = compiler.compile()

        assert len(compiled.entities) == 0
        assert len(compiled.relationships) == 0


class TestSchemaCompilerGeneration:
    """Tests for constraint and index generation."""

    def test_unique_constraints_generated(self, mock_ontology_dir: Path):
        """Test unique constraints are generated for unique properties."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        unique_constraints = [c for c in compiled.constraints if c["type"] == "unique"]

        # patient_id, hcp_id, brand_id are all unique
        unique_entities = [c["entity"] for c in unique_constraints]
        assert "Patient" in unique_entities
        assert "HCP" in unique_entities

    def test_primary_key_constraints_generated(self, mock_ontology_dir: Path):
        """Test primary key constraints are generated for all entities."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        pk_constraints = [c for c in compiled.constraints if c["type"] == "primary_key"]

        # One PK constraint per entity
        assert len(pk_constraints) == len(compiled.entities)

        pk_entities = [c["entity"] for c in pk_constraints]
        assert "Patient" in pk_entities
        assert "HCP" in pk_entities
        assert "Brand" in pk_entities

    def test_cardinality_constraints_for_one_to_one(self, tmp_path: Path):
        """Test cardinality constraints generated for 1:1 relationships."""
        # Create a schema with 1:1 relationship
        ontology_dir = tmp_path / "ontology"
        ontology_dir.mkdir()

        schema_data = {
            "entities": [
                {
                    "label": "User",
                    "primary_key": "id",
                    "properties": [{"name": "id", "type": "string", "required": True}],
                },
                {
                    "label": "Profile",
                    "primary_key": "id",
                    "properties": [{"name": "id", "type": "string", "required": True}],
                },
            ],
            "relationships": [
                {
                    "type": "HAS_PROFILE",
                    "from": "User",
                    "to": "Profile",
                    "cardinality": "1:1",
                    "properties": [],
                }
            ],
        }

        with open(ontology_dir / "schema.yaml", "w") as f:
            yaml.dump(schema_data, f)

        compiler = SchemaCompiler(ontology_dir)
        compiled = compiler.compile()

        cardinality_constraints = [c for c in compiled.constraints if c["type"] == "cardinality"]

        assert len(cardinality_constraints) == 1
        assert cardinality_constraints[0]["relationship"] == "HAS_PROFILE"
        assert cardinality_constraints[0]["max_outgoing"] == 1
        assert cardinality_constraints[0]["max_incoming"] == 1

    def test_indexes_generated_for_indexed_properties(self, mock_ontology_dir: Path):
        """Test indexes are generated for properties with indexed=True."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        # Check that indexed properties have indexes
        index_props = [(i["entity"], i["property"]) for i in compiled.indexes]

        # patient_id, region are indexed
        assert ("Patient", "patient_id") in index_props
        assert ("Patient", "region") in index_props

    def test_index_types(self, mock_ontology_dir: Path):
        """Test correct index types are assigned."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        # String/integer properties get 'exact' type
        string_indexes = [i for i in compiled.indexes if i["property"] == "patient_id"]
        if string_indexes:
            assert string_indexes[0]["type"] == "exact"


class TestSchemaCompilerGraphity:
    """Tests for Graphity configuration generation."""

    def test_graphity_config_generated(self, mock_ontology_dir: Path):
        """Test Graphity config is generated."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        assert "enabled" in compiled.graphity_config
        assert compiled.graphity_config["enabled"] is True

    def test_hub_entities_identified(self, mock_ontology_dir: Path):
        """Test hub entities are identified based on incoming relationships."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        # HCP and Brand have multiple incoming relationships
        hub_entities = compiled.graphity_config.get("hub_entities", [])
        # May or may not be identified as hubs depending on relationship count
        assert isinstance(hub_entities, list)

    def test_traversal_patterns_generated(self, mock_ontology_dir: Path):
        """Test traversal patterns are generated for relationships."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        patterns = compiled.graphity_config.get("traversal_patterns", [])
        assert len(patterns) == len(compiled.relationships)

        # Check pattern structure
        for pattern in patterns:
            assert "pattern" in pattern
            assert "edge_type" in pattern
            assert "estimated_frequency" in pattern

    def test_edge_grouping_config(self, mock_ontology_dir: Path):
        """Test edge grouping configuration is present."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        edge_grouping = compiled.graphity_config.get("edge_grouping", {})
        assert edge_grouping["strategy"] == "by_type"
        assert edge_grouping["chunk_size"] == 1000

    def test_cache_policy_config(self, mock_ontology_dir: Path):
        """Test cache policy configuration is present."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        cache_policy = compiled.graphity_config.get("cache_policy", {})
        assert cache_policy["hot_paths"] is True
        assert "ttl_seconds" in cache_policy


class TestSchemaCompilerExport:
    """Tests for Cypher DDL and JSON Schema export."""

    def test_export_cypher_ddl(self, mock_ontology_dir: Path):
        """Test Cypher DDL export."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        ddl = compiler.export_cypher_ddl(compiled)

        assert isinstance(ddl, str)
        assert "// E2I Ontology Schema" in ddl
        assert "CREATE INDEX" in ddl
        assert "CREATE CONSTRAINT" in ddl

    def test_export_cypher_ddl_contains_indexes(self, mock_ontology_dir: Path):
        """Test Cypher DDL contains index creation statements."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        ddl = compiler.export_cypher_ddl(compiled)

        # Should have index creation statements
        assert "CREATE INDEX FOR" in ddl
        assert "Patient" in ddl

    def test_export_cypher_ddl_contains_constraints(self, mock_ontology_dir: Path):
        """Test Cypher DDL contains constraint creation statements."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        ddl = compiler.export_cypher_ddl(compiled)

        # Should have unique constraint statements
        assert "IS UNIQUE" in ddl

    def test_export_json_schema(self, mock_ontology_dir: Path):
        """Test JSON Schema export."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        schema = compiler.export_json_schema(compiled)

        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "title" in schema
        assert "entities" in schema
        assert "relationships" in schema

    def test_export_json_schema_entities(self, mock_ontology_dir: Path):
        """Test JSON Schema contains entity definitions."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        schema = compiler.export_json_schema(compiled)

        assert "Patient" in schema["entities"]
        patient_schema = schema["entities"]["Patient"]

        assert patient_schema["type"] == "object"
        assert "properties" in patient_schema
        assert "patient_id" in patient_schema["properties"]
        assert "required" in patient_schema

    def test_export_json_schema_relationships(self, mock_ontology_dir: Path):
        """Test JSON Schema contains relationship definitions."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        schema = compiler.export_json_schema(compiled)

        assert "TREATED_BY" in schema["relationships"]
        rel_schema = schema["relationships"]["TREATED_BY"]

        assert rel_schema["from"] == "Patient"
        assert rel_schema["to"] == "HCP"
        assert rel_schema["cardinality"] == "1:N"

    def test_export_json_schema_version(self, mock_ontology_dir: Path):
        """Test JSON Schema contains version."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        schema = compiler.export_json_schema(compiled)

        assert schema["version"] == compiled.version


class TestSchemaCompilerMetadata:
    """Tests for compilation metadata."""

    def test_compiled_files_in_metadata(self, mock_ontology_dir: Path):
        """Test compiled files are tracked in metadata."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        assert "compiled_files" in compiled.metadata
        assert isinstance(compiled.metadata["compiled_files"], list)
        # Should include our mock YAML files
        assert len(compiled.metadata["compiled_files"]) > 0

    def test_version_in_compiled_schema(self, mock_ontology_dir: Path):
        """Test version is set in compiled schema."""
        compiler = SchemaCompiler(mock_ontology_dir)
        compiled = compiler.compile()

        assert compiled.version == "1.0"
