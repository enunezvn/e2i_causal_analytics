"""
Unit tests for src/ontology/validator.py

Tests the OntologyValidator class which validates compiled ontology schemas
for consistency, completeness, and adherence to E2I semantic memory requirements.

Test Classes:
- TestEnumsAndDataclasses: ValidationLevel, ValidationIssue, ValidationReport
- TestValidatorInit: Validator initialization
- TestEntityValidation: Entity schema validation
- TestRelationshipValidation: Relationship schema validation
- TestReferenceValidation: Entity reference validation
- TestNamingValidation: Naming convention validation
- TestIndexValidation: Index definition validation
- TestConstraintValidation: Constraint definition validation
- TestCardinalityValidation: Cardinality consistency validation
- TestPropertyTypeValidation: Property type consistency validation
- TestGraphityValidation: Graphity configuration validation
- TestReportGeneration: Report generation methods
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.ontology.validator import (
    OntologyValidator,
    ValidationLevel,
    ValidationIssue,
    ValidationReport,
)
from src.ontology.schema_compiler import (
    CompiledSchema,
    EntitySchema,
    RelationshipSchema,
    PropertySchema,
    PropertyType,
    CardinalityType,
)


class TestEnumsAndDataclasses:
    """Tests for ValidationLevel, ValidationIssue, and ValidationReport."""

    def test_validation_level_values(self):
        """Test ValidationLevel enum has expected values."""
        assert ValidationLevel.ERROR.value == "error"
        assert ValidationLevel.WARNING.value == "warning"
        assert ValidationLevel.INFO.value == "info"

    def test_validation_issue_defaults(self):
        """Test ValidationIssue has correct defaults."""
        issue = ValidationIssue(
            level=ValidationLevel.ERROR,
            category="test",
            entity_or_rel="TestEntity",
            message="Test message"
        )

        assert issue.details is None
        assert issue.fix_suggestion is None

    def test_validation_issue_with_all_fields(self):
        """Test ValidationIssue with all fields specified."""
        issue = ValidationIssue(
            level=ValidationLevel.WARNING,
            category="naming",
            entity_or_rel="BadEntity",
            message="Invalid naming",
            details={"current": "bad_name"},
            fix_suggestion="Use PascalCase"
        )

        assert issue.level == ValidationLevel.WARNING
        assert issue.details == {"current": "bad_name"}
        assert issue.fix_suggestion == "Use PascalCase"

    def test_validation_report_counts(self):
        """Test ValidationReport count properties."""
        issues = [
            ValidationIssue(ValidationLevel.ERROR, "a", "x", "err1"),
            ValidationIssue(ValidationLevel.ERROR, "b", "y", "err2"),
            ValidationIssue(ValidationLevel.WARNING, "c", "z", "warn1"),
            ValidationIssue(ValidationLevel.INFO, "d", "w", "info1"),
            ValidationIssue(ValidationLevel.INFO, "e", "v", "info2"),
        ]

        report = ValidationReport(
            passed=False,
            timestamp=datetime.now(),
            issues=issues,
            statistics={},
            schema_version="1.0"
        )

        assert report.error_count == 2
        assert report.warning_count == 1
        assert report.info_count == 2

    def test_validation_report_passed_true(self):
        """Test ValidationReport.passed when no errors."""
        report = ValidationReport(
            passed=True,
            timestamp=datetime.now(),
            issues=[ValidationIssue(ValidationLevel.WARNING, "a", "x", "warn")],
            statistics={},
            schema_version="1.0"
        )

        assert report.passed is True
        assert report.error_count == 0


class TestValidatorInit:
    """Tests for OntologyValidator initialization."""

    def test_init_with_compiled_schema(self, compiled_schema_for_validation: CompiledSchema):
        """Test validator initialization with compiled schema."""
        validator = OntologyValidator(compiled_schema_for_validation)

        assert validator.schema == compiled_schema_for_validation
        assert validator.issues == []

    def test_validator_reserved_properties(self):
        """Test validator has reserved properties defined."""
        assert 'id' in OntologyValidator.RESERVED_PROPERTIES
        assert 'uuid' in OntologyValidator.RESERVED_PROPERTIES
        assert 'created_at' in OntologyValidator.RESERVED_PROPERTIES


class TestEntityValidation:
    """Tests for entity schema validation."""

    def test_valid_schema_passes_validation(self, compiled_schema_for_validation: CompiledSchema):
        """Test that valid schema passes validation."""
        validator = OntologyValidator(compiled_schema_for_validation)
        report = validator.validate()

        # Valid schema should not have schema-level errors (may have warnings)
        schema_errors = [
            i for i in report.issues
            if i.level == ValidationLevel.ERROR and i.category == "schema"
        ]
        assert len(schema_errors) == 0

    def test_missing_primary_key_error(self, schema_with_missing_pk: CompiledSchema):
        """Test validation fails when primary key is not in properties."""
        validator = OntologyValidator(schema_with_missing_pk)
        report = validator.validate()

        assert report.passed is False
        assert report.error_count > 0

        pk_errors = [
            i for i in report.issues
            if 'Primary key' in i.message and i.level == ValidationLevel.ERROR
        ]
        assert len(pk_errors) == 1
        assert 'missing_pk' in pk_errors[0].message

    def test_duplicate_property_names_error(self):
        """Test validation fails for duplicate property names."""
        entity = EntitySchema(
            label='DuplicateProps',
            properties=[
                PropertySchema(name='id', property_type=PropertyType.STRING),
                PropertySchema(name='id', property_type=PropertyType.STRING),  # Duplicate
            ],
            primary_key='id'
        )

        schema = CompiledSchema(
            entities={'DuplicateProps': entity},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        dup_errors = [
            i for i in report.issues
            if 'Duplicate property' in i.message
        ]
        assert len(dup_errors) == 1

    def test_empty_entity_warning(self):
        """Test validation warns for entity with no properties."""
        entity = EntitySchema(
            label='EmptyEntity',
            properties=[],
            primary_key='id'
        )

        schema = CompiledSchema(
            entities={'EmptyEntity': entity},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        empty_warnings = [
            i for i in report.issues
            if 'no properties' in i.message
        ]
        assert len(empty_warnings) == 1


class TestRelationshipValidation:
    """Tests for relationship schema validation."""

    def test_self_referencing_relationship_info(self):
        """Test self-referencing relationship generates info message."""
        entity = EntitySchema(
            label='Node',
            properties=[PropertySchema(name='id', property_type=PropertyType.STRING)],
            primary_key='id'
        )

        rel = RelationshipSchema(
            type='RELATES_TO',
            from_label='Node',
            to_label='Node',  # Self-reference
            properties=[],
            cardinality=CardinalityType.MANY_TO_MANY
        )

        schema = CompiledSchema(
            entities={'Node': entity},
            relationships={'RELATES_TO': rel},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        self_ref_info = [
            i for i in report.issues
            if 'Self-referencing' in i.message
        ]
        assert len(self_ref_info) == 1
        assert self_ref_info[0].level == ValidationLevel.INFO

    def test_relationship_duplicate_properties_error(self):
        """Test validation fails for relationships with duplicate properties."""
        entity = EntitySchema(
            label='Entity',
            properties=[PropertySchema(name='id', property_type=PropertyType.STRING)],
            primary_key='id'
        )

        rel = RelationshipSchema(
            type='HAS_DUP',
            from_label='Entity',
            to_label='Entity',
            properties=[
                PropertySchema(name='weight', property_type=PropertyType.FLOAT),
                PropertySchema(name='weight', property_type=PropertyType.FLOAT),  # Dup
            ],
            cardinality=CardinalityType.ONE_TO_MANY
        )

        schema = CompiledSchema(
            entities={'Entity': entity},
            relationships={'HAS_DUP': rel},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        dup_errors = [
            i for i in report.issues
            if 'Duplicate property' in i.message
        ]
        assert len(dup_errors) == 1


class TestReferenceValidation:
    """Tests for entity reference validation."""

    def test_invalid_from_entity_error(self):
        """Test validation fails when relationship references nonexistent from entity."""
        entity = EntitySchema(
            label='ValidEntity',
            properties=[PropertySchema(name='id', property_type=PropertyType.STRING)],
            primary_key='id'
        )

        rel = RelationshipSchema(
            type='INVALID_REL',
            from_label='NonExistent',  # Invalid reference
            to_label='ValidEntity',
            properties=[],
            cardinality=CardinalityType.ONE_TO_MANY
        )

        schema = CompiledSchema(
            entities={'ValidEntity': entity},
            relationships={'INVALID_REL': rel},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        assert report.passed is False
        ref_errors = [
            i for i in report.issues
            if 'non-existent entity' in i.message
        ]
        assert len(ref_errors) == 1
        assert 'NonExistent' in ref_errors[0].message

    def test_invalid_to_entity_error(self):
        """Test validation fails when relationship references nonexistent to entity."""
        entity = EntitySchema(
            label='ValidEntity',
            properties=[PropertySchema(name='id', property_type=PropertyType.STRING)],
            primary_key='id'
        )

        rel = RelationshipSchema(
            type='INVALID_REL',
            from_label='ValidEntity',
            to_label='NonExistent',  # Invalid reference
            properties=[],
            cardinality=CardinalityType.ONE_TO_MANY
        )

        schema = CompiledSchema(
            entities={'ValidEntity': entity},
            relationships={'INVALID_REL': rel},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        assert report.passed is False


class TestNamingValidation:
    """Tests for naming convention validation."""

    def test_non_pascal_case_entity_warning(self, schema_with_invalid_naming: CompiledSchema):
        """Test validation warns for non-PascalCase entity labels."""
        validator = OntologyValidator(schema_with_invalid_naming)
        report = validator.validate()

        naming_warnings = [
            i for i in report.issues
            if 'PascalCase' in i.message
        ]
        assert len(naming_warnings) > 0
        assert naming_warnings[0].level == ValidationLevel.WARNING

    def test_non_upper_snake_case_relationship_warning(self, schema_with_invalid_naming: CompiledSchema):
        """Test validation warns for non-UPPER_SNAKE_CASE relationship types."""
        validator = OntologyValidator(schema_with_invalid_naming)
        report = validator.validate()

        naming_warnings = [
            i for i in report.issues
            if 'UPPER_SNAKE_CASE' in i.message
        ]
        assert len(naming_warnings) > 0

    def test_reserved_property_name_warning(self):
        """Test validation warns for reserved property names."""
        entity = EntitySchema(
            label='TestEntity',
            properties=[
                PropertySchema(name='id', property_type=PropertyType.STRING),  # Reserved
                PropertySchema(name='name', property_type=PropertyType.STRING),
            ],
            primary_key='id'
        )

        schema = CompiledSchema(
            entities={'TestEntity': entity},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        reserved_warnings = [
            i for i in report.issues
            if 'reserved name' in i.message
        ]
        assert len(reserved_warnings) == 1


class TestIndexValidation:
    """Tests for index definition validation."""

    def test_unindexed_primary_key_warning(self):
        """Test validation warns when primary key is not indexed."""
        entity = EntitySchema(
            label='UnindexedPK',
            properties=[
                PropertySchema(name='pk', property_type=PropertyType.STRING)
            ],
            primary_key='pk'
        )

        schema = CompiledSchema(
            entities={'UnindexedPK': entity},
            relationships={},
            constraints=[],
            indexes=[],  # No indexes
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        pk_warnings = [
            i for i in report.issues
            if 'should be indexed' in i.message and 'Primary key' in i.message
        ]
        assert len(pk_warnings) == 1

    def test_duplicate_index_warning(self):
        """Test validation warns for duplicate indexes."""
        entity = EntitySchema(
            label='Entity',
            properties=[
                PropertySchema(name='id', property_type=PropertyType.STRING, indexed=True)
            ],
            primary_key='id'
        )

        schema = CompiledSchema(
            entities={'Entity': entity},
            relationships={},
            constraints=[],
            indexes=[
                {'entity': 'Entity', 'property': 'id', 'type': 'exact'},
                {'entity': 'Entity', 'property': 'id', 'type': 'exact'},  # Duplicate
            ],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        dup_warnings = [
            i for i in report.issues
            if 'Duplicate indexes' in i.message
        ]
        assert len(dup_warnings) == 1


class TestConstraintValidation:
    """Tests for constraint definition validation."""

    def test_unique_on_nonexistent_property_error(self):
        """Test validation fails for unique constraint on nonexistent property."""
        entity = EntitySchema(
            label='Entity',
            properties=[
                PropertySchema(name='id', property_type=PropertyType.STRING)
            ],
            primary_key='id'
        )

        schema = CompiledSchema(
            entities={'Entity': entity},
            relationships={},
            constraints=[
                {'type': 'unique', 'entity': 'Entity', 'property': 'nonexistent'}
            ],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        constraint_errors = [
            i for i in report.issues
            if 'non-existent property' in i.message
        ]
        assert len(constraint_errors) == 1

    def test_unique_not_indexed_warning(self):
        """Test validation warns when unique property is not indexed."""
        entity = EntitySchema(
            label='Entity',
            properties=[
                PropertySchema(
                    name='email',
                    property_type=PropertyType.STRING,
                    unique=True,
                    indexed=False  # Should be indexed
                )
            ],
            primary_key='email'
        )

        schema = CompiledSchema(
            entities={'Entity': entity},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        perf_warnings = [
            i for i in report.issues
            if 'unique but not indexed' in i.message
        ]
        assert len(perf_warnings) == 1

    def test_invalid_min_max_constraint_error(self):
        """Test validation fails when min > max in constraints."""
        entity = EntitySchema(
            label='Entity',
            properties=[
                PropertySchema(
                    name='score',
                    property_type=PropertyType.FLOAT,
                    constraints={'min': 100, 'max': 0}  # Invalid: min > max
                )
            ],
            primary_key='score'
        )

        schema = CompiledSchema(
            entities={'Entity': entity},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        constraint_errors = [
            i for i in report.issues
            if 'min > max' in i.message
        ]
        assert len(constraint_errors) == 1


class TestCardinalityValidation:
    """Tests for cardinality consistency validation."""

    def test_multiple_one_to_one_warning(self):
        """Test validation warns for multiple 1:1 relationships to same target."""
        entity_a = EntitySchema(
            label='EntityA',
            properties=[PropertySchema(name='id', property_type=PropertyType.STRING)],
            primary_key='id'
        )
        entity_b = EntitySchema(
            label='EntityB',
            properties=[PropertySchema(name='id', property_type=PropertyType.STRING)],
            primary_key='id'
        )

        rel1 = RelationshipSchema(
            type='REL_ONE',
            from_label='EntityA',
            to_label='EntityB',
            properties=[],
            cardinality=CardinalityType.ONE_TO_ONE
        )
        rel2 = RelationshipSchema(
            type='REL_TWO',
            from_label='EntityA',
            to_label='EntityB',
            properties=[],
            cardinality=CardinalityType.ONE_TO_ONE
        )

        schema = CompiledSchema(
            entities={'EntityA': entity_a, 'EntityB': entity_b},
            relationships={'REL_ONE': rel1, 'REL_TWO': rel2},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        card_warnings = [
            i for i in report.issues
            if 'Multiple 1:1' in i.message
        ]
        assert len(card_warnings) == 1


class TestPropertyTypeValidation:
    """Tests for property type consistency validation."""

    def test_inconsistent_property_types_info(self):
        """Test validation reports inconsistent types for same property name."""
        entity_a = EntitySchema(
            label='EntityA',
            properties=[
                PropertySchema(name='id', property_type=PropertyType.STRING),
                PropertySchema(name='value', property_type=PropertyType.STRING)
            ],
            primary_key='id'
        )
        entity_b = EntitySchema(
            label='EntityB',
            properties=[
                PropertySchema(name='id', property_type=PropertyType.STRING),
                PropertySchema(name='value', property_type=PropertyType.INTEGER)  # Different type
            ],
            primary_key='id'
        )

        schema = CompiledSchema(
            entities={'EntityA': entity_a, 'EntityB': entity_b},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        type_info = [
            i for i in report.issues
            if 'inconsistent types' in i.message
        ]
        assert len(type_info) == 1
        assert type_info[0].level == ValidationLevel.INFO


class TestGraphityValidation:
    """Tests for Graphity configuration validation."""

    def test_graphity_disabled_info(self):
        """Test validation reports when Graphity is disabled."""
        schema = CompiledSchema(
            entities={},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        graphity_info = [
            i for i in report.issues
            if 'Graphity optimization is disabled' in i.message
        ]
        assert len(graphity_info) == 1

    def test_hub_entity_not_found_warning(self):
        """Test validation warns for hub entity not in schema."""
        schema = CompiledSchema(
            entities={},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={
                'enabled': True,
                'hub_entities': ['NonExistentHub']
            },
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        hub_warnings = [
            i for i in report.issues
            if 'Hub entity' in i.message and 'not found' in i.message
        ]
        assert len(hub_warnings) == 1


class TestStatistics:
    """Tests for statistics computation."""

    def test_statistics_computed(self, compiled_schema_for_validation: CompiledSchema):
        """Test statistics are computed correctly."""
        validator = OntologyValidator(compiled_schema_for_validation)
        report = validator.validate()

        assert 'entity_count' in report.statistics
        assert 'relationship_count' in report.statistics
        assert 'total_properties' in report.statistics
        assert 'index_count' in report.statistics
        assert 'constraint_count' in report.statistics
        assert 'graphity_enabled' in report.statistics

    def test_statistics_values(self, compiled_schema_for_validation: CompiledSchema):
        """Test statistics have correct values."""
        validator = OntologyValidator(compiled_schema_for_validation)
        report = validator.validate()

        # Based on mock_ontology_dir fixture: 3 entities, 3 relationships
        assert report.statistics['entity_count'] == 3
        assert report.statistics['relationship_count'] == 3


class TestReportGeneration:
    """Tests for report generation methods."""

    def test_generate_text_report(self, compiled_schema_for_validation: CompiledSchema):
        """Test text report generation."""
        validator = OntologyValidator(compiled_schema_for_validation)
        report = validator.validate()

        text_report = validator.generate_report(report, format='text')

        assert isinstance(text_report, str)
        assert "E2I ONTOLOGY VALIDATION REPORT" in text_report
        assert "Timestamp:" in text_report
        assert "Schema Version:" in text_report
        assert "Status:" in text_report

    def test_generate_markdown_report(self, compiled_schema_for_validation: CompiledSchema):
        """Test markdown report generation."""
        validator = OntologyValidator(compiled_schema_for_validation)
        report = validator.validate()

        md_report = validator.generate_report(report, format='markdown')

        assert isinstance(md_report, str)
        assert "# E2I Ontology Validation Report" in md_report
        assert "**Timestamp:**" in md_report
        assert "## Summary" in md_report
        assert "## Schema Statistics" in md_report

    def test_report_includes_errors(self):
        """Test report includes error messages."""
        # Create schema with known error
        schema = CompiledSchema(
            entities={'BadEntity': EntitySchema(
                label='BadEntity',
                properties=[],
                primary_key='missing_pk'
            )},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()
        text_report = validator.generate_report(report, format='text')

        assert "ERRORS:" in text_report or "Error" in text_report.lower()

    def test_report_includes_fix_suggestions(self):
        """Test report includes fix suggestions."""
        # Create schema with issue that has fix suggestion
        schema = CompiledSchema(
            entities={'BadEntity': EntitySchema(
                label='BadEntity',
                properties=[],
                primary_key='missing_pk'
            )},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()
        text_report = validator.generate_report(report, format='text')

        assert "Fix:" in text_report


class TestValidatorEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_schema_validates(self):
        """Test validation of empty schema."""
        schema = CompiledSchema(
            entities={},
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        # Empty schema should pass (no errors, may have warnings about missing entities)
        assert report.error_count == 0

    def test_large_schema_validates(self):
        """Test validation of large schema doesn't error."""
        # Create schema with many entities
        entities = {}
        for i in range(20):
            entities[f'Entity{i}'] = EntitySchema(
                label=f'Entity{i}',
                properties=[
                    PropertySchema(name='id', property_type=PropertyType.STRING)
                ],
                primary_key='id'
            )

        schema = CompiledSchema(
            entities=entities,
            relationships={},
            constraints=[],
            indexes=[],
            graphity_config={'enabled': False},
            version="1.0",
            metadata={}
        )

        validator = OntologyValidator(schema)
        report = validator.validate()

        # Should complete without error
        assert isinstance(report, ValidationReport)
        assert report.statistics['entity_count'] == 20
