"""
E2I Causal Analytics - Ontology Validator
Validates ontology schema consistency, relationship integrity, and data quality
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ontology.schema_compiler import (
    CardinalityType,
    CompiledSchema,
    PropertySchema,
    PropertyType,
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""

    ERROR = "error"  # Must be fixed
    WARNING = "warning"  # Should be reviewed
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """Single validation issue"""

    level: ValidationLevel
    category: str
    entity_or_rel: str
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report"""

    passed: bool
    timestamp: datetime
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    schema_version: str

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.INFO)


class OntologyValidator:
    """
    Validates compiled ontology schema for consistency, completeness,
    and adherence to E2I semantic memory requirements.
    """

    # Reserved property names that shouldn't be used
    RESERVED_PROPERTIES = {
        "id",
        "uuid",
        "created_at",
        "updated_at",
        "deleted_at",
        "version",
        "_type",
    }

    # Recommended naming conventions
    ENTITY_LABEL_PATTERN = r"^[A-Z][a-zA-Z0-9]*$"  # PascalCase
    RELATIONSHIP_TYPE_PATTERN = r"^[A-Z_]+$"  # UPPER_SNAKE_CASE
    PROPERTY_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"  # snake_case

    def __init__(self, compiled_schema: CompiledSchema):
        """
        Initialize validator with compiled schema

        Args:
            compiled_schema: Compiled ontology schema to validate
        """
        self.schema = compiled_schema
        self.issues: List[ValidationIssue] = []

    def validate(self) -> ValidationReport:
        """
        Run complete validation suite

        Returns:
            ValidationReport with all identified issues
        """
        logger.info("Starting ontology validation")
        self.issues = []

        # Run validation checks
        self._validate_entity_schemas()
        self._validate_relationship_schemas()
        self._validate_references()
        self._validate_naming_conventions()
        self._validate_indexes()
        self._validate_constraints()
        self._validate_cardinality()
        self._validate_property_types()
        self._validate_required_patterns()
        self._validate_graphity_config()

        # Compute statistics
        statistics = self._compute_statistics()

        # Determine if validation passed
        passed = self.error_count == 0

        report = ValidationReport(
            passed=passed,
            timestamp=datetime.now(),
            issues=self.issues,
            statistics=statistics,
            schema_version=self.schema.version,
        )

        logger.info(
            f"Validation complete: {self.error_count} errors, {self.warning_count} warnings"
        )

        return report

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)

    def _add_issue(
        self,
        level: ValidationLevel,
        category: str,
        entity_or_rel: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        fix_suggestion: Optional[str] = None,
    ) -> None:
        """Add validation issue to report"""
        issue = ValidationIssue(
            level=level,
            category=category,
            entity_or_rel=entity_or_rel,
            message=message,
            details=details,
            fix_suggestion=fix_suggestion,
        )
        self.issues.append(issue)

    def _validate_entity_schemas(self) -> None:
        """Validate entity schema definitions"""
        logger.debug("Validating entity schemas")

        for label, entity in self.schema.entities.items():
            # Check primary key exists
            pk_found = any(p.name == entity.primary_key for p in entity.properties)
            if not pk_found:
                self._add_issue(
                    ValidationLevel.ERROR,
                    "schema",
                    label,
                    f"Primary key '{entity.primary_key}' not found in properties",
                    fix_suggestion=f"Add property '{entity.primary_key}' to entity",
                )

            # Check for duplicate property names
            prop_names = [p.name for p in entity.properties]
            duplicates = {p for p in prop_names if prop_names.count(p) > 1}
            if duplicates:
                self._add_issue(
                    ValidationLevel.ERROR,
                    "schema",
                    label,
                    f"Duplicate property names: {duplicates}",
                    details={"duplicates": list(duplicates)},
                )

            # Check for empty entity (no properties)
            if not entity.properties:
                self._add_issue(
                    ValidationLevel.WARNING,
                    "completeness",
                    label,
                    "Entity has no properties defined",
                )

            # Validate individual properties
            for prop in entity.properties:
                self._validate_property(label, prop, is_entity=True)

    def _validate_relationship_schemas(self) -> None:
        """Validate relationship schema definitions"""
        logger.debug("Validating relationship schemas")

        for rel_type, rel in self.schema.relationships.items():
            # Check for self-referencing relationships
            if rel.from_label == rel.to_label:
                self._add_issue(
                    ValidationLevel.INFO,
                    "design",
                    rel_type,
                    f"Self-referencing relationship: {rel.from_label} -> {rel.from_label}",
                    details={"pattern": "recursive"},
                )

            # Check for duplicate properties
            if rel.properties:
                prop_names = [p.name for p in rel.properties]
                duplicates = {p for p in prop_names if prop_names.count(p) > 1}
                if duplicates:
                    self._add_issue(
                        ValidationLevel.ERROR,
                        "schema",
                        rel_type,
                        f"Duplicate property names: {duplicates}",
                    )

            # Validate relationship properties
            for prop in rel.properties:
                self._validate_property(rel_type, prop, is_entity=False)

    def _validate_property(self, parent_name: str, prop: PropertySchema, is_entity: bool) -> None:
        """Validate individual property definition"""
        # Check for reserved property names
        if prop.name in self.RESERVED_PROPERTIES:
            self._add_issue(
                ValidationLevel.WARNING,
                "naming",
                parent_name,
                f"Property '{prop.name}' uses reserved name",
                fix_suggestion="Consider renaming to avoid conflicts",
            )

        # Validate constraints
        if prop.constraints:
            if "min" in prop.constraints and "max" in prop.constraints:
                if prop.constraints["min"] > prop.constraints["max"]:
                    self._add_issue(
                        ValidationLevel.ERROR,
                        "constraints",
                        parent_name,
                        f"Property '{prop.name}': min > max in constraints",
                    )

        # Check unique on non-indexed properties
        if prop.unique and not prop.indexed:
            self._add_issue(
                ValidationLevel.WARNING,
                "performance",
                parent_name,
                f"Property '{prop.name}' is unique but not indexed",
                fix_suggestion="Add indexed: true for better performance",
            )

    def _validate_references(self) -> None:
        """Validate entity references in relationships"""
        logger.debug("Validating entity references")

        for rel_type, rel in self.schema.relationships.items():
            # Check from_label exists
            if rel.from_label not in self.schema.entities:
                self._add_issue(
                    ValidationLevel.ERROR,
                    "references",
                    rel_type,
                    f"References non-existent entity '{rel.from_label}'",
                    fix_suggestion=f"Add entity '{rel.from_label}' or fix reference",
                )

            # Check to_label exists
            if rel.to_label not in self.schema.entities:
                self._add_issue(
                    ValidationLevel.ERROR,
                    "references",
                    rel_type,
                    f"References non-existent entity '{rel.to_label}'",
                    fix_suggestion=f"Add entity '{rel.to_label}' or fix reference",
                )

    def _validate_naming_conventions(self) -> None:
        """Validate naming conventions"""
        logger.debug("Validating naming conventions")

        import re

        # Entity labels (PascalCase recommended)
        for label in self.schema.entities.keys():
            if not re.match(self.ENTITY_LABEL_PATTERN, label):
                self._add_issue(
                    ValidationLevel.WARNING,
                    "naming",
                    label,
                    "Entity label should be PascalCase",
                    details={"current": label},
                    fix_suggestion="Use PascalCase (e.g., CausalEstimate)",
                )

        # Relationship types (UPPER_SNAKE_CASE recommended)
        for rel_type in self.schema.relationships.keys():
            if not re.match(self.RELATIONSHIP_TYPE_PATTERN, rel_type):
                self._add_issue(
                    ValidationLevel.WARNING,
                    "naming",
                    rel_type,
                    "Relationship type should be UPPER_SNAKE_CASE",
                    details={"current": rel_type},
                    fix_suggestion="Use UPPER_SNAKE_CASE (e.g., HAS_IMPACT_ON)",
                )

        # Property names (snake_case recommended)
        for label, entity in self.schema.entities.items():
            for prop in entity.properties:
                if not re.match(self.PROPERTY_NAME_PATTERN, prop.name):
                    self._add_issue(
                        ValidationLevel.INFO,
                        "naming",
                        f"{label}.{prop.name}",
                        "Property name should be snake_case",
                    )

    def _validate_indexes(self) -> None:
        """Validate index definitions"""
        logger.debug("Validating indexes")

        # Check for indexed primary keys
        for entity in self.schema.entities.values():
            pk_indexed = any(
                idx["entity"] == entity.label and idx["property"] == entity.primary_key
                for idx in self.schema.indexes
            )
            if not pk_indexed:
                self._add_issue(
                    ValidationLevel.WARNING,
                    "performance",
                    entity.label,
                    f"Primary key '{entity.primary_key}' should be indexed",
                    fix_suggestion="Add index on primary key",
                )

        # Check for duplicate indexes
        index_keys = [(idx["entity"], idx["property"]) for idx in self.schema.indexes]
        duplicates = {k for k in index_keys if index_keys.count(k) > 1}
        if duplicates:
            self._add_issue(
                ValidationLevel.WARNING,
                "optimization",
                "indexes",
                f"Duplicate indexes found: {duplicates}",
                fix_suggestion="Remove duplicate index definitions",
            )

    def _validate_constraints(self) -> None:
        """Validate constraint definitions"""
        logger.debug("Validating constraints")

        # Check for conflicting constraints
        for constraint in self.schema.constraints:
            if constraint["type"] == "unique":
                entity = constraint["entity"]
                prop = constraint["property"]

                # Check if property exists
                if entity in self.schema.entities:
                    entity_schema = self.schema.entities[entity]
                    prop_exists = any(p.name == prop for p in entity_schema.properties)
                    if not prop_exists:
                        self._add_issue(
                            ValidationLevel.ERROR,
                            "constraints",
                            entity,
                            f"Unique constraint on non-existent property '{prop}'",
                        )

    def _validate_cardinality(self) -> None:
        """Validate relationship cardinality consistency"""
        logger.debug("Validating cardinality")

        # Build cardinality matrix
        entity_relationships: Dict[str, List[Tuple[str, str, CardinalityType]]] = {}

        for rel_type, rel in self.schema.relationships.items():
            # Track outgoing relationships
            if rel.from_label not in entity_relationships:
                entity_relationships[rel.from_label] = []
            entity_relationships[rel.from_label].append((rel_type, rel.to_label, rel.cardinality))

        # Check for potential cardinality conflicts
        for entity, rels in entity_relationships.items():
            # Multiple 1:1 relationships to same target
            targets = {}
            for rel_type, target, cardinality in rels:
                if cardinality == CardinalityType.ONE_TO_ONE:
                    if target in targets:
                        self._add_issue(
                            ValidationLevel.WARNING,
                            "cardinality",
                            entity,
                            f"Multiple 1:1 relationships to {target}",
                            details={"relationships": [targets[target], rel_type]},
                        )
                    targets[target] = rel_type

    def _validate_property_types(self) -> None:
        """Validate property type consistency"""
        logger.debug("Validating property types")

        # Track properties with same name across entities
        property_types: Dict[str, Set[PropertyType]] = {}

        for entity in self.schema.entities.values():
            for prop in entity.properties:
                if prop.name not in property_types:
                    property_types[prop.name] = set()
                property_types[prop.name].add(prop.property_type)

        # Flag properties with inconsistent types
        for prop_name, types in property_types.items():
            if len(types) > 1:
                self._add_issue(
                    ValidationLevel.INFO,
                    "consistency",
                    prop_name,
                    f"Property '{prop_name}' has inconsistent types: {types}",
                    details={"types": [t.value for t in types]},
                    fix_suggestion="Consider using consistent type or renaming",
                )

    def _validate_required_patterns(self) -> None:
        """Validate E2I-specific required patterns"""
        logger.debug("Validating E2I required patterns")

        # Check for temporal properties
        required_temporal_entities = ["CausalEstimate", "Experiment", "AgentActivity"]
        for entity_label in required_temporal_entities:
            if entity_label in self.schema.entities:
                entity = self.schema.entities[entity_label]
                has_timestamp = any(
                    "time" in p.name.lower() or "date" in p.name.lower() for p in entity.properties
                )
                if not has_timestamp:
                    self._add_issue(
                        ValidationLevel.WARNING,
                        "completeness",
                        entity_label,
                        "Temporal entity missing timestamp property",
                        fix_suggestion="Add created_at or timestamp property",
                    )

        # Check for required causal entities
        required_entities = ["HCP", "Patient", "Intervention", "Outcome"]
        for required in required_entities:
            if required not in self.schema.entities:
                self._add_issue(
                    ValidationLevel.WARNING,
                    "completeness",
                    "schema",
                    f"Missing recommended entity: {required}",
                    fix_suggestion=f"Add {required} entity for complete causal model",
                )

    def _validate_graphity_config(self) -> None:
        """Validate Graphity optimization configuration"""
        logger.debug("Validating Graphity config")

        config = self.schema.graphity_config

        if not config.get("enabled"):
            self._add_issue(
                ValidationLevel.INFO, "performance", "graphity", "Graphity optimization is disabled"
            )
            return

        # Check hub entities exist
        for hub in config.get("hub_entities", []):
            if hub not in self.schema.entities:
                self._add_issue(
                    ValidationLevel.WARNING,
                    "configuration",
                    "graphity",
                    f"Hub entity '{hub}' not found in schema",
                )

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute schema statistics"""
        total_properties = sum(len(e.properties) for e in self.schema.entities.values())
        total_rel_properties = sum(len(r.properties) for r in self.schema.relationships.values())

        return {
            "entity_count": len(self.schema.entities),
            "relationship_count": len(self.schema.relationships),
            "total_properties": total_properties,
            "total_relationship_properties": total_rel_properties,
            "index_count": len(self.schema.indexes),
            "constraint_count": len(self.schema.constraints),
            "graphity_enabled": self.schema.graphity_config.get("enabled", False),
        }

    def generate_report(self, report: ValidationReport, format: str = "text") -> str:
        """
        Generate human-readable validation report

        Args:
            report: Validation report to format
            format: Output format ('text' or 'markdown')

        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._generate_markdown_report(report)
        else:
            return self._generate_text_report(report)

    def _generate_text_report(self, report: ValidationReport) -> str:
        """Generate plain text report"""
        lines = []
        lines.append("=" * 70)
        lines.append("E2I ONTOLOGY VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {report.timestamp.isoformat()}")
        lines.append(f"Schema Version: {report.schema_version}")
        lines.append(f"Status: {'PASSED' if report.passed else 'FAILED'}")
        lines.append("")
        lines.append("Summary:")
        lines.append(f"  Errors:   {report.error_count}")
        lines.append(f"  Warnings: {report.warning_count}")
        lines.append(f"  Info:     {report.info_count}")
        lines.append("")

        # Statistics
        lines.append("Schema Statistics:")
        for key, value in report.statistics.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Issues by level
        for level in [ValidationLevel.ERROR, ValidationLevel.WARNING, ValidationLevel.INFO]:
            level_issues = [i for i in report.issues if i.level == level]
            if level_issues:
                lines.append(f"{level.value.upper()}S:")
                for issue in level_issues:
                    lines.append(f"  [{issue.category}] {issue.entity_or_rel}")
                    lines.append(f"    {issue.message}")
                    if issue.fix_suggestion:
                        lines.append(f"    Fix: {issue.fix_suggestion}")
                lines.append("")

        return "\n".join(lines)

    def _generate_markdown_report(self, report: ValidationReport) -> str:
        """Generate Markdown report"""
        lines = []
        lines.append("# E2I Ontology Validation Report")
        lines.append("")
        lines.append(f"**Timestamp:** {report.timestamp.isoformat()}")
        lines.append(f"**Schema Version:** {report.schema_version}")
        lines.append(f"**Status:** {'PASSED' if report.passed else 'FAILED'}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Errors:** {report.error_count}")
        lines.append(f"- **Warnings:** {report.warning_count}")
        lines.append(f"- **Info:** {report.info_count}")
        lines.append("")

        # Statistics
        lines.append("## Schema Statistics")
        lines.append("")
        for key, value in report.statistics.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

        # Issues
        if report.issues:
            lines.append("## Issues")
            lines.append("")

            for level in [ValidationLevel.ERROR, ValidationLevel.WARNING, ValidationLevel.INFO]:
                level_issues = [i for i in report.issues if i.level == level]
                if level_issues:
                    icon = (
                        "X"
                        if level == ValidationLevel.ERROR
                        else "!"
                        if level == ValidationLevel.WARNING
                        else "i"
                    )
                    lines.append(f"### [{icon}] {level.value.upper()}S")
                    lines.append("")
                    for issue in level_issues:
                        lines.append(f"**{issue.entity_or_rel}** [{issue.category}]")
                        lines.append(f"- {issue.message}")
                        if issue.fix_suggestion:
                            lines.append(f"- *Fix:* {issue.fix_suggestion}")
                        lines.append("")

        return "\n".join(lines)


# CLI interface
if __name__ == "__main__":
    import sys
    from pathlib import Path

    from src.ontology.schema_compiler import SchemaCompiler

    if len(sys.argv) < 2:
        print("Usage: validator.py <ontology_dir> [--format text|markdown]")
        sys.exit(1)

    ontology_dir = Path(sys.argv[1])
    output_format = "text"
    if len(sys.argv) > 2 and sys.argv[2] == "--format":
        output_format = sys.argv[3] if len(sys.argv) > 3 else "text"

    # Compile schema
    compiler = SchemaCompiler(ontology_dir)
    compiled = compiler.compile()

    # Validate
    validator = OntologyValidator(compiled)
    report = validator.validate()

    # Print report
    print(validator.generate_report(report, format=output_format))

    sys.exit(0 if report.passed else 1)
