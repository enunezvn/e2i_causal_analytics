"""
Schema Validator for Synthetic Data

Validates that generated data matches Supabase table schemas:
- Column presence and types
- ENUM value compliance
- NOT NULL constraints
- Foreign key references
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type
from enum import Enum

import pandas as pd

from ..config import (
    Brand,
    SpecialtyEnum,
    PracticeTypeEnum,
    RegionEnum,
    InsuranceTypeEnum,
    EngagementTypeEnum,
    DataSplit,
)


@dataclass
class ColumnSpec:
    """Specification for a single column."""

    name: str
    dtype: Type
    nullable: bool = True
    enum_values: Optional[Set[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_table: Optional[str] = None


@dataclass
class TableSchema:
    """Schema definition for a table."""

    table_name: str
    columns: Dict[str, ColumnSpec]
    primary_key: str
    foreign_keys: Dict[str, str] = field(default_factory=dict)  # column -> table


@dataclass
class ValidationError:
    """A single validation error."""

    table: str
    column: str
    error_type: str
    message: str
    row_indices: Optional[List[int]] = None
    sample_values: Optional[List[Any]] = None


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""

    is_valid: bool
    table_name: str
    total_rows: int
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(
        self,
        column: str,
        error_type: str,
        message: str,
        row_indices: Optional[List[int]] = None,
        sample_values: Optional[List[Any]] = None,
    ) -> None:
        """Add a validation error."""
        self.errors.append(
            ValidationError(
                table=self.table_name,
                column=column,
                error_type=error_type,
                message=message,
                row_indices=row_indices[:10] if row_indices else None,  # Limit samples
                sample_values=sample_values[:5] if sample_values else None,
            )
        )
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)


# =============================================================================
# TABLE SCHEMAS
# =============================================================================

# Helper to convert enum to set of values
def _enum_values(enum_class: Type[Enum]) -> Set[str]:
    return {e.value for e in enum_class}


# HCP Profiles Schema
HCP_PROFILES_SCHEMA = TableSchema(
    table_name="hcp_profiles",
    primary_key="hcp_id",
    columns={
        "hcp_id": ColumnSpec(name="hcp_id", dtype=str, nullable=False, is_primary_key=True),
        "npi": ColumnSpec(name="npi", dtype=str, nullable=True),
        "specialty": ColumnSpec(
            name="specialty",
            dtype=str,
            nullable=False,
            enum_values=_enum_values(SpecialtyEnum),
        ),
        "practice_type": ColumnSpec(
            name="practice_type",
            dtype=str,
            nullable=False,
            enum_values=_enum_values(PracticeTypeEnum),
        ),
        "geographic_region": ColumnSpec(
            name="geographic_region",
            dtype=str,
            nullable=False,
            enum_values=_enum_values(RegionEnum),
        ),
        "years_experience": ColumnSpec(
            name="years_experience", dtype=(int, float), nullable=True, min_value=0, max_value=60
        ),
        "academic_hcp": ColumnSpec(
            name="academic_hcp", dtype=(int, bool), nullable=False
        ),
        "total_patient_volume": ColumnSpec(
            name="total_patient_volume", dtype=(int, float), nullable=True, min_value=0
        ),
        "brand": ColumnSpec(
            name="brand", dtype=str, nullable=False, enum_values=_enum_values(Brand)
        ),
    },
)

# Patient Journeys Schema
PATIENT_JOURNEYS_SCHEMA = TableSchema(
    table_name="patient_journeys",
    primary_key="patient_journey_id",
    foreign_keys={"hcp_id": "hcp_profiles"},
    columns={
        "patient_journey_id": ColumnSpec(
            name="patient_journey_id", dtype=str, nullable=False, is_primary_key=True
        ),
        "patient_id": ColumnSpec(name="patient_id", dtype=str, nullable=False),
        "hcp_id": ColumnSpec(
            name="hcp_id",
            dtype=str,
            nullable=False,
            is_foreign_key=True,
            foreign_table="hcp_profiles",
        ),
        "brand": ColumnSpec(
            name="brand", dtype=str, nullable=False, enum_values=_enum_values(Brand)
        ),
        "journey_start_date": ColumnSpec(name="journey_start_date", dtype=str, nullable=False),
        "data_split": ColumnSpec(
            name="data_split",
            dtype=str,
            nullable=False,
            enum_values=_enum_values(DataSplit),
        ),
        "disease_severity": ColumnSpec(
            name="disease_severity", dtype=(int, float), nullable=False, min_value=0, max_value=10
        ),
        "academic_hcp": ColumnSpec(name="academic_hcp", dtype=(int, bool), nullable=False),
        "engagement_score": ColumnSpec(
            name="engagement_score", dtype=(int, float), nullable=False, min_value=0, max_value=10
        ),
        "treatment_initiated": ColumnSpec(
            name="treatment_initiated", dtype=(int, bool), nullable=False
        ),
        "days_to_treatment": ColumnSpec(
            name="days_to_treatment", dtype=(int, float), nullable=True, min_value=0
        ),
        "geographic_region": ColumnSpec(
            name="geographic_region",
            dtype=str,
            nullable=True,
            enum_values=_enum_values(RegionEnum),
        ),
        "insurance_type": ColumnSpec(
            name="insurance_type",
            dtype=str,
            nullable=True,
            enum_values=_enum_values(InsuranceTypeEnum),
        ),
        "age_at_diagnosis": ColumnSpec(
            name="age_at_diagnosis", dtype=(int, float), nullable=True, min_value=0, max_value=120
        ),
    },
)

# Engagement Events Schema
ENGAGEMENT_EVENTS_SCHEMA = TableSchema(
    table_name="engagement_events",
    primary_key="engagement_id",
    foreign_keys={"hcp_id": "hcp_profiles"},
    columns={
        "engagement_id": ColumnSpec(
            name="engagement_id", dtype=str, nullable=False, is_primary_key=True
        ),
        "hcp_id": ColumnSpec(
            name="hcp_id",
            dtype=str,
            nullable=False,
            is_foreign_key=True,
            foreign_table="hcp_profiles",
        ),
        "brand": ColumnSpec(
            name="brand", dtype=str, nullable=False, enum_values=_enum_values(Brand)
        ),
        "engagement_date": ColumnSpec(name="engagement_date", dtype=str, nullable=False),
        "engagement_type": ColumnSpec(
            name="engagement_type",
            dtype=str,
            nullable=False,
            enum_values=_enum_values(EngagementTypeEnum),
        ),
        "engagement_quality": ColumnSpec(
            name="engagement_quality", dtype=(int, float), nullable=True, min_value=0, max_value=1
        ),
    },
)

# Registry of all schemas
TABLE_SCHEMAS: Dict[str, TableSchema] = {
    "hcp_profiles": HCP_PROFILES_SCHEMA,
    "patient_journeys": PATIENT_JOURNEYS_SCHEMA,
    "engagement_events": ENGAGEMENT_EVENTS_SCHEMA,
}


class SchemaValidator:
    """
    Validates DataFrame schema against Supabase table definitions.

    Usage:
        validator = SchemaValidator()
        result = validator.validate(df, "patient_journeys")
        if not result.is_valid:
            for error in result.errors:
                print(f"{error.column}: {error.message}")
    """

    def __init__(self, schemas: Optional[Dict[str, TableSchema]] = None):
        """
        Initialize validator with schemas.

        Args:
            schemas: Custom schemas. Defaults to TABLE_SCHEMAS.
        """
        self.schemas = schemas or TABLE_SCHEMAS

    def validate(
        self,
        df: pd.DataFrame,
        table_name: str,
        reference_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> SchemaValidationResult:
        """
        Validate a DataFrame against its table schema.

        Args:
            df: DataFrame to validate
            table_name: Name of the table to validate against
            reference_dfs: Dict of table_name -> DataFrame for FK validation

        Returns:
            SchemaValidationResult with validation status and errors
        """
        if table_name not in self.schemas:
            result = SchemaValidationResult(
                is_valid=False, table_name=table_name, total_rows=len(df)
            )
            result.add_error("_table", "unknown_table", f"Unknown table: {table_name}")
            return result

        schema = self.schemas[table_name]
        result = SchemaValidationResult(
            is_valid=True, table_name=table_name, total_rows=len(df)
        )

        # 1. Check for missing required columns
        self._validate_required_columns(df, schema, result)

        # 2. Validate each column
        for col_name, col_spec in schema.columns.items():
            if col_name in df.columns:
                self._validate_column(df, col_name, col_spec, result)

        # 3. Validate foreign keys if reference data provided
        if reference_dfs:
            self._validate_foreign_keys(df, schema, reference_dfs, result)

        return result

    def _validate_required_columns(
        self, df: pd.DataFrame, schema: TableSchema, result: SchemaValidationResult
    ) -> None:
        """Check that all non-nullable columns are present."""
        for col_name, col_spec in schema.columns.items():
            if not col_spec.nullable and col_name not in df.columns:
                result.add_error(
                    col_name,
                    "missing_column",
                    f"Required column '{col_name}' is missing",
                )

    def _validate_column(
        self,
        df: pd.DataFrame,
        col_name: str,
        col_spec: ColumnSpec,
        result: SchemaValidationResult,
    ) -> None:
        """Validate a single column."""
        col = df[col_name]

        # Check nullability
        if not col_spec.nullable:
            null_mask = col.isna()
            if null_mask.any():
                null_indices = df.index[null_mask].tolist()
                result.add_error(
                    col_name,
                    "null_value",
                    f"Column '{col_name}' contains {null_mask.sum()} NULL values (not allowed)",
                    row_indices=null_indices,
                )

        # Check enum values
        if col_spec.enum_values is not None:
            non_null = col.dropna()
            if len(non_null) > 0:
                invalid_mask = ~non_null.isin(col_spec.enum_values)
                if invalid_mask.any():
                    invalid_values = non_null[invalid_mask].unique().tolist()
                    result.add_error(
                        col_name,
                        "invalid_enum",
                        f"Invalid enum values in '{col_name}': {invalid_values}. "
                        f"Allowed: {col_spec.enum_values}",
                        sample_values=invalid_values,
                    )

        # Check numeric ranges
        if col_spec.min_value is not None or col_spec.max_value is not None:
            non_null = col.dropna()
            if len(non_null) > 0 and pd.api.types.is_numeric_dtype(non_null):
                if col_spec.min_value is not None:
                    below_min = non_null < col_spec.min_value
                    if below_min.any():
                        result.add_error(
                            col_name,
                            "below_min",
                            f"Values in '{col_name}' below minimum {col_spec.min_value}",
                            sample_values=non_null[below_min].head().tolist(),
                        )
                if col_spec.max_value is not None:
                    above_max = non_null > col_spec.max_value
                    if above_max.any():
                        result.add_error(
                            col_name,
                            "above_max",
                            f"Values in '{col_name}' above maximum {col_spec.max_value}",
                            sample_values=non_null[above_max].head().tolist(),
                        )

    def _validate_foreign_keys(
        self,
        df: pd.DataFrame,
        schema: TableSchema,
        reference_dfs: Dict[str, pd.DataFrame],
        result: SchemaValidationResult,
    ) -> None:
        """Validate foreign key constraints."""
        for fk_column, ref_table in schema.foreign_keys.items():
            if fk_column not in df.columns:
                continue

            if ref_table not in reference_dfs:
                result.add_warning(
                    f"Cannot validate FK '{fk_column}' -> '{ref_table}': "
                    "reference table not provided"
                )
                continue

            ref_df = reference_dfs[ref_table]
            ref_schema = self.schemas.get(ref_table)
            if ref_schema is None:
                continue

            pk_column = ref_schema.primary_key
            if pk_column not in ref_df.columns:
                result.add_warning(
                    f"Cannot validate FK '{fk_column}': "
                    f"PK '{pk_column}' not in reference table"
                )
                continue

            valid_pks = set(ref_df[pk_column].dropna())
            fk_values = df[fk_column].dropna()
            orphans = ~fk_values.isin(valid_pks)

            if orphans.any():
                orphan_values = fk_values[orphans].unique().tolist()
                result.add_error(
                    fk_column,
                    "orphan_fk",
                    f"Orphaned FK values in '{fk_column}': {len(orphan_values)} unique values "
                    f"not found in '{ref_table}.{pk_column}'",
                    sample_values=orphan_values[:5],
                )

    def validate_all(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, SchemaValidationResult]:
        """
        Validate all datasets at once with FK resolution.

        Args:
            datasets: Dict of table_name -> DataFrame

        Returns:
            Dict of table_name -> SchemaValidationResult
        """
        results = {}
        for table_name, df in datasets.items():
            results[table_name] = self.validate(
                df, table_name, reference_dfs=datasets
            )
        return results

    def get_validation_summary(
        self, results: Dict[str, SchemaValidationResult]
    ) -> Dict:
        """Get a summary of validation results."""
        total_errors = sum(len(r.errors) for r in results.values())
        total_warnings = sum(len(r.warnings) for r in results.values())

        return {
            "all_valid": all(r.is_valid for r in results.values()),
            "total_tables": len(results),
            "valid_tables": sum(1 for r in results.values() if r.is_valid),
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "tables": {
                name: {
                    "is_valid": r.is_valid,
                    "rows": r.total_rows,
                    "errors": len(r.errors),
                    "warnings": len(r.warnings),
                }
                for name, r in results.items()
            },
        }
