# Data Layer Contracts - CohortConstructor Addition

**Purpose**: Extension to data-contracts.md for CohortConstructor agent data specifications.

---

## CohortConstructor Data Contract (V1.0)

### Overview

The CohortConstructor agent defines eligible patient cohorts using explicit clinical criteria before ML training. This contract specifies the data structures for cohort definitions, executions, and eligibility tracking.

### Cohort Definition Schema

```python
from typing import TypedDict, Literal, Optional, Any
from datetime import datetime

class Criterion(TypedDict):
    """Single eligibility criterion."""
    field: str                          # Patient data field name
    operator: str                       # "==", ">=", "in", "between", "contains"
    value: Any                          # Comparison value
    description: Optional[str]          # Human-readable description
    clinical_rationale: Optional[str]   # Clinical justification

class CohortConfig(TypedDict):
    """Complete cohort configuration."""
    
    # Identity
    brand: str                          # "Remibrutinib" | "Fabhalta" | "Kisqali"
    indication: str                     # "csu" | "pnh" | "hr_her2"
    cohort_name: str                    # "Remibrutinib CSU Eligible Population"
    
    # Criteria
    inclusion_criteria: list[Criterion] # AND logic
    exclusion_criteria: list[Criterion] # AND NOT logic
    
    # Temporal requirements
    lookback_days: int                  # Default: 180
    followup_days: int                  # Default: 90
    index_date_field: str               # Default: "diagnosis_date"
    
    # Data quality
    required_fields: list[str]          # Fields that must be present
    
    # Metadata
    version: str                        # Semantic version
    created_date: str                   # ISO 8601 timestamp
    status: str                         # "draft" | "active" | "locked"
```

### Patient Data Schema

```python
PATIENT_DATA_SCHEMA = {
    # Identity (required)
    "patient_journey_id": "string",     # Unique patient identifier
    "patient_id": "string",             # De-identified patient ID
    
    # Demographics (required)
    "age_at_diagnosis": "int",          # Age in years
    "gender": "categorical",            # "M" | "F" | "O"
    
    # Clinical (brand-specific required fields)
    "diagnosis_code": "string",         # ICD-10 code
    "diagnosis_date": "datetime",       # Index date for temporal validation
    
    # Temporal tracking (required)
    "first_observation_date": "datetime",  # Earliest data point
    "last_observation_date": "datetime",   # Latest data point
    
    # Brand-specific clinical fields (conditional)
    # Remibrutinib CSU:
    "urticaria_severity_uas7": "int",      # UAS7 score (0-42)
    "antihistamine_failures_count": "int", # Number of failures
    "is_pregnant": "bool",
    "immunodeficiency_severe": "bool",
    
    # Fabhalta PNH:
    "complement_inhibitor_status": "categorical",  # "naive" | "switching" | "continuing"
    "ldh_level": "float",                          # LDH as multiple of ULN
    "active_infection": "bool",
    "pregnancy_or_breastfeeding": "bool",
    
    # Kisqali HR+/HER2-:
    "hr_status": "categorical",                    # "positive" | "negative" | "unknown"
    "her2_status": "categorical",                  # "positive" | "negative" | "unknown"
    "ecog_performance_status": "int",              # 0-5 scale
    "prior_cdk_inhibitor": "bool",
    "metastatic_cns": "bool"
}
```

### Quality Constraints

```python
COHORT_QUALITY_RULES = {
    "patient_journey_id": {
        "unique": True,
        "format": r"^pj_[a-zA-Z0-9]{10}$"
    },
    "diagnosis_date": {
        "min_date": "2015-01-01",  # Data availability start
        "max_date": "today",
        "timezone": "UTC"
    },
    "age_at_diagnosis": {
        "min_value": 0,
        "max_value": 120,
        "allow_null": False
    },
    "urticaria_severity_uas7": {
        "min_value": 0,
        "max_value": 42,
        "allow_null": True  # Not all brands require this
    },
    "lookback_days": {
        "min_value": 1,
        "max_value": 730,  # Max 2 years lookback
        "allow_null": False
    },
    "followup_days": {
        "min_value": 1,
        "max_value": 730,  # Max 2 years followup
        "allow_null": False
    }
}
```

---

## Cohort Output Schema

### Eligibility Result Schema

```python
class EligibilityLogEntry(TypedDict):
    """Per-criterion eligibility tracking."""
    criterion_name: str                 # Field name or description
    criterion_type: str                 # "inclusion" | "exclusion"
    criterion_order: int                # Application order (1, 2, 3, ...)
    removed_count: int                  # Patients removed by this criterion
    remaining_count: int                # Patients remaining after criterion
    description: str                    # Human-readable criterion

class TemporalValidationResult(TypedDict):
    """Temporal eligibility validation."""
    passed: bool                        # All checks passed
    lookback_exclusions: int            # Patients w/o sufficient lookback
    followup_exclusions: int            # Patients w/o sufficient followup
    total_temporal_exclusions: int      # Total removed by temporal

COHORT_OUTPUT_SCHEMA = {
    # Cohort specification
    "cohort_id": "string",              # Unique cohort identifier
    "cohort_name": "string",
    "brand": "string",
    "indication": "string",
    "config_version": "string",
    "config_hash": "string",            # SHA256 for reproducibility
    
    # Eligible population
    "eligible_patient_ids": "list[string]",  # Patient journey IDs
    "eligible_patient_count": "int",
    
    # Statistics
    "total_input_patients": "int",
    "exclusion_rate": "float",          # 0.0-1.0
    
    # Audit trail
    "eligibility_log": "list[EligibilityLogEntry]",
    "temporal_validation": "TemporalValidationResult",
    
    # Execution metadata
    "execution_id": "string",
    "execution_timestamp": "datetime",
    "execution_time_ms": "float"
}
```

---

## Database Tables Contract

### ml_cohort_definitions

```sql
CREATE TABLE ml_cohort_definitions (
    cohort_id TEXT PRIMARY KEY,
    experiment_id TEXT,  -- Links to ml_experiments
    brand TEXT NOT NULL,
    indication TEXT NOT NULL,
    cohort_name TEXT NOT NULL,
    
    -- Criteria (JSONB for flexibility)
    inclusion_criteria JSONB NOT NULL,
    exclusion_criteria JSONB NOT NULL,
    
    -- Temporal requirements
    lookback_days INTEGER NOT NULL,
    followup_days INTEGER NOT NULL,
    index_date_field TEXT NOT NULL,
    
    -- Required fields list
    required_fields JSONB,
    
    -- Version control
    config_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,  -- SHA256 for reproducibility
    
    -- Metadata
    created_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by TEXT,
    status TEXT NOT NULL,  -- "draft" | "active" | "locked" | "deprecated"
    
    CONSTRAINT valid_brand CHECK (brand IN ('Remibrutinib', 'Fabhalta', 'Kisqali')),
    CONSTRAINT valid_status CHECK (status IN ('draft', 'active', 'locked', 'deprecated')),
    CONSTRAINT positive_temporal CHECK (lookback_days > 0 AND followup_days > 0)
);

-- Indexes
CREATE INDEX idx_cohort_brand_indication ON ml_cohort_definitions(brand, indication);
CREATE INDEX idx_cohort_experiment ON ml_cohort_definitions(experiment_id);
CREATE INDEX idx_cohort_status ON ml_cohort_definitions(status);
```

### ml_cohort_executions

```sql
CREATE TABLE ml_cohort_executions (
    execution_id TEXT PRIMARY KEY,
    cohort_id TEXT NOT NULL REFERENCES ml_cohort_definitions(cohort_id),
    
    -- Execution timestamp
    execution_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Population statistics
    total_input_patients INTEGER NOT NULL,
    eligible_patient_count INTEGER NOT NULL,
    exclusion_rate DECIMAL(5,4) NOT NULL,  -- 0.0000-1.0000
    
    -- Temporal validation
    temporal_validation_passed BOOLEAN NOT NULL,
    lookback_exclusions INTEGER,
    followup_exclusions INTEGER,
    
    -- Performance
    execution_time_ms INTEGER NOT NULL,
    validate_config_ms DECIMAL(10,2),
    apply_criteria_ms DECIMAL(10,2),
    validate_temporal_ms DECIMAL(10,2),
    generate_metadata_ms DECIMAL(10,2),
    
    -- Status
    status TEXT NOT NULL,  -- "completed" | "failed" | "partial"
    error_message TEXT,
    
    CONSTRAINT positive_counts CHECK (
        total_input_patients >= 0 AND 
        eligible_patient_count >= 0 AND
        eligible_patient_count <= total_input_patients
    ),
    CONSTRAINT valid_exclusion_rate CHECK (exclusion_rate >= 0.0 AND exclusion_rate <= 1.0),
    CONSTRAINT valid_status CHECK (status IN ('completed', 'failed', 'partial'))
);

-- Indexes
CREATE INDEX idx_execution_cohort ON ml_cohort_executions(cohort_id);
CREATE INDEX idx_execution_timestamp ON ml_cohort_executions(execution_timestamp DESC);
CREATE INDEX idx_execution_status ON ml_cohort_executions(status);
```

### ml_cohort_eligibility_log

```sql
CREATE TABLE ml_cohort_eligibility_log (
    log_id SERIAL PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES ml_cohort_executions(execution_id),
    
    -- Criterion details
    criterion_name TEXT NOT NULL,
    criterion_type TEXT NOT NULL,  -- "inclusion" | "exclusion"
    criterion_order INTEGER NOT NULL,
    criterion_description TEXT,
    
    -- Application results
    removed_count INTEGER NOT NULL,
    remaining_count INTEGER NOT NULL,
    
    -- Metadata
    applied_timestamp TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT valid_criterion_type CHECK (criterion_type IN ('inclusion', 'exclusion')),
    CONSTRAINT positive_counts CHECK (removed_count >= 0 AND remaining_count >= 0)
);

-- Indexes
CREATE INDEX idx_eligibility_execution ON ml_cohort_eligibility_log(execution_id);
CREATE INDEX idx_eligibility_order ON ml_cohort_eligibility_log(execution_id, criterion_order);
```

### ml_patient_cohort_assignments

```sql
CREATE TABLE ml_patient_cohort_assignments (
    patient_journey_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL REFERENCES ml_cohort_definitions(cohort_id),
    execution_id TEXT NOT NULL REFERENCES ml_cohort_executions(execution_id),
    
    -- Eligibility
    is_eligible BOOLEAN NOT NULL,
    
    -- Failure tracking (for excluded patients)
    failed_criteria JSONB,  -- List of criterion names that failed
    temporal_failure_reason TEXT,  -- "insufficient_lookback" | "insufficient_followup" | null
    
    -- Metadata
    assignment_timestamp TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (patient_journey_id, cohort_id, execution_id)
);

-- Indexes
CREATE INDEX idx_assignment_patient ON ml_patient_cohort_assignments(patient_journey_id);
CREATE INDEX idx_assignment_cohort ON ml_patient_cohort_assignments(cohort_id);
CREATE INDEX idx_assignment_eligible ON ml_patient_cohort_assignments(cohort_id, is_eligible);
```

---

## Data Validation Contract

### Cohort-Specific Validation Rules

```python
COHORT_VALIDATION_RULES = {
    "cohort_not_empty": {
        "check": "eligible_patient_count >= 100",  # Minimum viable cohort
        "action_on_fail": "reject"
    },
    "exclusion_rate_reasonable": {
        "check": "0.05 <= exclusion_rate <= 0.95",  # Not extreme
        "action_on_fail": "warn"
    },
    "temporal_coverage": {
        "check": "temporal_validation_passed == True",
        "action_on_fail": "warn_and_track"
    },
    "config_locked": {
        "check": "status == 'locked' for production runs",
        "action_on_fail": "reject"
    },
    "required_fields_present": {
        "check": "All required_fields exist in patient data",
        "action_on_fail": "reject"
    }
}

class CohortValidator:
    """Contract for cohort validation."""
    
    def validate_config(self, config: CohortConfig) -> ValidationResult:
        """Validate cohort configuration structure."""
        pass
    
    def validate_patient_data(
        self, 
        df: pd.DataFrame, 
        required_fields: list[str]
    ) -> ValidationResult:
        """Validate patient data contains required fields."""
        pass
    
    def validate_cohort_size(
        self, 
        eligible_count: int,
        min_threshold: int = 100
    ) -> ValidationResult:
        """Validate cohort meets minimum size."""
        pass
    
    def validate_exclusion_rate(
        self,
        exclusion_rate: float
    ) -> ValidationResult:
        """Validate exclusion rate is reasonable (not too extreme)."""
        pass
```

---

## Integration with Existing Contracts

### Relationship to ML Pipeline

```python
# CohortConstructor sits between scope_definer and data_preparer
PIPELINE_FLOW = {
    "scope_definer": {
        "outputs": ["scope_spec", "experiment_id", "brand", "indication"],
        "next": "cohort_constructor"
    },
    "cohort_constructor": {
        "inputs": ["scope_spec", "patient_data"],
        "outputs": ["cohort_spec", "eligible_patient_ids"],
        "next": "data_preparer"
    },
    "data_preparer": {
        "inputs": ["cohort_spec", "eligible_patient_ids"],
        "note": "QC gate validates eligible cohort ONLY",
        "next": "model_selector"
    }
}
```

### Data Flow Constraints

```python
# CRITICAL: data_preparer must only validate eligible cohort
CONSTRAINT_ELIGIBLE_COHORT_ONLY = """
data_preparer receives eligible_patient_ids from cohort_constructor.
QC validation MUST be applied to this subset ONLY, not the entire
patient_journeys table. This ensures:

1. QC validates clinically relevant population
2. Training data excludes ineligible patients by design
3. No data leakage from excluded patients
4. Regulatory compliance (cohort pre-specification)
"""

# Validation order
VALIDATION_SEQUENCE = [
    "1. cohort_constructor defines eligible population",
    "2. data_preparer validates quality of eligible population",
    "3. model_trainer trains on eligible + quality-validated subset"
]
```

---

## Pandera Schema Contract (Extension)

### CohortConstructor Pandera Schemas

```python
from src.mlops.pandera_schemas import PANDERA_SCHEMA_REGISTRY

# Add cohort-specific schemas
COHORT_SCHEMAS = {
    "cohort_definitions": CohortDefinitionsSchema,
    "cohort_executions": CohortExecutionsSchema,
    "patient_cohort_assignments": PatientCohortAssignmentsSchema
}

class CohortDefinitionsSchema(DataFrameModel):
    """Schema for ml_cohort_definitions table."""
    
    cohort_id: Series[str] = Field(nullable=False, unique=True)
    brand: Series[str] = Field(nullable=False, isin=["Remibrutinib", "Fabhalta", "Kisqali"])
    indication: Series[str] = Field(nullable=False)
    cohort_name: Series[str] = Field(nullable=False)
    lookback_days: Series[int] = Field(nullable=False, ge=1, le=730)
    followup_days: Series[int] = Field(nullable=False, ge=1, le=730)
    config_version: Series[str] = Field(nullable=False)
    status: Series[str] = Field(nullable=False, isin=["draft", "active", "locked", "deprecated"])
    
    class Config:
        name = "cohort_definitions"
        strict = False
        coerce = True

class CohortExecutionsSchema(DataFrameModel):
    """Schema for ml_cohort_executions table."""
    
    execution_id: Series[str] = Field(nullable=False, unique=True)
    cohort_id: Series[str] = Field(nullable=False)
    total_input_patients: Series[int] = Field(nullable=False, ge=0)
    eligible_patient_count: Series[int] = Field(nullable=False, ge=0)
    exclusion_rate: Series[float] = Field(nullable=False, ge=0.0, le=1.0)
    temporal_validation_passed: Series[bool] = Field(nullable=False)
    execution_time_ms: Series[int] = Field(nullable=False, ge=0)
    status: Series[str] = Field(nullable=False, isin=["completed", "failed", "partial"])
    
    class Config:
        name = "cohort_executions"
        strict = False
        coerce = True
```

---

## Contract Obligations

| Component | Obligation |
|-----------|------------|
| CohortConstructor | Produce cohort_spec and eligible_patient_ids conforming to schema |
| CohortConstructor | Write to 4 database tables (definitions, executions, log, assignments) |
| CohortConstructor | Enforce temporal validation (lookback/followup) |
| data_preparer | Accept eligible_patient_ids and validate ONLY those patients |
| data_preparer | Do NOT run QC on entire patient_journeys table |
| model_trainer | Train on intersection of (eligible_patient_ids AND qc_passed_patients) |
| All Tier 0 Agents | Respect cohort_spec as source of truth for population definition |

---

## Change Management

### Breaking Changes
Changes that require code updates:
1. Adding required fields to patient data schema
2. Changing operator semantics
3. Modifying database table structures
4. Changing eligibility log format

### Non-Breaking Changes
Changes that don't require code updates:
1. Adding optional patient data fields
2. Adding new supported operators
3. Adding new brand configurations
4. Relaxing validation rules

### Deprecation Process
1. Announce cohort definition deprecation with 90-day notice (regulatory requirement)
2. Lock deprecated configs (status = 'deprecated')
3. Provide migration guide to new version
4. Maintain archived definitions for reproducibility

---

**Last Updated**: 2026-01-15
**Version**: 1.1 (added CohortConstructor contract)
**Owner**: E2I Tier 0 Team

**Changes in V1.1**:
- Added CohortConstructor Data Contract
- Added cohort_definitions, cohort_executions, cohort_eligibility_log, patient_cohort_assignments schemas
- Added Pandera schemas for cohort tables
- Specified integration constraints with data_preparer

**Instructions**:
1. All cohort processing code MUST conform to these contracts
2. Validate cohort schemas in CI/CD pipeline
3. Update version when making changes
4. Notify stakeholders of breaking changes with 90-day notice
