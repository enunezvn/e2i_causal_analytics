# CohortConstructor Contract

**Tier**: 0 (ML Foundation)
**Agent**: `cohort_constructor`
**Position**: Between `scope_definer` and `data_preparer`
**Purpose**: Define eligible patient populations with explicit clinical criteria before ML training

---

## Overview

CohortConstructor applies inclusion/exclusion criteria to define eligible patient cohorts for ML experiments. It ensures:
- Only clinically valid patients enter the ML pipeline
- Complete audit trail for regulatory compliance
- Reproducible cohort definitions
- Temporal eligibility validation (lookback/followup periods)

**Pipeline Position:**
```
scope_definer → cohort_constructor → data_preparer → model_selector → ...
```

**Latency Budget:** <120s for 100K patients

---

## Input Contract

### CohortConstructorInput

```yaml
# cohort_constructor_input.yaml
cohort_request:
  request_id: string              # UUID
  timestamp: datetime             # ISO 8601
  
  # From scope_definer
  query: string                   # Original user query
  scope_spec:
    experiment_id: string         # "exp_remibrutinib_csu_20260115"
    problem_type: string          # "binary_classification" | "regression"
    target_variable: string       # "patient_conversion"
    brand: string                 # "Remibrutinib" | "Fabhalta" | "Kisqali"
    indication: string            # "csu" | "pnh" | "hr_her2"
    
  # Cohort configuration (optional - uses pre-built if not provided)
  cohort_config:
    cohort_name: string           # "Remibrutinib CSU Eligible Population"
    
    # Inclusion criteria (AND logic)
    inclusion_criteria:
      - field: string             # "age_at_diagnosis"
        operator: string          # ">=", "==", "in", "between", "contains"
        value: any                # 18 OR [18, 65] OR ["L50.0", "L50.1"]
        description: string       # "Age ≥18 years"
        clinical_rationale: string # "Adult population per label"
        
    # Exclusion criteria (AND NOT logic)
    exclusion_criteria:
      - field: string
        operator: string
        value: any
        description: string
        clinical_rationale: string
        
    # Temporal requirements
    lookback_days: int            # Default: 180
    followup_days: int            # Default: 90
    index_date_field: string      # Default: "diagnosis_date"
    
    # Data quality
    required_fields: list[string] # ["age_at_diagnosis", "diagnosis_code", ...]
    
    # Metadata
    version: string               # "1.0.0"
    
  # Data source
  patient_data_source:
    table: string                 # "patient_journeys"
    filters: object               # {"brand": "Remibrutinib"} (optional pre-filter)
```

### Required Input Keys

```python
REQUIRED_KEYS = [
    "request_id",
    "scope_spec",
    "scope_spec.experiment_id",
    "scope_spec.brand",
    "scope_spec.indication",
    "patient_data_source"
]
```

### Validation Rules

1. **Brand Valid**: Must be "Remibrutinib" | "Fabhalta" | "Kisqali"
2. **Indication Valid**: Must match brand (e.g., "csu" for Remibrutinib)
3. **Operators Valid**: Must be in SUPPORTED_OPERATORS
4. **Temporal Parameters**: lookback_days > 0, followup_days > 0
5. **Required Fields**: All required_fields must exist in patient data

---

## Output Contract

### CohortConstructorOutput

```yaml
# cohort_constructor_output.yaml
cohort_response:
  request_id: string              # Echo from input
  experiment_id: string
  timestamp: datetime
  processing_time_ms: int
  
  # === COHORT SPECIFICATION ===
  cohort_spec:
    cohort_id: string             # "cohort_remibrutinib_csu_v1_20260115"
    cohort_name: string
    brand: string
    indication: string
    
    # Criteria applied
    inclusion_criteria:
      - field: string
        operator: string
        value: any
        description: string
        clinical_rationale: string
        
    exclusion_criteria:
      - field: string
        operator: string
        value: any
        description: string
        clinical_rationale: string
        
    # Temporal requirements
    lookback_days: int
    followup_days: int
    index_date_field: string
    
    # Version control
    version: string               # "1.0.0"
    config_hash: string           # SHA256 of configuration
    status: string                # "active" | "locked"
    
  # === ELIGIBLE POPULATION ===
  eligible_patients:
    patient_ids: list[string]     # List of eligible patient_journey_ids
    count: int                    # Total eligible patients
    
  # === ELIGIBILITY STATISTICS ===
  eligibility_stats:
    total_input_patients: int     # Starting population
    eligible_patient_count: int   # Final cohort size
    exclusion_rate: float         # 0.0-1.0 (% excluded)
    
    # Per-criterion tracking
    eligibility_log:
      - criterion_name: string    # "age_at_diagnosis"
        criterion_type: string    # "inclusion" | "exclusion"
        criterion_order: int      # 1, 2, 3, ...
        removed_count: int        # Patients removed by this criterion
        remaining_count: int      # Patients remaining after this criterion
        description: string       # "Age ≥18 years"
        
    # Temporal validation
    temporal_validation:
      passed: bool                # All temporal checks passed
      lookback_exclusions: int    # Patients removed by insufficient lookback
      followup_exclusions: int    # Patients removed by insufficient followup
      
  # === EXECUTION METADATA ===
  execution_metadata:
    execution_id: string          # "exec_20260115_143022"
    execution_timestamp: string   # ISO 8601
    execution_time_ms: float      # Total execution time
    
    # Node latencies
    node_latencies:
      validate_config_ms: float
      apply_criteria_ms: float
      validate_temporal_ms: float
      generate_metadata_ms: float
      
    # Database records created
    database_records:
      cohort_definition_id: string
      cohort_execution_id: string
      eligibility_log_entries: int
      patient_assignments: int
      
  # === SUMMARY ===
  summary:
    summary_report: string        # Human-readable summary
    # Example:
    # "Cohort constructed for Remibrutinib CSU population.
    #  Input: 50,000 patients
    #  Eligible: 8,500 patients (17.0%)
    #  Excluded: 41,500 patients (83.0%)
    #  Top exclusions:
    #  - Insufficient UAS7 severity: 30,000 (60%)
    #  - Age <18: 8,000 (16%)
    #  - Pregnancy: 3,500 (7%)"
    
    recommended_actions: list[string]
    # Examples:
    # - "Proceed to data_preparer with 8,500 eligible patients"
    # - "Consider relaxing UAS7 threshold if cohort too small"
    
  # === HANDOFF DATA (for data_preparer) ===
  handoff:
    cohort_id: string
    eligible_patient_ids: list[string]
    cohort_spec: object           # Full cohort specification
    quality_checks_required: list[string]
    # Examples:
    # - "validate_temporal_completeness"
    # - "check_feature_availability"
    
  # === STATUS ===
  status: string                  # "completed" | "partial" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys

```python
REQUIRED_KEYS = [
    "cohort_spec",
    "eligible_patients",
    "eligibility_stats",
    "summary"
]
```

### Validation Rules

1. **Cohort ID Valid**: Must follow format "cohort_{brand}_{indication}_v{version}_{timestamp}"
2. **Eligible Count**: eligible_patient_count >= 0
3. **Exclusion Rate**: 0.0 <= exclusion_rate <= 1.0
4. **Eligibility Log Complete**: One entry per criterion applied
5. **Temporal Validation**: If temporal checks enabled, temporal_validation.passed must be present

---

## State Contract

### CohortConstructorState (Complete)

```python
from typing import TypedDict, Literal, Optional

class CohortConstructorState(TypedDict):
    """Complete state definition for cohort_constructor agent."""
    
    # === INPUT (from scope_definer) ===
    query: str
    scope_spec: Optional[dict]
    experiment_id: Optional[str]
    brand: Optional[str]              # "Remibrutinib" | "Fabhalta" | "Kisqali"
    indication: Optional[str]          # "csu" | "pnh" | "hr_her2"
    
    # === CONFIGURATION ===
    cohort_config: Optional[dict]      # CohortConfig as dict
    lookback_days: int                 # Default: 180
    followup_days: int                 # Default: 90
    index_date_field: str              # Default: "diagnosis_date"
    
    # === DATA INPUTS ===
    patient_data_source: Optional[str] # "patient_journeys"
    patient_data: Optional[list[dict]] # Raw patient records
    total_input_patients: int          # Input population size
    
    # === NODE 1: validate_config OUTPUTS ===
    config_valid: bool
    config_validation_errors: list[str]
    required_fields: list[str]
    supported_operators: list[str]
    
    # === NODE 2: apply_criteria OUTPUTS ===
    eligible_patients: Optional[list[dict]]
    eligible_patient_count: int
    exclusion_rate: float              # 0.0-1.0
    eligibility_log: list[dict]
    # Structure: [{
    #     "criterion_name": str,
    #     "criterion_type": "inclusion" | "exclusion",
    #     "criterion_order": int,
    #     "removed_count": int,
    #     "remaining_count": int,
    #     "description": str
    # }]
    
    # === NODE 3: validate_temporal OUTPUTS ===
    temporal_validation_passed: bool
    temporal_exclusions: int
    lookback_failures: int
    followup_failures: int
    temporally_eligible_patients: Optional[list[dict]]
    
    # === NODE 4: generate_metadata OUTPUTS ===
    cohort_spec: Optional[dict]
    cohort_id: Optional[str]
    execution_id: Optional[str]
    config_version: str
    config_hash: Optional[str]         # SHA256 for reproducibility
    summary_report: Optional[str]
    recommended_actions: list[str]
    
    # === DATABASE IDs ===
    cohort_definition_id: Optional[str]
    cohort_execution_id: Optional[str]
    eligibility_log_entry_count: int
    patient_assignment_count: int
    
    # === HANDOFF DATA ===
    handoff_data: Optional[dict]       # For data_preparer
    quality_checks_required: list[str]
    
    # === NODE LATENCIES ===
    validate_config_latency_ms: float
    apply_criteria_latency_ms: float
    validate_temporal_latency_ms: float
    generate_metadata_latency_ms: float
    
    # === METADATA (Standard) ===
    execution_timestamp: Optional[str] # ISO 8601
    execution_time_ms: float           # Total execution time
    
    # === ERROR HANDLING ===
    errors: list[dict]                 # Structured errors
    warnings: list[str]                # Non-blocking warnings
    status: str                        # "completed" | "failed" | "partial"
```

---

## Additional TypedDicts

### Criterion

```python
class Criterion(TypedDict):
    """Single cohort eligibility criterion."""
    
    field: str                         # "age_at_diagnosis"
    operator: str                      # ">=", "==", "in", "between", "contains"
    value: Any                         # 18 OR [18, 65] OR ["L50.0", "L50.1"]
    description: Optional[str]         # "Age ≥18 years"
    clinical_rationale: Optional[str]  # "Adult population per label"
```

### CohortConfig

```python
class CohortConfig(TypedDict):
    """Complete cohort configuration."""
    
    # Identity
    brand: str
    indication: str
    cohort_name: str
    
    # Criteria
    inclusion_criteria: list[Criterion]
    exclusion_criteria: list[Criterion]
    
    # Temporal
    lookback_days: int
    followup_days: int
    index_date_field: str
    
    # Data quality
    required_fields: list[str]
    
    # Metadata
    version: str
    created_date: str                  # ISO 8601
    status: str                        # "draft" | "active" | "locked"
```

### EligibilityLogEntry

```python
class EligibilityLogEntry(TypedDict):
    """Single criterion application result."""
    
    criterion_name: str
    criterion_type: str                # "inclusion" | "exclusion"
    criterion_order: int
    removed_count: int
    remaining_count: int
    description: str
```

### TemporalValidationResult

```python
class TemporalValidationResult(TypedDict):
    """Temporal eligibility validation result."""
    
    passed: bool
    lookback_exclusions: int
    followup_exclusions: int
    total_temporal_exclusions: int
```

---

## Supported Operators

| Operator | Symbol | Example | Use Case |
|----------|--------|---------|----------|
| EQUAL | `==` | `age == 18` | Exact match (rarely used for age) |
| NOT_EQUAL | `!=` | `gender != 'M'` | Exclusion by value |
| GREATER | `>` | `severity > 7` | Strict threshold |
| GREATER_EQUAL | `>=` | `age >= 18` | Minimum age/threshold |
| LESS | `<` | `age < 65` | Maximum threshold |
| LESS_EQUAL | `<=` | `severity <= 10` | Maximum severity |
| IN | `in` | `code in ['L50.0', 'L50.1']` | Multiple allowed values |
| NOT_IN | `not_in` | `code not_in ['L30.0']` | Multiple excluded values |
| BETWEEN | `between` | `age between [18, 65]` | Inclusive range |
| CONTAINS | `contains` | `notes contains 'urticaria'` | Text search |

---

## Error Codes

### CohortConstructor-Specific Errors

| Code | Category | Description | Recovery |
|------|----------|-------------|----------|
| `CC_001` | VALIDATION | Invalid cohort config | Return validation errors, block execution |
| `CC_002` | DATA_MISSING | Required fields missing in patient data | List missing fields, request data fix |
| `CC_003` | EMPTY_COHORT | All patients excluded (cohort size = 0) | Return eligibility log, suggest criteria relaxation |
| `CC_004` | TEMPORAL_INSUFFICIENT | <100 patients with sufficient temporal data | Report temporal gaps, suggest requirement adjustment |
| `CC_005` | OPERATOR_UNSUPPORTED | Unknown operator in criterion | List supported operators, reject criterion |
| `CC_006` | BRAND_UNSUPPORTED | Unknown brand specified | List supported brands (Remibrutinib, Fabhalta, Kisqali) |
| `CC_007` | DATABASE_WRITE_FAILED | Failed to write cohort to database | Log error, continue with in-memory cohort |

### Error Response Structure

```python
{
    "error_code": "CC_003",
    "category": "EMPTY_COHORT",
    "message": "All 50,000 patients excluded by eligibility criteria",
    "details": {
        "top_exclusions": [
            {"criterion": "urticaria_severity_uas7 >= 16", "removed": 30000},
            {"criterion": "age_at_diagnosis >= 18", "removed": 8000},
            {"criterion": "is_pregnant == False", "removed": 3500}
        ]
    },
    "recovery_suggestions": [
        "Consider relaxing UAS7 threshold from 16 to 12",
        "Review pregnancy exclusion necessity",
        "Validate data quality (e.g., missing UAS7 values)"
    ],
    "timestamp": "2026-01-15T14:30:22Z"
}
```

---

## Pre-Built Configurations

### 1. Remibrutinib CSU

```yaml
brand: Remibrutinib
indication: csu
cohort_name: "Remibrutinib CSU Eligible Population"

inclusion_criteria:
  - field: age_at_diagnosis
    operator: ">="
    value: 18
    description: "Age ≥18 years"
    clinical_rationale: "Adult population per FDA label"
    
  - field: diagnosis_code
    operator: "in"
    value: ["L50.0", "L50.1", "L50.8", "L50.9"]
    description: "ICD-10: CSU diagnosis"
    clinical_rationale: "Chronic spontaneous urticaria diagnosis codes"
    
  - field: urticaria_severity_uas7
    operator: ">="
    value: 16
    description: "UAS7 ≥16 (moderate-to-severe)"
    clinical_rationale: "Trial inclusion threshold"
    
  - field: antihistamine_failures_count
    operator: ">="
    value: 1
    description: "≥1 antihistamine failure"
    clinical_rationale: "Refractory to standard therapy"

exclusion_criteria:
  - field: is_pregnant
    operator: "=="
    value: True
    description: "Pregnancy"
    clinical_rationale: "Safety exclusion per label"
    
  - field: immunodeficiency_severe
    operator: "=="
    value: True
    description: "Severe immunodeficiency"
    clinical_rationale: "Safety concern"

lookback_days: 180
followup_days: 90
required_fields:
  - age_at_diagnosis
  - diagnosis_code
  - urticaria_severity_uas7
  - antihistamine_failures_count
  - is_pregnant
  - immunodeficiency_severe
```

### 2. Fabhalta PNH

```yaml
brand: Fabhalta
indication: pnh
cohort_name: "Fabhalta PNH Eligible Population"

inclusion_criteria:
  - field: diagnosis_code
    operator: "=="
    value: "D59.5"
    description: "ICD-10: PNH diagnosis"
    
  - field: complement_inhibitor_status
    operator: "in"
    value: ["naive", "switching"]
    description: "Complement inhibitor naive or switching"
    
  - field: ldh_level
    operator: ">="
    value: 1.5
    description: "LDH ≥1.5x ULN"

exclusion_criteria:
  - field: active_infection
    operator: "=="
    value: True
    description: "Active infection"
    
  - field: pregnancy_or_breastfeeding
    operator: "=="
    value: True
    description: "Pregnancy or breastfeeding"

lookback_days: 365
followup_days: 180
```

### 3. Kisqali HR+/HER2-

```yaml
brand: Kisqali
indication: hr_her2
cohort_name: "Kisqali HR+/HER2- Breast Cancer Eligible Population"

inclusion_criteria:
  - field: diagnosis_code
    operator: "in"
    value: ["C50.0", "C50.1", "C50.9"]
    description: "ICD-10: Breast cancer"
    
  - field: hr_status
    operator: "=="
    value: "positive"
    description: "HR+ status"
    
  - field: her2_status
    operator: "=="
    value: "negative"
    description: "HER2- status"
    
  - field: ecog_performance_status
    operator: "<="
    value: 2
    description: "ECOG ≤2"

exclusion_criteria:
  - field: prior_cdk_inhibitor
    operator: "=="
    value: True
    description: "Prior CDK4/6 inhibitor"
    
  - field: metastatic_cns
    operator: "=="
    value: True
    description: "CNS metastases"

lookback_days: 180
followup_days: 365
```

---

## Inter-Agent Communication

### cohort_constructor → data_preparer Handoff

```yaml
# Handoff structure
handoff:
  from: cohort_constructor
  to: data_preparer
  
  data_passed:
    cohort_spec: object           # Complete cohort definition
    eligible_patient_ids: list[string]
    cohort_id: string
    execution_id: string
    eligible_patient_count: int
    exclusion_rate: float
    
  context:
    quality_checks_required:
      - "validate_temporal_completeness"
      - "check_feature_availability_for_eligible_patients"
      
  usage:
    "data_preparer validates quality of eligible cohort ONLY,
     not the entire patient population. QC gate applies to
     cohort members only."
```

---

## Quality Assurance

### Cohort Construction Quality Checks

```python
class CohortQualityContract:
    """Quality checks specific to cohort construction."""
    
    @staticmethod
    def validate_cohort_not_empty(eligible_count: int) -> bool:
        """Ensure cohort has at least 100 patients (minimum viable)."""
        return eligible_count >= 100
    
    @staticmethod
    def validate_exclusion_rate_reasonable(exclusion_rate: float) -> bool:
        """Ensure exclusion rate is not extreme (e.g., >99% or <5%)."""
        return 0.05 <= exclusion_rate <= 0.95
    
    @staticmethod
    def validate_temporal_coverage(temporal_passed_rate: float) -> bool:
        """Ensure at least 50% of inclusion-eligible have sufficient temporal data."""
        return temporal_passed_rate >= 0.50
    
    @staticmethod
    def validate_config_locked(config: CohortConfig) -> bool:
        """For regulatory submissions, config must be locked before execution."""
        return config['status'] == 'locked'
```

---

## Validation Tests

```bash
# Input contract validation
pytest tests/integration/test_cohort_constructor_contracts.py::test_input_contract

# Output contract validation
pytest tests/integration/test_cohort_constructor_contracts.py::test_output_contract

# State contract validation
pytest tests/integration/test_cohort_constructor_contracts.py::test_state_contract

# Pre-built configurations
pytest tests/integration/test_cohort_constructor_contracts.py::test_remibrutinib_config
pytest tests/integration/test_cohort_constructor_contracts.py::test_fabhalta_config
pytest tests/integration/test_cohort_constructor_contracts.py::test_kisqali_config

# Error handling
pytest tests/integration/test_cohort_constructor_contracts.py::test_empty_cohort_error
pytest tests/integration/test_cohort_constructor_contracts.py::test_missing_fields_error

# Performance
pytest tests/integration/test_cohort_constructor_contracts.py::test_latency_100k_patients

# Handoff validation
pytest tests/integration/test_cohort_constructor_contracts.py::test_handoff_to_data_preparer
```

---

## Change Log

| Date | Change |
|------|--------|
| 2026-01-15 | Initial contract creation for CohortConstructor (Tier 0) |

---

**Version**: 1.0
**Last Updated**: 2026-01-15
**Owner**: E2I ML Foundation Team
