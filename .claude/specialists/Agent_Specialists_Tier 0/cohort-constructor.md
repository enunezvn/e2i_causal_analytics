# CohortConstructor - Agent Specialist

**Architecture Tier**: 0 (ML Foundation Layer)
**Agent Name**: `cohort_constructor`
**Agent Type**: Standard (tool-heavy, SLA-bound, no LLM)
**Position**: Between `scope_definer` and `data_preparer`
**Primary Purpose**: Construct eligible patient cohorts based on explicit clinical criteria with regulatory-compliant audit logging

---

## 🎯 Agent Overview

### Role in E2I Architecture

CohortConstructor is a **Tier 0 ML Foundation agent** that defines eligible patient populations before model training. It ensures:
- **Regulatory Compliance**: Explicit, defensible cohort definitions
- **Data Quality**: Only clinically valid patients enter the ML pipeline
- **Reproducibility**: Locked, versioned cohort specifications
- **Transparency**: Complete audit trail for regulatory submissions

### Position in Tier 0 Pipeline

```
scope_definer → cohort_constructor → data_preparer → model_selector → ...
                     │                    │
                     └─ Eligible patients ─┘
                     │
                     └─ Cohort specification
                     └─ Eligibility audit log
```

**Critical Flow**:
1. `scope_definer` provides ML problem definition
2. `cohort_constructor` applies eligibility criteria to define population
3. `data_preparer` validates quality of eligible population ONLY
4. `model_trainer` trains on eligible cohort

---

## 📋 Agent Classification

| Attribute | Value |
|-----------|-------|
| **Tier** | 0 (ML Foundation) |
| **Type** | Standard (Fast Path) |
| **LLM Usage** | None (rule-based, deterministic) |
| **SLA** | <120s for 100K patients |
| **Primary Path** | Fast (computational only) |
| **Memory Usage** | Working Memory (Redis) + Procedural (Supabase) |
| **Observability** | Full spans via observability_connector |

---

## 🔄 Workflow Design

### Node Structure (4 Nodes Sequential)

```
1. validate_config    (Validate cohort configuration)
    ↓
2. apply_criteria     (Apply inclusion/exclusion rules)
    ↓
3. validate_temporal  (Enforce lookback/followup periods)
    ↓
4. generate_metadata  (Audit log, summary statistics)
```

### Node Specifications

#### Node 1: validate_config
- **Purpose**: Validate cohort configuration completeness
- **Inputs**: ScopeSpec, CohortConfig
- **Outputs**: Validated config, required fields list
- **Validations**:
  - All required fields present
  - Criteria have valid operators
  - Temporal parameters valid (lookback/followup > 0)
  - Brand/indication supported
- **Latency**: <100ms

#### Node 2: apply_criteria
- **Purpose**: Apply inclusion and exclusion criteria sequentially
- **Inputs**: Patient DataFrame, validated config
- **Processing**:
  1. Apply each inclusion criterion (AND logic)
  2. Apply each exclusion criterion (AND NOT logic)
  3. Track removed patients per criterion
- **Outputs**: Eligible patient subset, eligibility log
- **Latency**: <60s for 100K patients
- **Operators Supported**:
  - EQUAL (`==`): `age == 18`
  - NOT_EQUAL (`!=`): `gender != 'M'`
  - GREATER_EQUAL (`>=`): `age >= 18`
  - LESS_EQUAL (`<=`): `severity <= 10`
  - IN (`in`): `code in ['L50.0', 'L50.1']`
  - NOT_IN (`not_in`): `code not_in ['L30.0']`
  - BETWEEN (`between`): `age between [18, 65]`
  - CONTAINS (`contains`): `notes contains 'urticaria'`

#### Node 3: validate_temporal
- **Purpose**: Enforce temporal eligibility requirements
- **Inputs**: Eligible patients, temporal config
- **Validations**:
  - Sufficient lookback period (e.g., 180 days pre-diagnosis)
  - Sufficient followup period (e.g., 90 days post-diagnosis)
  - Index date validity
- **Outputs**: Temporally valid patients, temporal exclusion log
- **Latency**: <30s for 100K patients

#### Node 4: generate_metadata
- **Purpose**: Generate comprehensive metadata and audit trail
- **Inputs**: Final eligible cohort, eligibility log
- **Outputs**: CohortMetadata, summary report
- **Computed Metrics**:
  - Total input patients
  - Eligible patients count
  - Exclusion rate
  - Per-criterion removal counts
  - Execution timestamp, duration
  - Configuration hash for reproducibility
- **Latency**: <5s

---

## 📊 State Definition

### CohortConstructorState (TypedDict)

```python
class CohortConstructorState(TypedDict):
    """State for cohort_constructor agent - Tier 0 contract."""
    
    # === INPUT (from scope_definer) ===
    query: str                              # User query
    scope_spec: Optional[dict]              # From scope_definer
    experiment_id: Optional[str]            # ML experiment ID
    brand: Optional[str]                    # "Remibrutinib" | "Fabhalta" | "Kisqali"
    indication: Optional[str]               # "csu" | "pnh" | "hr_her2"
    
    # === CONFIGURATION ===
    cohort_config: Optional[dict]           # CohortConfig as dict
    lookback_days: int                      # Default: 180
    followup_days: int                      # Default: 90
    index_date_field: str                   # Default: "diagnosis_date"
    
    # === DATA INPUTS ===
    patient_data_source: Optional[str]      # "patient_journeys" table
    patient_data: Optional[list[dict]]      # Raw patient records
    total_input_patients: int               # Input population size
    
    # === PROCESSING OUTPUTS ===
    eligible_patients: Optional[list[dict]] # Patients meeting criteria
    eligible_patient_count: int             # Final cohort size
    exclusion_rate: float                   # % excluded (0.0-1.0)
    
    # === ELIGIBILITY LOG ===
    eligibility_log: list[dict]             # Per-criterion tracking
    # Structure: [{
    #     "criterion_name": str,
    #     "criterion_type": "inclusion" | "exclusion",
    #     "removed_count": int,
    #     "remaining_count": int,
    #     "description": str
    # }]
    
    # === TEMPORAL VALIDATION ===
    temporal_validation_passed: bool        # All temporal checks passed
    temporal_exclusions: int                # Patients removed by temporal
    
    # === AUDIT METADATA ===
    cohort_spec: Optional[dict]             # Final cohort specification
    cohort_id: Optional[str]                # Unique cohort identifier
    execution_id: Optional[str]             # Unique execution identifier
    config_version: str                     # Configuration version
    config_hash: Optional[str]              # SHA256 of config for reproducibility
    
    # === SUMMARY ===
    summary_report: Optional[str]           # Human-readable summary
    recommended_actions: list[str]          # Next steps
    
    # === METADATA (Standard) ===
    execution_timestamp: Optional[str]      # ISO 8601 timestamp
    execution_time_ms: float                # Node execution latency
    
    # === ERROR HANDLING ===
    errors: list[dict]                      # Structured errors
    warnings: list[str]                     # Non-blocking warnings
    status: str                             # "completed" | "failed" | "partial"
```

---

## 🔧 Algorithm Specifications

### Inclusion Criteria Application (AND Logic)

For each inclusion criterion:
```python
def apply_inclusion_criterion(df: pd.DataFrame, criterion: Criterion) -> pd.DataFrame:
    """Apply single inclusion criterion, remove non-matching patients."""
    
    field = criterion.field
    operator = criterion.operator
    value = criterion.value
    
    # Track before count
    before_count = len(df)
    
    # Apply operator
    if operator == Operator.EQUAL:
        eligible_mask = df[field] == value
    elif operator == Operator.GREATER_EQUAL:
        eligible_mask = df[field] >= value
    elif operator == Operator.IN:
        eligible_mask = df[field].isin(value)
    elif operator == Operator.BETWEEN:
        eligible_mask = df[field].between(value[0], value[1])
    # ... other operators
    
    # Keep only matching patients
    df_eligible = df[eligible_mask]
    
    # Track removal
    removed = before_count - len(df_eligible)
    log_entry = {
        "criterion": criterion.description,
        "criterion_type": "inclusion",
        "removed": removed,
        "remaining": len(df_eligible)
    }
    
    return df_eligible, log_entry
```

### Exclusion Criteria Application (AND NOT Logic)

For each exclusion criterion:
```python
def apply_exclusion_criterion(df: pd.DataFrame, criterion: Criterion) -> pd.DataFrame:
    """Apply single exclusion criterion, remove matching patients."""
    
    field = criterion.field
    operator = criterion.operator
    value = criterion.value
    
    before_count = len(df)
    
    # Apply operator (inverted logic)
    if operator == Operator.EQUAL:
        exclusion_mask = df[field] == value
        df_eligible = df[~exclusion_mask]  # Keep non-matching
    elif operator == Operator.IN:
        exclusion_mask = df[field].isin(value)
        df_eligible = df[~exclusion_mask]
    # ... other operators
    
    removed = before_count - len(df_eligible)
    log_entry = {
        "criterion": criterion.description,
        "criterion_type": "exclusion",
        "removed": removed,
        "remaining": len(df_eligible)
    }
    
    return df_eligible, log_entry
```

### Temporal Validation Formula

```python
def validate_temporal_eligibility(
    patient: pd.Series,
    index_date_field: str,
    lookback_days: int,
    followup_days: int
) -> bool:
    """Validate patient has sufficient lookback and followup data."""
    
    index_date = patient[index_date_field]
    first_observation = patient['first_observation_date']
    last_observation = patient['last_observation_date']
    
    # Lookback requirement: first_observation <= index_date - lookback_days
    lookback_satisfied = (index_date - first_observation).days >= lookback_days
    
    # Followup requirement: last_observation >= index_date + followup_days
    followup_satisfied = (last_observation - index_date).days >= followup_days
    
    return lookback_satisfied and followup_satisfied
```

---

## 📈 Performance Requirements

### Latency SLA: <120s for 100K patients

| Node | Target | Typical |
|------|--------|---------|
| validate_config | <100ms | ~50ms |
| apply_criteria | <60s | ~30s |
| validate_temporal | <30s | ~15s |
| generate_metadata | <5s | ~2s |
| **Total** | **<120s** | **~50s** |

### Resource Constraints

- **Memory**: <5GB for 100K patients (~50KB/patient)
- **CPU**: Single-threaded (parallelization future optimization)
- **Disk I/O**: Minimal (in-memory processing)

### Scalability

- **1K patients**: ~0.5s
- **10K patients**: ~5s
- **100K patients**: ~50s
- **1M patients**: ~500s (batch processing recommended)

---

## 🗄️ Database Integration

### Tables Written

#### 1. `ml_cohort_definitions` (Procedural Memory)

```sql
CREATE TABLE ml_cohort_definitions (
    cohort_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    brand TEXT,
    indication TEXT,
    cohort_name TEXT,
    inclusion_criteria JSONB,
    exclusion_criteria JSONB,
    lookback_days INTEGER,
    followup_days INTEGER,
    config_version TEXT,
    config_hash TEXT,
    created_timestamp TIMESTAMP,
    status TEXT  -- "active" | "locked" | "deprecated"
);
```

#### 2. `ml_cohort_executions`

```sql
CREATE TABLE ml_cohort_executions (
    execution_id TEXT PRIMARY KEY,
    cohort_id TEXT REFERENCES ml_cohort_definitions(cohort_id),
    execution_timestamp TIMESTAMP,
    total_input_patients INTEGER,
    eligible_patient_count INTEGER,
    exclusion_rate DECIMAL(5,4),
    execution_time_ms INTEGER,
    status TEXT  -- "completed" | "failed"
);
```

#### 3. `ml_cohort_eligibility_log`

```sql
CREATE TABLE ml_cohort_eligibility_log (
    log_id SERIAL PRIMARY KEY,
    execution_id TEXT REFERENCES ml_cohort_executions(execution_id),
    criterion_name TEXT,
    criterion_type TEXT,  -- "inclusion" | "exclusion"
    removed_count INTEGER,
    remaining_count INTEGER,
    criterion_order INTEGER
);
```

#### 4. `ml_patient_cohort_assignments`

```sql
CREATE TABLE ml_patient_cohort_assignments (
    patient_journey_id TEXT,
    cohort_id TEXT,
    execution_id TEXT,
    is_eligible BOOLEAN,
    failed_criteria JSONB,  -- List of failed criterion names
    PRIMARY KEY (patient_journey_id, cohort_id, execution_id)
);
```

---

## 🔗 Integration Contracts

### Input Contract (from scope_definer)

```python
@dataclass
class CohortConstructorInput:
    """Input from scope_definer."""
    
    query: str
    scope_spec: dict  # Includes brand, indication, target definition
    experiment_id: str
    patient_data_source: str  # Table name
```

### Output Contract (to data_preparer)

```python
@dataclass
class CohortConstructorOutput:
    """Output for data_preparer."""
    
    cohort_spec: dict  # Complete cohort definition
    eligible_patient_ids: list[str]  # Patient IDs meeting criteria
    cohort_id: str
    execution_id: str
    eligible_patient_count: int
    exclusion_rate: float
    summary_report: str
    eligibility_log: list[dict]
```

---

## 🧪 Pre-Built Configurations

### 1. Remibrutinib CSU (Chronic Spontaneous Urticaria)

```python
REMIBRUTINIB_CSU_CONFIG = {
    "inclusion_criteria": [
        {"field": "age_at_diagnosis", "operator": ">=", "value": 18,
         "description": "Age ≥18 years"},
        {"field": "diagnosis_code", "operator": "in", "value": ["L50.0", "L50.1", "L50.8", "L50.9"],
         "description": "ICD-10: CSU diagnosis"},
        {"field": "urticaria_severity_uas7", "operator": ">=", "value": 16,
         "description": "UAS7 ≥16 (moderate-to-severe)"},
        {"field": "antihistamine_failures_count", "operator": ">=", "value": 1,
         "description": "≥1 antihistamine failure"}
    ],
    "exclusion_criteria": [
        {"field": "is_pregnant", "operator": "==", "value": True,
         "description": "Pregnancy"},
        {"field": "immunodeficiency_severe", "operator": "==", "value": True,
         "description": "Severe immunodeficiency"},
        {"field": "urticaria_type", "operator": "==", "value": "physical_only",
         "description": "Physical urticaria only"}
    ],
    "lookback_days": 180,
    "followup_days": 90
}
```

### 2. Fabhalta PNH (Paroxysmal Nocturnal Hemoglobinuria)

```python
FABHALTA_PNH_CONFIG = {
    "inclusion_criteria": [
        {"field": "diagnosis_code", "operator": "==", "value": "D59.5",
         "description": "ICD-10: PNH diagnosis"},
        {"field": "complement_inhibitor_status", "operator": "in", 
         "value": ["naive", "switching"],
         "description": "Complement inhibitor naive or switching"},
        {"field": "ldh_level", "operator": ">=", "value": 1.5,
         "description": "LDH ≥1.5x ULN"}
    ],
    "exclusion_criteria": [
        {"field": "active_infection", "operator": "==", "value": True,
         "description": "Active infection"},
        {"field": "pregnancy_or_breastfeeding", "operator": "==", "value": True,
         "description": "Pregnancy or breastfeeding"}
    ],
    "lookback_days": 365,
    "followup_days": 180
}
```

### 3. Kisqali HR+/HER2- (Breast Cancer)

```python
KISQALI_CONFIG = {
    "inclusion_criteria": [
        {"field": "diagnosis_code", "operator": "in", 
         "value": ["C50.0", "C50.1", "C50.9"],
         "description": "ICD-10: Breast cancer"},
        {"field": "hr_status", "operator": "==", "value": "positive",
         "description": "HR+ status"},
        {"field": "her2_status", "operator": "==", "value": "negative",
         "description": "HER2- status"},
        {"field": "ecog_performance_status", "operator": "<=", "value": 2,
         "description": "ECOG ≤2"}
    ],
    "exclusion_criteria": [
        {"field": "prior_cdk_inhibitor", "operator": "==", "value": True,
         "description": "Prior CDK4/6 inhibitor"},
        {"field": "metastatic_cns", "operator": "==", "value": True,
         "description": "CNS metastases"}
    ],
    "lookback_days": 180,
    "followup_days": 365
}
```

---

## 🚨 Error Handling

### Error Categories

| Code | Category | Description | Recovery |
|------|----------|-------------|----------|
| `CC_001` | VALIDATION | Invalid cohort config | Return validation errors, block execution |
| `CC_002` | DATA_MISSING | Required fields missing | List missing fields, request data correction |
| `CC_003` | EMPTY_COHORT | All patients excluded | Return eligibility log, suggest criteria relaxation |
| `CC_004` | TEMPORAL_INSUFFICIENT | Insufficient temporal data | Report temporal gaps, adjust requirements |
| `CC_005` | OPERATOR_UNSUPPORTED | Unknown operator | List supported operators, reject criterion |

### Fallback Strategy

```python
# No computational fallback (deterministic rules)
# If cohort construction fails, block pipeline
# CRITICAL: Do NOT proceed to data_preparer without cohort
```

---

## 📝 Memory Architecture

### Working Memory (Redis)

```python
WORKING_MEMORY_KEYS = [
    "cohort_constructor:current_execution:{execution_id}",
    "cohort_constructor:patient_eligibility_cache:{cohort_id}"
]
```

### Procedural Memory (Supabase)

- `ml_cohort_definitions` - Cohort specifications
- `ml_cohort_executions` - Execution history
- `ml_cohort_eligibility_log` - Criterion-level audit
- `ml_patient_cohort_assignments` - Patient-level eligibility

### Semantic Memory (FalkorDB)

Not used (cohort construction is procedural, not semantic).

---

## 🔍 Observability

### Spans Emitted

```python
SPANS = [
    "cohort_constructor.validate_config",
    "cohort_constructor.apply_criteria",
    "cohort_constructor.validate_temporal",
    "cohort_constructor.generate_metadata",
    "cohort_constructor.full_pipeline"
]
```

### Metrics Tracked

- `execution_time_ms` per node
- `eligible_patient_count`
- `exclusion_rate`
- `per_criterion_removal_counts`
- `temporal_exclusion_count`

---

## 🧪 Testing Requirements

### Unit Tests (20+)

- `test_equal_operator`
- `test_greater_equal_operator`
- `test_in_operator`
- `test_between_operator`
- `test_contains_operator`
- `test_inclusion_criterion_application`
- `test_exclusion_criterion_application`
- `test_temporal_lookback_validation`
- `test_temporal_followup_validation`
- `test_config_validation`
- `test_empty_cohort_handling`

### Integration Tests (10+)

- `test_remibrutinib_config_execution`
- `test_fabhalta_config_execution`
- `test_kisqali_config_execution`
- `test_full_pipeline_100k_patients`
- `test_database_writes`
- `test_handoff_to_data_preparer`

### Performance Tests

- `test_latency_1k_patients` (<1s)
- `test_latency_10k_patients` (<10s)
- `test_latency_100k_patients` (<120s)
- `test_memory_usage_100k_patients` (<5GB)

---

## 📖 Related Documentation

### Tier 0 Documentation
- [tier0-overview.md](tier0-overview.md) - Complete Tier 0 architecture
- [tier0-contracts.md](tier0-contracts.md) - Tier 0 integration contracts
- [AGENT_IMPLEMENTATION_PROTOCOL.md](AGENT_IMPLEMENTATION_PROTOCOL.md) - Implementation guide

### E2I Context
- [summary-v4.md](context/summary-v4.md) - E2I project overview
- [kpi-dictionary.md](context/kpi-dictionary.md) - KPI definitions
- [data-architecture.md](context/data-architecture.md) - Data schema

### Implementation Files
- `cohort_constructor.py` - Production implementation
- `cohort_schema.sql` - Database schema
- `test_cohort_constructor.py` - Test suite

---

## ✅ Deployment Checklist

- [ ] Cohort configurations validated for all 3 brands
- [ ] Database migrations applied (4 new tables)
- [ ] Integration tests passing (scope_definer → cohort_constructor → data_preparer)
- [ ] Performance SLA validated (<120s for 100K patients)
- [ ] Observability spans emitting correctly
- [ ] Audit logging comprehensive for regulatory compliance
- [ ] Error handling covers all failure modes
- [ ] Documentation complete (specialist, contracts, handoffs)

---

**Version**: V1.0
**Last Updated**: 2026-01-15
**Tier**: 0 (ML Foundation)
**Agent Type**: Standard (Fast Path, No LLM)
**SLA**: <120s for 100K patients
**Critical Workflows**: Cohort Eligibility, Temporal Validation, Audit Logging
