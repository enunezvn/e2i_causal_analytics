# CohortConstructor - E2I Tier 0 Agent

## Overview

CohortConstructor is a production-ready agent for the E2I (Evidence-to-Impact) Causal Analytics system that constructs eligible patient cohorts based on explicit clinical criteria. It provides regulatory-compliant cohort definition with comprehensive audit logging for pharmaceutical real-world evidence studies.

### Key Features

✅ **Explicit Eligibility Criteria** - Rule-based inclusion/exclusion with clinical rationale
✅ **Temporal Validation** - Lookback and follow-up period enforcement  
✅ **Multi-Brand Support** - Pre-configured for Remibrutinib, Fabhalta, Kisqali
✅ **Audit Trail** - Comprehensive logging for regulatory compliance
✅ **Flexible Operators** - Support for ==, !=, >=, <=, in, between, contains
✅ **E2I Integration** - Ready for Tier 0 ML Foundation pipeline
✅ **Database Integration** - Supabase storage for cohort definitions and executions

---

## Quick Start

### Installation

```bash
pip install pandas numpy supabase
```

### Basic Usage

```python
from cohort_constructor import create_cohort_quick
import pandas as pd

# Load patient data
patient_df = pd.read_parquet('patient_journeys.parquet')

# Construct cohort
eligible_df, metadata = create_cohort_quick('remibrutinib', 'csu', patient_df)

print(f"Eligible: {len(eligible_df):,} / {len(patient_df):,}")
print(f"Exclusion Rate: {metadata['exclusion_rate']:.1%}")
```

---

## Architecture

### Position in E2I Tier 0

```
Tier 0: ML Foundation
├─ scope_definer
├─ cohort_constructor  ← NEW AGENT
├─ data_preparer (receives eligible patients only)
├─ model_selector
├─ model_trainer
├─ feature_analyzer
├─ model_deployer
└─ observability_connector
```

### Agent Specifications

- **Agent Name**: `cohort_constructor`
- **Tier**: 0 (ML Foundation)
- **Type**: Standard (tool-heavy, SLA-bound)
- **SLA**: 120 seconds
- **Inputs**: ScopeSpec, PatientData
- **Outputs**: CohortSpec, EligiblePatients, EligibilityLog

---

## Configuration

### Pre-Built Configurations

#### 1. Remibrutinib CSU (Chronic Spontaneous Urticaria)

```python
from cohort_constructor import CohortConfig

config = CohortConfig.from_brand('remibrutinib', 'csu')

# Inclusion Criteria:
# - Age ≥ 18 years
# - ICD-10: L50.0, L50.1, L50.8, L50.9 (CSU diagnosis)
# - UAS7 ≥ 16 (moderate-to-severe)
# - ≥1 antihistamine failure
# - ≥90 days follow-up

# Exclusion Criteria:
# - Pregnancy
# - Severe immunodeficiency
# - Physical urticaria only
```

#### 2. Fabhalta PNH/C3G (Rare Disease)

```python
config = CohortConfig.from_brand('fabhalta', 'pnh')

# Inclusion: PNH diagnosis, complement inhibitor naive/switching
# Exclusion: Pregnancy, active infection
```

#### 3. Kisqali HR+/HER2- Breast Cancer

```python
config = CohortConfig.from_brand('kisqali', 'hr_positive_bc')

# Inclusion: HR+/HER2-, advanced/metastatic stage
# Exclusion: Pregnancy, severe liver impairment
```

### Custom Configuration

```python
from cohort_constructor import CohortConfig, Criterion, Operator

config = CohortConfig(
    brand='my_drug',
    indication='my_indication',
    cohort_name='My Custom Cohort',
    inclusion_criteria=[
        Criterion(
            field='age_at_diagnosis',
            operator=Operator.GREATER_EQUAL,
            value=18,
            description='Adult patients',
            clinical_rationale='Safety not established in pediatrics'
        ),
        Criterion(
            field='disease_severity',
            operator=Operator.BETWEEN,
            value=[5, 10],
            description='Moderate-to-severe disease'
        )
    ],
    exclusion_criteria=[
        Criterion(
            field='pregnancy_flag',
            operator=Operator.EQUAL,
            value=True,
            description='Exclude pregnant patients'
        )
    ],
    lookback_days=180,
    followup_days=90
)
```

---

## Usage Examples

### Example 1: Standalone Cohort Construction

```python
from cohort_constructor import CohortConstructor, CohortConfig
import pandas as pd

# Load data
df = pd.read_csv('patients.csv')

# Create configuration
config = CohortConfig.from_brand('remibrutinib', 'csu')

# Construct cohort
constructor = CohortConstructor(config)
eligible_df, metadata = constructor.construct_cohort(df)

# Print summary
print(constructor.summary_report(metadata))

# Save results
eligible_df.to_parquet('eligible_patients.parquet')
config.to_json('cohort_config.json')
```

### Example 2: Compare Multiple Cohorts

```python
from cohort_constructor import compare_cohorts, CohortConfig

# Create variations
config_strict = CohortConfig.from_brand('remibrutinib', 'csu')

config_relaxed = CohortConfig.from_brand('remibrutinib', 'csu')
config_relaxed.inclusion_criteria[2].value = 10  # Lower severity threshold

# Compare
comparison = compare_cohorts(df, [config_strict, config_relaxed])
print(comparison)

#    cohort_name                         eligible_population  exclusion_rate
# 0  Remibrutinib CSU Eligible          28450               0.187
# 1  Remibrutinib CSU Relaxed           32100               0.082
```

### Example 3: E2I Tier 0 Integration

```python
from e2i_tier0_integration import CohortConstructorAgent

agent = CohortConstructorAgent()

input_data = {
    'scope_spec': {
        'brand': 'remibrutinib',
        'indication': 'csu',
        'target_outcome': 'treatment_initiated'
    },
    'patient_data_source': 'patient_journeys',
    'use_existing_config': True
}

output = agent.run(input_data)

if output['success']:
    eligible_df = output['eligible_patients']
    metadata = output['metadata']
    print(f"✓ Cohort constructed: {len(eligible_df):,} eligible")
else:
    print(f"✗ Failed: {output['error_message']}")
```

---

## Database Schema

### Tables Created

```sql
-- 1. Cohort definitions
CREATE TABLE ml_cohort_definitions (
    cohort_id TEXT PRIMARY KEY,
    brand TEXT,
    indication TEXT,
    inclusion_criteria JSONB,
    exclusion_criteria JSONB,
    eligible_population INTEGER,
    exclusion_rate DECIMAL(5,4),
    ...
);

-- 2. Execution log
CREATE TABLE ml_cohort_executions (
    execution_id TEXT PRIMARY KEY,
    cohort_id TEXT REFERENCES ml_cohort_definitions,
    execution_timestamp TIMESTAMP,
    eligible_row_count INTEGER,
    execution_time_seconds DECIMAL(10,3),
    ...
);

-- 3. Eligibility audit log
CREATE TABLE ml_cohort_eligibility_log (
    log_id SERIAL PRIMARY KEY,
    cohort_id TEXT,
    execution_id TEXT,
    criterion_name TEXT,
    criterion_type TEXT,
    removed_count INTEGER,
    remaining_count INTEGER,
    ...
);

-- 4. Patient assignments
CREATE TABLE ml_patient_cohort_assignments (
    patient_journey_id TEXT,
    cohort_id TEXT,
    execution_id TEXT,
    is_eligible BOOLEAN,
    failed_criteria JSONB,
    PRIMARY KEY (patient_journey_id, cohort_id, execution_id)
);
```

### Setup Database

```bash
# Run schema creation
psql -h your-host -U your-user -d your-db -f cohort_schema.sql

# Or with Supabase CLI
supabase db push
```

---

## Supported Operators

| Operator | Symbol | Example | Description |
|----------|--------|---------|-------------|
| EQUAL | `==` | `age == 18` | Exact match |
| NOT_EQUAL | `!=` | `gender != 'M'` | Not equal |
| GREATER | `>` | `severity > 7` | Strictly greater |
| GREATER_EQUAL | `>=` | `age >= 18` | Greater or equal |
| LESS | `<` | `age < 65` | Strictly less |
| LESS_EQUAL | `<=` | `severity <= 10` | Less or equal |
| IN | `in` | `code in ['L50.0', 'L50.1']` | Member of list |
| NOT_IN | `not_in` | `code not_in ['L30.0']` | Not in list |
| BETWEEN | `between` | `age between [18, 65]` | Inclusive range |
| CONTAINS | `contains` | `notes contains 'urticaria'` | String contains |

---

## API Reference

### CohortConstructor

Main class for cohort construction.

**Methods:**

```python
__init__(config: CohortConfig)
    Initialize constructor with configuration

construct_cohort(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]
    Main pipeline: apply criteria, validate temporal, generate metadata
    Returns: (eligible_df, metadata)

is_eligible(patient_row: pd.Series) -> bool
    Check if single patient meets eligibility criteria
    Returns: True if eligible

summary_report(metadata: Dict) -> str
    Generate human-readable summary report
    Returns: Formatted text report
```

### CohortConfig

Configuration dataclass for cohort definitions.

**Factory Methods:**

```python
CohortConfig.from_brand(brand: str, indication: str) -> CohortConfig
    Create pre-configured cohort for brand/indication

CohortConfig.from_json(filepath: str) -> CohortConfig
    Load configuration from JSON file
```

**Methods:**

```python
to_dict() -> Dict
    Convert config to dictionary

to_json(filepath: str)
    Save config to JSON file
```

### Criterion

Single eligibility criterion.

**Attributes:**

```python
field: str                    # Data field name
operator: Operator           # Comparison operator
value: Any                   # Comparison value
description: Optional[str]   # Human-readable description
clinical_rationale: Optional[str]  # Clinical justification
```

---

## Testing

### Run Test Suite

```bash
python test_cohort_constructor.py
```

### Test Coverage

- ✅ Operator application (==, >=, in, between, etc.)
- ✅ Temporal eligibility validation
- ✅ Multi-brand configuration
- ✅ Metadata generation
- ✅ Edge cases (empty data, missing fields, all excluded)
- ✅ Performance benchmarks (100K patients in <30s)
- ✅ Real-world scenarios

### Example Test Output

```
==========================================
COHORT CONSTRUCTOR - TEST SUITE
==========================================

test_equal_operator ... ok
test_greater_equal_operator ... ok
test_in_operator ... ok
test_lookback_validation ... ok
test_remibrutinib_config ... ok
test_metadata_structure ... ok
test_large_dataset_performance ... ok
   Performance: 100,000 patients in 8.23s

==========================================
TEST SUMMARY
==========================================
Tests run: 28
Failures: 0
Errors: 0
Success rate: 100.0%
```

---

## Performance

### Benchmarks

| Dataset Size | Execution Time | Memory Usage |
|--------------|----------------|--------------|
| 1,000 patients | 0.15 sec | ~5 MB |
| 10,000 patients | 0.82 sec | ~45 MB |
| 100,000 patients | 8.2 sec | ~420 MB |
| 1,000,000 patients | 95 sec | ~4.2 GB |

**Environment:** Standard laptop (16GB RAM, Intel i7)

### Optimization Tips

1. **Pre-filter data**: Apply broad filters (e.g., brand) before cohort construction
2. **Batch processing**: For millions of patients, process in chunks
3. **Database indices**: Index frequently queried fields (brand, diagnosis codes)
4. **Temporal validation**: Skip if not needed for your use case

---

## Troubleshooting

### Common Issues

**Issue 1: Missing required fields**

```python
ValueError: Missing required fields: ['age_at_diagnosis']
```

**Solution:** Ensure all required fields are present in input DataFrame

```python
# Check required fields
config = CohortConfig.from_brand('remibrutinib', 'csu')
print(config.required_fields)

# Verify data has fields
assert all(f in df.columns for f in config.required_fields)
```

**Issue 2: No eligible patients**

```python
# All patients excluded, eligible_df is empty
```

**Solution:** Review eligibility log to identify exclusion reasons

```python
_, metadata = constructor.construct_cohort(df)

for entry in metadata['eligibility_log']:
    print(f"{entry['criterion']}: removed {entry['removed']}")
    
# Output:
# age_at_diagnosis: removed 50 (950 remaining)
# urticaria_severity_uas7: removed 800 (150 remaining)
# antihistamine_failures_count: removed 150 (0 remaining)  ← Issue here
```

**Issue 3: Temporal validation fails**

```python
# Many patients excluded due to lookback/follow-up
```

**Solution:** Adjust temporal requirements or ensure data completeness

```python
config.lookback_days = 90  # Reduce from 180
config.followup_days = 60  # Reduce from 90
```

---

## Regulatory Compliance

### Audit Trail

CohortConstructor generates comprehensive audit logs suitable for regulatory submissions (FDA, EMA):

1. **Cohort Definition** - Explicit inclusion/exclusion criteria with clinical rationale
2. **Eligibility Log** - Step-by-step record of criteria application
3. **Patient Assignments** - Which patients were deemed eligible and why
4. **Execution Metadata** - When, by whom, with what configuration

### Pre-Specification

For regulatory submissions, cohort definitions must be pre-specified:

```python
# 1. Define cohort BEFORE looking at data
config = CohortConfig.from_brand('remibrutinib', 'csu')

# 2. Save configuration
config.to_json('cohort_spec_preregistered.json')

# 3. Lock configuration (prevent changes)
config.version = '1.0.0-FINAL'
config.status = 'locked'

# 4. Execute on data
constructor = CohortConstructor(config)
eligible_df, metadata = constructor.construct_cohort(df)

# 5. Store results in database for auditability
```

---

## Roadmap

### V1.1 (Planned)
- [ ] Integration with DoWhy for causal graph validation
- [ ] Propensity score matching for control cohort
- [ ] SNOMED CT / LOINC code support
- [ ] Natural language criterion parsing

### V2.0 (Future)
- [ ] CohortNet integration for automatic phenotype discovery
- [ ] Hybrid rule-based + learned cohort definitions
- [ ] Multi-cohort overlap analysis
- [ ] Clinical trial matching engine

---

## Contributing

### Development Setup

```bash
git clone https://github.com/your-org/e2i-cohort-constructor.git
cd e2i-cohort-constructor

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_cohort_constructor.py

# Run examples
python cohort_constructor.py
```

### Adding New Brand Configurations

```python
# In cohort_constructor.py, add new method:

@staticmethod
def _my_new_drug_config() -> 'CohortConfig':
    """My New Drug cohort configuration"""
    
    inclusion = [
        Criterion(field='age', operator=Operator.GREATER_EQUAL, value=18),
        # Add more criteria...
    ]
    
    exclusion = [
        # Add exclusion criteria...
    ]
    
    return CohortConfig(
        brand='my_new_drug',
        indication='my_indication',
        cohort_name='My New Drug Eligible Population',
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion
    )

# Update from_brand() method to include new drug
```

---

## License

Copyright © 2026 E2I Causal Analytics Team

---

## Support

For questions or issues:
- GitHub Issues: [github.com/your-org/e2i/issues]
- Email: support@e2i-analytics.com
- Documentation: [docs.e2i-analytics.com]

---

## Citation

If using CohortConstructor in research, please cite:

```bibtex
@software{cohort_constructor_2026,
  title={CohortConstructor: Rule-Based Patient Cohort Definition for Pharmaceutical RWE},
  author={E2I Causal Analytics Team},
  year={2026},
  url={https://github.com/your-org/e2i-cohort-constructor}
}
```

---

**Built for:** E2I (Evidence-to-Impact) Causal Analytics System
**Version:** 1.0.0
**Last Updated:** January 2026
