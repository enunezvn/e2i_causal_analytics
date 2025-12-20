# Brand Context

**Last Updated**: 2025-12-18
**Update Cadence**: Quarterly or when brand strategy changes
**Owner**: Commercial Strategy Team
**Purpose**: Brand-specific context for causal analytics

## Overview

This document provides brand-specific context for the E2I Causal Analytics platform. Each brand has unique characteristics that affect causal analysis and interpretation.

---

## Brand Configuration Files

Brand-specific settings are managed through configuration files in the `config/` directory. These files define brand attributes, KPIs, and analysis parameters used throughout the platform.

### Configuration File Locations

| File | Purpose | Brand Coverage | Last Updated |
|------|---------|----------------|--------------|
| `config/kpi_definitions.yaml` | KPI calculations, thresholds, frequencies | All 3 brands | 2024-12-08 |
| `config/agent_config.yaml` | Agent configurations, brand contexts | All 3 brands | 2024-12-08 |
| `config/brand_profiles.yaml` | Brand-specific metadata (if exists) | All 3 brands | TBD |

### KPI Definitions (config/kpi_definitions.yaml)

**Structure**:
```yaml
kpis:
  - id: "BR-001"  # Brand-specific KPI
    name: "kisqali_conversion_rate"
    brand: "kisqali"
    category: "brand_specific"
    formula: "nrx / identified_patients"
    # ... additional fields
```

**Brand-Specific KPIs**:
- **BR-001**: Kisqali Conversion Rate (Identified → NRx)
- **BR-002**: Fabhalta Switching Rate (C5i → Fabhalta)
- **BR-003**: Remibrutinib Time-to-Biologic (Diagnosis → Treatment)
- **BR-004**: Kisqali Persistence-Adjusted TRx
- **BR-005**: Fabhalta Transfusion-Free Rate

**Usage**:
```python
from src.repositories.kpi_repository import KPIRepository

kpi_repo = KPIRepository()
kisqali_kpis = kpi_repo.get_kpis_by_brand("kisqali")
conversion_rate = kpi_repo.calculate_kpi("BR-001", date_range)
```

### Agent Configurations (config/agent_config.yaml)

**Brand Context in Agents**:
Each agent's configuration includes brand-specific context for tailored analysis.

**Example (orchestrator agent)**:
```yaml
orchestrator:
  context:
    brand_knowledge:
      kisqali:
        indication: "HR+/HER2- advanced breast cancer"
        patient_population: "Adult women with HR+/HER2- advanced or metastatic breast cancer"
        key_drivers: ["HCP targeting", "patient support", "persistence"]
      fabhalta:
        indication: "Paroxysmal nocturnal hemoglobinuria (PNH)"
        patient_population: "Adults with PNH, including C5i-experienced"
        key_drivers: ["disease awareness", "HCP education", "switching"]
      remibrutinib:
        indication: "Chronic Spontaneous Urticaria (CSU)"
        patient_population: "Adults with CSU inadequately controlled on H1 antihistamines"
        key_drivers: ["HCP awareness", "patient identification", "access"]
```

**Usage in Code**:
```python
from src.utils.config_loader import load_agent_config

config = load_agent_config()
brand_context = config['orchestrator']['context']['brand_knowledge']['kisqali']
```

### Database Brand Configuration

**Brand Data Tables**:
- `patient_journeys.brand` - VARCHAR brand identifier
- `hcp_profiles.brand_focus` - JSON array of brand specialties
- `business_metrics.brand` - Brand-specific metric tracking

**Brand Values** (standardized):
- `kisqali` - Kisqali (Ribociclib)
- `fabhalta` - Fabhalta (Iptacopan)
- `remibrutinib` - Remibrutinib

**Query Example**:
```sql
-- Get Kisqali patient journeys
SELECT * FROM patient_journeys
WHERE brand = 'kisqali'
  AND created_at >= CURRENT_DATE - INTERVAL '90 days';

-- Get HCPs with Fabhalta focus
SELECT * FROM hcp_profiles
WHERE brand_focus ? 'fabhalta';  -- JSON contains operator
```

### Adding a New Brand

**Steps to add new brand configuration**:

1. **Update `config/kpi_definitions.yaml`**:
   ```yaml
   - id: "BR-006"
     name: "new_brand_key_metric"
     brand: "new_brand"
     category: "brand_specific"
     formula: "..."
   ```

2. **Update `config/agent_config.yaml`**:
   ```yaml
   orchestrator:
     context:
       brand_knowledge:
         new_brand:
           indication: "..."
           patient_population: "..."
           key_drivers: [...]
   ```

3. **Add brand context to this document**:
   - Create new brand section following Kisqali/Fabhalta/Remibrutinib pattern
   - Document causal DAG, segments, KPIs

4. **Update database schema** (if needed):
   ```sql
   -- Add brand-specific tables or columns
   ALTER TABLE patient_journeys ADD CONSTRAINT check_brand
   CHECK (brand IN ('kisqali', 'fabhalta', 'remibrutinib', 'new_brand'));
   ```

---

## Kisqali (Ribociclib)

### Brand Profile

| Property | Value |
|----------|-------|
| **Therapeutic Area** | Oncology |
| **Indication** | HR+/HER2- advanced breast cancer |
| **Patient Population** | Adult women with hormone receptor-positive, HER2-negative advanced or metastatic breast cancer |
| **Treatment Setting** | First-line and subsequent lines of therapy |
| **Combination Therapy** | Combined with aromatase inhibitor or fulvestrant |

### Key Performance Indicators

| KPI | Description | Target | Causal Drivers |
|-----|-------------|--------|----------------|
| NRx | New prescriptions | +15% YoY | HCP engagement, patient identification |
| TRx | Total prescriptions | Maintain leadership | Persistence, refill rate |
| NBRx | New-to-brand prescriptions | +10% YoY | Competitive switching |
| Market Share | Share of CDK4/6 inhibitor market | >35% | HCP preference, access |
| Adherence Rate | Patient compliance | >80% | Patient support programs |

### Causal Relationships (DAG)

```
┌──────────────────────────────────────────────────────────────────┐
│                    KISQALI CAUSAL DAG                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [HCP Targeting] ───────┬───────────► [NRx]                      │
│         │               │              ▲                          │
│         ▼               │              │                          │
│  [HCP Engagement] ──────┼──────────────┘                         │
│         │               │                                         │
│         ▼               │                                         │
│  [Brand Perception] ────┘                                         │
│                                                                   │
│  [Patient ID Programs] ─────────────► [Patient Volume]           │
│         │                                    │                    │
│         ▼                                    ▼                    │
│  [Early Detection] ─────────────────► [Treatment Start]          │
│                                                                   │
│  [Patient Support] ────────────────► [Adherence] ──► [TRx]      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Segment Definitions

| Segment | Definition | Size | Expected Response |
|---------|------------|------|-------------------|
| High-Volume Oncologists | >50 BC patients/year, academic affiliations | 2,500 | High |
| Community Oncologists | 10-50 BC patients/year, private practice | 8,000 | Medium |
| Nurse Navigators | Breast cancer nurse navigators | 3,500 | Medium-High |
| Emerging HCPs | <10 BC patients/year, new to CDK4/6 | 5,000 | Variable |

### Historical Experiment Outcomes

| Experiment | Treatment | Outcome | Effect Size | Confidence |
|------------|-----------|---------|-------------|------------|
| Q2 2024 Targeting Pilot | Enhanced HCP targeting | NRx | +18% | 95% CI [12%, 24%] |
| Q3 2024 Nurse Navigator | Navigator engagement | Persistence | +8% | 95% CI [5%, 11%] |
| Q4 2024 Patient Support | Enhanced support calls | Adherence | +12% | 95% CI [8%, 16%] |

---

## Fabhalta (Iptacopan)

### Brand Profile

| Property | Value |
|----------|-------|
| **Therapeutic Area** | Rare Disease / Hematology |
| **Indication** | Paroxysmal nocturnal hemoglobinuria (PNH) |
| **Patient Population** | Adults with PNH, including C5i-experienced |
| **Treatment Setting** | First oral treatment for PNH |
| **Mechanism** | Factor B inhibitor (proximal complement) |

### Key Performance Indicators

| KPI | Description | Target | Causal Drivers |
|-----|-------------|--------|----------------|
| Patient Starts | New patients initiated | 500+ Year 1 | Disease awareness, HCP education |
| Switching Rate | From C5 inhibitors | 30% of eligible | Breakthrough therapy positioning |
| Hemoglobin Response | Patients achieving Hb normalization | >70% | Proper dosing, adherence |
| Transfusion Avoidance | Patients avoiding transfusions | >80% | Disease control |

### Causal Relationships (DAG)

```
┌──────────────────────────────────────────────────────────────────┐
│                    FABHALTA CAUSAL DAG                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Disease Awareness] ──────────────► [Diagnosis Rate]            │
│         │                                   │                     │
│         ▼                                   ▼                     │
│  [HCP Education] ──────────────────► [Referral Rate]             │
│         │                                   │                     │
│         ▼                                   ▼                     │
│  [Treatment Decision] ◄────────────── [Patient ID]               │
│         │                                                         │
│         ▼                                                         │
│  [Fabhalta Start] ─────────────────► [Clinical Outcomes]         │
│         │                                   │                     │
│         ▼                                   ▼                     │
│  [Adherence] ──────────────────────► [Hb Response]               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Segment Definitions

| Segment | Definition | Size | Expected Response |
|---------|------------|------|-------------------|
| PNH Specialists | Hematologists with PNH expertise | 200 | Very High |
| Transplant Centers | BMT centers treating PNH | 150 | High |
| Community Hematologists | General hematology practice | 1,500 | Medium |
| Rare Disease Centers | Comprehensive rare disease programs | 100 | High |

### Key Considerations for Causal Analysis

- **Small population**: PNH is rare (~5,000 patients in US), requiring careful statistical approaches
- **Specialty concentration**: Most patients managed by few specialists
- **Prior therapy effects**: C5i experience affects response expectations
- **Real-world evidence critical**: Limited RCT generalizability

---

## Remibrutinib

### Brand Profile

| Property | Value |
|----------|-------|
| **Therapeutic Area** | Immunology |
| **Indication** | Chronic Spontaneous Urticaria (CSU) |
| **Patient Population** | Adults with CSU inadequately controlled on H1 antihistamines |
| **Treatment Setting** | Second-line after antihistamine failure |
| **Mechanism** | BTK inhibitor |

### Key Performance Indicators

| KPI | Description | Target | Causal Drivers |
|-----|-------------|--------|----------------|
| NRx | New prescriptions | Launch trajectory | HCP awareness, patient identification |
| Market Penetration | Share of CSU biologics market | 25% Year 1 | Positioning vs. omalizumab |
| Urticaria Control | UAS7=0 achievement | >40% | Proper patient selection |
| Persistence | Patients continuing at 6 months | >70% | Efficacy, tolerability |

### Causal Relationships (DAG)

```
┌──────────────────────────────────────────────────────────────────┐
│                  REMIBRUTINIB CAUSAL DAG                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [HCP Education] ──────────────────► [Awareness]                 │
│         │                                 │                       │
│         ▼                                 ▼                       │
│  [Patient ID] ◄───────────────────── [Diagnosis]                 │
│         │                                                         │
│         ▼                                                         │
│  [Antihistamine Failure] ──────────► [Biologic Consideration]    │
│         │                                 │                       │
│         ▼                                 ▼                       │
│  [Access/Coverage] ◄──────────────── [Prescribing Decision]      │
│         │                                 │                       │
│         ▼                                 ▼                       │
│  [Remibrutinib Start] ─────────────► [Urticaria Control]         │
│         │                                                         │
│         ▼                                                         │
│  [Persistence] ────────────────────► [Long-term Control]         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Segment Definitions

| Segment | Definition | Size | Expected Response |
|---------|------------|------|-------------------|
| Allergists/Immunologists | CSU specialty focus | 3,000 | High |
| Dermatologists | High CSU volume | 5,000 | Medium-High |
| Primary Care | Referring PCPs | 20,000 | Low (referral focus) |
| Academic Centers | Clinical trial sites | 100 | Very High |

### Launch-Specific Considerations

- **Launch dynamics**: Causal effects differ pre/post launch
- **Competitive landscape**: omalizumab established, ligelizumab pending
- **Payer dynamics**: Prior authorization requirements affect access
- **KOL influence**: Academic opinion leaders drive early adoption

---

## Cross-Brand Learnings

### Transferable Causal Insights

| Learning | Source Brand | Applicable Brands | Evidence |
|----------|--------------|-------------------|----------|
| HCP targeting effect | Kisqali | All | +15-20% NRx lift |
| Patient support adherence | Kisqali | All | +10-15% adherence |
| Specialist concentration | Fabhalta | Remibrutinib | Top 20% drive 80% |
| KOL early adoption | All | All | 2x faster adoption |

### Common Confounders

| Confounder | Description | Mitigation |
|------------|-------------|------------|
| Territory potential | Baseline market size | Stratified analysis |
| Prior brand experience | Historical Novartis engagement | Control for history |
| Payer mix | Insurance coverage variation | Include payer controls |
| Competitive activity | Competitor engagement | Monitor competitive spend |

---

## Data Validation Queries

SQL queries to validate brand data quality and consistency. Run these queries periodically to ensure data integrity.

### Brand Coverage Validation

**Check patient journey brand distribution**:
```sql
-- Verify all brands have patient journeys
SELECT
    brand,
    COUNT(*) as journey_count,
    MIN(created_at) as earliest_journey,
    MAX(created_at) as latest_journey
FROM patient_journeys
GROUP BY brand
ORDER BY brand;

-- Expected: All 3 brands present with reasonable counts
-- Alert if any brand has 0 journeys or very old latest_journey
```

**Check HCP brand focus**:
```sql
-- HCPs by brand focus
SELECT
    brand_elem.value as brand,
    COUNT(DISTINCT hcp_id) as hcp_count,
    COUNT(DISTINCT territory) as territory_count,
    COUNT(DISTINCT region) as region_count
FROM hcp_profiles,
     jsonb_array_elements_text(brand_focus) AS brand_elem(value)
GROUP BY brand_elem.value
ORDER BY brand_elem.value;

-- Expected: Kisqali >> Remibrutinib > Fabhalta (due to population sizes)
```

### Brand Data Quality Checks

**Check for orphaned records (no brand assignment)**:
```sql
-- Patient journeys without brand
SELECT COUNT(*) as orphaned_journeys
FROM patient_journeys
WHERE brand IS NULL OR brand = '';

-- Expected: 0 (all journeys must have brand)

-- Treatment events without brand context
SELECT COUNT(*) as orphaned_treatments
FROM treatment_events te
LEFT JOIN patient_journeys pj ON te.journey_id = pj.journey_id
WHERE pj.brand IS NULL;

-- Expected: 0 (all treatments linked to branded journeys)
```

**Validate brand values are standardized**:
```sql
-- Check for non-standard brand values
SELECT DISTINCT brand, COUNT(*) as occurrence_count
FROM patient_journeys
WHERE brand NOT IN ('kisqali', 'fabhalta', 'remibrutinib')
GROUP BY brand;

-- Expected: Empty result (all brands should be lowercase, standardized)

-- Check for case inconsistencies
SELECT
    brand,
    LOWER(brand) as normalized_brand,
    COUNT(*) as count
FROM patient_journeys
WHERE brand != LOWER(brand)
GROUP BY brand, LOWER(brand);

-- Expected: Empty result (no case inconsistencies)
```

### KPI Calculation Validation

**Validate brand-specific KPI calculations**:
```sql
-- BR-001: Kisqali Conversion Rate
WITH kisqali_metrics AS (
    SELECT
        COUNT(DISTINCT CASE WHEN status = 'identified' THEN patient_id END) as identified,
        COUNT(DISTINCT CASE WHEN status = 'converted' THEN patient_id END) as converted
    FROM patient_journeys
    WHERE brand = 'kisqali'
        AND created_at >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT
    identified,
    converted,
    ROUND(converted::NUMERIC / NULLIF(identified, 0), 4) as conversion_rate,
    CASE
        WHEN identified = 0 THEN 'No identified patients'
        WHEN converted::NUMERIC / NULLIF(identified, 0) < 0.10 THEN 'Below threshold (10%)'
        WHEN converted::NUMERIC / NULLIF(identified, 0) > 0.30 THEN 'Above threshold (30%)'
        ELSE 'Within expected range'
    END as status
FROM kisqali_metrics;

-- Expected: Conversion rate between 10-30%
```

```sql
-- BR-002: Fabhalta Switching Rate (from C5i)
WITH fabhalta_switching AS (
    SELECT
        COUNT(DISTINCT CASE
            WHEN prior_therapy LIKE '%C5i%' THEN patient_id
        END) as c5i_eligible,
        COUNT(DISTINCT CASE
            WHEN prior_therapy LIKE '%C5i%' AND brand = 'fabhalta' THEN patient_id
        END) as switched_to_fabhalta
    FROM patient_journeys
    WHERE created_at >= CURRENT_DATE - INTERVAL '90 days'
)
SELECT
    c5i_eligible,
    switched_to_fabhalta,
    ROUND(switched_to_fabhalta::NUMERIC / NULLIF(c5i_eligible, 0), 4) as switching_rate,
    CASE
        WHEN c5i_eligible = 0 THEN 'No C5i patients'
        WHEN switched_to_fabhalta::NUMERIC / NULLIF(c5i_eligible, 0) < 0.20 THEN 'Below target (30%)'
        ELSE 'On track'
    END as status
FROM fabhalta_switching;

-- Expected: Switching rate approaching 30%
```

### Cross-Brand Consistency Checks

**Verify segment definitions are consistent**:
```sql
-- Check for HCPs assigned to incompatible brand-segment combinations
SELECT
    h.hcp_id,
    h.specialty,
    brand_elem.value as brand,
    h.segment
FROM hcp_profiles h,
     jsonb_array_elements_text(h.brand_focus) AS brand_elem(value)
WHERE
    (brand_elem.value = 'fabhalta' AND h.segment NOT IN ('PNH Specialists', 'Transplant Centers', 'Community Hematologists', 'Rare Disease Centers'))
    OR (brand_elem.value = 'kisqali' AND h.segment NOT IN ('High-Volume Oncologists', 'Community Oncologists', 'Nurse Navigators', 'Emerging HCPs'))
    OR (brand_elem.value = 'remibrutinib' AND h.segment NOT IN ('Allergists/Immunologists', 'Dermatologists', 'Primary Care', 'Academic Centers'));

-- Expected: Empty result (all HCPs in valid segments for their brands)
```

**Check for cross-brand contamination in experiments**:
```sql
-- Verify experiment units don't cross brands
SELECT
    experiment_id,
    COUNT(DISTINCT brand) as brand_count,
    array_agg(DISTINCT brand) as brands
FROM (
    SELECT
        ae.experiment_id,
        pj.brand
    FROM agent_activities ae
    JOIN patient_journeys pj ON ae.entity_id = pj.patient_id::TEXT
    WHERE ae.activity_type = 'experiment_assignment'
) sub
GROUP BY experiment_id
HAVING COUNT(DISTINCT brand) > 1;

-- Expected: Empty result (experiments should not mix brands)
```

### Business Metric Validation

**Compare business_metrics table to calculated values**:
```sql
-- Validate NRx metric consistency
WITH calculated_nrx AS (
    SELECT
        brand,
        DATE_TRUNC('month', created_at) as month,
        COUNT(DISTINCT patient_id) as calc_nrx
    FROM patient_journeys
    WHERE status = 'new_rx'
    GROUP BY brand, month
),
stored_nrx AS (
    SELECT
        brand,
        DATE_TRUNC('month', timestamp) as month,
        value as stored_nrx
    FROM business_metrics
    WHERE metric_name = 'nrx'
)
SELECT
    c.brand,
    c.month,
    c.calc_nrx,
    s.stored_nrx,
    ABS(c.calc_nrx - s.stored_nrx) as difference,
    CASE
        WHEN ABS(c.calc_nrx - s.stored_nrx) > 10 THEN 'ALERT: Large discrepancy'
        WHEN s.stored_nrx IS NULL THEN 'ALERT: Missing stored metric'
        ELSE 'OK'
    END as validation_status
FROM calculated_nrx c
LEFT JOIN stored_nrx s ON c.brand = s.brand AND c.month = s.month
WHERE c.month >= CURRENT_DATE - INTERVAL '6 months'
ORDER BY c.brand, c.month DESC;

-- Expected: All "OK" or small differences (<5)
```

### Causal DAG Validation

**Verify causal paths exist for all brand-driver combinations**:
```sql
-- Check causal_paths coverage by brand and key driver
SELECT
    brand,
    source_node,
    COUNT(*) as path_count,
    COUNT(DISTINCT target_node) as unique_targets,
    AVG(effect_size) as avg_effect
FROM causal_paths
WHERE brand IS NOT NULL
GROUP BY brand, source_node
ORDER BY brand, path_count DESC;

-- Expected: Each brand has paths for all key drivers from brand profile
-- Kisqali: HCP targeting, patient support, persistence
-- Fabhalta: disease awareness, HCP education, switching
-- Remibrutinib: HCP awareness, patient identification, access
```

**Check for missing critical causal relationships**:
```sql
-- Kisqali critical paths
SELECT 'Kisqali - Missing critical paths' as alert
WHERE NOT EXISTS (
    SELECT 1 FROM causal_paths
    WHERE brand = 'kisqali'
    AND source_node IN ('hcp_targeting', 'patient_support', 'persistence')
);

-- Fabhalta critical paths
SELECT 'Fabhalta - Missing critical paths' as alert
WHERE NOT EXISTS (
    SELECT 1 FROM causal_paths
    WHERE brand = 'fabhalta'
    AND source_node IN ('disease_awareness', 'hcp_education', 'switching')
);

-- Remibrutinib critical paths
SELECT 'Remibrutinib - Missing critical paths' as alert
WHERE NOT EXISTS (
    SELECT 1 FROM causal_paths
    WHERE brand = 'remibrutinib'
    AND source_node IN ('hcp_awareness', 'patient_identification', 'access')
);

-- Expected: No results (all critical paths should exist)
```

### Recommended Validation Cadence

| Validation Type | Frequency | Owner | Action if Failed |
|-----------------|-----------|-------|------------------|
| Brand coverage | Daily | Data Engineering | Alert if any brand missing >24hrs |
| Data quality checks | Daily | Data Engineering | Fix orphaned records immediately |
| KPI calculations | Weekly | Analytics Team | Investigate discrepancies >5% |
| Cross-brand consistency | Weekly | Analytics Team | Review segment assignments |
| Business metric validation | Weekly | Analytics Team | Reconcile calculated vs stored |
| Causal DAG validation | Monthly | Causal Analytics Team | Rebuild missing paths |

### Automated Validation Script

**Example script** (save as `scripts/validate_brand_data.py`):
```python
#!/usr/bin/env python3
"""
Brand data validation script.
Run weekly to validate brand data quality.
"""
import sys
from sqlalchemy import create_engine, text
from config import DATABASE_URL

def run_validation_query(engine, name, query, expected_result='empty'):
    """Run validation query and check result."""
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()

        if expected_result == 'empty' and len(rows) > 0:
            print(f"❌ FAILED: {name}")
            print(f"   Found {len(rows)} rows, expected 0")
            return False
        elif expected_result == 'non_empty' and len(rows) == 0:
            print(f"❌ FAILED: {name}")
            print(f"   Found 0 rows, expected data")
            return False
        else:
            print(f"✅ PASSED: {name}")
            return True

def main():
    engine = create_engine(DATABASE_URL)

    validations = [
        ("Brand coverage", "SELECT brand FROM patient_journeys WHERE brand IS NULL", 'empty'),
        ("Brand standardization", "SELECT brand FROM patient_journeys WHERE brand NOT IN ('kisqali', 'fabhalta', 'remibrutinib')", 'empty'),
        # Add more validations...
    ]

    passed = sum(run_validation_query(engine, name, query, expected)
                 for name, query, expected in validations)
    total = len(validations)

    print(f"\n{passed}/{total} validations passed")
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
```
