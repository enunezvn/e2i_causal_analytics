# Vocabulary Gaps Remediation Plan

**Created**: 2026-01-24
**Completed**: 2026-01-24
**Status**: ✅ COMPLETED
**Priority**: Medium
**Estimated Effort**: 16-24 hours

---

## Executive Summary

This plan addresses 8 vocabulary gaps identified in the E2I Causal Analytics vocabulary completeness assessment. The gaps are prioritized by impact and grouped into 3 implementation waves to minimize risk and maximize efficiency.

---

## Priority Matrix

| Gap | Priority | Impact | Effort | Risk | Wave |
|-----|----------|--------|--------|------|------|
| Agent List Sync | **HIGH** | High (data consistency) | 1h | Medium | 1 |
| Journey Stage Standardization | **HIGH** | High (data consistency) | 3h | Medium | 1 |
| Territory States | Low | Low (optional feature) | 1h | Low | 2 |
| Competitor Brands | Low | Low (future feature) | 30m | Low | 2 |
| Channel Vocabulary | Low | Low (optional feature) | 1h | Low | 2 |
| Payer Vocabulary | Low | Low (future feature) | 30m | Low | 2 |
| ICD-10 Mappings | Medium | Medium (clinical integration) | 1.5h | Low | 3 |
| NDC Codes | Medium | Medium (drug identification) | 1h | Low | 3 |

---

## Wave 1: Consistency Fixes (Critical)

### 1.1 Agent List Synchronization

**Problem**: `config/ontology/node_types.yaml` uses outdated PascalCase agent names that don't match the current 18-agent architecture in `config/domain_vocabulary.yaml`.

**Current State** (`node_types.yaml` line ~369):
```yaml
agent_name:
  values: [CausalImpactAgent, HeterogeneousOptimizerAgent, ExperimentDesigner, ...]
```

**Target State** (matching domain_vocabulary.yaml Section 2):
```yaml
agent_name:
  type: enum
  required: true
  unique: true
  values:
    # Tier 0: ML Foundation (7 agents)
    - scope_definer
    - data_preparer
    - feature_analyzer
    - model_selector
    - model_trainer
    - model_deployer
    - observability_connector
    # Tier 1: Coordination (2 agents)
    - orchestrator
    - tool_composer
    # Tier 2: Causal Analytics (4 agents)
    - causal_impact
    - heterogeneous_optimizer
    - gap_analyzer
    - experiment_designer
    # Tier 3: Monitoring (3 agents)
    - drift_monitor
    - data_quality_monitor
    - health_score
    # Tier 4: Prediction (3 agents)
    - prediction_synthesizer
    - risk_assessor
    - resource_optimizer
    # Tier 5: Self-Improvement (2 agents)
    - explainer
    - feedback_learner
  description: "Agent name (snake_case format matching factory registry)"
```

**Files to Modify**:
1. `config/ontology/node_types.yaml` - Update Agent node agent_name enum
2. `config/ontology/agent_tools.yaml` - Verify agent name references match

**Validation**:
```bash
python -c "from src.ontology.vocabulary_registry import VocabularyRegistry; v = VocabularyRegistry.load(); print(v.get_agent_names())"
pytest tests/unit/test_ontology/ -v
```

---

### 1.2 Journey Stage Standardization

**Problem**: Three different journey stage vocabularies exist across files.

**Analysis**:

| File | Values |
|------|--------|
| `domain_vocabulary.yaml` (patient_journey_stages) | diagnosis, treatment_naive, first_line, second_line, maintenance, discontinuation, switch |
| `node_types.yaml` & `core_attributes.yaml` | aware, considering, prescribed, first_fill, adherent, discontinued, maintained |
| `domain_vocabulary.yaml` (semantic_graph) | unaware, aware, considering, trialing, adopted, adherent, discontinued |

**Resolution**: These are **two different conceptual models** that should coexist:

1. **Patient Engagement Journey** (funnel stages) - Current in `node_types.yaml`:
   - aware, considering, prescribed, first_fill, adherent, discontinued, maintained
   - Used for: Patient node properties, trigger targeting

2. **Treatment Line Journey** (therapy progression) - Rename in `domain_vocabulary.yaml`:
   - Rename `patient_journey_stages` → `treatment_line_stages`
   - Values: diagnosis, treatment_naive, first_line, second_line, maintenance, discontinuation, switch
   - Used for: Clinical cohort analysis, therapy sequencing

**Changes Required**:

1. **`config/domain_vocabulary.yaml`**:
   ```yaml
   # RENAME from patient_journey_stages to:
   treatment_line_stages:
     description: "Treatment line progression stages (clinical cohort analysis)"
     note: "For patient engagement funnel stages, see core_attributes.yaml"
     values:
       - diagnosis
       - treatment_naive
       - first_line
       - second_line
       - maintenance
       - discontinuation
       - switch
   ```

2. **`config/domain_vocabulary.yaml`** (semantic_graph section):
   ```yaml
   # Align with core_attributes.yaml:
   journey_stages:
     - aware
     - considering
     - prescribed
     - first_fill
     - adherent
     - discontinued
     - maintained
   ```

3. **`src/ontology/vocabulary_registry.py`**:
   - Add `get_engagement_journey_stages()` method
   - Add `get_treatment_line_stages()` method
   - Keep `get_journey_stages()` for backward compatibility

**Validation**:
```bash
pytest tests/unit/test_ontology/test_vocabulary_registry.py -v
pytest tests/unit/test_rag/test_entity_extractor.py -v
```

---

## Wave 2: Minor Enhancements (Low Priority)

### 2.1 Territory States Mapping

**Add to `config/domain_vocabulary.yaml`**:

```yaml
state_to_region_mapping:
  description: "US state to region mapping for territory analysis"
  mapping:
    northeast:
      - CT  # Connecticut
      - ME  # Maine
      - MA  # Massachusetts
      - NH  # New Hampshire
      - NJ  # New Jersey
      - NY  # New York
      - PA  # Pennsylvania
      - RI  # Rhode Island
      - VT  # Vermont
    south:
      - AL  # Alabama
      - AR  # Arkansas
      - DE  # Delaware
      - FL  # Florida
      - GA  # Georgia
      - KY  # Kentucky
      - LA  # Louisiana
      - MD  # Maryland
      - MS  # Mississippi
      - NC  # North Carolina
      - OK  # Oklahoma
      - SC  # South Carolina
      - TN  # Tennessee
      - TX  # Texas
      - VA  # Virginia
      - WV  # West Virginia
      - DC  # District of Columbia
    midwest:
      - IL  # Illinois
      - IN  # Indiana
      - IA  # Iowa
      - KS  # Kansas
      - MI  # Michigan
      - MN  # Minnesota
      - MO  # Missouri
      - NE  # Nebraska
      - ND  # North Dakota
      - OH  # Ohio
      - SD  # South Dakota
      - WI  # Wisconsin
    west:
      - AK  # Alaska
      - AZ  # Arizona
      - CA  # California
      - CO  # Colorado
      - HI  # Hawaii
      - ID  # Idaho
      - MT  # Montana
      - NV  # Nevada
      - NM  # New Mexico
      - OR  # Oregon
      - UT  # Utah
      - WA  # Washington
      - WY  # Wyoming
```

---

### 2.2 Competitor Brands

**Add to `config/domain_vocabulary.yaml`**:

```yaml
competitor_brands:
  description: "Known competitor brands by therapeutic area"
  note: "For competitive intelligence analysis"
  by_therapeutic_area:
    csu_btk_inhibitors:
      - Xolair       # omalizumab (current standard)
      - fenebrutinib # competitor BTK inhibitor
    pnh_complement:
      - Soliris      # eculizumab
      - Ultomiris    # ravulizumab
    breast_cancer_cdk46:
      - Ibrance      # palbociclib
      - Verzenio     # abemaciclib
```

---

### 2.3 Channel Vocabulary

**Add to `config/domain_vocabulary.yaml`**:

```yaml
marketing_channels:
  description: "Marketing and engagement channels for HCP/patient outreach"
  channels:
    digital:
      - email
      - website
      - social_media
      - digital_ads
      - webinar
      - edetailing
    field:
      - in_person
      - phone
      - lunch_and_learn
      - speaker_program
    crm:
      - crm_alert
      - mobile_app
      - portal
    print:
      - direct_mail
      - journal_ads
      - samples
```

---

### 2.4 Payer Vocabulary

**Add to `config/domain_vocabulary.yaml`**:

```yaml
payer_categories:
  description: "Payer categories for formulary and access analysis"
  categories:
    commercial:
      description: "Commercial/employer-sponsored plans"
      subcategories:
        - national_plans
        - regional_plans
        - pbm_managed
    government:
      description: "Government-funded programs"
      subcategories:
        - medicare_part_d
        - medicaid
        - tricare
    specialty:
      description: "Specialty pharmacy channels"
      subcategories:
        - specialty_pharmacy
        - hub_services
        - buy_and_bill
```

---

## Wave 3: Future Enhancements

### 3.1 ICD-10 Mappings

**Add to `config/domain_vocabulary.yaml`**:

```yaml
brand_icd10_mappings:
  description: "ICD-10 diagnosis codes for brand indications"
  note: "For cohort construction and clinical data integration"
  mappings:
    Remibrutinib:
      primary_indication: "Chronic Spontaneous Urticaria (CSU)"
      icd10_codes:
        - L50.1  # Idiopathic urticaria
        - L50.8  # Other urticaria
        - L50.9  # Urticaria, unspecified
    Fabhalta:
      primary_indication: "Paroxysmal Nocturnal Hemoglobinuria (PNH)"
      icd10_codes:
        - D59.5  # Paroxysmal nocturnal hemoglobinuria
    Kisqali:
      primary_indication: "HR+/HER2- advanced breast cancer"
      icd10_codes:
        - C50.9  # Breast cancer, unspecified
        - C50.1  # Central portion of breast
        - C50.2  # Upper-inner quadrant
        - C50.3  # Lower-inner quadrant
        - C50.4  # Upper-outer quadrant
        - C50.5  # Lower-outer quadrant
```

---

### 3.2 NDC Codes

**Add to `config/domain_vocabulary.yaml`**:

```yaml
brand_ndc_codes:
  description: "NDC drug codes for brand products"
  note: "For pharmacy claims data integration"
  format: "11-digit NDC (5-4-2 format)"
  mappings:
    Kisqali:
      drug_name: "ribociclib"
      ndc_codes:
        - "00078-0903-51"  # 200mg tablets, 63-count
        - "00078-0903-21"  # 200mg tablets, 21-count
        - "00078-0903-42"  # 200mg tablets, 42-count
```

---

## Implementation Checklist

### Pre-Implementation

- [x] Backup current vocabulary files
- [x] Document current state for comparison

### Wave 1 Implementation

- [x] Update agent_name enum in `config/ontology/node_types.yaml` (21 snake_case agents)
- [x] Verify agent_tools.yaml references
- [x] Rename patient_journey_stages → treatment_line_stages in domain_vocabulary.yaml
- [x] Align semantic_graph journey_stages with core_attributes.yaml
- [x] Update vocabulary_registry.py with new methods (get_treatment_line_stages, get_engagement_stages, etc.)
- [x] Run validation tests

### Wave 2 Implementation

- [x] Add state_to_region_mapping section (51 states → 4 regions)
- [x] Add competitor_brands section (6 competitors across 3 therapeutic areas)
- [x] Add marketing_channels section (16 channels across 4 types)
- [x] Add payer_categories section (3 categories with subcategories)
- [x] Run validation tests

### Wave 3 Implementation

- [x] Add brand_icd10_mappings section (3 brands with ICD-10 codes)
- [x] Add brand_ndc_codes section (Kisqali NDC codes)
- [x] Run validation tests

### Post-Implementation

- [x] Bump version to 5.1.0
- [x] Update version_history in metadata
- [x] Run full test suite (69 vocabulary registry tests pass)
- [x] Update VOCABULARY_COMPLETENESS_ASSESSMENT.md

---

## Validation Results (2026-01-24)

### Droplet Validation Batches

**Batch 1: YAML Schema Validation** ✅
- Version: 5.1.0
- Sections: 88
- Agent names: 21 (snake_case)

**Batch 2: Vocabulary Registry Tests** ✅
- Initial: 66 passed, 2 failed
- Fixed tests expecting old journey stage behavior
- Final: 69 passed, 0 failed

**Batch 3: Enum Synchronization** ✅
- Agent names: 21 match between vocabulary and node_types
- Region values: 4 match (northeast, south, midwest, west)
- Journey stages: 7 match (engagement funnel)

**Batch 4: V5.1.0 Accessor Validation** ✅
- All new accessor methods work correctly
- State-to-region mapping: 51 states across 4 regions
- Marketing channels: 16 across 4 types
- Competitor brands: 6 across 3 therapeutic areas
- ICD-10 codes: 3 brands mapped
- NDC codes: Kisqali mapped

### Test Fixes Applied

1. `test_get_journey_stages_returns_list` - Updated to check engagement funnel stages
2. `test_get_enum_values_journey_stage_type` - Updated to check engagement stages
3. Added `test_get_treatment_line_stages_returns_list` - New test for clinical progression stages

---

## Validation Commands

```bash
# Schema validation
python -c "import yaml; yaml.safe_load(open('config/domain_vocabulary.yaml'))"
python -c "import yaml; yaml.safe_load(open('config/ontology/node_types.yaml'))"

# Vocabulary registry tests
pytest tests/unit/test_ontology/test_vocabulary_registry.py -v

# Entity extraction tests
pytest tests/unit/test_rag/test_entity_extractor.py -v

# Full test suite (memory-safe)
make test
```

---

## Rollback Plan

If issues are detected:

```bash
# Restore from backup
cp config/archived/domain_vocabulary_v5.0.0_backup.yaml config/domain_vocabulary.yaml
cp config/archived/node_types_backup.yaml config/ontology/node_types.yaml

# Clear vocabulary cache
python -c "from src.ontology.vocabulary_registry import VocabularyRegistry; VocabularyRegistry.clear_cache()"

# Restart services (if deployed)
sudo systemctl restart e2i-api
```

---

## Files Affected

| File | Changes |
|------|---------|
| `config/domain_vocabulary.yaml` | All vocabulary additions |
| `config/ontology/node_types.yaml` | Agent name enum update |
| `config/ontology/agent_tools.yaml` | Agent name reference verification |
| `src/ontology/vocabulary_registry.py` | New accessor methods |
| `config/cohort_vocabulary.yaml` | Cross-reference updates |

---

## Success Criteria

1. ✅ All YAML files pass schema validation (88 sections, valid structure)
2. ✅ All existing tests pass (69/69 vocabulary registry tests)
3. ✅ Agent names synchronized across files (21 snake_case agents)
4. ✅ Journey stages clearly differentiated (engagement vs treatment line)
5. ✅ Vocabulary completeness achieved (all 8 gaps addressed)
