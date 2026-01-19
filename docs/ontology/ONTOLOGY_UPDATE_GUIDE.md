# Ontology Update & Evolution Guide

## 📊 Overview: Two Types of Updates

The ontology system handles two fundamentally different types of updates:

### **1. INSTANCE-LEVEL UPDATES** (Continuous, Automated)
New data within existing schema → Add nodes/edges without changing structure

### **2. SCHEMA-LEVEL UPDATES** (Periodic, Manual)
New types of entities/relationships → Modify the ontology structure itself

---

## 🔄 INSTANCE-LEVEL UPDATES (Daily Operations)

### **Scenario 1: New Patient Data Arrives**

**Input**: Daily batch from IQVIA, Flatiron, Optum
```json
{
  "patient_id": "PAT99887766",
  "journey_stage": "prescribed",
  "region": "northeast",
  "prescribed_brand": "remibrutinib",
  "prescribing_hcp_npi": "1234567890",
  "prescription_date": "2026-01-13"
}
```

**Automatic Process**:

```python
# 1. VALIDATE against vocabulary
from config.domain_vocabulary import get_loader, get_node_types

loader = get_loader()
validator = GraphValidator(get_node_types())

# Validate patient data
is_valid, errors = validator.validate_node('Patient', {
    'patient_id': 'PAT99887766',
    'journey_stage': 'prescribed',  # ✅ Valid enum from vocabulary
    'region': 'northeast'           # ✅ Valid enum from vocabulary
})

if not is_valid:
    raise ValidationError(errors)

# 2. CREATE NODES if they don't exist
graph.query("""
    MERGE (p:Patient {patient_id: 'PAT99887766'})
    SET p.journey_stage = 'prescribed',
        p.region = 'northeast',
        p.updated_at = datetime()
""")

graph.query("""
    MERGE (h:HCP {npi: '1234567890'})
    ON CREATE SET h.hcp_id = 'HCP' + h.npi
""")

graph.query("""
    MERGE (b:Brand {brand_name: 'remibrutinib'})
""")

# 3. CREATE EDGES
graph.query("""
    MATCH (p:Patient {patient_id: 'PAT99887766'}),
          (h:HCP {npi: '1234567890'}),
          (b:Brand {brand_name: 'remibrutinib'})
    MERGE (p)-[t:TREATED_BY]->(h)
    MERGE (p)-[rx:PRESCRIBED]->(b)
    SET rx.prescription_date = datetime('2026-01-13'),
        rx.status = 'active'
""")

# 4. RUN VALIDATION RULES
validation_rules = get_validation_rules()

# Check: Does patient have only one primary HCP?
result = graph.query("""
    MATCH (p:Patient {patient_id: 'PAT99887766'})-[r:TREATED_BY {is_primary_hcp: true}]->()
    RETURN COUNT(r) AS primary_count
""")

if result[0]['primary_count'] > 1:
    raise ValidationError("Patient has multiple primary HCPs")

# 5. TRIGGER INFERENCE RULES (if scheduled)
# These run automatically (daily/weekly/monthly)
# Example: indirect_treatment rule will infer HCP->Brand relationship
```

**Result**: 
- ✅ New Patient node created
- ✅ New edges created (TREATED_BY, PRESCRIBED)
- ✅ Validation rules enforced
- ✅ Inference rules will discover new relationships on next run

**No schema change needed** - all entities fit existing structure.

---

### **Scenario 2: Inference Rules Discover New Relationships**

**Automatic Process** (runs on schedule):

```python
from ontology_output.run_inference_rules import InferenceEngine

engine = InferenceEngine(db=graph, graph_name='e2i_semantic')

# Run daily at 2 AM
results = engine.run_indirect_treatment()
# Output: "indirect_treatment completed: 47 new PRESCRIBES edges created"

# This discovers:
# Patient→HCP + Patient→Brand ⇒ HCP→Brand (inferred prescribing relationship)
```

**Example**:
```cypher
# Before inference:
(Patient:PAT123)-[:TREATED_BY]->(HCP:12345)
(Patient:PAT123)-[:PRESCRIBED]->(Brand:remibrutinib)

# After inference (automatic):
(HCP:12345)-[:PRESCRIBES {inferred: true}]->(Brand:remibrutinib)
```

**No human intervention required** - inference rules run on cron schedule.

---

### **Scenario 3: Agents Generate New Triggers**

**Automatic Process**:

```python
# CausalImpactAgent discovers new causal relationship
causal_path = {
    'path_id': 'CP87654321',
    'source_variable': 'rep_visit_frequency',
    'target_variable': 'hcp_prescription_volume',
    'effect_size': 0.15,
    'confidence': 0.82,
    'method_used': 'econml_dml',
    'validation_status': 'validated'
}

# 1. VALIDATE against vocabulary
is_valid, errors = validator.validate_node('CausalPath', causal_path)

# 2. CREATE CausalPath node
graph.query("""
    CREATE (cp:CausalPath {
        path_id: 'CP87654321',
        source_variable: 'rep_visit_frequency',
        target_variable: 'hcp_prescription_volume',
        effect_size: 0.15,
        confidence: 0.82,
        method_used: 'econml_dml',
        validation_status: 'validated',
        gate_decision: 'proceed',  # ✅ confidence >= 0.75
        created_at: datetime()
    })
""")

# 3. ActionGeneratorAgent creates trigger
trigger = {
    'trigger_id': 'TRG11223344',
    'trigger_type': 'recommendation',  # ✅ Valid from vocabulary
    'priority': 'high',
    'status': 'pending',
    'message': 'Increase rep visit frequency to HCP12345',
    'expiration_date': '2026-02-13'
}

# 4. CREATE edges for provenance
graph.query("""
    MATCH (agent:Agent {agent_name: 'CausalImpactAgent'}),
          (cp:CausalPath {path_id: 'CP87654321'})
    CREATE (agent)-[:DISCOVERED {discovery_date: datetime()}]->(cp)
""")

graph.query("""
    MATCH (agent:Agent {agent_name: 'ActionGeneratorAgent'}),
          (trigger:Trigger {trigger_id: 'TRG11223344'})
    CREATE (agent)-[:GENERATED {generation_date: datetime()}]->(trigger)
""")
```

**Result**:
- ✅ New CausalPath node (discovered relationship)
- ✅ New Trigger node (recommended action)
- ✅ Provenance tracked (which agent discovered/generated what)
- ✅ All validated against vocabulary

---

### **Scenario 4: Feedback Loop Updates (Model Drift Detection)**

**Automatic Process** (runs weekly):

```python
# ModelDriftDetectorAgent detects data drift
drift_result = {
    'model_id': 'script_conversion_v2',
    'drift_type': 'data_drift',
    'detection_method': 'psi',
    'drift_score': 0.35,
    'severity': 'warning'
}

# ✅ All values validated against vocabulary:
# - drift_type in ['data_drift', 'model_drift', 'concept_drift']
# - detection_method in ['psi', 'ks_test', 'chi_square', 'jensen_shannon']
# - severity calculated from drift_score using vocabulary thresholds

if drift_score > 0.25:  # From vocabulary: psi_thresholds.significant_shift
    # Create alert
    create_drift_alert(model_id, severity='warning')
    
    # Flag model for retraining
    update_model_status(model_id, status='needs_retraining')
```

**Result**:
- ✅ Drift detected using methods from vocabulary
- ✅ Severity classified using vocabulary thresholds
- ✅ Model status updated
- ✅ Alerts generated

---

## 🔧 SCHEMA-LEVEL UPDATES (Manual, Periodic)

### **When Schema Changes Are Needed**

**Indicators**:
1. **New entity type** not covered by existing 8 node types
2. **New relationship type** not covered by existing 15 edge types
3. **New property** needed on existing node/edge type
4. **New validation rule** for data quality
5. **New inference rule** for relationship discovery

### **Scenario 5: Adding New Brand (Novartis launches new drug)**

**New Data**:
```yaml
# New brand: Cosentyx (not in current vocabulary)
brand: cosentyx
therapeutic_area: immunology
indication: "Psoriatic arthritis"
```

**Update Process**:

```bash
# 1. UPDATE VOCABULARY (manual)
# Edit: core/01_entities.yaml

brands:
  remibrutinib:
    therapeutic_area: "Immunology"
    indication: "Chronic Spontaneous Urticaria (CSU)"
    
  fabhalta:
    therapeutic_area: "Hematology"
    indication: "Paroxysmal Nocturnal Hemoglobinuria (PNH)"
    
  kisqali:
    therapeutic_area: "Oncology"
    indication: "HR+/HER2- breast cancer"
    
  # NEW BRAND ADDED
  cosentyx:
    therapeutic_area: "Immunology"
    indication: "Psoriatic arthritis"
    abbreviation: "COS"
    status: "active"

# 2. UPDATE NODE TYPE ENUM (manual)
# Edit: ontology/01_node_types.yaml

Brand:
  properties:
    brand_name:
      type: enum
      values: [remibrutinib, fabhalta, kisqali, cosentyx]  # ADDED cosentyx
```

```bash
# 3. RECOMPILE SCHEMA
cd config/domain_vocabulary/
python apply_vocabulary_to_ontology.py --action schema

# Output:
# Generated 9 constraints (unchanged)
# Generated 45 indexes (unchanged)
# ✅ Schema updated with new brand enum
```

```python
# 4. APPLY TO FALKORDB (manual - production deployment)
# No structural change needed - just new enum value
# Existing Brand nodes still work
# New Brand node can be created:

graph.query("""
    CREATE (b:Brand {
        brand_name: 'cosentyx',
        therapeutic_area: 'immunology',
        indication: 'Psoriatic arthritis',
        created_at: datetime()
    })
""")
```

```bash
# 5. VERSION UPDATE
# Edit: index.yaml

_metadata:
  version: "6.1.0"  # Increment MINOR version
  last_updated: "2026-01-15"
  
  changelog:
    - version: "6.1.0"
      date: "2026-01-15"
      changes:
        - "Added new brand: cosentyx"
        - "Updated Brand.brand_name enum"
```

**Result**:
- ✅ New brand added to vocabulary
- ✅ Schema updated
- ✅ Version incremented
- ✅ All validation now includes new brand

**Deployment Timeline**: 1-2 days (testing + approval)

---

### **Scenario 6: Adding New Node Type (Prescriber Network)**

**Business Need**: Track prescriber influence networks explicitly

**New Entity**:
```yaml
# NEW NODE TYPE
PrescriberNetwork:
  description: "Network of prescribers who influence each other"
  properties:
    network_id:
      type: string
      required: true
      unique: true
      pattern: "^NET[0-9]{8}$"
    network_type:
      type: enum
      values: [peer_group, referral_network, hospital_system, practice_group]
    member_count:
      type: integer
      min: 2
```

**Update Process**:

```bash
# 1. ADD NODE TYPE TO VOCABULARY
# Edit: ontology/01_node_types.yaml
# Add full PrescriberNetwork definition

# 2. ADD EDGE TYPES
# Edit: ontology/02_edge_types.yaml

MEMBER_OF:
  description: "HCP is member of prescriber network"
  source: HCP
  target: PrescriberNetwork
  cardinality: "N:M"

# 3. UPDATE FALKORDB CONFIG
# Edit: ontology/05_falkordb_config.yaml

constraints:
  unique_properties:
    - node: PrescriberNetwork
      property: network_id

indexes:
  PrescriberNetwork:
    - network_id
    - network_type
    - created_at
```

```bash
# 4. RECOMPILE SCHEMA
python apply_vocabulary_to_ontology.py --action schema

# Output:
# Generated 10 constraints (+1 new)
# Generated 48 indexes (+3 new)
```

```python
# 5. APPLY TO FALKORDB (production)
# This is a BREAKING CHANGE if constraints conflict

# Apply new constraints
graph.query("""
    GRAPH.CONSTRAINT CREATE FOR (n:PrescriberNetwork) 
    REQUIRE n.network_id IS UNIQUE
""")

# Apply new indexes
graph.query("GRAPH.IDX CREATE FOR (n:PrescriberNetwork) ON (n.network_id)")
graph.query("GRAPH.IDX CREATE FOR (n:PrescriberNetwork) ON (n.network_type)")
```

```bash
# 6. VERSION UPDATE
# Edit: index.yaml

_metadata:
  version: "7.0.0"  # Increment MAJOR version (new node type)
  last_updated: "2026-02-01"
  
  changelog:
    - version: "7.0.0"
      date: "2026-02-01"
      changes:
        - "BREAKING: Added new node type PrescriberNetwork"
        - "Added edge type MEMBER_OF"
        - "Added 1 constraint, 3 indexes"
```

**Result**:
- ✅ New node type added
- ✅ New edge type added
- ✅ Schema extended
- ✅ Version incremented (MAJOR)

**Deployment Timeline**: 1-2 weeks (testing, approval, migration)

---

### **Scenario 7: Adding New Inference Rule**

**Business Need**: Detect emerging HCP opinion leaders

**New Rule**:
```yaml
# Edit: ontology/03_inference_rules.yaml

inference_rules:
  # ... existing rules ...
  
  emerging_opinion_leader:
    description: "Identify HCPs with rapidly growing influence"
    enabled: true
    frequency: "weekly"
    priority: 6
    
    cypher_query: |
      MATCH (h:HCP)-[i:INFLUENCES]->(:HCP)
      WHERE i.influence_strength >= 0.70
      WITH h, COUNT(i) AS influence_count,
           AVG(i.influence_strength) AS avg_strength
      WHERE influence_count >= 5
        AND avg_strength >= 0.75
      SET h.opinion_leader = true,
          h.opinion_leader_score = influence_count * avg_strength
      RETURN h.hcp_id, h.opinion_leader_score
    
    parameters:
      min_influence_strength: 0.70
      min_influence_count: 5
      min_avg_strength: 0.75
```

```bash
# REGENERATE INFERENCE SCHEDULER
python apply_vocabulary_to_ontology.py --action inference

# Output:
# ✅ Inference scheduler written to ontology_output/run_inference_rules.py
# New method: run_emerging_opinion_leader()
```

```python
# DEPLOY NEW SCHEDULER (production)
# Replace existing run_inference_rules.py with updated version

# NEW CRON JOB (add to crontab)
# Run new rule weekly on Sundays at 4 AM
0 4 * * 0 python ontology_output/run_inference_rules.py --rule emerging_opinion_leader
```

**Result**:
- ✅ New inference rule added
- ✅ Scheduler updated
- ✅ Cron job scheduled
- ✅ HCPs automatically flagged as opinion leaders

**Deployment Timeline**: 1 week (testing, approval)

---

## 📅 Update Schedules

### **Automatic (No Human Intervention)**

| Update Type | Frequency | Triggered By |
|-------------|-----------|--------------|
| **New nodes/edges** | Continuous | Data ingestion pipelines |
| **Inference rules** | Daily/Weekly/Monthly | Cron scheduler |
| **Drift detection** | Weekly | ModelDriftDetectorAgent |
| **Outcome labeling** | Daily | OutcomeTruthLabelerAgent |
| **Trigger generation** | Real-time | ActionGeneratorAgent |

### **Manual (Requires Approval)**

| Update Type | Frequency | Review Process |
|-------------|-----------|----------------|
| **New entity types** | Quarterly | Architecture review |
| **New relationship types** | Quarterly | Architecture review |
| **New validation rules** | Monthly | Data quality review |
| **New inference rules** | Monthly | Data science review |
| **Property additions** | Monthly | Schema review |

---

## 🔍 Monitoring & Observability

### **Tracking Ontology Changes**

```python
# All schema changes logged to SchemaVersion node
graph.query("""
    MERGE (sv:SchemaVersion {version: '7.0.0'})
    SET sv.deployed_at = datetime(),
        sv.deployed_by = 'e.estrada@novartis.com',
        sv.changes = [
            'Added PrescriberNetwork node type',
            'Added MEMBER_OF edge type'
        ]
""")

# Query change history
history = graph.query("""
    MATCH (sv:SchemaVersion)
    RETURN sv.version, sv.deployed_at, sv.changes
    ORDER BY sv.deployed_at DESC
    LIMIT 10
""")
```

### **Validation Metrics**

```python
# Track validation failures
validation_metrics = {
    'total_validations': 1000000,
    'validation_failures': 143,
    'failure_rate': 0.000143,
    'top_failures': [
        {'rule': 'journey_stage_progression', 'count': 67},
        {'rule': 'specialty_brand_alignment', 'count': 42},
        {'rule': 'confidence_threshold', 'count': 34}
    ]
}

# Alert if failure rate > threshold
if validation_metrics['failure_rate'] > 0.001:  # 0.1%
    send_alert("High validation failure rate detected")
```

### **Inference Rule Performance**

```python
# Track inference rule execution
inference_metrics = {
    'rule': 'causal_chain',
    'execution_time_seconds': 45.2,
    'rows_affected': 3421,
    'errors': 0,
    'last_run': '2026-01-13T03:00:00Z'
}

# Alert if execution time exceeds threshold
if inference_metrics['execution_time_seconds'] > 300:  # 5 minutes
    send_alert("Inference rule 'causal_chain' taking too long")
```

---

## 🔐 Governance & Approval Workflow

### **Schema Change Approval Matrix**

| Change Type | Approval Required | Review Process |
|-------------|-------------------|----------------|
| **New enum value** | Data Steward | 1-2 days |
| **New property** | Data Architect | 3-5 days |
| **New node type** | Architecture Board | 1-2 weeks |
| **New edge type** | Architecture Board | 1-2 weeks |
| **New inference rule** | Data Science Lead | 1 week |
| **New validation rule** | Data Quality Lead | 1 week |

### **Change Request Template**

```markdown
# Ontology Change Request

**Request ID**: OCR-2026-001
**Requested By**: E. Estrada
**Date**: 2026-01-15
**Priority**: Medium

## Change Summary
Add new brand "cosentyx" to vocabulary

## Business Justification
Novartis launched Cosentyx for psoriatic arthritis. Need to track prescriptions and HCP engagement.

## Technical Details
- Module: core/01_entities.yaml
- Change Type: New enum value
- Breaking Change: No
- Impact: Low (additive only)

## Testing Plan
1. Add cosentyx to staging vocabulary
2. Create test Brand node
3. Run validation tests
4. Deploy to production

## Approval
- [ ] Data Steward: _______________
- [ ] Data Architect: _______________
- [ ] Deployment Date: _______________
```

---

## 🚨 Edge Cases & Error Handling

### **Case 1: Data Arrives Before Schema Updated**

```python
# New brand data arrives but vocabulary not updated yet
new_data = {
    'brand_name': 'cosentyx',  # ❌ Not in vocabulary enum
    'therapeutic_area': 'immunology'
}

# Validation FAILS
is_valid, errors = validator.validate_node('Brand', new_data)
# errors: ["Invalid enum value for brand_name: cosentyx"]

# SOLUTION: Queue data for retry
queue_for_retry(new_data, retry_after='2026-01-16')  # After schema update
```

### **Case 2: Inference Rule Creates Invalid Data**

```python
# Inference rule tries to create edge with invalid properties
graph.query("""
    MATCH (h:HCP)-[i:INFLUENCES]->(h2:HCP)
    WHERE i.influence_strength >= 0.70
    CREATE (h)-[:HIGHLY_INFLUENCES {strength: 1.5}]->(h2)  # ❌ strength > 1.0
""")

# SOLUTION: Add validation to inference rules
inference_rules:
  hcp_influence_propagation:
    post_validation:
      - check: "strength >= 0.0 AND strength <= 1.0"
        action: "Delete invalid edges"
```

### **Case 3: Schema Change Breaks Existing Data**

```python
# Change property type from string to enum
# BEFORE:
Patient.age_group: string (any value)

# AFTER:
Patient.age_group: enum [pediatric, young_adult, adult, senior]

# Existing data: "25-35 years old" ❌ Not in enum

# SOLUTION: Migration script
migration_script = """
    MATCH (p:Patient)
    WHERE NOT p.age_group IN ['pediatric', 'young_adult', 'adult', 'senior']
    SET p.age_group = CASE
        WHEN toInteger(split(p.age_group, '-')[0]) < 18 THEN 'pediatric'
        WHEN toInteger(split(p.age_group, '-')[0]) < 36 THEN 'young_adult'
        WHEN toInteger(split(p.age_group, '-')[0]) < 65 THEN 'adult'
        ELSE 'senior'
    END
"""
```

---

## ✅ Best Practices Summary

### **For Instance-Level Updates (Daily)**
1. ✅ Always validate against vocabulary before creating nodes/edges
2. ✅ Use inference rules to discover relationships automatically
3. ✅ Monitor validation failure rates
4. ✅ Log all changes for audit trail

### **For Schema-Level Updates (Periodic)**
1. ✅ Follow approval workflow before making changes
2. ✅ Version vocabulary modules using semantic versioning
3. ✅ Test schema changes in staging before production
4. ✅ Provide migration scripts for breaking changes
5. ✅ Document all changes in changelog
6. ✅ Update dependent systems (agents, dashboards, reports)

### **For Long-Term Maintenance**
1. ✅ Review vocabulary quarterly for obsolete terms
2. ✅ Monitor inference rule performance
3. ✅ Archive deprecated node/edge types
4. ✅ Keep documentation in sync with vocabulary
5. ✅ Train team on vocabulary updates

---

## 📞 Questions?

**For instance-level updates**: Contact Data Engineering Team  
**For schema-level updates**: Submit Ontology Change Request to Architecture Board  
**For inference rules**: Contact Data Science Team  
**For validation rules**: Contact Data Quality Team
