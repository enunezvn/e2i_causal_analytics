# E2I Schema Changes for GEPA Migration
## V4.1 → V4.2 Migration Summary

---

## Quick Answer: NO Deletions Required

**All existing tables are retained.** GEPA is an *additive* change to your infrastructure.

---

## What STAYS (V4.1 → V4.2)

| Category | Tables | Status |
|----------|--------|--------|
| Core Data | 8 tables | ✅ Unchanged |
| ML Split Management | 4 tables | ✅ Unchanged |
| KPI Gap Tables | 6 tables | ✅ Unchanged |
| ML Foundation | 8 tables | ✅ Unchanged |
| Validation | 2 tables | ✅ Unchanged |
| **Existing Total** | **28 tables** | ✅ **All retained** |

### Why Keep Everything?

1. **`ml_experiments`** - Still tracks MLflow experiments (GEPA logs here too)
2. **`ml_observability_spans`** - Opik tracing still needed for GEPA runs
3. **`causal_validations`** - GEPA metrics directly reference refutation results
4. **`expert_reviews`** - DAG approval still gates Causal Impact agent
5. **`agent_activities`** - Still tracks all agent executions

---

## What's ADDED (New in V4.2)

### New ENUMs (4)

```sql
optimizer_type          -- miprov2, gepa, bootstrap_fewshot, ...
gepa_budget_preset      -- light, medium, heavy, custom
optimization_status     -- pending, running, completed, failed, ...
ab_test_variant         -- baseline, gepa, gepa_v2, control
```

### New Tables (5)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `prompt_optimization_runs` | Track GEPA sessions | agent_name, optimizer_type, baseline_score, optimized_score, improvement_percent |
| `optimized_instructions` | Store versioned prompts | instruction_text, version, is_active, val_score |
| `optimized_tool_descriptions` | Tool optimization outputs | tool_name, description_text, is_active |
| `prompt_ab_tests` | A/B test configuration | baseline_instruction_id, treatment_instruction_id, traffic_split |
| `prompt_ab_test_observations` | A/B test data | variant, score, latency_ms |

### New Views (4)

```sql
v_active_instructions      -- Current active instructions per agent
v_optimization_summary     -- Aggregated optimization results
v_ab_test_results          -- A/B test outcomes
v_optimizer_comparison     -- GEPA vs MIPROv2 comparison
```

### Table Count Change

```
V4.1: 28 tables
V4.2: 33 tables (+5 new)
```

---

## What's UPDATED (Vocabulary)

### domain_vocabulary.yaml: V3.1.0 → V3.2.0

**Added sections:**
- `optimizer_types`
- `gepa_budget_presets`
- `optimization_statuses`
- `ab_test_variants`
- `gepa_candidate_selection_strategies`
- `agent_types` (explicit standard/hybrid/deep)
- `gepa_metric_components`
- `reflection_models`

**Updated sections:**
- `agents` - Added `gepa_config` per agent with budget/tool_optimization settings
- `database_tables` - Added `prompt_optimization` category

**Unchanged sections:**
- All V3.1.0 validation ENUMs (refutation_test_types, validation_statuses, gate_decisions, expert_review_types)
- brands, regions, model_stages, mlops_tools, memory_types

---

## Migration Instructions

### Step 1: Run SQL Migration

```bash
# From your project root
psql -d e2i_database -f migrations/011_gepa_optimization_tables.sql

# Or via your migration tool
alembic upgrade head  # If using Alembic
```

### Step 2: Update Vocabulary File

```bash
# Replace domain_vocabulary.yaml
cp config/domain_vocabulary_v3.2.0.yaml config/domain_vocabulary.yaml

# Or merge the new sections into your existing file
```

### Step 3: Verify Migration

```sql
-- Check new tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_name LIKE 'prompt_%' OR table_name LIKE 'optimized_%';

-- Check new ENUMs exist
SELECT typname 
FROM pg_type 
WHERE typname IN ('optimizer_type', 'gepa_budget_preset', 'optimization_status', 'ab_test_variant');

-- Check views
SELECT viewname FROM pg_views WHERE viewname LIKE 'v_%instruction%' OR viewname LIKE 'v_%optimization%';
```

### Step 4: Validate Existing Data

```sql
-- Ensure existing tables unaffected
SELECT COUNT(*) FROM causal_validations;  -- Should match pre-migration count
SELECT COUNT(*) FROM ml_experiments;      -- Should match pre-migration count
```

---

## Rollback Plan

If issues arise:

```sql
-- Drop new views
DROP VIEW IF EXISTS v_active_instructions;
DROP VIEW IF EXISTS v_optimization_summary;
DROP VIEW IF EXISTS v_ab_test_results;
DROP VIEW IF EXISTS v_optimizer_comparison;

-- Drop new tables (reverse order)
DROP TABLE IF EXISTS prompt_ab_test_observations;
DROP TABLE IF EXISTS prompt_ab_tests;
DROP TABLE IF EXISTS optimized_tool_descriptions;
DROP TABLE IF EXISTS optimized_instructions;
DROP TABLE IF EXISTS prompt_optimization_runs;

-- Drop new ENUMs
DROP TYPE IF EXISTS ab_test_variant;
DROP TYPE IF EXISTS optimization_status;
DROP TYPE IF EXISTS gepa_budget_preset;
DROP TYPE IF EXISTS optimizer_type;

-- Revert vocabulary file
cp config/domain_vocabulary_v3.1.0.yaml config/domain_vocabulary.yaml
```

---

## Integration Points

### How New Tables Connect to Existing Schema

```
┌─────────────────────────┐
│  ml_experiments         │ ←── MLflow still logs here
│  (existing)             │
└───────────┬─────────────┘
            │ mlflow_run_id
            ▼
┌─────────────────────────┐
│  prompt_optimization_   │ ←── GEPA optimization sessions
│  runs (NEW)             │
└───────────┬─────────────┘
            │ run_id
            ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  optimized_instructions │     │  optimized_tool_        │
│  (NEW)                  │     │  descriptions (NEW)     │
└───────────┬─────────────┘     └─────────────────────────┘
            │ instruction_id
            ▼
┌─────────────────────────┐
│  prompt_ab_tests (NEW)  │
└───────────┬─────────────┘
            │ test_id
            ▼
┌─────────────────────────┐
│  prompt_ab_test_        │
│  observations (NEW)     │
└─────────────────────────┘
```

### GEPA Metrics → Existing Tables

```python
# CausalImpactGEPAMetric reads from existing tables:
causal_validations     # Refutation test results
expert_reviews         # DAG approval status
agent_activities       # Execution traces
```

---

## Summary

| Action | Count | Details |
|--------|-------|---------|
| Tables DELETED | 0 | None |
| Tables UNCHANGED | 28 | All V4.1 tables |
| Tables ADDED | 5 | GEPA optimization tracking |
| ENUMs ADDED | 4 | Optimizer types, statuses |
| Views ADDED | 4 | Optimization summaries |
| Vocabulary UPDATED | 1 file | V3.1.0 → V3.2.0 |

**Total post-migration: 33 tables**

---

*Migration Version: 011*  
*Schema Version: V4.2*  
*Vocabulary Version: V3.2.0*
