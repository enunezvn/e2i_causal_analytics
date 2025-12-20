# E2I ML-Compliant Data Package V3.0 - KPI Coverage Summary

## Overview

This package provides a **clean drop/recreate** approach to address all 12 KPI calculability gaps identified in the domain_vocabulary.yaml review.

## Recommendation: Clean Drop & Recreate ✅

**Why clean reinstall is better for proof-of-concept:**
1. Agent enum changes are fundamental (8 → 11 agents)
2. Multiple new tables needed (6 new tables)
3. New fields across existing tables
4. Simpler to validate complete coverage
5. No migration complexity for dummy data

---

## Files Delivered

### 1. SQL Schema
**File:** `e2i_ml_complete_v3_schema.sql` (~58 KB)

Run this in Supabase SQL Editor to create all tables.

```sql
-- To do a clean install, uncomment the DROP statements in PART 0
-- Then run the entire script
```

### 2. Data Generator  
**File:** `e2i_ml_complete_v3_generator.py` (~62 KB)

Python script that generates complete synthetic data with all fields populated.

```bash
python e2i_ml_complete_v3_generator.py
```

### 3. Data Loader
**File:** `e2i_ml_complete_v3_loader.py` (~14 KB)

Loads JSON data into Supabase.

```bash
export SUPABASE_URL='https://your-project.supabase.co'
export SUPABASE_KEY='your-service-role-key'
python e2i_ml_complete_v3_loader.py --data-dir ./e2i_ml_complete_v3_data
```

### 4. Data Files
**Directory:** `e2i_ml_complete_v3_data/` (~25 MB)

| File | Records | Description |
|------|---------|-------------|
| `e2i_ml_v3_split_registry.json` | 1 | ML split configuration |
| `e2i_ml_v3_preprocessing_metadata.json` | 1 | Training stats |
| `e2i_ml_v3_reference_universe.json` | 84 | Coverage targets |
| `e2i_ml_v3_hcp_profiles.json` | 50 | Healthcare providers |
| `e2i_ml_v3_patient_journeys.json` | 200 | Patient journeys |
| `e2i_ml_v3_treatment_events.json` | 1,501 | Treatment events |
| `e2i_ml_v3_ml_predictions.json` | 626 | ML predictions |
| `e2i_ml_v3_triggers.json` | 606 | HCP triggers |
| `e2i_ml_v3_agent_activities.json` | 2,323 | Agent activities |
| `e2i_ml_v3_business_metrics.json` | 3,168 | Business metrics |
| `e2i_ml_v3_causal_paths.json` | 50 | Causal paths |
| `e2i_ml_v3_user_sessions.json` | 7,494 | **NEW** User sessions |
| `e2i_ml_v3_data_source_tracking.json` | 1,520 | **NEW** Source tracking |
| `e2i_ml_v3_ml_annotations.json` | 153 | **NEW** Annotations |
| `e2i_ml_v3_etl_pipeline_metrics.json` | 1,216 | **NEW** ETL metrics |
| `e2i_ml_v3_hcp_intent_surveys.json` | 360 | **NEW** Intent surveys |
| `e2i_ml_v3_train.json` | varies | Train split data |
| `e2i_ml_v3_validation.json` | varies | Validation split data |
| `e2i_ml_v3_test.json` | varies | Test split data |

**Total Records: 19,353**

---

## KPI Gap Resolution

### WS1: Data Coverage & Quality (6 gaps → 0 gaps)

| KPI | Previous Status | Solution | New Status |
|-----|-----------------|----------|------------|
| **Cross-source Match Rate** | ❌ GAP | Added `data_source_tracking` table with `match_rate_vs_*` fields | ✅ DIRECT |
| **Stacking Lift** | ❌ GAP | Added `stacking_lift_percentage`, `stacking_eligible_records` to `data_source_tracking` | ✅ DIRECT |
| **Data Lag (Median)** | ❌ GAP | Added `source_timestamp`, `ingestion_timestamp`, `data_lag_hours` to `patient_journeys` | ✅ DIRECT |
| **Label Quality (IAA)** | ❌ GAP | Added `ml_annotations` table with `iaa_group_id` for inter-annotator agreement | ✅ DIRECT |
| **Time-to-Release (TTR)** | ❌ GAP | Added `etl_pipeline_metrics` table with `time_to_release_hours` | ✅ DIRECT |
| **Source Coverage** | ⚠️ PARTIAL | Added `reference_universe` table for target counts | ✅ DERIVED |

### WS1: Model Performance (3 gaps → 0 gaps)

| KPI | Previous Status | Solution | New Status |
|-----|-----------------|----------|------------|
| **PR-AUC** | ❌ GAP | Added `model_pr_auc` field to `ml_predictions` | ✅ DIRECT |
| **Recall@Top-K** | ❌ GAP | Added `rank_metrics` JSONB with recall/precision at K values | ✅ DIRECT |
| **Brier Score** | ❌ GAP | Added `brier_score` field to `ml_predictions` | ✅ DIRECT |

### WS2: Trigger Performance (1 gap → 0 gaps)

| KPI | Previous Status | Solution | New Status |
|-----|-----------------|----------|------------|
| **Change-Fail Rate (CFR)** | ❌ GAP | Added `previous_trigger_id`, `change_type`, `change_failed`, `change_outcome_delta` to `triggers` | ✅ DERIVED |

### WS3: Business Impact (1 gap → 0 gaps)

| KPI | Previous Status | Solution | New Status |
|-----|-----------------|----------|------------|
| **Active Users (MAU/WAU)** | ❌ GAP | Added `user_sessions` table with complete session tracking | ✅ DIRECT |

### Brand-Specific (1 gap → 0 gaps)

| KPI | Previous Status | Solution | New Status |
|-----|-----------------|----------|------------|
| **Remi Intent-to-Prescribe Δ** | ❌ GAP | Added `hcp_intent_surveys` table with `intent_to_prescribe_score`, `intent_to_prescribe_change` | ✅ DIRECT |

---

## KPI Helper Views Created

The schema includes ready-to-use views for KPI calculations:

```sql
-- WS1 KPIs
v_kpi_cross_source_match    -- Cross-source match rates by date/source
v_kpi_stacking_lift         -- Stacking lift percentages
v_kpi_data_lag              -- Data lag statistics (avg, median, p95)
v_kpi_label_quality         -- Label quality and IAA metrics
v_kpi_time_to_release       -- TTR by pipeline

-- WS2 KPIs
v_kpi_change_fail_rate      -- Change-fail rate by change type

-- WS3 KPIs
v_kpi_active_users          -- MAU, WAU, DAU counts

-- Brand-Specific
v_kpi_intent_to_prescribe   -- Intent scores by brand/month
```

---

## 11-Agent Architecture

The schema includes the updated agent registry with tiered architecture:

| Tier | Agents | Purpose |
|------|--------|---------|
| **1: Coordination** | orchestrator | Query routing, synthesis |
| **2: Causal Analytics** | causal_impact, gap_analyzer, heterogeneous_optimizer | Core E2I mission |
| **3: Monitoring** | drift_monitor, experiment_designer, health_score | System health |
| **4: ML Predictions** | prediction_synthesizer, resource_optimizer | Model operations |
| **5: Self-Improvement** | explainer, feedback_learner | NLV + RAG support |

---

## Installation Steps

### Option A: Clean Install (Recommended for PoC)

1. **Backup existing data (if any)**
   ```sql
   -- Run pg_dump or export via Supabase dashboard
   ```

2. **Drop existing tables**
   - Uncomment the DROP statements in `PART 0` of the schema
   - Run the schema in Supabase SQL Editor

3. **Run complete schema**
   ```sql
   -- Run e2i_ml_complete_v3_schema.sql in Supabase SQL Editor
   ```

4. **Load data**
   ```bash
   pip install supabase faker --break-system-packages
   python e2i_ml_complete_v3_loader.py --data-dir ./e2i_ml_complete_v3_data
   ```

5. **Verify**
   ```sql
   -- Check table counts
   SELECT 'patient_journeys' as tbl, count(*) FROM patient_journeys
   UNION ALL SELECT 'user_sessions', count(*) FROM user_sessions
   UNION ALL SELECT 'data_source_tracking', count(*) FROM data_source_tracking;
   
   -- Run leakage audit
   SELECT * FROM run_leakage_audit('<split_config_id>');
   ```

### Option B: Migration (If preserving existing data)

Use the `006_update_agent_enum.sql` migration for agent changes, then manually add:
- New tables: `user_sessions`, `data_source_tracking`, `ml_annotations`, `etl_pipeline_metrics`, `hcp_intent_surveys`, `reference_universe`
- New columns to existing tables (see schema PART 3)

---

## Split Distribution

| Split | Patients | Ratio | Date Range |
|-------|----------|-------|------------|
| Train | 134 | 67.0% | Jan-Jun 2025 |
| Validation | 50 | 25.0% | Jul-Aug 2025 |
| Test | 15 | 7.5% | Sep 2025 |
| Holdout | 1 | 0.5% | Oct 2025 |

---

## Summary: Before vs After

| Metric | Before (V2) | After (V3) |
|--------|-------------|------------|
| **KPI Gaps** | 12 | 0 |
| **Tables** | 12 | 18 (+6 new) |
| **Agent Count** | 8 | 11 |
| **Coverage** | 74% | 100% |
| **Total Records** | ~10K | ~19K |

**All 46 KPIs are now calculable from the data schema.**
