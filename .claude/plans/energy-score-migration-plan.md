# Migration Replacement Plan: 023 → 011 Energy Score Tables

**Created**: 2025-12-27
**Status**: ✅ COMPLETED
**Complexity**: Medium (8 phases, ~16 tasks)
**Completed**: 2025-12-27

---

## Phase 0: Setup (First Task)

- [x] **0.1** Copy this plan to project directory

---

## Executive Summary

Replace incorrect migration `023_energy_score_tables.sql` with `011_energy_score_enhancement.sql`. The code (`mlflow_tracker.py`) already expects migration 011's schema, making 023 incompatible.

**Key Issue**: Migration 011 has a FK bug that must be fixed before applying.

---

## Critical Files

| File | Action | Notes |
|------|--------|-------|
| `database/causal/011_energy_score_enhancement.sql` | **FIX + APPLY** | ✅ Fixed FK at line 29, applied |
| `database/ml/023_energy_score_tables.sql` | **DELETE** | ✅ Removed from git |
| `src/causal_engine/energy_score/mlflow_tracker.py` | VERIFY | ✅ Uses 011 schema (lines 296-337) |
| `tests/unit/test_agents/test_causal_impact/test_energy_score_selection.py` | TEST | ✅ 18/18 tests passed |

---

## Phase 1: Pre-Migration Checks ✅

**Goal**: Verify database state before making changes

### Tasks:
- [x] **1.1** Check if 023 ENUM types exist in database → **Found all 3 types**
- [x] **1.2** Check if estimator_evaluations table exists → **Yes, with 023 schema (query_id)**
- [x] **1.3** Count existing data (if any) → **0 rows (safe to rollback)**
- [x] **1.4** Verify git status → **023 tracked, 011 untracked**

**Result**: Migration 023 was applied to database with 0 data rows

---

## Phase 2: Fix Migration 011 ✅

**Goal**: Fix FK reference bug before applying

### Bug Fixed:
```sql
-- Line 29 ORIGINAL (WRONG):
experiment_id UUID NOT NULL REFERENCES ml_experiments(experiment_id) ON DELETE CASCADE,

-- Line 29 FIXED (CORRECT):
experiment_id UUID NOT NULL REFERENCES ml_experiments(id) ON DELETE CASCADE,
```

### Tasks:
- [x] **2.1** Edit `database/causal/011_energy_score_enhancement.sql`
- [x] **2.2** Change `ml_experiments(experiment_id)` → `ml_experiments(id)` at line 29

---

## Phase 3: Rollback 023 ✅

**Goal**: Remove 023 schema if it was applied to database

### Tasks:
- [x] **3.1** Create rollback script `database/ml/rollback_023.sql`
- [x] **3.2** Execute rollback via Supabase MCP
- [x] **3.3** Verify rollback succeeded (table/types gone) → **Confirmed empty results**

---

## Phase 4: Apply Migration 011 ✅

**Goal**: Apply the correct schema to database

### Tasks:
- [x] **4.1** Apply fixed 011 migration via Supabase MCP
- [x] **4.2** Verify table created with 28 correct columns
- [x] **4.3** Verify FK constraint references ml_experiments(id)
- [x] **4.4** Verify views created (v_estimator_performance, v_energy_score_trends, v_selection_comparison)

---

## Phase 5: Git Cleanup ✅

**Goal**: Update git tracking to reflect correct migration

### Tasks:
- [x] **5.1** Remove incorrect migration from git → `git rm database/ml/023_energy_score_tables.sql`
- [x] **5.2** Add correct migration to git → `git add database/causal/011_energy_score_enhancement.sql`
- [x] **5.3** Remove Zone.Identifier files (Windows metadata)
- [x] **5.4** Commit changes → `2b2bf86`

**Commit**: `fix(database): replace 023 with 011 energy score migration`

---

## Phase 6: Testing (Small Batches) ✅

**Goal**: Verify system works after migration

### Batch 1: Unit Tests - Energy Score
- [x] **6.1** Run energy score selection tests → **18/18 PASSED**

### Batch 2: Unit Tests - Causal Engine
- [x] **6.2** Run causal engine unit tests → **123/123 PASSED**

### Batch 3: Causal Impact Agent Tests
- [x] **6.3** Run causal impact agent tests → **163/163 PASSED**

**Total Tests Passed**: 304

---

## Phase 7: Post-Migration Validation ✅

**Goal**: Confirm everything is working correctly

### Tasks:
- [x] **7.1** Verify CHECK constraint allows all 9 estimator types
- [x] **7.2** Run Supabase security advisors (no new issues for estimator_evaluations)
- [x] **7.3** Update this plan to COMPLETED status

---

## Quick Reference: Schema Comparison

| Aspect | 023 (DELETED) | 011 (APPLIED) |
|--------|--------------|-------------|
| PK | `id` | `evaluation_id` |
| Context Link | `query_id`, `session_id` | `experiment_id` FK |
| Estimator Types | ENUM (4 types) | VARCHAR + CHECK (9 types) |
| Code Match | NO | YES (mlflow_tracker.py) |
| Location | database/ml/ | database/causal/ |

---

## Execution Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 0: Setup | < 1 min | ✅ |
| Phase 1: Pre-checks | 2 min | ✅ |
| Phase 2: Fix FK | 1 min | ✅ |
| Phase 3: Rollback | 2 min | ✅ |
| Phase 4: Apply | 3 min | ✅ |
| Phase 5: Git cleanup | 2 min | ✅ |
| Phase 6: Testing | 3 min | ✅ |
| Phase 7: Validation | 2 min | ✅ |
| **Total** | **~15 min** | ✅ |
