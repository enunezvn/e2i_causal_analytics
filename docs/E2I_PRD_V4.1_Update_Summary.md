# E2I PRD V4.1 Update Summary
## Changes Required in E2I_Causal_Analytics_PRD

---

## Document Version Update

**Current:** Document Version: 4.0  
**New:** Document Version: 4.1

---

## Section 3.1: ML-Compliant Schema

### Change 1: Section Header and Table Count
**Location:** Section 3.1  
**Current:**
> The V4 schema includes 26 tables...

**New:**
> The V4.1 schema includes **28 tables** with complete ML split tracking, MLOps integration, and causal validation infrastructure.

### Change 2: Add Validation Tables Section
**Location:** After "V4 ML Foundation Tables" section  
**Add:**

> **V4.1 New Tables for Causal Validation**
> 
> | **Table** | **Purpose** | **Integration** |
> |-----------|-------------|-----------------|
> | causal_validations | DoWhy refutation test results (5 test types) with gate decisions | Causal Impact agent Node 3 (Refutation) |
> | expert_reviews | Domain expert DAG approval tracking for regulatory compliance | GraphBuilder node expert check |

---

## Section 6.1: System Architecture

### Change 3: Data Layer Update
**Location:** Architecture diagram, Data Layer  
**Current:**
```
│ Data Layer (V4.0 - 26 Tables) │
```

**New:**
```
│ Data Layer (V4.1 - 28 Tables) │
```

---

## Section 2.2: Technical Stack

### Change 4: Add Validation to Causal Inference Libraries
**Location:** Causal Inference Libraries table  
**Add row:**

> | DoWhy | Causal graph construction, assumption validation, effect estimation, **refutation tests (V4.1)** |

---

## Section 10: Risk Assessment

### Change 5: Update Causal Validity Risk Mitigation
**Location:** Risk Matrix, "Causal Validity" row  
**Current:**
> **Mitigation:** DoWhy refutation tests, multiple methods, confidence intervals

**New:**
> **Mitigation:** DoWhy refutation tests (automated via `causal_validations` table with 5 test types), expert review workflow via `expert_reviews` table, gate decisions (proceed/review/block), confidence scoring

---

## Appendix Updates

### Change 6: Schema Appendix
**Add to Appendix C:**

> **Causal Validation Tables (V4.1)**
> 
> ```
> causal_validations
> ├── validation_id (PK)
> ├── estimate_id (FK → causal_paths or ml_experiments)
> ├── test_type (ENUM: placebo, common_cause, subset, bootstrap, sensitivity)
> ├── status (ENUM: passed, failed, warning, skipped)
> ├── confidence_score (0-1)
> ├── gate_decision (ENUM: proceed, review, block)
> └── details_json
> 
> expert_reviews
> ├── review_id (PK)
> ├── review_type (ENUM: dag_approval, methodology_review, quarterly_audit, ad_hoc)
> ├── dag_version_hash (SHA256)
> ├── approval_status
> ├── checklist_json
> └── valid_until
> ```

### Change 7: Migration Reference
**Update Appendix C migration list:**

> - 010_causal_validation_tables.sql - V4.1: Validation infrastructure (4 ENUMs, 2 tables, 4 views, 4 functions)

---

## Summary of V4.1 Changes

| Area | Change | Impact |
|------|--------|--------|
| Schema | 26 → 28 tables | +2 validation tables |
| ENUMs | +4 new types | refutation_test_type, validation_status, gate_decision, expert_review_type |
| Views | +4 new views | v_validation_summary, v_active_expert_approvals, v_blocked_estimates, v_pending_expert_reviews |
| Functions | +4 new functions | is_dag_approved(), get_validation_gate(), get_validation_confidence(), can_use_estimate() |
| Agents | causal_impact enhanced | Added VALIDATION intent, 5-node workflow with gate |
| domain_vocabulary | v3.0.0 → v3.1.0 | Added validation ENUMs, aliases, intent indicators |

---

## Files Modified in V4.1 Release

| File | Version Change | Key Updates |
|------|---------------|-------------|
| `e2i_nlv_project_structure.md` | v4.0 → v4.1 | Added validation tables, repositories, models, tests |
| `config/domain_vocabulary.yaml` | v3.0.0 → v3.1.0 | Added 4 validation ENUM sections, aliases, intent |
| `database/migrations/010_causal_validation_tables.sql` | NEW | Complete validation infrastructure |
| `src/database/models/enums.py` | Updated | +4 validation ENUMs |
| `src/database/models/validation_models.py` | NEW | Pydantic schemas |
| `src/database/repositories/causal_validation.py` | NEW | CRUD operations |
| `src/database/repositories/expert_review.py` | NEW | CRUD operations |
| `src/causal_engine/refutation.py` | Enhanced | RefutationRunner with persistence |
| `src/causal_engine/models/validation_models.py` | NEW | RefutationResult, ValidationSuite |
| `src/api/routes/validation.py` | NEW | /validation/* endpoints |
| `frontend/src/components/validation/*` | NEW | UI components |
| `tests/unit/test_validation/*` | NEW | Unit tests |
| `tests/synthetic/*` | NEW | Synthetic benchmark tests |

---

## Implementation Checklist

1. ☐ Run migration `010_causal_validation_tables.sql` on Supabase
2. ☐ Update `domain_vocabulary.yaml` to v3.1.0
3. ☐ Add Python ENUMs to `src/database/models/enums.py`
4. ☐ Create `validation_models.py` Pydantic schemas
5. ☐ Create `causal_validation.py` repository
6. ☐ Create `expert_review.py` repository
7. ☐ Enhance `RefutationRunner` to persist results
8. ☐ Wire causal_impact agent Node 3 to RefutationRunner
9. ☐ Create `/validation/*` API routes
10. ☐ Create frontend validation components
11. ☐ Update PRD document with changes above
12. ☐ Run synthetic benchmark tests

---

*Generated: December 2025*  
*For: E2I Causal Analytics V4.1 Release*
