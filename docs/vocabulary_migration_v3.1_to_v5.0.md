# E2I Vocabulary Migration Guide: v3.1.0 → v5.0.0

## Overview

This document tracks the evolution of E2I domain vocabularies from the legacy v3.1.0 format (1 file) through v4.x iterations (6 files) to the consolidated v5.0.0 format (1 file).

**Migration Date**: 2025-12-28
**Consolidated File**: `config/domain_vocabulary.yaml` (v5.0.0)
**Files Archived**: 6 legacy vocabulary files → `config/archived/`

---

## Migration Timeline

### v3.1.0 → v4.2.0 (2024-11-01 to 2024-12-27)

**What Changed**:
- Added Tool Composer vocabularies (Section 3)
- Added DSPy/MIPROv2 integration (Section 11)
- Added Energy Score enhancement (Section 12)
- Added GEPA optimization (Section 13)
- Simplified tier structure (removed verbose SLA seconds)
- Created 5 specialized extension files

**Files Added**:
- `domain_vocabulary_v3_2_additions.yaml` (Energy Score, 157 lines)
- `Feedback Loop domain vocabulary.yml` (Concept drift, 519 lines)
- `Ragas-Opik Integration Domain Vocabulary.yml` (Evaluation, 351 lines)
- `003_memory_vocabulary.yaml` (Memory graph, 528 lines)

**Key Features Added**:
- Tool Composer & Orchestrator Classifier (v4.2.0)
- Energy Score Enhancement for causal estimator selection (v4.2.2)
- GEPA Prompt Optimization replacing MIPROv2 (v4.3.0)
- Feedback loop mechanism for concept drift detection
- Ragas-Opik integration for agent evaluation
- Semantic knowledge graph entity/relationship types

---

### v4.2.0 → v5.0.0 (2025-12-28)

**What Changed**:
- **Consolidated 6 files → 1 file**
- Eliminated 1,569 lines of duplicate content (37.5% reduction: 4,179 → 2,610 lines)
- Added Section 14: Feedback Loop & Concept Drift
- Added Section 15: Agent Evaluation & Observability
- Expanded Section 6: Memory Architecture with semantic graph details
- Merged Energy Score vocabularies into Section 12

**Files Archived**:
All 6 legacy vocabulary files moved to `config/archived/`:
- `domain_vocabulary_v3.1.0.yaml` (1,521 lines, v3.1.0) - Superseded by v4.2.0
- `domain_vocabulary_v4.2.0.yaml` (1,103 lines, v4.3.0) - Merged as base of v5.0.0
- `domain_vocabulary_v3_2_additions.yaml` (157 lines, v3.2.0) - Merged into Section 12
- `Feedback Loop domain vocabulary.yml` (519 lines, v3.2.0) - Merged into Section 14
- `Ragas-Opik Integration Domain Vocabulary.yml` (351 lines, v1.0.0) - Merged into Section 15
- `003_memory_vocabulary.yaml` (528 lines, v1.0) - Merged into Section 6

**Duplicate Content Eliminated**:
- `brands` (Remibrutinib, Fabhalta, Kisqali) - Appeared in ALL 6 files → Now in Section 1 only
- `regions` (northeast, south, midwest, west) - Appeared in ALL 6 files → Now in Section 1 only
- `agent_tiers` (6-tier structure) - In 3 files → Now in Section 2 only
- `time_periods` (Q1-Q4, YTD, MTD) - In 2 files → Now in Section 8 only
- `agents` - 18-agent authoritative list in Section 2, outdated 11-agent list removed

---

## Breaking Changes

### File Path Updates Required

**CRITICAL**: 4 code files with hardcoded paths must be updated:

#### 1. src/memory/004_cognitive_workflow.py (lines 184, 224)

**OLD**:
```python
vocab_path = Path(__file__).parent / "003_memory_vocabulary.yaml"
```

**NEW**:
```python
vocab_path = Path(__file__).parent.parent.parent / "config" / "domain_vocabulary.yaml"
```

**Why**: Memory vocabulary merged into Section 6 of consolidated file

---

#### 2. tests/integration/test_gepa_integration.py (line 422)

**OLD**:
```python
vocab_path = config_dir / "domain_vocabulary_v4.2.0.yaml"
```

**NEW**:
```python
vocab_path = config_dir / "domain_vocabulary.yaml"
```

**Why**: Version-specific file replaced with generic name

---

#### 3. scripts/gepa_integration_test.py (lines 437, 444, 466)

**OLD** (3 occurrences):
```python
vocab_path = Path(__file__).parent.parent / "config" / "domain_vocabulary_v4.2.0.yaml"
```

**NEW** (3 occurrences):
```python
vocab_path = Path(__file__).parent.parent / "config" / "domain_vocabulary.yaml"
```

**Why**: Version-specific file replaced with generic name

---

#### 4. scripts/seed_falkordb.py (line 41)

**OLD**:
```python
# Load vocabulary from domain_vocabulary_v4.2.0.yaml
```

**NEW**:
```python
# Load vocabulary from domain_vocabulary.yaml
```

**Why**: Comment update for clarity

---

### No Breaking Changes For

The following will continue to work without modification:

- **Generic references** (52 files): Code using `"domain_vocabulary.yaml"` (non-versioned)
- **Database ENUMs**: All 10 ENUM values preserved (no schema migration needed)
- **Entity extraction patterns**: All entity types preserved in Section 9
- **Agent routing logic**: All 18 agents preserved in Section 2
- **KPI definitions**: All 46 KPIs preserved in Section 7

---

## Database ENUM Validation

### 10 ENUMs Must Match Vocabulary

The consolidated vocabulary preserves all database ENUM values. Use the validation script to verify:

```bash
python scripts/validate_vocabulary_enum_sync.py
```

**ENUMs Validated**:

| ENUM Name | SQL File | Vocabulary Section | Values |
|-----------|----------|-------------------|---------|
| `brand_type` | e2i_ml_complete_v3_schema.sql | Section 1 (brands) | remibrutinib, fabhalta, kisqali |
| `region_type` | e2i_ml_complete_v3_schema.sql | Section 1 (regions) | northeast, south, midwest, west |
| `agent_tier_type` | e2i_ml_complete_v3_schema.sql | Section 2 (agent_tiers) | tier_0 - tier_5 |
| `agent_name_type_v2` | e2i_ml_complete_v3_schema.sql | Section 2 (agents) | 18 agent names |
| `refutation_test_type` | 010_causal_validation_tables.sql | Section 4 (refutation_tests) | placebo_test, etc. |
| `validation_status` | 010_causal_validation_tables.sql | Section 4 (validation_status) | 4 statuses |
| `gate_decision` | 010_causal_validation_tables.sql | Section 4 (gate_decisions) | 4 decisions |
| `expert_review_type` | 010_causal_validation_tables.sql | Section 4 (expert_review_types) | 3 review types |
| `memory_event_type` | (memory schema) | Section 6 (memory_event_types) | user_events, agent_events, system_events |
| `e2i_agent_name` | (agent schema) | Section 2 (agents) | 18 agent names |

**Pre-Deployment Check**:
```bash
# Run validation script
python scripts/validate_vocabulary_enum_sync.py

# Expected output:
# ✅ All ENUMs match vocabulary definitions
```

**If Validation Fails**:
- Check `database/` SQL files for ENUM definition mismatches
- Update vocabulary or database schema to align
- Re-run validation before deployment

---

## Consolidated File Structure (v5.0.0)

### 15 Sections Total (~2,610 lines)

```yaml
# =============================================================================
# SECTION 1: Core Business Entities
# =============================================================================
# brands, regions, specialties, practice_types, data_sources
# ✅ Single source of truth for brands/regions (eliminated duplicates)

# =============================================================================
# SECTION 2: Agent Architecture (18 Agents, 6 Tiers)
# =============================================================================
# agent_tiers, agents (18-agent definitions)
# ✅ Authoritative agent list (outdated 11-agent list excluded)

# =============================================================================
# SECTION 3: Tool Composer & Orchestrator Classifier
# =============================================================================
# query_facets, facet_combinations, orchestrator_decisions

# =============================================================================
# SECTION 4: Causal Validation (Tier 2 Integration)
# =============================================================================
# refutation_tests, validation_status, gate_decisions, expert_review_types

# =============================================================================
# SECTION 5: ML Foundation & MLOps
# =============================================================================
# ml_layers, prediction_types, feature_types, model_types, mlflow_entities

# =============================================================================
# SECTION 6: Memory Architecture (EXPANDED)
# =============================================================================
# memory_types, memory_backends
# ✅ EXPANDED with semantic_graph_entity_types, semantic_graph_relationship_types,
#    memory_event_types, memory_intent_vocabulary, memory_templates

# =============================================================================
# SECTION 7: Visualization & KPIs (46 KPIs)
# =============================================================================
# kpi_categories, workstream_1_kpis, workstream_2_kpis, workstream_3_kpis

# =============================================================================
# SECTION 8: Time References
# =============================================================================
# time_periods, fiscal_quarters
# ✅ Single definition (eliminated duplicates)

# =============================================================================
# SECTION 9: Entity Patterns (for NLP Extraction)
# =============================================================================
# brand_synonyms, region_patterns, kpi_synonyms

# =============================================================================
# SECTION 10: Error Handling
# =============================================================================
# error_types, severity_levels

# =============================================================================
# SECTION 11: DSPy Integration & MIPROv2
# =============================================================================
# dspy_modules, optimizer_types, evaluation_metrics

# =============================================================================
# SECTION 12: Energy Score Enhancement
# =============================================================================
# energy_score_methods, causal_estimator_types
# ✅ Merged from domain_vocabulary_v3_2_additions.yaml

# =============================================================================
# SECTION 13: GEPA Prompt Optimization
# =============================================================================
# gepa_budget_presets, gepa_metrics, gepa_integration

# =============================================================================
# SECTION 14: Feedback Loop & Concept Drift (NEW)
# =============================================================================
# label_lag, observation_window, ground_truth, concept_drift, feature_drift
# outcome_labels (POSITIVE, NEGATIVE, INDETERMINATE, EXCLUDED, PENDING)
# truth_definitions (5 model-specific)
# edge_case_taxonomy (9 categories)
# database_objects (tables, functions, views)
# metrics_glossary (8 metrics)
# agent_integration (drift_monitor, feedback_learner, experiment_monitor, health_score)
# ✅ Added from Feedback Loop domain vocabulary.yml

# =============================================================================
# SECTION 15: Agent Evaluation & Observability (NEW)
# =============================================================================
# fundamental_distinction (THE WHAT vs THE WHY)
# ragas_domain_terms (7 core metrics: faithfulness, answer_relevancy, etc.)
# opik_domain_terms (tracing concepts: Trace, Span, Trace ID, etc.)
# integration_terms (online_evaluation, offline_evaluation, golden_set)
# rag_pipeline_terms (Retriever, Generator, Reranker, Embedder)
# score_interpretation (0.8-1.0 Excellent, 0.7-0.8 Good, etc.)
# ✅ Added from Ragas-Opik Integration Domain Vocabulary.yml
```

---

## Rollback Procedure

If issues arise after consolidation, use the following rollback steps:

### 1. Create Pre-Consolidation Tag (Done Before Consolidation)

```bash
git tag vocabulary-pre-consolidation-v5.0.0
```

### 2. If Rollback Needed

**Option A: Revert the consolidation commit**
```bash
# Find the consolidation commit hash
git log --oneline --grep="consolidate 6 vocabulary files"

# Revert the commit
git revert <consolidation-commit-hash>
```

**Option B: Restore from tag**
```bash
# Checkout config directory from pre-consolidation state
git checkout vocabulary-pre-consolidation-v5.0.0 -- config/
```

### 3. Restore Code References

After rollback, restore hardcoded paths in 4 files:
```bash
git checkout vocabulary-pre-consolidation-v5.0.0 -- src/memory/004_cognitive_workflow.py
git checkout vocabulary-pre-consolidation-v5.0.0 -- tests/integration/test_gepa_integration.py
git checkout vocabulary-pre-consolidation-v5.0.0 -- scripts/gepa_integration_test.py
git checkout vocabulary-pre-consolidation-v5.0.0 -- scripts/seed_falkordb.py
```

### 4. Verify Rollback

```bash
# Verify files restored
ls -la config/

# Should see:
# - domain_vocabulary_v4.2.0.yaml
# - domain_vocabulary_v3_2_additions.yaml
# - Feedback Loop domain vocabulary.yml
# - Ragas-Opik Integration Domain Vocabulary.yml
# - 003_memory_vocabulary.yaml

# Run tests to verify system works
make test
```

---

## Validation Checklist

### Pre-Consolidation ✅
- [x] All 6 files analyzed for unique content
- [x] Overlap matrix completed
- [x] Database ENUMs cross-referenced
- [x] Plan created and approved

### Consolidation ✅
- [x] No data loss (all unique sections preserved)
- [x] YAML validity (`yamllint` passes)
- [x] Schema compliance verified
- [x] No duplicate keys
- [x] All cross-references resolve

### Code Migration (In Progress)
- [ ] 4 hardcoded paths updated
- [ ] All code importing vocabulary runs without error
- [ ] Entity extraction (NLP) still works
- [ ] Agent routing still works
- [ ] Memory graph construction works

### Database
- [ ] 10 ENUMs validated against vocabulary
- [ ] Validation script created and passes
- [ ] No database migration needed (values unchanged)

### Testing
- [ ] Unit tests pass (217 test files)
- [ ] Integration tests pass
- [ ] Memory graph tests pass
- [ ] GEPA integration tests pass

### Documentation
- [x] Migration document created (this file)
- [ ] Archive README created
- [ ] Changelog updated
- [ ] Git tag created

---

## Success Metrics

| Metric | Baseline | Target | Result |
|--------|----------|--------|--------|
| File count | 6 files | 1 file | ✅ 1 active + 6 archived |
| Total lines | 4,179 lines | ≤ 2,800 lines | ✅ ~2,610 lines |
| Duplicate content | ~1,569 lines | 0 lines | ✅ No duplicates |
| Code references updated | 0 of 4 | 4 of 4 | ⏳ In progress |
| Test suite pass rate | 100% | 100% | ⏳ Pending |
| Database ENUM sync | Manual | Automated | ⏳ Script created |

---

## Archive Access

### Archived Files Location

All legacy vocabulary files are preserved in:
```
config/archived/
├── README.md
├── domain_vocabulary_v3.1.0.yaml
├── domain_vocabulary_v4.2.0.yaml
├── domain_vocabulary_v3_2_additions.yaml
├── Feedback Loop domain vocabulary.yml
├── Ragas-Opik Integration Domain Vocabulary.yml
└── 003_memory_vocabulary.yaml
```

### Retrieve Legacy Content

**View archived file**:
```bash
cat config/archived/domain_vocabulary_v4.2.0.yaml
```

**View from git history**:
```bash
git show vocabulary-pre-consolidation-v5.0.0:config/domain_vocabulary_v4.2.0.yaml
```

**Compare versions**:
```bash
# Compare v4.2.0 vs v5.0.0
diff config/archived/domain_vocabulary_v4.2.0.yaml config/domain_vocabulary.yaml
```

---

## FAQ

### Q: Why consolidate into a single file?

**A**: The 6-file structure created significant duplicate content (37.5% of total lines):
- `brands`, `regions`, `agent_tiers`, `time_periods` appeared in ALL 6 files
- Cross-file references were inconsistent
- Agent list had 2 conflicting versions (11 vs 18 agents)
- Maintenance burden increased with each new vocabulary extension

Single file provides:
- ✅ Single source of truth
- ✅ No duplicate content
- ✅ Easier maintenance
- ✅ Consistent cross-references

---

### Q: Will this break existing code?

**A**: Only 4 files with hardcoded paths need updates:
- `src/memory/004_cognitive_workflow.py` (2 lines)
- `tests/integration/test_gepa_integration.py` (1 line)
- `scripts/gepa_integration_test.py` (3 lines)
- `scripts/seed_falkordb.py` (1 comment)

52 other files using generic `"domain_vocabulary.yaml"` will continue to work without changes.

---

### Q: How do I validate database ENUMs match the vocabulary?

**A**: Run the validation script before deployment:
```bash
python scripts/validate_vocabulary_enum_sync.py
```

This checks all 10 database ENUMs against vocabulary definitions and reports any mismatches.

---

### Q: What if I need content from a legacy file?

**A**: All legacy files are preserved in `config/archived/`. You can:
1. View directly: `cat config/archived/<filename>`
2. View from git: `git show vocabulary-pre-consolidation-v5.0.0:config/<filename>`
3. See `config/archived/README.md` for index

---

### Q: Can I rollback the consolidation?

**A**: Yes, see **Rollback Procedure** section above. The pre-consolidation state is tagged as `vocabulary-pre-consolidation-v5.0.0`.

---

### Q: Where is the Feedback Loop vocabulary content now?

**A**: Merged into **Section 14: Feedback Loop & Concept Drift** of the consolidated file.

Key terms:
- `label_lag`, `observation_window`, `ground_truth`, `concept_drift`
- `outcome_labels`: POSITIVE, NEGATIVE, INDETERMINATE, EXCLUDED, PENDING
- `truth_definitions`: 5 model-specific definitions
- `edge_case_taxonomy`: 9 edge case categories
- `database_objects`: Tables, functions, views for feedback loop

---

### Q: Where is the Ragas-Opik vocabulary content now?

**A**: Merged into **Section 15: Agent Evaluation & Observability** of the consolidated file.

Key terms:
- **THE WHAT** (Ragas): quantitative metrics (faithfulness, answer_relevancy, etc.)
- **THE WHY** (Opik): qualitative traces (Trace, Span, Trace ID)
- Integration workflows, score interpretation, RAG pipeline components

---

### Q: Where is the memory graph vocabulary content now?

**A**: Merged into **Section 6: Memory Architecture** of the consolidated file.

Added content:
- `semantic_graph_entity_types`: 16 entity types
- `semantic_graph_relationship_types`: 12 relationship types
- `memory_event_types`: 3 event categories
- `memory_intent_vocabulary`: 5 intent types
- `memory_templates`: 3 template types

---

## Related Documentation

- **Consolidation Plan**: `.claude/plans/vocabulary_consolidation_plan.md`
- **Archive Index**: `config/archived/README.md`
- **Validation Script**: `scripts/validate_vocabulary_enum_sync.py`
- **Consolidated Vocabulary**: `config/domain_vocabulary.yaml` (v5.0.0)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v5.0.0 | 2025-12-28 | Consolidated 6 files into single source of truth |
| v4.3.0 | 2024-12-XX | Added GEPA Prompt Optimization |
| v4.2.2 | 2024-12-XX | Added Energy Score Enhancement |
| v4.2.0 | 2024-12-XX | Added Tool Composer & Orchestrator Classifier |
| v3.2.0 | 2024-XX-XX | Added Feedback Loop & Ragas-Opik extensions |
| v3.1.0 | 2024-11-XX | Base vocabulary with 18-agent architecture |
| v3.0.0 | 2024-XX-XX | Initial V3 schema vocabulary |

---

**Last Updated**: 2025-12-28
**Migration Status**: ✅ Consolidation complete, validation in progress
**Contact**: E2I Team
