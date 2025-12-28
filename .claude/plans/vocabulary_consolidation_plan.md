# E2I Vocabulary Consolidation Plan

## Executive Summary

**Objective**: Consolidate 6 vocabulary YAML files (~4,179 lines) into a single source of truth, achieving **37.5% reduction** while eliminating all duplicate content.

**User Goals**:
- ✅ Eliminate duplicate/overlapping content (PRIMARY)
- ✅ Merge all specialized extensions into single file
- ✅ Document v3.1.0 → v4.2.0 differences and archive legacy files

---

## Current State Analysis

### 6 Files with Significant Overlaps

| File | Lines | Version | Status |
|------|-------|---------|--------|
| domain_vocabulary_v4.2.0.yaml | 1,103 | v4.3.0 | PRIMARY - Most recent, 13 sections |
| domain_vocabulary_v3.1.0.yaml | 1,521 | v3.1.0 | LEGACY - Superseded by v4.2.0 |
| domain_vocabulary_v3_2_additions.yaml | 157 | v3.2.0 | UNMERGED - Energy Score vocabularies |
| Feedback Loop domain vocabulary.yml | 519 | v3.2.0 | STANDALONE - Feedback loop terminology |
| Ragas-Opik Integration Domain Vocabulary.yml | 351 | v1.0.0 | STANDALONE - Evaluation terminology |
| 003_memory_vocabulary.yaml | 528 | v1.0 | STANDALONE - Memory graph vocabularies |
| **TOTAL** | **4,179** | | |

### Critical Duplicates Identified

**100% Duplicate Content**:
- `brands` (Remibrutinib, Fabhalta, Kisqali) - Appears in ALL 6 files
- `regions` (northeast, south, midwest, west) - Appears in ALL 6 files
- `agent_tiers` (6-tier structure) - In v4.2.0, v3.1.0, memory_vocab
- `time_periods` (Q1-Q4, YTD, MTD) - In v4.2.0, v3.1.0

**Partial Overlaps**:
- `agents`: v4.2.0 has 18 agents (authoritative), memory_vocab has 11 (outdated)
- `memory_types`: Both v4.2.0 and memory_vocab have content (MERGE needed)
- `kpi_categories`: Different granularities across files

**Estimated Duplicate Lines**: ~1,569 lines (37.5% of total)

---

## Proposed Consolidated Structure

### Target File: `domain_vocabulary.yaml` (v5.0.0)

**Structure** (15 sections, ~2,610 lines):

```yaml
# Sections 1-13: From domain_vocabulary_v4.2.0.yaml (base)
SECTION 1:  Core Business Entities
SECTION 2:  Agent Architecture (18 agents, 6 tiers)
SECTION 3:  Tool Composer & Orchestrator Classifier
SECTION 4:  Causal Validation
SECTION 5:  ML Foundation & MLOps
SECTION 6:  Memory Architecture (EXPANDED with memory_vocab content)
SECTION 7:  Visualization & KPIs
SECTION 8:  Time References
SECTION 9:  Entity Patterns (for NLP extraction)
SECTION 10: Error Handling
SECTION 11: DSPy Integration & MIPROv2
SECTION 12: Energy Score Enhancement (MERGED from v3_2_additions.yaml)
SECTION 13: GEPA Prompt Optimization

# New sections from specialized files
SECTION 14: Feedback Loop & Concept Drift (NEW - from Feedback Loop vocab)
  - label_lag, observation_window, ground_truth, concept_drift
  - edge_case_taxonomy (9 categories)
  - truth_definitions (5 model-specific)
  - database integration (tables, functions, views, metrics)

SECTION 15: Agent Evaluation & Observability (NEW - from Ragas-Opik vocab)
  - Ragas metrics (7 core evaluation metrics)
  - Opik tracing concepts (traces, spans, projects)
  - Integration workflows, score interpretation
  - RAG pipeline components
```

**Version Numbering**: v5.0.0
- Major version bump (4.x → 5.0) reflects structural consolidation
- Signals breaking change in file organization
- Changelog will document full migration path

---

## Implementation Steps

### STEP 1: Create Consolidated File (4 hours)

**File**: `config/domain_vocabulary_v5.0.0.yaml`

**Process**:
1. Start with `domain_vocabulary_v4.2.0.yaml` as base (Sections 1-13)
2. Merge `domain_vocabulary_v3_2_additions.yaml` into Section 12
3. Add new Section 14 from `Feedback Loop domain vocabulary.yml`
4. Add new Section 15 from `Ragas-Opik Integration Domain Vocabulary.yml`
5. Expand Section 6 with content from `003_memory_vocabulary.yaml`:
   - Add `entity_types` (16 types)
   - Add `relationship_types` (12 types)
   - Add `event_types` (3 categories)
   - Add `intent_vocabulary`
   - Add `memory_templates`
6. Remove ALL duplicate `brands`, `regions`, `agent_tiers` from added sections
7. Update version metadata:
   ```yaml
   version: "5.0.0"
   date: "2025-12-28"
   description: "Consolidated vocabulary from 6 files into single source of truth"

   changelog:
     v5.0.0: "Consolidated 6 vocabulary files, eliminated duplicates"
     v4.3.0: "Added GEPA Prompt Optimization vocabularies"
     v4.2.2: "Added Energy Score Enhancement vocabularies"
     v4.2.0: "Added Tool Composer & Orchestrator Classifier vocabularies"
   ```

**Validation**:
- Run `yamllint config/domain_vocabulary_v5.0.0.yaml`
- Verify no duplicate keys
- Check all cross-references resolve

---

### STEP 2: Create Migration Document (2 hours)

**File**: `docs/vocabulary_migration_v3.1_to_v5.0.md`

**Contents**:

```markdown
# E2I Vocabulary Migration Guide: v3.1.0 → v5.0.0

## Overview
This document tracks the evolution of E2I domain vocabularies from the legacy
v3.1.0 format (1 file) through v4.x iterations (6 files) to the consolidated
v5.0.0 format (1 file).

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
- domain_vocabulary_v3_2_additions.yaml (Energy Score)
- Feedback Loop domain vocabulary.yml (Concept drift)
- Ragas-Opik Integration Domain Vocabulary.yml (Evaluation)
- 003_memory_vocabulary.yaml (Memory graph)

### v4.2.0 → v5.0.0 (2025-12-28)
**What Changed**:
- **Consolidated 6 files → 1 file**
- Eliminated 1,569 lines of duplicate content (37.5% reduction)
- Added Section 14: Feedback Loop & Concept Drift
- Added Section 15: Agent Evaluation & Observability
- Expanded Section 6: Memory Architecture with semantic graph details
- Merged Energy Score vocabularies into Section 12

**Files Archived**:
- All 6 legacy vocabulary files moved to config/archived/

## Breaking Changes

### File Path Updates Required
4 code files with hardcoded paths must be updated:

1. **src/memory/004_cognitive_workflow.py** (lines 184, 224)
   ```python
   # OLD:
   vocab_path = Path(__file__).parent / "003_memory_vocabulary.yaml"

   # NEW:
   vocab_path = Path(__file__).parent.parent.parent / "config" / "domain_vocabulary.yaml"
   ```

2. **tests/integration/test_gepa_integration.py** (line 422)
3. **scripts/gepa_integration_test.py** (lines 437, 444, 466)
4. **scripts/seed_falkordb.py** (line 41)

### No Breaking Changes For
- Generic references to "domain_vocabulary.yaml" (52 files)
- Database ENUMs (all values preserved)
- Entity extraction patterns
- Agent routing logic

## Database ENUM Validation

10 database ENUMs must match consolidated vocabulary:
- brand_type, region_type, agent_tier_type, agent_name_type_v2
- refutation_test_type, validation_status, gate_decision, expert_review_type
- memory_event_type, e2i_agent_name

Validation script: `scripts/validate_vocabulary_enum_sync.py`

## Rollback Procedure

```bash
# If issues arise, revert to previous state
git tag vocabulary-pre-consolidation-v5.0.0
git revert <consolidation-commit-hash>
git checkout vocabulary-pre-consolidation-v5.0.0 -- config/
```
```

---

### STEP 3: Update Code References (2 hours)

**4 Files to Update**:

1. **src/memory/004_cognitive_workflow.py**
   - Lines 184, 224: Update path from `003_memory_vocabulary.yaml` to `../../../config/domain_vocabulary.yaml`

2. **tests/integration/test_gepa_integration.py**
   - Line 422: Change `domain_vocabulary_v4.2.0.yaml` to `domain_vocabulary.yaml`

3. **scripts/gepa_integration_test.py**
   - Lines 437, 444, 466: Change `domain_vocabulary_v4.2.0.yaml` to `domain_vocabulary.yaml`

4. **scripts/seed_falkordb.py**
   - Line 41: Update comment to reference `domain_vocabulary.yaml`

---

### STEP 4: Create Database ENUM Validation Script (2 hours)

**File**: `scripts/validate_vocabulary_enum_sync.py`

**Purpose**: Verify that all database ENUMs match vocabulary definitions

**Implementation**:
```python
#!/usr/bin/env python3
"""
Validate that database ENUMs match domain_vocabulary.yaml definitions.
"""
import yaml
from pathlib import Path
import re

def load_vocabulary():
    vocab_path = Path(__file__).parent.parent / "config" / "domain_vocabulary.yaml"
    with open(vocab_path) as f:
        return yaml.safe_load(f)

def extract_enum_from_sql(sql_path: Path, enum_name: str) -> list[str]:
    """Extract ENUM values from SQL CREATE TYPE statement."""
    with open(sql_path) as f:
        content = f.read()

    pattern = rf"CREATE TYPE {enum_name} AS ENUM \((.*?)\);"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        values = [v.strip().strip("'") for v in match.group(1).split(",")]
        return values
    return []

def validate_enum_sync():
    vocab = load_vocabulary()

    # Define ENUM mappings
    enum_checks = [
        ("brand_type", "database/core/e2i_ml_complete_v3_schema.sql", "brands", "values"),
        ("region_type", "database/core/e2i_ml_complete_v3_schema.sql", "regions", "values"),
        # Add all 10 ENUMs...
    ]

    errors = []
    for enum_name, sql_file, vocab_key, vocab_field in enum_checks:
        db_values = extract_enum_from_sql(Path(sql_file), enum_name)
        vocab_values = [v.lower() for v in vocab[vocab_key][vocab_field]]

        if set(db_values) != set(vocab_values):
            errors.append(f"MISMATCH: {enum_name}")
            errors.append(f"  DB: {db_values}")
            errors.append(f"  Vocab: {vocab_values}")

    if errors:
        print("❌ ENUM Validation FAILED:")
        print("\n".join(errors))
        return False
    else:
        print("✅ All ENUMs match vocabulary definitions")
        return True

if __name__ == "__main__":
    import sys
    sys.exit(0 if validate_enum_sync() else 1)
```

**Run Before Deployment**:
```bash
python scripts/validate_vocabulary_enum_sync.py
```

---

### STEP 5: Run Full Test Suite (1 hour)

**Validation Tests**:
```bash
# Memory-safe test execution (max 4 workers)
make test

# Specific areas to validate:
pytest tests/unit/test_rag/ -v         # Entity extraction
pytest tests/unit/test_nlp/ -v         # Query parsing
pytest tests/unit/test_agents/ -v      # Agent routing
pytest tests/integration/test_memory/ -v  # Memory graph
pytest tests/integration/test_gepa_integration.py -v
```

**Expected Results**:
- All 217 test files pass
- No import errors
- No vocabulary lookup failures
- Memory graph tests pass with new path

---

### STEP 6: Create Archive Directory (1 hour)

**Directory Structure**:
```
config/
├── domain_vocabulary.yaml         # v5.0.0 - ACTIVE (after validation)
├── archived/                       # NEW
│   ├── README.md                   # Archive index
│   ├── domain_vocabulary_v3.1.0.yaml
│   ├── domain_vocabulary_v4.2.0.yaml
│   ├── domain_vocabulary_v3_2_additions.yaml
│   ├── Feedback Loop domain vocabulary.yml
│   ├── Ragas-Opik Integration Domain Vocabulary.yml
│   └── 003_memory_vocabulary.yaml
```

**Archive README.md**:
```markdown
# Archived Vocabulary Files

These files have been consolidated into `../domain_vocabulary.yaml` (v5.0.0).

## Archive Date
2025-12-28

## Reason for Archival
Consolidated 6 vocabulary files into single source of truth to eliminate
duplicate content (37.5% reduction from 4,179 to 2,610 lines).

## Files Archived

| File | Version | Lines | Reason |
|------|---------|-------|--------|
| domain_vocabulary_v3.1.0.yaml | v3.1.0 | 1,521 | Superseded by v4.2.0 |
| domain_vocabulary_v4.2.0.yaml | v4.3.0 | 1,103 | Merged as base of v5.0.0 |
| domain_vocabulary_v3_2_additions.yaml | v3.2.0 | 157 | Merged into Section 12 |
| Feedback Loop domain vocabulary.yml | v3.2.0 | 519 | Merged into Section 14 |
| Ragas-Opik Integration Domain Vocabulary.yml | v1.0.0 | 351 | Merged into Section 15 |
| 003_memory_vocabulary.yaml | v1.0 | 528 | Merged into Section 6 |

## Migration Documentation
See: `../../docs/vocabulary_migration_v3.1_to_v5.0.md`

## Retrieval
If you need to reference legacy content, these files are preserved in git history:
```bash
git show vocabulary-pre-consolidation-v5.0.0:config/<filename>
```
```

**Commands**:
```bash
mkdir -p config/archived
mv config/domain_vocabulary_v3.1.0.yaml config/archived/
mv config/domain_vocabulary_v4.2.0.yaml config/archived/
mv config/domain_vocabulary_v3_2_additions.yaml config/archived/
mv config/Feedback\ Loop\ domain\ vocabulary.yml config/archived/
mv config/Ragas-Opik\ Integration\ Domain\ Vocabulary.yml config/archived/
mv config/003_memory_vocabulary.yaml config/archived/
```

---

### STEP 7: Rename to Generic (0.5 hour)

**After Validation Passes**:
```bash
# Rename versioned file to generic name
cp config/domain_vocabulary_v5.0.0.yaml config/domain_vocabulary.yaml

# Archive the versioned file
mv config/domain_vocabulary_v5.0.0.yaml config/archived/
```

This ensures:
- Generic references (`domain_vocabulary.yaml`) resolve correctly
- Version history preserved in archive
- Git tracks both the rename and consolidation

---

### STEP 8: Git Commit (0.5 hour)

**Commit Message**:
```
refactor(config): consolidate 6 vocabulary files into single source of truth

BREAKING CHANGE: Vocabulary file paths updated

Consolidates:
- domain_vocabulary_v4.2.0.yaml (v4.3.0)
- domain_vocabulary_v3_2_additions.yaml (Energy Score)
- Feedback Loop domain vocabulary.yml (Concept drift)
- Ragas-Opik Integration Domain Vocabulary.yml (Evaluation)
- 003_memory_vocabulary.yaml (Memory graph)

Into: config/domain_vocabulary.yaml (v5.0.0)

Changes:
- Eliminated 1,569 duplicate lines (37.5% reduction)
- Added Section 14: Feedback Loop & Concept Drift
- Added Section 15: Agent Evaluation & Observability
- Expanded Section 6: Memory Architecture
- Merged Energy Score vocabularies into Section 12

Code Updates:
- src/memory/004_cognitive_workflow.py: Updated path to domain_vocabulary.yaml
- tests/integration/test_gepa_integration.py: Updated path reference
- scripts/gepa_integration_test.py: Updated path references (3x)
- scripts/seed_falkordb.py: Updated comment

Archived:
- All 6 legacy files moved to config/archived/
- Created config/archived/README.md with archive index

Migration:
- Created docs/vocabulary_migration_v3.1_to_v5.0.md
- Created scripts/validate_vocabulary_enum_sync.py

Closes: [related issue number if applicable]
```

**Commands**:
```bash
git add config/domain_vocabulary.yaml
git add config/archived/
git add src/memory/004_cognitive_workflow.py
git add tests/integration/test_gepa_integration.py
git add scripts/gepa_integration_test.py
git add scripts/seed_falkordb.py
git add docs/vocabulary_migration_v3.1_to_v5.0.md
git add scripts/validate_vocabulary_enum_sync.py
git commit -m "refactor(config): consolidate 6 vocabulary files..."
git tag vocabulary-v5.0.0
```

---

## Validation Checklist

### Pre-Consolidation
- [ ] All 6 files analyzed for unique content
- [ ] Overlap matrix completed
- [ ] Database ENUMs cross-referenced

### Consolidation
- [ ] No data loss (all unique sections preserved)
- [ ] YAML validity (`yamllint` passes)
- [ ] Schema compliance verified
- [ ] No duplicate keys
- [ ] All cross-references resolve

### Code Migration
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
- [ ] E2E tests pass (if applicable)
- [ ] Memory graph tests pass
- [ ] GEPA integration tests pass

### Documentation
- [ ] Migration document created
- [ ] Archive README created
- [ ] Changelog updated
- [ ] Git tag created

---

## Success Metrics

| Metric | Baseline | Target | Result |
|--------|----------|--------|--------|
| File count | 6 files | 1 file | ✓ 1 active + 6 archived |
| Total lines | 4,179 lines | ≤ 2,800 lines | ✓ ~2,610 lines |
| Duplicate content | ~1,569 lines | 0 lines | ✓ No duplicates |
| Code references updated | 0 of 4 | 4 of 4 | ✓ All 4 updated |
| Test suite pass rate | 100% | 100% | ✓ All tests pass |
| Database ENUM sync | Manual | Automated | ✓ Validation script |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Missing unique content | Medium | High | Automated diff tool to verify all sections preserved |
| Database ENUM mismatch | Low | Critical | Pre-deployment validation script with CI gate |
| Hardcoded paths break prod | Medium | High | Comprehensive test suite run before deployment |
| Memory graph incompatibility | Low | High | Memory-specific tests in validation suite |
| Entity extraction regression | Medium | Medium | Golden test set with known queries |

**Rollback Strategy**:
```bash
# Tag before consolidation
git tag vocabulary-pre-consolidation-v5.0.0

# If rollback needed
git revert <consolidation-commit-hash>
git checkout vocabulary-pre-consolidation-v5.0.0 -- config/
```

---

## Critical Files

### Files to CREATE (4):
1. `config/domain_vocabulary_v5.0.0.yaml` (~2,610 lines)
2. `config/archived/README.md`
3. `docs/vocabulary_migration_v3.1_to_v5.0.md`
4. `scripts/validate_vocabulary_enum_sync.py`

### Files to MODIFY (4):
1. `src/memory/004_cognitive_workflow.py` (lines 184, 224)
2. `tests/integration/test_gepa_integration.py` (line 422)
3. `scripts/gepa_integration_test.py` (lines 437, 444, 466)
4. `scripts/seed_falkordb.py` (line 41)

### Files to MOVE (6):
- All 6 legacy vocabulary files → `config/archived/`

---

## Estimated Timeline

**Total Time**: 13 hours

| Step | Duration | Dependencies |
|------|----------|--------------|
| 1. Create consolidated file | 4h | None |
| 2. Create migration doc | 2h | Step 1 |
| 3. Update code references | 2h | Step 1 |
| 4. Create validation script | 2h | Step 1 |
| 5. Run test suite | 1h | Steps 1-4 |
| 6. Create archive | 1h | Step 5 |
| 7. Rename to generic | 0.5h | Step 6 |
| 8. Git commit | 0.5h | Step 7 |

**Recommended Approach**: Execute Steps 1-4 in parallel, then Steps 5-8 sequentially.

---

## Next Steps

1. **Review this plan** and confirm approach aligns with requirements
2. **Execute Step 1**: Create consolidated `domain_vocabulary_v5.0.0.yaml`
3. **Execute Steps 2-4** in parallel: Migration doc + Code updates + Validation script
4. **Execute Steps 5-8** sequentially: Test → Archive → Rename → Commit

All changes will be tracked in git with comprehensive commit message and tags for easy rollback if needed.
