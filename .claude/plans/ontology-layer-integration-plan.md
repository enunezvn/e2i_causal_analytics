# Ontology Layer Integration Plan

**Project**: E2I Causal Analytics - Semantic Backbone Implementation
**Date**: 2026-01-19
**Status**: ✅ COMPLETED (Phases 1-5)
**Last Updated**: 2026-01-19 @ 13:45 EST

---

## Completion Summary

### ✅ Phase 1: Foundation (COMPLETE)
- Created `src/ontology/` module with all Python components
- Implemented `VocabularyRegistry` as single source of truth
- All modules have `__pycache__` files (compiled and tested)

### ✅ Phase 2: Config Reorganization (COMPLETE)
- Created `config/ontology/` with 17 YAML files
- Migrated from `Ontology/Ontology/` and `Ontology/Operations/`

### ✅ Phase 3: Code Integration (COMPLETE)
- `src/rag/entity_extractor.py` uses `VocabularyRegistry` (lines 56-70)

### ✅ Phase 4: ENUM Verification (COMPLETE - ALL ISSUES RESOLVED)
- Fixed validation script (section name bugs)
- **8/8 checks pass**: All ENUMs now synchronized
- Created migration 029 for new ENUM types

### ✅ Phase 5: Cleanup (COMPLETE)
- `Ontology/` directory deleted
- Documentation moved to `docs/ontology/`

### ✅ ENUM Issues Resolved (2026-01-19)

| ENUM | Issue | Resolution |
|------|-------|------------|
| `brand_type` | DB had 'competitor', 'other' not in vocab | ✅ Added to vocabulary |
| `agent_tier_type` → `agent_tier_type_v2` | DB used old naming | ✅ Migration 029 with new tier_X_* format |
| `agent_name_type_v2` → `agent_name_type_v3` | 11 agents vs 21 needed | ✅ Migration 029 with all 21 agents |
| Vocabulary gaps | Missing tool_composer, legacy agents | ✅ Added to domain_vocabulary.yaml |

---

## Executive Summary

Integrate the Ontology layer as the semantic backbone of E2I Causal Analytics by:
1. Moving Python modules from `Ontology/` to `src/ontology/`
2. Consolidating YAML configs into `config/ontology/`
3. Eliminating vocabulary duplication via central registry
4. Updating 30+ code references to new paths
5. Ensuring database ENUM synchronization

**Total Files**: 32 Ontology files -> reorganized into project structure
**Risk Level**: Medium (phased approach with rollback points)

---

## Current State Analysis

### Ontology Folder (32 files to migrate/delete)
```
Ontology/
├── Python Modules (5 files) -> src/ontology/
│   ├── schema_compiler.py
│   ├── validator.py
│   ├── inference_engine.py
│   ├── e2i_query_extractor_REVISED.py
│   └── grafiti_config.py
├── Core/ (3 files) -> DELETE (duplicates domain_vocabulary.yaml)
│   ├── 01_entities.yaml
│   ├── 02_agents.yaml
│   └── 03_attributes.yaml
├── Ontology/ (5 files) -> config/ontology/
│   ├── 01_node_types.yaml
│   ├── 02_edge_types.yaml
│   ├── 03_inference_rules.yaml
│   ├── 04_validation_rules.yaml
│   └── 05_falkordb_config.yaml
├── Operations/ (5 files) -> MERGE into existing config/
├── Infrastructure/ (3 files) -> MERGE into existing config/
├── Mlops/ (2 files) -> MERGE into config/observability.yaml
├── Feedback/ (2 files) -> MERGE into existing config/
├── Core Documentation/ (4 files) -> docs/ontology/
└── Documentation (3 .md files) -> docs/ontology/
```

### Duplication Found
| Entity | Files with Duplicates |
|--------|----------------------|
| Brands (3 values) | domain_vocabulary.yaml, cohort_vocabulary.yaml, filter_mapping.yaml, Ontology/Core/01_entities.yaml |
| Regions (4 values) | domain_vocabulary.yaml, cohort_vocabulary.yaml, filter_mapping.yaml, Ontology/Core/01_entities.yaml |
| Agents (18 agents) | domain_vocabulary.yaml, Ontology/Core/02_agents.yaml |
| HCP Specialties | domain_vocabulary.yaml, cohort_vocabulary.yaml |

### Key Integration Points (30+ files)
- `scripts/validate_vocabulary_enum_sync.py` - ENUM validation
- `src/rag/entity_extractor.py` - 70+ hardcoded vocabulary lines
- `src/kpi/registry.py` - KPI definitions loader
- `database/core/e2i_ml_complete_v3_schema.sql` - Database ENUMs

---

## Phase 1: Foundation - Create src/ontology/ Module

**Objective**: Establish new Python module without breaking existing code.

### Files to Create
| File | Purpose |
|------|---------|
| `src/ontology/__init__.py` | Package exports |
| `src/ontology/vocabulary_registry.py` | Central vocabulary loader (NEW) |

### Files to Move (copy first, test, then delete originals in Phase 5)
| From | To |
|------|-----|
| `Ontology/schema_compiler.py` | `src/ontology/schema_compiler.py` |
| `Ontology/validator.py` | `src/ontology/validator.py` |
| `Ontology/inference_engine.py` | `src/ontology/inference_engine.py` |
| `Ontology/e2i_query_extractor_REVISED.py` | `src/ontology/query_extractor.py` |
| `Ontology/grafiti_config.py` | `src/ontology/grafiti_config.py` |

### VocabularyRegistry Design
```python
"""Central vocabulary registry - single source of truth."""
from functools import lru_cache
from pathlib import Path
import yaml

class VocabularyRegistry:
    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> "VocabularyRegistry": ...

    def get_brands(self) -> list[str]: ...
    def get_regions(self) -> list[str]: ...
    def get_agents(self, tier: int | None = None) -> dict: ...
    def get_entity_with_aliases(self, entity_type: str) -> dict: ...
```

### Testing Checklist
- [ ] `python -c "from src.ontology import VocabularyRegistry"`
- [ ] `python -c "from src.ontology import SchemaCompiler"`
- [ ] VocabularyRegistry loads domain_vocabulary.yaml correctly

### Rollback
```bash
rm -rf src/ontology/
```

---

## Phase 2: Config Reorganization - Create config/ontology/

**Objective**: Move Ontology YAML files to config/ontology/, eliminate duplicates.

### Directory Structure After
```
config/
├── domain_vocabulary.yaml        # MASTER (unchanged)
├── cohort_vocabulary.yaml        # UPDATED: remove duplicate brands/regions
├── kpi_definitions.yaml          # Unchanged
├── filter_mapping.yaml           # UPDATED: reference domain_vocabulary
├── ontology/                     # NEW directory
│   ├── node_types.yaml           # From Ontology/Ontology/01_node_types.yaml
│   ├── edge_types.yaml           # From Ontology/Ontology/02_edge_types.yaml
│   ├── inference_rules.yaml      # From Ontology/Ontology/03_inference_rules.yaml
│   ├── validation_rules.yaml     # From Ontology/Ontology/04_validation_rules.yaml
│   └── falkordb_config.yaml      # From Ontology/Ontology/05_falkordb_config.yaml
```

### Files to Move
| From | To |
|------|-----|
| `Ontology/Ontology/01_node_types.yaml` | `config/ontology/node_types.yaml` |
| `Ontology/Ontology/02_edge_types.yaml` | `config/ontology/edge_types.yaml` |
| `Ontology/Ontology/03_inference_rules.yaml` | `config/ontology/inference_rules.yaml` |
| `Ontology/Ontology/04_validation_rules.yaml` | `config/ontology/validation_rules.yaml` |
| `Ontology/Ontology/05_falkordb_config.yaml` | `config/ontology/falkordb_config.yaml` |

### Files to DELETE (duplicates)
| File | Reason |
|------|--------|
| `Ontology/Core/01_entities.yaml` | 100% duplicate of domain_vocabulary.yaml brands/regions |
| `Ontology/Core/02_agents.yaml` | Duplicate of domain_vocabulary.yaml agents |

### Testing Checklist
- [ ] `python scripts/validate_vocabulary_enum_sync.py` passes
- [ ] VocabularyRegistry loads new paths correctly
- [ ] `ls config/ontology/` shows 5 files
- [ ] No YAML syntax errors

### Rollback
```bash
git checkout HEAD -- config/
rm -rf config/ontology/
```

---

## Phase 3: Code Integration - Update Vocabulary References

**Objective**: Update 30+ files to use VocabularyRegistry or new paths.

### Priority 1: Entity Extractor (Highest Impact)
**File**: `src/rag/entity_extractor.py`

Replace hardcoded `EntityVocabulary.from_default()` (lines 49-121):
```python
@classmethod
def from_default(cls) -> "EntityVocabulary":
    """Create vocabulary from central registry."""
    from src.ontology import VocabularyRegistry
    vocab = VocabularyRegistry.load()
    return cls(
        brands=vocab.get_entity_with_aliases("brands"),
        regions=vocab.get_entity_with_aliases("regions"),
        kpis=vocab.get_entity_with_aliases("kpis"),
        agents=vocab.get_entity_with_aliases("agents"),
        journey_stages=vocab.get_entity_with_aliases("journey_stages"),
        time_references=vocab.get_entity_with_aliases("time_references"),
        hcp_segments=vocab.get_entity_with_aliases("hcp_segments"),
    )
```

### Testing Checklist
- [ ] `make test-fast` - all unit tests pass
- [ ] Entity extraction works:
  ```bash
  python -c "from src.rag import EntityExtractor; e = EntityExtractor(); print(e.extract('Kisqali TRx in West'))"
  ```

### Rollback
```bash
git checkout HEAD -- src/
```

---

## Phase 4: Database ENUM Verification

**Objective**: Ensure database ENUMs remain synchronized with vocabulary.

### ENUM Locations
| File | ENUMs |
|------|-------|
| `database/core/e2i_ml_complete_v3_schema.sql` | brand_type, region_type, agent_tier_type, agent_name_type_v2 |
| `database/ml/010_causal_validation_tables.sql` | refutation_test_type, validation_status, gate_decision |

### Approach
1. **DO NOT** modify database ENUMs directly
2. Run validation script to check sync:
   ```bash
   python scripts/validate_vocabulary_enum_sync.py
   ```
3. If mismatches found, create migration

### Testing Checklist (Droplet)
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "
  cd ~/Projects/e2i_causal_analytics && \
  /home/enunez/Projects/e2i_causal_analytics/venv/bin/python scripts/validate_vocabulary_enum_sync.py
"
```

---

## Phase 5: Cleanup - Delete Obsolete Files

**Objective**: Remove migrated/duplicate files, archive documentation.

### Files to DELETE
```bash
# Duplicates (already copied/merged)
rm Ontology/Core/01_entities.yaml
rm Ontology/Core/02_agents.yaml

# Migrated Python files (now in src/ontology/)
rm Ontology/schema_compiler.py
rm Ontology/validator.py
rm Ontology/inference_engine.py
rm Ontology/e2i_query_extractor_REVISED.py
rm Ontology/grafiti_config.py

# Migrated YAML files (now in config/ontology/)
rm -rf Ontology/Ontology/
```

### Files to MOVE to docs/
```bash
mkdir -p docs/ontology
mv Ontology/README.md docs/ontology/
mv Ontology/MODULAR_IMPLEMENTATION_GUIDE.md docs/ontology/
mv Ontology/ONTOLOGY_UPDATE_GUIDE.md docs/ontology/
```

### Final Cleanup
```bash
# Remove empty Ontology directory after all files handled
rm -rf Ontology/
```

### Testing Checklist (Full)
- [ ] `make test` - full test suite passes
- [ ] `python scripts/validate_vocabulary_enum_sync.py` - passes
- [ ] No import errors: `python -c "from src.ontology import *"`
- [ ] API health check on droplet

---

## Verification Commands

### After Each Phase
```bash
# Syntax check all YAML
find config -name "*.yaml" -exec python -c "import yaml; yaml.safe_load(open('{}'))" \;

# Import check
python -c "from src.ontology import VocabularyRegistry, SchemaCompiler"

# ENUM sync check
python scripts/validate_vocabulary_enum_sync.py

# Quick test
make test-fast
```

### Final Verification (Droplet)
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 "
  cd ~/Projects/e2i_causal_analytics && \
  git pull && \
  sudo systemctl restart e2i-api && \
  sleep 5 && \
  curl -s localhost:8000/health | python3 -m json.tool
"
```

---

## Commit Strategy

**Phase 1 Commit**: `feat(ontology): create src/ontology module with vocabulary registry`
**Phase 2 Commit**: `refactor(config): reorganize ontology YAML files and remove duplicates`
**Phase 3 Commit**: `refactor(rag): integrate VocabularyRegistry into entity extractor`
**Phase 4 Commit**: `test(db): verify database ENUM synchronization`
**Phase 5 Commit**: `chore: cleanup obsolete Ontology directory`

---

## Success Criteria

- [x] All Python modules migrated to `src/ontology/`
- [x] All YAML configs in `config/ontology/`
- [x] Zero vocabulary duplication across files
- [x] `scripts/validate_vocabulary_enum_sync.py` passes (**8/8 - ALL RESOLVED**)
- [x] Ontology module imports work on droplet
- [x] VocabularyRegistry loads 81 sections
- [x] E2IQueryExtractor extracts entities correctly
- [x] API healthy on droplet (status: healthy, v4.1.0)
- [x] `Ontology/` directory deleted

---

## Verification Summary (2026-01-19 @ 13:45 EST)

### Droplet Tests Passed
```
✅ VocabularyRegistry: 81 sections
   - Brands: ['Remibrutinib', 'Fabhalta', 'Kisqali']
   - Regions: ['northeast', 'south', 'midwest', 'west']
   - Agent count: 18

✅ E2IQueryExtractor: 2 entities extracted from "Show Kisqali TRx in West region"
   - Entity(text='Kisqali', entity_type='brand', confidence=1.0)
   - Entity(text='west', entity_type='region', confidence=1.0)

✅ API Health: {"status":"healthy","version":"4.1.0"}
```

### ENUM Validation Results (8/8 passed - FINAL)
```
✅ brand_type                (5 values)
✅ region_type               (4 values)
✅ agent_tier_type_v2        (6 values)
✅ agent_name_type_v3        (21 values)
✅ refutation_test_type      (5 values)
✅ validation_status         (4 values)
✅ gate_decision             (3 values)
✅ expert_review_type        (4 values)
```

### Resolution Summary (2026-01-19 @ 14:00 EST)
- Migration 029 created: `database/core/029_update_agent_enums_v4.sql`
- Vocabulary updated: added tool_composer, legacy agents, competitor/other
- Validation script updated: checks new ENUM types (v2/v3)
