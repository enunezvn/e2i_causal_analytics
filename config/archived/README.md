# Archived Vocabulary Files

These files have been consolidated into `../domain_vocabulary.yaml` (v5.0.0) as part of the vocabulary consolidation effort completed on 2025-12-28.

## Archive Date

**2025-12-28**

## Reason for Archival

Consolidated 6 vocabulary files into a single source of truth to eliminate duplicate content, achieving **37.5% reduction** (from 4,179 to ~2,610 lines).

### Duplicates Eliminated

- **brands** (Remibrutinib, Fabhalta, Kisqali) - Appeared in ALL 6 files → Now in Section 1 only
- **regions** (northeast, south, midwest, west) - Appeared in ALL 6 files → Now in Section 1 only
- **agent_tiers** (6-tier structure) - In 3 files → Now in Section 2 only
- **time_periods** (Q1-Q4, YTD, MTD) - In 2 files → Now in Section 8 only
- **agents** - Outdated 11-agent list removed; authoritative 18-agent list in Section 2

---

## Files Archived

| File | Version | Lines | Reason |
|------|---------|-------|--------|
| domain_vocabulary_v3.1.0.yaml | v3.1.0 | 1,521 | Superseded by v4.2.0 |
| domain_vocabulary_v4.2.0.yaml | v4.3.0 | 1,103 | Merged as base of v5.0.0 (Sections 1-13) |
| domain_vocabulary_v3_2_additions.yaml | v3.2.0 | 157 | Merged into Section 12 (Energy Score Enhancement) |
| Feedback Loop domain vocabulary.yml | v3.2.0 | 519 | Merged into Section 14 (Feedback Loop & Concept Drift) |
| Ragas-Opik Integration Domain Vocabulary.yml | v1.0.0 | 351 | Merged into Section 15 (Agent Evaluation & Observability) |
| 003_memory_vocabulary.yaml | v1.0 | 528 | Merged into Section 6 (Memory Architecture expansion) |
| domain_vocabulary_v5.0.0.yaml | v5.0.0 | 2,334 | Versioned file (use ../domain_vocabulary.yaml instead) |
| **TOTAL** | | **4,179** | **Consolidated → 2,610 lines** |

---

## Consolidated File Structure (v5.0.0)

The new unified `../domain_vocabulary.yaml` contains **15 sections**:

1. **Core Business Entities** - brands, regions, specialties, segments
2. **Agent Architecture** - 18 agents in 6 tiers
3. **Tool Composer & Orchestrator Classifier** - query facets, orchestrator decisions
4. **Causal Validation** - refutation tests, validation gates
5. **ML Foundation & MLOps** - prediction types, feature types, MLflow entities
6. **Memory Architecture** ✅ EXPANDED - semantic graph entities/relationships, event types, intents
7. **Visualization & KPIs** - 46 KPIs across 3 workstreams
8. **Time References** - time periods, fiscal quarters
9. **Entity Patterns** - NLP extraction patterns
10. **Error Handling** - error types, severity levels
11. **DSPy Integration & MIPROv2** - optimization modules
12. **Energy Score Enhancement** ✅ MERGED - causal estimator selection
13. **GEPA Prompt Optimization** - budget presets, metrics
14. **Feedback Loop & Concept Drift** ✅ NEW - ground truth labeling, drift detection
15. **Agent Evaluation & Observability** ✅ NEW - Ragas metrics, Opik tracing

---

## Migration Documentation

For complete migration details, see:
- **Migration Guide**: `../../docs/vocabulary_migration_v3.1_to_v5.0.md`
- **Consolidation Plan**: `../../.claude/plans/vocabulary_consolidation_plan.md`

---

## Retrieval

### View Archived File

```bash
# From project root
cat config/archived/domain_vocabulary_v4.2.0.yaml
```

### View from Git History

```bash
# Before consolidation
git show vocabulary-pre-consolidation-v5.0.0:config/domain_vocabulary_v4.2.0.yaml

# Specific version tag
git show vocabulary-v4.2.0:config/domain_vocabulary_v4.2.0.yaml
```

### Compare Versions

```bash
# Compare v4.2.0 vs v5.0.0
diff config/archived/domain_vocabulary_v4.2.0.yaml config/domain_vocabulary.yaml

# Show only section headers
diff config/archived/domain_vocabulary_v4.2.0.yaml config/domain_vocabulary.yaml | grep "^[<>].*#.*SECTION"
```

---

## Breaking Changes

### Code Path Updates (Completed)

4 files were updated with new vocabulary paths:

1. **src/memory/004_cognitive_workflow.py** (lines 184, 224)
   - OLD: `Path(__file__).parent / "003_memory_vocabulary.yaml"`
   - NEW: `Path(__file__).parent.parent.parent / "config" / "domain_vocabulary.yaml"`

2. **tests/integration/test_gepa_integration.py** (line 422)
   - OLD: `"domain_vocabulary_v4.2.0.yaml"`
   - NEW: `"domain_vocabulary.yaml"`

3. **scripts/gepa_integration_test.py** (lines 437, 444, 466)
   - OLD: `"domain_vocabulary_v4.2.0.yaml"` (3 occurrences)
   - NEW: `"domain_vocabulary.yaml"` (3 occurrences)

4. **scripts/seed_falkordb.py** (line 41)
   - OLD: `# DOMAIN DATA - From domain_vocabulary_v4.2.0.yaml`
   - NEW: `# DOMAIN DATA - From domain_vocabulary.yaml`

### No Breaking Changes For

- **Generic references** (52 files): Code using `"domain_vocabulary.yaml"` (non-versioned)
- **Database ENUMs**: All ENUM values preserved (minor differences documented)
- **Entity extraction**: All entity types preserved
- **Agent routing**: All 18 agents preserved

---

## Validation

### Database ENUM Validation

Run the validation script to check database/vocabulary synchronization:

```bash
python scripts/validate_vocabulary_enum_sync.py
```

**Known Differences** (documented, not errors):
- `brand_type`: Database has additional values "competitor", "other" beyond core brands
- `agent_tier_type`: Legacy tier names in database vs new v5.0.0 tier naming
- Agent name mappings: Vocabulary structure differs from database flat list

### Test Suite

Validate code changes work correctly:

```bash
# Memory-safe test execution (max 4 workers)
make test

# Specific areas
pytest tests/unit/test_rag/ -v         # Entity extraction
pytest tests/unit/test_nlp/ -v         # Query parsing
pytest tests/integration/test_memory/ -v  # Memory graph
pytest tests/integration/test_gepa_integration.py -v  # GEPA integration
```

---

## Rollback Procedure

If issues arise after consolidation:

### 1. Revert Consolidation Commit

```bash
# Find consolidation commit
git log --oneline --grep="consolidate 6 vocabulary files"

# Revert
git revert <commit-hash>
```

### 2. Restore from Pre-Consolidation Tag

```bash
git checkout vocabulary-pre-consolidation-v5.0.0 -- config/
```

### 3. Restore Code References

```bash
git checkout vocabulary-pre-consolidation-v5.0.0 -- \
  src/memory/004_cognitive_workflow.py \
  tests/integration/test_gepa_integration.py \
  scripts/gepa_integration_test.py \
  scripts/seed_falkordb.py
```

### 4. Verify Rollback

```bash
# Check files restored
ls -la config/

# Run tests
make test
```

---

## Version History

| Version | Date | Key Changes |
|---------|------|------------|
| **v5.0.0** | 2025-12-28 | **Consolidated 6 files → 1 file** |
| | | - 37.5% reduction (4,179 → 2,610 lines) |
| | | - Added Section 14: Feedback Loop & Concept Drift |
| | | - Added Section 15: Agent Evaluation & Observability |
| | | - Expanded Section 6: Memory Architecture |
| **v4.3.0** | 2024-12-XX | Added GEPA Prompt Optimization (Section 13) |
| **v4.2.2** | 2024-12-XX | Added Energy Score Enhancement (Section 12) |
| **v4.2.0** | 2024-12-XX | Added Tool Composer & Orchestrator Classifier (Section 3) |
| **v3.2.0** | 2024-XX-XX | Added Feedback Loop & Ragas-Opik extensions |
| **v3.1.0** | 2024-11-XX | Base vocabulary with 18-agent architecture |

---

## Contact

For questions about the consolidation or archived files:
- **E2I Team**
- **Migration Documentation**: `docs/vocabulary_migration_v3.1_to_v5.0.md`
- **Consolidation Plan**: `.claude/plans/vocabulary_consolidation_plan.md`

---

**Last Updated**: 2025-12-28
**Archive Status**: ✅ Complete
**Active File**: `../domain_vocabulary.yaml` (v5.0.0)
