# CohortConstructor Archive

This folder contains **archived reference materials** from the initial CohortConstructor planning and prototyping phase.

## Production Implementation

The production CohortConstructor agent is now implemented at:
- **Agent Code**: `src/agents/cohort_constructor/`
- **Tests**: `tests/unit/test_agents/test_cohort_constructor/`
- **Database Migration**: `database/ml/028_cohort_constructor_tables.sql`

## Reorganized Files

Files from this folder have been moved to their proper locations:

| Original File | New Location |
|--------------|--------------|
| `cohort-constructor-specialist.md` | `.claude/specialists/Agent_Specialists_Tier 0/cohort-constructor.md` |
| `cohort-constructor-contract.md` | `.claude/contracts/Tier-Specific Contracts/` |
| `cohort-constructor-data-contract.md` | `.claude/contracts/Tier-Specific Contracts/` |
| `cohort-constructor-handoff.yaml` | `.claude/contracts/Tier-Specific Contracts/` |
| `README_CohortConstructor.md` | `docs/agents/cohort_constructor/README.md` |
| `CohortConstructor_vs_CohortNet_Comparison.md` | `docs/agents/cohort_constructor/` |
| `cohort_observability_integration_guide.md` | `docs/agents/cohort_constructor/` |
| `cohort_ontology_update_workflow.md` | `docs/agents/cohort_constructor/` |
| `cohort_vocabulary.yaml` | `config/agents/cohort_constructor.yaml` |

## Archive Contents

The `archive/` subfolder contains the original prototype code and reference materials for historical reference only. **Do not use these files for development** - use the production implementation in `src/agents/cohort_constructor/` instead.
