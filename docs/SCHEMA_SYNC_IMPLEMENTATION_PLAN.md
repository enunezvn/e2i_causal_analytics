# E2I Schema Synchronization Implementation Plan

**Created**: 2025-12-20
**Based On**: `docs/SUPABASE_SCHEMA_AUDIT_REPORT.md`
**Target**: Update `E2I_Data_Schema_Comprehensive_v1.2.json` to match Supabase reality

---

## Executive Summary

| Gap Type | Count | Priority | Estimated Effort |
|----------|-------|----------|------------------|
| Missing ENUM in DB | 1 | P0 - Critical | 5 min |
| Undocumented Tables | 16 | P1 - High | 2 hours |
| Undocumented ENUMs | 11 | P1 - High | 30 min |
| Undocumented Views | 5 | P2 - Medium | 30 min |
| Undocumented Functions | 30 | P2 - Medium | 1.5 hours |

**Total Estimated Effort**: ~5 hours across 6 phases

---

## Phase Overview

```
Phase 0: Critical Fix (P0)
├── Create shap_analysis_type ENUM in Supabase
└── Verify 011_realtime_shap_audit.sql migration

Phase 1A: Document Memory System (P1)
├── 5 tables: episodic_memories, procedural_memories, semantic_memory_cache,
│             cognitive_cycles, memory_statistics
├── 4 ENUMs: cognitive_phase, memory_importance, memory_type, procedure_type
└── 8 functions

Phase 1B: Document DSPy Integration (P1)
├── 4 tables: dspy_optimization_runs, dspy_evaluation_results,
│             dspy_prompt_templates, dspy_training_examples
├── 1 ENUM: dspy_optimization_phase
└── 6 functions

Phase 1C: Document Feature Store (P1)
├── 3 tables: feature_definitions, feature_values, feature_groups
├── 3 ENUMs: feature_data_type, feature_status, feature_value_status
├── 2 views: feature_freshness_monitor, feature_values_latest
└── 5 functions

Phase 1D: Document Audit Chain System (P1)
├── 2 tables: audit_chain_entries, audit_chain_verification_log
├── 2 views: v_audit_chain_integrity, v_audit_chain_summary
└── 4 functions

Phase 2: Document Remaining Entities (P2)
├── 2 tables: investigation_hops, learning_signals
├── 3 ENUMs: agent_name_enum, e2i_agent_name, learning_signal_type, optimization_status
├── 1 view: v_causal_validation_chain
└── 7 functions (trigger + other)

Phase 3: Finalize & Validate (P3)
├── Update schema metadata (version, counts, timestamps)
├── Validate JSON structure
├── Generate diff report
└── Update CHANGELOG
```

---

## Phase 0: Critical Fix - Create Missing ENUM

**Priority**: P0 - CRITICAL
**Duration**: ~5 minutes
**Blocker**: Migration `011_realtime_shap_audit.sql` fails without this

### Task 0.1: Create shap_analysis_type ENUM

**SQL to Execute**:
```sql
-- Create shap_analysis_type ENUM before running 011_realtime_shap_audit.sql
CREATE TYPE shap_analysis_type AS ENUM (
    'global',
    'local',
    'interaction',
    'summary'
);

-- After creation, the migration can add 'local_realtime':
-- ALTER TYPE shap_analysis_type ADD VALUE IF NOT EXISTS 'local_realtime';
```

### Task 0.2: Verify Migration

```bash
# After ENUM creation, verify migration can run
psql $DATABASE_URL -f database/ml/011_realtime_shap_audit.sql
```

### Task 0.3: Add to Documentation

Add to `enum_definitions.ml_enums` in JSON:
```json
"shap_analysis_type": ["global", "local", "interaction", "summary", "local_realtime"]
```

---

## Phase 1A: Document Memory System

**Priority**: P1 - High
**Duration**: ~30 minutes
**Tables**: 5 | **ENUMs**: 4 | **Functions**: 8

### Tables to Document

#### 1. episodic_memories
```json
{
  "description": "Stores episodic memories for agent learning and context retrieval",
  "primary_key": "memory_id",
  "category": "memory_system",
  "fields": {
    "memory_id": "uuid",
    "agent_name": "enum[agent_name_type]",
    "session_id": "uuid",
    "content": "text",
    "content_embedding": "vector(1536)",
    "importance": "enum[memory_importance]",
    "memory_type": "enum[memory_type]",
    "context_json": "jsonb",
    "created_at": "timestamp",
    "accessed_at": "timestamp",
    "access_count": "integer",
    "decay_factor": "float",
    "is_consolidated": "boolean"
  }
}
```

#### 2. procedural_memories
```json
{
  "description": "Stores successful action sequences and procedures for agent reuse",
  "primary_key": "procedure_id",
  "category": "memory_system",
  "fields": {
    "procedure_id": "uuid",
    "procedure_name": "string",
    "procedure_type": "enum[procedure_type]",
    "agent_name": "enum[agent_name_type]",
    "steps_json": "jsonb",
    "success_rate": "float",
    "execution_count": "integer",
    "avg_duration_ms": "integer",
    "preconditions_json": "jsonb",
    "postconditions_json": "jsonb",
    "created_at": "timestamp",
    "last_executed_at": "timestamp"
  }
}
```

#### 3. semantic_memory_cache
```json
{
  "description": "Cached semantic knowledge for fast retrieval",
  "primary_key": "cache_id",
  "category": "memory_system",
  "fields": {
    "cache_id": "uuid",
    "key": "string",
    "value_json": "jsonb",
    "embedding": "vector(1536)",
    "source": "string",
    "ttl_seconds": "integer",
    "created_at": "timestamp",
    "expires_at": "timestamp",
    "hit_count": "integer"
  }
}
```

#### 4. cognitive_cycles
```json
{
  "description": "Tracks cognitive processing cycles for agent introspection",
  "primary_key": "cycle_id",
  "category": "memory_system",
  "fields": {
    "cycle_id": "uuid",
    "session_id": "uuid",
    "agent_name": "enum[agent_name_type]",
    "phase": "enum[cognitive_phase]",
    "input_context": "jsonb",
    "processing_result": "jsonb",
    "duration_ms": "integer",
    "token_count": "integer",
    "created_at": "timestamp"
  }
}
```

#### 5. memory_statistics
```json
{
  "description": "Aggregated statistics for memory system monitoring",
  "primary_key": "stat_id",
  "category": "memory_system",
  "fields": {
    "stat_id": "uuid",
    "agent_name": "enum[agent_name_type]",
    "stat_date": "date",
    "episodic_count": "integer",
    "procedural_count": "integer",
    "semantic_cache_count": "integer",
    "avg_retrieval_ms": "float",
    "cache_hit_rate": "float",
    "consolidation_count": "integer",
    "pruned_count": "integer"
  }
}
```

### ENUMs to Document

```json
"memory_enums": {
  "cognitive_phase": ["perception", "attention", "encoding", "retrieval", "decision", "action"],
  "memory_importance": ["critical", "high", "medium", "low", "trivial"],
  "memory_type": ["experience", "fact", "procedure", "context", "feedback"],
  "procedure_type": ["tool_sequence", "query_pattern", "analysis_workflow", "error_recovery"]
}
```

### Functions to Document

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `store_episodic_memory` | Store new episodic memory | agent, session, content, importance | uuid |
| `store_procedural_memory` | Store new procedure | agent, name, steps, preconditions | uuid |
| `search_episodic_memories` | Semantic search over episodic memories | query_embedding, agent, limit | setof episodic_memories |
| `search_semantic_memories` | Search semantic cache | query_embedding, limit | setof semantic_memory_cache |
| `get_related_memories` | Get related memories by similarity | memory_id, limit | setof episodic_memories |
| `consolidate_memories` | Consolidate old memories | agent, threshold_days | integer (count) |
| `prune_memories` | Remove low-importance memories | agent, max_count | integer (pruned) |
| `update_memory_statistics` | Update daily stats | stat_date | void |

---

## Phase 1B: Document DSPy Integration

**Priority**: P1 - High
**Duration**: ~25 minutes
**Tables**: 4 | **ENUMs**: 1 | **Functions**: 6

### Tables to Document

#### 1. dspy_optimization_runs
```json
{
  "description": "Tracks DSPy prompt optimization runs for feedback learner",
  "primary_key": "run_id",
  "category": "dspy_integration",
  "fields": {
    "run_id": "uuid",
    "agent_name": "enum[agent_name_type]",
    "optimization_type": "string",
    "phase": "enum[dspy_optimization_phase]",
    "status": "enum[optimization_status]",
    "config_json": "jsonb",
    "metrics_json": "jsonb",
    "best_prompt_id": "uuid",
    "improvement_pct": "float",
    "started_at": "timestamp",
    "completed_at": "timestamp",
    "error_message": "text"
  }
}
```

#### 2. dspy_evaluation_results
```json
{
  "description": "Stores evaluation results for DSPy optimization iterations",
  "primary_key": "result_id",
  "category": "dspy_integration",
  "fields": {
    "result_id": "uuid",
    "run_id": "uuid",
    "iteration": "integer",
    "prompt_template_id": "uuid",
    "evaluation_metrics": "jsonb",
    "sample_outputs": "jsonb",
    "score": "float",
    "created_at": "timestamp"
  }
}
```

#### 3. dspy_prompt_templates
```json
{
  "description": "Stores versioned prompt templates for DSPy modules",
  "primary_key": "template_id",
  "category": "dspy_integration",
  "fields": {
    "template_id": "uuid",
    "agent_name": "enum[agent_name_type]",
    "module_name": "string",
    "template_version": "integer",
    "template_content": "text",
    "variables": "jsonb",
    "is_active": "boolean",
    "performance_score": "float",
    "usage_count": "integer",
    "created_at": "timestamp",
    "activated_at": "timestamp"
  }
}
```

#### 4. dspy_training_examples
```json
{
  "description": "Training examples for DSPy few-shot learning",
  "primary_key": "example_id",
  "category": "dspy_integration",
  "fields": {
    "example_id": "uuid",
    "agent_name": "enum[agent_name_type]",
    "module_name": "string",
    "input_json": "jsonb",
    "output_json": "jsonb",
    "quality_score": "float",
    "source": "string",
    "is_validated": "boolean",
    "created_at": "timestamp"
  }
}
```

### ENUMs to Document

```json
"dspy_enums": {
  "dspy_optimization_phase": ["initialization", "sampling", "evaluation", "optimization", "validation", "deployment"],
  "optimization_status": ["pending", "running", "completed", "failed", "cancelled"]
}
```

### Functions to Document

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `create_dspy_optimization_run` | Create new optimization run | agent, config | uuid |
| `update_dspy_optimization_status` | Update run status | run_id, status, metrics | void |
| `store_dspy_evaluation_result` | Store evaluation iteration | run_id, iteration, metrics, score | uuid |
| `get_dspy_prompt_template` | Get active template | agent, module | dspy_prompt_templates |
| `get_dspy_training_examples` | Get training examples | agent, module, limit | setof dspy_training_examples |
| `archive_dspy_run` | Archive completed runs | older_than_days | integer |

---

## Phase 1C: Document Feature Store

**Priority**: P1 - High
**Duration**: ~25 minutes
**Tables**: 3 | **ENUMs**: 3 | **Views**: 2 | **Functions**: 5

### Tables to Document

#### 1. feature_groups
```json
{
  "description": "Logical groupings of related features",
  "primary_key": "group_id",
  "category": "feature_store",
  "fields": {
    "group_id": "uuid",
    "group_name": "string",
    "description": "text",
    "entity_type": "string",
    "owner": "string",
    "tags": "array<string>",
    "created_at": "timestamp",
    "updated_at": "timestamp"
  }
}
```

#### 2. feature_definitions
```json
{
  "description": "Feature metadata and definitions for the feature store",
  "primary_key": "feature_id",
  "category": "feature_store",
  "fields": {
    "feature_id": "uuid",
    "feature_name": "string",
    "group_id": "uuid",
    "data_type": "enum[feature_data_type]",
    "description": "text",
    "computation_logic": "text",
    "source_tables": "array<string>",
    "freshness_sla_hours": "integer",
    "status": "enum[feature_status]",
    "version": "integer",
    "created_at": "timestamp",
    "updated_at": "timestamp"
  }
}
```

#### 3. feature_values
```json
{
  "description": "Materialized feature values with timestamps",
  "primary_key": "value_id",
  "category": "feature_store",
  "fields": {
    "value_id": "uuid",
    "feature_id": "uuid",
    "entity_id": "string",
    "entity_type": "string",
    "value": "jsonb",
    "computed_at": "timestamp",
    "valid_from": "timestamp",
    "valid_until": "timestamp",
    "status": "enum[feature_value_status]",
    "source_version": "string"
  }
}
```

### ENUMs to Document

```json
"feature_store_enums": {
  "feature_data_type": ["numeric", "categorical", "boolean", "array", "json", "embedding"],
  "feature_status": ["draft", "active", "deprecated", "archived"],
  "feature_value_status": ["current", "stale", "expired", "invalid"]
}
```

### Views to Document

#### feature_freshness_monitor
```json
{
  "description": "Monitors feature freshness against SLA",
  "columns": [
    "feature_id", "feature_name", "group_name", "freshness_sla_hours",
    "last_computed_at", "hours_since_update", "is_stale", "entity_count"
  ]
}
```

#### feature_values_latest
```json
{
  "description": "Latest feature values per entity",
  "columns": [
    "feature_id", "feature_name", "entity_id", "entity_type",
    "value", "computed_at", "is_current"
  ]
}
```

### Functions to Document

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `create_feature_definition` | Create new feature | name, group_id, data_type, logic | uuid |
| `get_feature_values` | Get feature values for entity | feature_id, entity_id | setof feature_values |
| `bulk_insert_feature_values` | Bulk insert values | values_json | integer |
| `update_feature_status` | Update feature status | feature_id, status | void |
| `calculate_feature_freshness` | Calculate freshness | feature_id | jsonb |

---

## Phase 1D: Document Audit Chain System

**Priority**: P1 - High
**Duration**: ~20 minutes
**Tables**: 2 | **Views**: 2 | **Functions**: 4

### Tables to Document

#### 1. audit_chain_entries
```json
{
  "description": "Immutable audit log entries with cryptographic chaining",
  "primary_key": "entry_id",
  "category": "audit_chain",
  "fields": {
    "entry_id": "uuid",
    "chain_id": "uuid",
    "sequence_number": "bigint",
    "entity_type": "string",
    "entity_id": "string",
    "action": "string",
    "actor_id": "string",
    "actor_type": "string",
    "payload_hash": "string(64)",
    "previous_hash": "string(64)",
    "entry_hash": "string(64)",
    "payload": "jsonb",
    "created_at": "timestamp"
  }
}
```

#### 2. audit_chain_verification_log
```json
{
  "description": "Log of audit chain verification checks",
  "primary_key": "verification_id",
  "category": "audit_chain",
  "fields": {
    "verification_id": "uuid",
    "chain_id": "uuid",
    "verified_at": "timestamp",
    "verified_by": "string",
    "is_valid": "boolean",
    "entries_checked": "integer",
    "first_invalid_entry": "uuid",
    "error_message": "text"
  }
}
```

### Views to Document

#### v_audit_chain_integrity
```json
{
  "description": "Real-time integrity status of audit chains",
  "columns": [
    "chain_id", "entity_type", "entry_count", "first_entry_at",
    "last_entry_at", "last_verified_at", "is_valid", "verification_status"
  ]
}
```

#### v_audit_chain_summary
```json
{
  "description": "Summary statistics for audit chains",
  "columns": [
    "entity_type", "chain_count", "total_entries", "oldest_entry",
    "newest_entry", "unverified_count", "invalid_count"
  ]
}
```

### Functions to Document

| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `append_audit_entry` | Append entry to chain | chain_id, entity, action, payload | uuid |
| `verify_audit_chain` | Verify chain integrity | chain_id | boolean |
| `get_audit_chain_for_entity` | Get audit trail | entity_type, entity_id | setof audit_chain_entries |
| `calculate_chain_hash` | Calculate entry hash | payload, previous_hash | string |

---

## Phase 2: Document Remaining Entities

**Priority**: P2 - Medium
**Duration**: ~30 minutes

### Remaining Tables (2)

#### investigation_hops
```json
{
  "description": "Tracks investigation steps in multi-hop reasoning",
  "primary_key": "hop_id",
  "category": "agent_system",
  "fields": {
    "hop_id": "uuid",
    "investigation_id": "uuid",
    "hop_number": "integer",
    "source_entity": "string",
    "target_entity": "string",
    "relationship": "string",
    "evidence": "jsonb",
    "confidence": "float",
    "created_at": "timestamp"
  }
}
```

#### learning_signals
```json
{
  "description": "Captures learning signals from user feedback and outcomes",
  "primary_key": "signal_id",
  "category": "feedback_system",
  "fields": {
    "signal_id": "uuid",
    "source_type": "string",
    "source_id": "uuid",
    "signal_type": "enum[learning_signal_type]",
    "signal_value": "float",
    "context_json": "jsonb",
    "agent_name": "enum[agent_name_type]",
    "processed": "boolean",
    "created_at": "timestamp"
  }
}
```

### Remaining ENUMs (3)

```json
"agent_system_enums": {
  "agent_name_enum": "(duplicate of agent_name_type - verify if distinct)",
  "e2i_agent_name": "(verify if distinct from agent_name_type)",
  "learning_signal_type": ["positive_feedback", "negative_feedback", "implicit_accept", "implicit_reject", "outcome_success", "outcome_failure"]
}
```

### Remaining View (1)

#### v_causal_validation_chain
```json
{
  "description": "Links causal estimates to their validation history",
  "columns": [
    "estimate_id", "estimate_source", "brand", "treatment_variable",
    "outcome_variable", "validation_count", "latest_gate", "approval_status"
  ]
}
```

### Remaining Functions (7)

#### Trigger Functions (4)
| Function | Purpose |
|----------|---------|
| `update_updated_at_column` | Auto-update timestamp on row modification |
| `notify_feature_update` | Notify listeners on feature value change |
| `validate_audit_entry` | Validate entry before insert |
| `cascade_status_change` | Cascade status updates to dependents |

#### Other Functions (3)
| Function | Purpose |
|----------|---------|
| `get_learning_signals` | Retrieve unprocessed learning signals |
| `store_investigation_hop` | Store investigation step |
| `calculate_cognitive_metrics` | Calculate agent cognitive metrics |

---

## Phase 3: Finalize & Validate

**Priority**: P3 - Low
**Duration**: ~30 minutes

### Task 3.1: Update Schema Metadata

```json
{
  "schema_version": "1.2",
  "last_updated": "2025-12-20",
  "total_tables": 55,
  "total_enums": 41,
  "total_views": 63,
  "total_functions": 51
}
```

### Task 3.2: Add Version History Entry

```json
"1.2": {
  "date": "2025-12-20",
  "changes": [
    "Added Memory System category (5 tables, 4 ENUMs, 8 functions)",
    "Added DSPy Integration category (4 tables, 2 ENUMs, 6 functions)",
    "Added Feature Store category (3 tables, 3 ENUMs, 2 views, 5 functions)",
    "Added Audit Chain category (2 tables, 2 views, 4 functions)",
    "Added 2 agent/feedback system tables",
    "Added shap_analysis_type ENUM",
    "Added 7 trigger/utility functions",
    "Table count: 28 → 55 (documented: 39 → 55)",
    "ENUM count: 30 → 41",
    "View count: 58 → 63",
    "Function count: 21 → 51"
  ]
}
```

### Task 3.3: Validate JSON Structure

```bash
# Validate JSON syntax
python -c "import json; json.load(open('docs/E2I_Data_Schema_Comprehensive_v1.2.json'))"

# Validate against expected counts
python scripts/validate_schema_counts.py
```

### Task 3.4: Generate Diff Report

Create summary of changes from v1.1 → v1.2 for review.

---

## Execution Checklist

### Phase 0: Critical Fix
- [ ] Create `shap_analysis_type` ENUM in Supabase
- [ ] Run `011_realtime_shap_audit.sql` migration
- [ ] Verify ENUM in database
- [ ] Document ENUM in JSON schema

### Phase 1A: Memory System
- [ ] Document 5 tables
- [ ] Document 4 ENUMs
- [ ] Document 8 functions
- [ ] Verify against Supabase

### Phase 1B: DSPy Integration
- [ ] Document 4 tables
- [ ] Document 2 ENUMs
- [ ] Document 6 functions
- [ ] Verify against Supabase

### Phase 1C: Feature Store
- [ ] Document 3 tables
- [ ] Document 3 ENUMs
- [ ] Document 2 views
- [ ] Document 5 functions
- [ ] Verify against Supabase

### Phase 1D: Audit Chain
- [ ] Document 2 tables
- [ ] Document 2 views
- [ ] Document 4 functions
- [ ] Verify against Supabase

### Phase 2: Remaining Entities
- [ ] Document 2 tables
- [ ] Verify/document 3 ENUMs
- [ ] Document 1 view
- [ ] Document 7 functions
- [ ] Verify against Supabase

### Phase 3: Finalize
- [ ] Update schema metadata
- [ ] Add version history
- [ ] Validate JSON syntax
- [ ] Validate counts match
- [ ] Generate diff report
- [ ] Update CHANGELOG

---

## Notes for Implementation

### Context Window Management

Each phase is designed to fit in a single context window (~100k tokens). When executing:

1. **Load only the current phase** - Don't load all phases at once
2. **Verify after each phase** - Query Supabase to confirm accuracy
3. **Commit after each phase** - Save progress incrementally
4. **Clear context between phases** - Start fresh for each phase

### Supabase MCP Verification

Use these queries to verify each category:

```sql
-- Verify table exists
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' AND table_name = 'TABLE_NAME';

-- Verify ENUM values
SELECT enum_range(NULL::ENUM_NAME);

-- Verify view exists
SELECT viewname FROM pg_views WHERE schemaname = 'public' AND viewname = 'VIEW_NAME';

-- Verify function exists
SELECT routine_name FROM information_schema.routines
WHERE routine_schema = 'public' AND routine_name = 'FUNCTION_NAME';
```

### File Locations

- **Current Schema**: `docs/E2I_Data_Schema_Comprehensive_v1.1.json`
- **Target Schema**: `docs/E2I_Data_Schema_Comprehensive_v1.2.json`
- **Audit Report**: `docs/SUPABASE_SCHEMA_AUDIT_REPORT.md`
- **This Plan**: `docs/SCHEMA_SYNC_IMPLEMENTATION_PLAN.md`

---

**Plan Created By**: Claude Code
**Creation Date**: 2025-12-20
