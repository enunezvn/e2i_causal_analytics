# Code Reference Index

**Last Updated**: 2025-12-18
**Purpose**: Cross-reference guide linking documentation to code locations
**Owner**: E2I Development Team
**Update Frequency**: After major code changes or restructuring

---

## Quick Navigation

This index helps you find code quickly when working with E2I context and specialist files. Each entry links documentation concepts to specific file paths and line numbers.

**Format**: `concept: path/to/file.py:line_number`

---

## Agents

### Tier 1: Orchestrator

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/orchestrator-agent.md`

| Concept | Code Location |
|---------|---------------|
| Main router | src/agents/orchestrator/router_v42.py:1 |
| Query classifier | src/agents/orchestrator/classifier/query_classifier.py:1 |
| Intent classification | src/agents/orchestrator/classifier/intent_classifier.py:1 |
| Complexity scorer | src/agents/orchestrator/classifier/complexity_scorer.py:1 |
| Tool routing | src/agents/orchestrator/tools/tool_router.py:1 |
| Agent state | src/agents/orchestrator/models/orchestrator_state.py:1 |
| Configuration | config/agent_config.yaml:326 |

### Tier 2: Causal Impact

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/causal-impact.md`

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:365 |
| Database support | database/core/001_core_schema.sql:120 (causal_paths table) |

**Status**: Config-only (no implementation yet)

### Tier 2: Gap Analyzer

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/gap-analyzer.md`

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:420 |

**Status**: Config-only (no implementation yet)

### Tier 2: Heterogeneous Optimizer

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/heterogeneous-optimizer.md`

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:448 |

**Status**: Config-only (no implementation yet)

### Tier 3: Drift Monitor

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md`

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:484 |

**Status**: Config-only (no implementation yet)

### Tier 3: Experiment Designer

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md`

| Concept | Code Location |
|---------|---------------|
| Main agent | src/agents/experiment_designer/__init__.py:1 |
| Intervention simulator | src/agents/experiment_designer/tools/simulate_intervention_tool.py:1 |
| Twin fidelity validator | src/agents/experiment_designer/tools/validate_twin_fidelity_tool.py:1 |
| Configuration | config/agent_config.yaml:518 |

### Tier 3: Health Score

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/health-score.md`

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:560 |

**Status**: Config-only (no implementation yet)

### Tier 4: Prediction Synthesizer

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/prediction-synthesizer.md`

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:599 |

**Status**: Config-only (no implementation yet)

### Tier 4: Resource Optimizer

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/resource-optimizer.md`

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:638 |

**Status**: Config-only (no implementation yet)

### Tier 5: Explainer

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/explainer.md`

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:672 |

**Status**: Config-only (no implementation yet)

### Tier 5: Feedback Learner

**Documentation**: `.claude/specialists/Agent_Specialists_Tiers 1-5/feedback-learner.md`

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:712 |

**Status**: Config-only (no implementation yet)

### Tool Composer (Not in Original Spec)

**Documentation**: `.claude/specialists/tool-composer.md`

| Concept | Code Location |
|---------|---------------|
| Main orchestrator | src/agents/tool_composer/composer.py:40 |
| Query decomposer | src/agents/tool_composer/decomposer.py:77 |
| Tool planner | src/agents/tool_composer/planner.py:1 |
| Plan executor | src/agents/tool_composer/executor.py:1 |
| Response synthesizer | src/agents/tool_composer/synthesizer.py:1 |
| Schemas | src/agents/tool_composer/schemas.py:1 |
| Tool registry | src/agents/tool_composer/tool_registrations.py:1 |
| Models | src/agents/tool_composer/models/composition_models.py:1 |
| Prompts | src/agents/tool_composer/prompts.py:1 |
| Database | database/ml/013_tool_composer_tables.sql:1 |
| Agent instructions | src/agents/tool_composer/CLAUDE.md:1 |

---

## Core Systems

### NLP Layer

**Documentation**: `.claude/specialists/system/nlp.md`

| Concept | Code Location |
|---------|---------------|
| Query processor | src/nlp/query_processor.py:1 |
| Intent classifier | src/nlp/intent_classifier.py:1 |
| Entity extractor | src/nlp/entity_extractor.py:1 |
| Query rewriter | src/nlp/query_rewriter.py:1 |
| Ambiguity resolver | src/nlp/ambiguity_resolver.py:1 |
| FastText trainer | src/nlp/e2i_fasttext_trainer.py:1 |
| Entity models | src/nlp/models/entity_models.py:1 |
| Query models | src/nlp/models/query_models.py:1 |
| Domain vocabulary | config/domain_vocabulary.yaml:1 |

### Causal Engine

**Documentation**: `.claude/specialists/system/causal.md`

| Concept | Code Location |
|---------|---------------|
| Causal directory | src/causal/:1 |

**Status**: ⚠️ Partial implementation (needs completion)

### RAG System

**Documentation**: `.claude/specialists/system/rag.md`

| Concept | Code Location |
|---------|---------------|
| RAG directory | src/rag/:1 |

**Status**: ✅ Operational RAG only (no medical literature)

### Digital Twin System

**Documentation**: `.claude/specialists/system/digital-twin.md`

| Concept | Code Location |
|---------|---------------|
| Twin generator | src/digital_twin/twin_generator.py:40 |
| Simulation engine | src/digital_twin/simulation_engine.py:39 |
| Fidelity tracker | src/digital_twin/fidelity_tracker.py:1 |
| Twin repository | src/digital_twin/twin_repository.py:1 |
| Twin models | src/digital_twin/models/twin_models.py:1 |
| Simulation models | src/digital_twin/models/simulation_models.py:1 |
| Database | database/ml/012_digital_twin_tables.sql:1 |

### API Layer

**Documentation**: `.claude/specialists/system/api.md`

| Concept | Code Location |
|---------|---------------|
| Main FastAPI app | src/api/main.py:1 |
| Routes directory | src/api/routes/:1 |

### Frontend

**Documentation**: `.claude/specialists/system/frontend.md`

| Concept | Code Location |
|---------|---------------|
| Frontend directory | src/frontend/:1 |

### Database

**Documentation**: `.claude/specialists/system/database.md`

| Concept | Code Location |
|---------|---------------|
| Core schema | database/core/001_core_schema.sql:1 |
| V3 KPI gap tables | database/core/002_kpi_gap_tables.sql:1 |
| ML foundation schema | database/ml/007_ml_foundation_schema.sql:1 |
| ML split registry | database/ml/008_ml_split_registry.sql:1 |
| ML preprocessing | database/ml/009_ml_preprocessing_metadata.sql:1 |
| Causal validation | database/ml/010_causal_validation_tables.sql:1 |
| Audit chain | database/audit/011_audit_chain_tables.sql:1 |
| Digital twins | database/ml/012_digital_twin_tables.sql:1 |
| Tool composer | database/ml/013_tool_composer_tables.sql:1 |
| Setup script | scripts/setup_db.py:1 |
| Data loader | scripts/load_v3_data.py:1 |

### Memory System

**Documentation**: (No specialist file yet - covered in summary-v4.md)

| Concept | Code Location |
|---------|---------------|
| Memory directory | src/memory/:1 |
| Cognitive workflow | src/memory/004_cognitive_workflow.py:1 |
| Memory backends | src/memory/006_memory_backends_v1_3.py:1 |

### Utilities

**Documentation**: (No specialist file - general utilities)

| Concept | Code Location |
|---------|---------------|
| Audit chain | src/utils/audit_chain.py:1 |

---

## Tier 0: ML Foundation Agents

All Tier 0 agents are **config-only** (no code implementation yet).

**Documentation**: `.claude/specialists/ml_foundation/`

### Scope Definer

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:96 |
| Database | database/ml/007_ml_foundation_schema.sql:1 (ml_experiments table) |
| Specialist doc | .claude/specialists/ml_foundation/scope_definer.md:1 |

**Status**: Config-only

### Data Preparer

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:124 |
| Database tables | database/ml/007_ml_foundation_schema.sql:20 (ml_data_quality_reports, ml_feature_store) |
| Specialist doc | .claude/specialists/ml_foundation/data_preparer.md:1 |

**Status**: Config-only
**Priority**: HIGHEST (next implementation)

### Feature Analyzer

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:156 |
| Database | database/ml/007_ml_foundation_schema.sql:60 (ml_shap_analyses) |
| Specialist doc | .claude/specialists/ml_foundation/feature_analyzer.md:1 |

**Status**: Config-only

### Model Selector

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:188 |
| Database | database/ml/007_ml_foundation_schema.sql:40 (ml_model_registry) |
| Specialist doc | .claude/specialists/ml_foundation/model_selector.md:1 |

**Status**: Config-only

### Model Trainer

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:225 |
| Database | database/ml/007_ml_foundation_schema.sql:80 (ml_training_runs) |
| Specialist doc | .claude/specialists/ml_foundation/model_trainer.md:1 |

**Status**: Config-only

### Model Deployer

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:258 |
| Database | database/ml/007_ml_foundation_schema.sql:100 (ml_deployments) |
| Specialist doc | .claude/specialists/ml_foundation/model_deployer.md:1 |

**Status**: Config-only

### Observability Connector

| Concept | Code Location |
|---------|---------------|
| Configuration | config/agent_config.yaml:295 |
| Database | database/ml/007_ml_foundation_schema.sql:120 (ml_observability_spans) |
| Specialist doc | .claude/specialists/ml_foundation/observability_connector.md:1 |

**Status**: Config-only

---

## MLOps Tools

**Documentation**: `.claude/context/mlops-tools.md`

| Tool | Version | Config Location |
|------|---------|-----------------|
| MLflow | ≥2.16.0 | config/agent_config.yaml:832 |
| Opik | ≥0.2.0 | config/agent_config.yaml:838 |
| Great Expectations | ≥1.0.0 | config/agent_config.yaml:843 |
| Feast | ≥0.40.0 | config/agent_config.yaml:848 |
| Optuna | ≥3.6.0 | config/agent_config.yaml:853 |
| SHAP | ≥0.46.0 | config/agent_config.yaml:858 |
| BentoML | ≥1.3.0 | config/agent_config.yaml:863 |

**SHAP Integration** (only partial implementation):
- src/mlops/shap_explainer_realtime.py:1

---

## Configuration Files

### Agent Configuration

| Concept | Code Location |
|---------|---------------|
| All 18 agents | config/agent_config.yaml:1 |
| MLOps tools | config/agent_config.yaml:832 |

### KPI Definitions

**Documentation**: `.claude/context/kpi-dictionary.md`

| Concept | Code Location |
|---------|---------------|
| 46 KPI definitions | config/kpi_definitions.yaml:1 |
| Database implementation | See kpi-dictionary.md for table mappings |

### Brand Context

**Documentation**: `.claude/context/brand-context.md`

| Concept | Code Location |
|---------|---------------|
| Brand configurations | config/brand_configs/:1 |
| Domain vocabulary | config/domain_vocabulary.yaml:1 |

---

## Database Tables Reference

### Core Tables (V3)

| Table | Purpose | Schema File | Line |
|-------|---------|-------------|------|
| patient_journeys | Patient treatment journeys | database/core/001_core_schema.sql | 10 |
| treatment_events | Treatment interventions | database/core/001_core_schema.sql | 40 |
| hcp_profiles | Healthcare provider profiles | database/core/001_core_schema.sql | 70 |
| triggers | Business event triggers | database/core/001_core_schema.sql | 100 |
| ml_predictions | Model predictions | database/core/001_core_schema.sql | 130 |
| agent_activities | Agent execution logs | database/core/001_core_schema.sql | 160 |
| business_metrics | Business KPI tracking | database/core/001_core_schema.sql | 190 |
| causal_paths | Discovered causal relationships | database/core/001_core_schema.sql | 220 |

### V3 KPI Gap Tables

| Table | Purpose | Schema File | Line |
|-------|---------|-------------|------|
| user_sessions | MAU/WAU/DAU tracking | database/core/002_kpi_gap_tables.sql | 10 |
| data_source_tracking | Cross-source matching, stacking lift | database/core/002_kpi_gap_tables.sql | 30 |
| ml_annotations | Label quality, inter-annotator agreement | database/core/002_kpi_gap_tables.sql | 50 |
| etl_pipeline_metrics | Time-to-release | database/core/002_kpi_gap_tables.sql | 70 |
| hcp_intent_surveys | Intent delta | database/core/002_kpi_gap_tables.sql | 90 |
| reference_universe | Coverage calculations | database/core/002_kpi_gap_tables.sql | 110 |

### V4 ML Foundation Tables

| Table | Purpose | Schema File | Line |
|-------|---------|-------------|------|
| ml_experiments | ML experiment scoping | database/ml/007_ml_foundation_schema.sql | 10 |
| ml_data_quality_reports | Data quality tracking | database/ml/007_ml_foundation_schema.sql | 30 |
| ml_feature_store | Feature store | database/ml/007_ml_foundation_schema.sql | 50 |
| ml_model_registry | Model versioning | database/ml/007_ml_foundation_schema.sql | 70 |
| ml_training_runs | Training execution | database/ml/007_ml_foundation_schema.sql | 90 |
| ml_shap_analyses | SHAP interpretability | database/ml/007_ml_foundation_schema.sql | 110 |
| ml_deployments | Deployment tracking | database/ml/007_ml_foundation_schema.sql | 130 |
| ml_observability_spans | Agent tracing | database/ml/007_ml_foundation_schema.sql | 150 |

### Additional ML Tables

| Table | Purpose | Schema File | Line |
|-------|---------|-------------|------|
| ml_split_registry | Train/val/test splits | database/ml/008_ml_split_registry.sql | 10 |
| ml_patient_split_assignments | Patient-level splits | database/ml/008_ml_split_registry.sql | 30 |
| ml_preprocessing_metadata | Feature engineering metadata | database/ml/009_ml_preprocessing_metadata.sql | 10 |
| ml_leakage_audit | Data leakage prevention | database/ml/009_ml_preprocessing_metadata.sql | 30 |
| causal_validation_gates | Causal validation | database/ml/010_causal_validation_tables.sql | 10 |

### Audit Tables

| Table | Purpose | Schema File | Line |
|-------|---------|-------------|------|
| audit_chain | Immutable audit trail | database/audit/011_audit_chain_tables.sql | 10 |

### Digital Twin Tables

| Table | Purpose | Schema File | Line |
|-------|---------|-------------|------|
| digital_twin_models | Twin generator models | database/ml/012_digital_twin_tables.sql | 10 |
| digital_twins | Individual synthetic entities | database/ml/012_digital_twin_tables.sql | 30 |
| twin_populations | Twin population groups | database/ml/012_digital_twin_tables.sql | 50 |
| twin_population_members | Many-to-many mapping | database/ml/012_digital_twin_tables.sql | 70 |
| simulation_runs | Intervention simulations | database/ml/012_digital_twin_tables.sql | 90 |
| fidelity_tracking | Twin-to-reality alignment | database/ml/012_digital_twin_tables.sql | 110 |
| fidelity_snapshots | Periodic fidelity aggregates | database/ml/012_digital_twin_tables.sql | 130 |

### Tool Composer Tables

| Table | Purpose | Schema File | Line |
|-------|---------|-------------|------|
| tool_composer_executions | Composition tracking | database/ml/013_tool_composer_tables.sql | 10 |
| tool_composer_step_logs | Step-level logging | database/ml/013_tool_composer_tables.sql | 30 |
| tool_registry | Registered tools | database/ml/013_tool_composer_tables.sql | 50 |

---

## Scripts

### Database Setup

| Script | Purpose | Location |
|--------|---------|----------|
| Database setup | Creates all tables | scripts/setup_db.py:1 |
| V3 data loader | Loads sample data | scripts/load_v3_data.py:1 |

### Validation

| Script | Purpose | Location |
|--------|---------|----------|
| KPI coverage validator | Validates 46 KPIs calculable | scripts/validate_kpi_coverage.py:1 |
| Data leakage auditor | Checks for leakage | scripts/run_leakage_audit.py:1 |

---

## Tests

### Unit Tests

| Component | Test Location |
|-----------|---------------|
| NLP | tests/unit/test_nlp/ |
| Tool Composer | tests/unit/test_tool_composer/ |
| Digital Twin | tests/unit/test_digital_twin/ |

### Integration Tests

| Component | Test Location |
|-----------|---------------|
| Tool Composer E2E | tests/integration/test_tool_composer_e2e.py:1 |
| Digital Twin E2E | tests/integration/test_digital_twin_e2e.py:1 |

---

## Context Files

### Summary Files

| File | Purpose | When to Update |
|------|---------|----------------|
| summary-v4.md | Project overview and architecture | After major code changes |
| implementation-status.md | Agent/module implementation tracking | After completing agents |
| CHANGELOG.md | Version history | After releases |

### Domain-Specific

| File | Purpose | When to Update |
|------|---------|----------------|
| brand-context.md | Brand-specific information | After brand additions |
| kpi-dictionary.md | KPI definitions and formulas | After KPI additions |
| experiment-history.md | Historical experiments | After experiment completion |
| mlops-tools.md | MLOps tool configuration | After tool updates |

### Audit & Reference

| File | Purpose | When to Update |
|------|---------|----------------|
| CONTEXT_AUDIT_REPORT.md | Documentation audit findings | Monthly |
| CODE_REFERENCE.md | This file - code cross-references | After restructuring |

---

## Quick Lookup Commands

### Find Agent Code
```bash
# List all agent directories
ls -1 src/agents/

# Find agent by name
find src/agents -name "*orchestrator*"
```

### Find Database Schemas
```bash
# List all migration files
ls -1 database/*/

# Find table definition
grep -r "CREATE TABLE ml_experiments" database/
```

### Find Configuration
```bash
# View agent config
cat config/agent_config.yaml | grep -A 20 "orchestrator:"

# View KPI definitions
cat config/kpi_definitions.yaml
```

### Find Tests
```bash
# List all test files
find tests -name "test_*.py"

# Run specific test suite
pytest tests/unit/test_tool_composer/
```

---

## Maintenance

### When to Update This File

1. **After code restructuring** - Moving files to new locations
2. **After adding new agents** - Add agent references
3. **After database migrations** - Update table references
4. **Monthly** - Review and update line numbers if files changed significantly

### Update Process

1. Run `find src -name "*.py" -type f` to verify file locations
2. Check line numbers for key classes/functions
3. Update relevant sections
4. Commit with message: "docs: update CODE_REFERENCE with [changes]"

---

**Last Updated**: 2025-12-18
**Next Review**: 2026-01-18
**Maintained By**: E2I Development Team
