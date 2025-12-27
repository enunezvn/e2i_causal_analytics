# E2I Causal Analytics - Change Log

**Purpose**: Track significant changes to the E2I Causal Analytics platform
**Scope**: Code changes, configuration updates, documentation improvements, infrastructure changes
**Format**: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
**Versioning**: Follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

### Added - GEPA Prompt Optimization (V4.3)
- **src/optimization/gepa/**: New GEPA optimizer module replacing MIPROv2
  - `metrics/base.py`: E2IGEPAMetric protocol with ScoreWithFeedback support
  - `metrics/causal_impact_metric.py`: Tier 2 Hybrid agent metric
  - `metrics/experiment_designer_metric.py`: Tier 3 Hybrid agent metric
  - `metrics/feedback_learner_metric.py`: Tier 5 Deep agent metric
  - `metrics/standard_agent_metric.py`: Standard agents metric
  - `optimizer_setup.py`: GEPA optimizer factory (light/medium/heavy budgets)
  - `versioning.py`: Module versioning with save/load/rollback
  - `ab_test.py`: A/B testing for optimized modules
  - `tools/causal_tools.py`: 9 DoWhy/EconML tools for joint optimization
  - `integration/mlflow_integration.py`: GEPAMLflowCallback
  - `integration/opik_integration.py`: GEPAOpikTracer
  - `integration/ragas_feedback.py`: RAGASFeedbackProvider
- **src/agents/feedback_learner/dspy_integration.py**: GEPA optimizer support
  - `OptimizerType = Literal["miprov2", "gepa"]` for optimizer selection
  - GEPA as default with MIPROv2 fallback
- **src/rag/cognitive_rag_dspy.py**: GEPA support for Cognitive RAG phases
  - Optimizer dispatch to GEPA or MIPROv2
  - RAGAS metric integration for RAG evaluation
- **database/ml/023_gepa_optimization_tables.sql**: Database schema
  - Tables: gepa_optimization_runs, gepa_module_versions, gepa_ab_experiments
  - Views: v_gepa_performance_summary
- **config/gepa_config.yaml**: GEPA configuration
  - Budget presets (light, medium, heavy)
  - Agent-specific optimization settings
- **config/domain_vocabulary_v4.2.0.yaml**: Added GEPA vocabulary (Section 13)
  - optimizer_types, gepa_budget_presets, ab_test_variants
  - gepa_candidate_selection_strategies, gepa_metric_components
  - reflection_models, gepa_agent_config, prompt_optimization_tables
- **scripts/gepa_pilot.py**: Phase 1 pilot for Causal Impact
- **scripts/gepa_phase2_hybrid.py**: Phase 2 all hybrid agents
- **scripts/gepa_integration_test.py**: Integration test suite
- **Tests**: Comprehensive GEPA test coverage (60+ tests)
  - `tests/unit/test_optimization/test_gepa_metrics.py`: Metric unit tests
  - `tests/unit/test_optimization/test_gepa_optimizer.py`: Optimizer unit tests
  - `tests/integration/test_gepa_integration.py`: Integration tests
  - All tests use xdist_group markers for DSPy import safety

### Added - Energy Score-based Estimator Selection (V4.2)
- **src/causal_engine/energy_score/**: New module for quality-based estimator selection
  - `score_calculator.py`: EnergyScoreCalculator with 3 weighted components
    - treatment_balance (35%): Covariate balance between treatment groups
    - outcome_fit (45%): Model fit quality for outcome prediction
    - propensity_calibration (20%): Propensity score calibration
  - `estimator_selector.py`: EstimatorSelector supporting 4 estimators
    - causal_forest, linear_dml, drlearner, ols
    - Selection strategies: best_energy, first_success, ensemble
  - `mlflow_tracker.py`: MLflow integration for energy score logging
- **src/agents/causal_impact/nodes/estimation.py**: Enhanced with energy score selection
  - `_select_estimator_with_energy_score()`: Multi-estimator evaluation
  - `_get_quality_tier()`: Quality tier classification (excellent/good/acceptable/poor/unreliable)
  - Backward compatible: explicit `method` parameter uses legacy path
- **src/agents/causal_impact/state.py**: Extended state types
  - `EnergyScoreData`: TypedDict for energy score components
  - Extended `EstimationResult` with selection_strategy, selected_estimator, energy_score fields
  - Extended `CausalImpactState` with energy_score_enabled, estimator_selection_result
- **src/agents/causal_impact/graph.py**: MLflow tracking updates
  - New metrics: energy_score, energy_score_gap, n_estimators_evaluated, per-estimator scores
  - New tags: selection_strategy, selected_estimator, energy_score_quality_tier
- **src/agents/causal_impact/dspy_integration.py**: DSPy training signal updates
  - 7 new energy score fields in CausalAnalysisTrainingSignal
  - `compute_reward()` updated with 5% energy score weight
  - `update_energy_score()` method in CausalImpactSignalCollector
- **database/ml/023_energy_score_tables.sql**: Database migration
  - 3 ENUMs: estimator_type, selection_strategy, quality_tier
  - Table: estimator_evaluations (tracks all estimator evaluations)
  - Views: v_estimator_performance, v_energy_score_trends, v_selection_comparison
  - Function: log_estimator_evaluation()
- **tests/unit/test_agents/test_causal_impact/test_energy_score_selection.py**: 18 new tests
  - Tests for energy score selection, legacy mode, quality tiers, backward compatibility
- **.claude/plans/energy-score-integration-plan.md**: Implementation plan document

### Added - Self-Improvement & Evaluation (RAGAS-Opik Integration)
- **database/ml/022_self_improvement_tables.sql**: 5 new tables for self-improvement loop
  - `evaluation_results`: Store rubric evaluation outcomes
  - `retrieval_configurations`: RAG retrieval settings
  - `prompt_configurations`: Prompt version management
  - `improvement_actions`: Track auto-updates and outcomes
  - `experiment_knowledge_store`: DSPy experiment results cache
  - 2 new ENUMs: `improvement_type`, `improvement_priority`
- **src/agents/feedback_learner/evaluation/**: Rubric evaluator module
  - `rubric_evaluator.py`: AI-as-judge evaluation with Claude API
  - `criteria.py`: 5 weighted criteria (causal_validity, actionability, evidence_chain, regulatory_awareness, uncertainty_communication)
  - `models.py`: Pydantic models (RubricEvaluation, CriterionScore, ImprovementDecision, PatternFlag)
  - Dual-mode evaluation: LLM scoring + heuristic fallback
- **src/agents/feedback_learner/nodes/rubric_node.py**: LangGraph node for rubric evaluation
  - Integrates into 7-phase feedback_learner pipeline
  - Conditional routing based on decision thresholds
- **src/rag/opik_integration.py**: Opik tracing utilities
  - `OpikEvaluationTracer`: Centralized tracing for evaluations
  - `log_ragas_scores_to_opik()`: RAGAS metric logging
  - `log_rubric_scores_to_opik()`: Rubric score logging
  - `CombinedEvaluationResult`: Unified evaluation report
  - Circuit breaker for Opik unavailability
- **src/rag/evaluation.py**: Enhanced RAGASEvaluator
  - `RAGASEvaluator`: RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
  - `RAGEvaluationPipeline`: Batch evaluation pipeline
  - Opik trace integration
- **config/self_improvement.yaml**: Self-improvement configuration
  - Rubric criteria and weights
  - Decision thresholds (acceptable >= 4.0, suggestion >= 3.0, auto_update >= 2.0)
  - Safety controls (cooldowns, rate limits, north-star guardrails)
  - Improvable components registry
- **src/agents/feedback_learner/config/**: Configuration loader
  - `loader.py`: Pydantic-validated YAML config loading with caching
  - `SelfImprovementConfig`: Complete config model hierarchy
- **Tests**: Comprehensive test coverage
  - `tests/unit/test_agents/test_feedback_learner/test_rubric_evaluator.py`
  - `tests/unit/test_agents/test_feedback_learner/test_rubric_node.py`
  - `tests/integration/test_self_improvement_integration.py`
  - 148 tests passing for feedback_learner module

### Added - Documentation
- **tool-composer.md**: Complete specialist instructions for Tool Composer (.claude/specialists/tool-composer.md)
  - 4-phase pipeline documentation (decompose, plan, execute, synthesize)
  - Module responsibilities and integration points
  - Database schema reference
  - Testing requirements and common modifications
- **digital-twin.md**: Complete specialist instructions for Digital Twin system (.claude/specialists/system/digital-twin.md)
  - Twin generation, simulation engine, fidelity tracking
  - ML model selection and feature engineering
  - Database schema reference
  - Integration with Experiment Designer
- **CODE_REFERENCE.md**: Cross-reference index linking documentation to code (.claude/context/CODE_REFERENCE.md)
  - All 18 agents mapped to code locations
  - Database tables with schema file references
  - Configuration files index
  - Quick lookup commands

### Changed - Documentation
- **summary-v4.md**: Updated documentation index
  - Removed "Missing Documentation" section
  - Added new specialist files to index
  - Updated to show 100% documentation completion
  - Added 4 new entries to Update Log

### Fixed
- Documentation gaps identified in CONTEXT_AUDIT_REPORT.md now resolved

---

## [4.1.0] - 2025-12-18

### Added - Documentation & Context
- **implementation-status.md**: Comprehensive implementation tracking document for all 18 agents, modules, database tables, and MLOps tools
- **CONTEXT_AUDIT_REPORT.md**: Detailed audit findings from context file review
- **CHANGELOG.md**: This file - systematic change tracking
- **summary-v4.md**: Implementation Status section showing 65% system completion
- **summary-v4.md**: Tool Composer domain documentation (multi-tool orchestration)
- **summary-v4.md**: Digital Twin domain documentation (patient journey simulation)
- **summary-v4.md**: 6 new integration points (tool_composer, digital_twin, audit_chain)
- **experiment-history.md**: MLflow Integration section explaining business vs technical tracking
- **experiment-history.md**: December 2025 placeholder section
- **experiment-history.md**: Update cadence and maintenance guidelines
- **brand-context.md**: Brand Configuration Files section documenting config/, database, and agent configs
- **brand-context.md**: Data Validation Queries section with SQL validation queries
- **kpi-dictionary.md**: Complete 46 KPI reference with formulas, tables, thresholds
- **kpi-dictionary.md**: Database Implementation Reference section mapping KPIs to tables/views

### Changed - Documentation
- **summary-v4.md**: Updated Documentation Index with missing documentation tracking
- **summary-v4.md**: Refreshed Recent Changes to 2025-12-18
- **mlops-tools.md**: Updated all 7 tool version numbers to match requirements.txt
- **mlops-tools.md**: Added comprehensive Implementation Status section
- **All context files**: Added header metadata (Last Updated, Update Cadence, Owner, Purpose)

### Removed - Documentation
- **summary-mlops.md**: Deleted unmodified template file (content covered by summary-v4.md and mlops-tools.md)

### Added - Modules & Database
- **tool_composer module** (src/agents/tool_composer/): Multi-tool orchestration for complex queries
  - Decomposer: Breaks queries into atomic sub-tasks
  - Planner: Determines execution order and dependencies
  - Composer: Combines tools into execution graph
  - Executor: Runs tool chains with error recovery
  - Synthesizer: Aggregates multi-tool results
- **digital_twin module** (src/digital_twin/): Patient journey simulation
  - twin_generator.py: Creates patient twins from historical data
  - simulation_engine.py: Runs what-if scenarios
  - fidelity_tracker.py: Monitors twin-to-reality alignment
  - twin_repository.py: Persists and retrieves twins
- **Database tables**:
  - tool_composer_tables.sql (database/ml/013_tool_composer_tables.sql)
  - digital_twin_tables.sql (database/ml/012_digital_twin_tables.sql)
  - audit_chain_tables.sql (database/audit/011_audit_chain_tables.sql)

---

## [4.0.0] - 2024-12-08

### Added - V4 ML Foundation
- **Tier 0 Agents (7 agents)**: Configuration-only, no code yet
  - scope_definer: ML experiment scoping
  - data_preparer: Data quality and feature store
  - feature_analyzer: SHAP-based feature analysis
  - model_selector: Model selection and comparison
  - model_trainer: Training pipeline with hyperparameter optimization
  - model_deployer: Model deployment and registry
  - observability_connector: Agent execution tracing
- **MLOps Tools Configuration** (7 tools):
  - MLflow ≥2.16.0 (experiment tracking)
  - Opik ≥0.2.0 (agent observability)
  - Great Expectations ≥1.0.0 (data quality)
  - Feast ≥0.40.0 (feature store)
  - Optuna ≥3.6.0 (hyperparameter optimization)
  - SHAP ≥0.46.0 (model interpretability)
  - BentoML ≥1.3.0 (model serving)
- **Database Schema V4** (8 new tables):
  - ml_experiments (scope_definer)
  - ml_data_quality_reports (data_preparer)
  - ml_feature_store (data_preparer, Feast integration)
  - ml_model_registry (model_selector, model_deployer)
  - ml_training_runs (model_trainer)
  - ml_shap_analyses (feature_analyzer)
  - ml_deployments (model_deployer)
  - ml_observability_spans (observability_connector)
- **Agent Configuration**: Complete agent_config.yaml with all 18 agents configured
- **KPI Definitions**: 46 KPIs across 6 categories (100% calculable from database)

### Added - Documentation
- **mlops-tools.md**: Complete MLOps tool documentation with configurations and usage
- **kpi-dictionary.md**: KPI reference document (enhanced in 4.1.0)
- **experiment-history.md**: Historical experiment tracking (enhanced in 4.1.0)
- **brand-context.md**: Brand-specific context (enhanced in 4.1.0)
- **summary-v4.md**: V4 architecture summary (enhanced in 4.1.0)

---

## [3.0.0] - 2024-11-15

### Added - V3 Core System
- **Tier 1-5 Agents (11 agents)**: Multi-tier agent architecture
  - **Tier 1**: orchestrator (✅ IMPLEMENTED)
  - **Tier 2**: causal_impact, gap_analyzer, heterogeneous_optimizer (❌ config-only)
  - **Tier 3**: drift_monitor, experiment_designer (✅ IMPLEMENTED), health_score (❌ config-only)
  - **Tier 4**: prediction_synthesizer, resource_optimizer (❌ config-only)
  - **Tier 5**: explainer, feedback_learner (❌ config-only)
- **Database Schema V3** (8 core tables):
  - patient_journeys: Patient treatment journeys
  - treatment_events: Treatment interventions
  - hcp_profiles: Healthcare provider profiles
  - triggers: Business event triggers
  - ml_predictions: Model predictions
  - agent_activities: Agent execution logs
  - business_metrics: Business KPI tracking
  - causal_paths: Discovered causal relationships
- **V3 KPI Gap Tables** (6 additional tables):
  - user_sessions (MAU/WAU/DAU tracking)
  - data_source_tracking (cross-source matching, stacking lift)
  - ml_annotations (label quality, inter-annotator agreement)
  - etl_pipeline_metrics (time-to-release)
  - hcp_intent_surveys (intent delta)
  - reference_universe (coverage calculations)
- **Core Modules**:
  - orchestrator (src/agents/orchestrator/) - ✅ Production-ready
  - experiment_designer (src/agents/experiment_designer/) - ✅ Production-ready
  - causal_engine (src/causal_engine/) - ⚠️ Partial implementation
  - rag (src/rag/) - ✅ Operational RAG only (NO medical literature)
  - nlp (src/nlp/) - ✅ Business entity extraction (NOT medical NER)
  - api (src/api/) - ✅ FastAPI backend
  - memory (src/memory/) - ✅ Tri-memory architecture

### Added - Infrastructure
- **Technology Stack**:
  - Backend: Python 3.11+, FastAPI
  - ML: LangGraph, Claude API
  - Causal: DoWhy, EconML, NetworkX
  - Database: Supabase (PostgreSQL + pgvector)
  - Cache: Redis
  - Graph: FalkorDB

---

## [2.0.0] - 2024-09-20

### Added - Brand Expansion
- **Fabhalta (Iptacopan)** brand support:
  - PNH indication and patient population
  - Rare disease specialist segments
  - KOL engagement causal DAG
- **Remibrutinib** brand support:
  - CSU indication and patient population
  - Allergist/immunologist segments
  - Launch-specific considerations

### Added - Features
- **Cross-Brand Learnings**: Transferable causal insights across brands
- **Experiment History**: EXP-2024-001 through EXP-2024-005 documented
- **Brand-Specific KPIs**: BR-001 through BR-005

---

## [1.0.0] - 2024-06-01

### Added - Initial Release
- **Kisqali (Ribociclib)** brand support:
  - HR+/HER2- breast cancer indication
  - HCP targeting and patient support programs
  - Causal DAG for NRx drivers
- **Base Analytics Engine**:
  - Natural language query processing
  - Basic causal inference capabilities
  - KPI tracking framework
- **Database Foundation**:
  - Initial patient_journeys schema
  - HCP profiles and engagement tracking
  - Treatment events recording
- **API Layer**:
  - REST API for query submission
  - Basic authentication
  - Query result retrieval

---

## Version History Summary

| Version | Date | Focus | Status |
|---------|------|-------|--------|
| **4.1.0** | 2025-12-18 | Context improvements, documentation audit | ✅ Released |
| **4.0.0** | 2024-12-08 | ML Foundation (Tier 0), MLOps integration | ⚠️ Config-only (agents pending) |
| **3.0.0** | 2024-11-15 | Multi-tier agents, V3 schema | ⚠️ Partial (3 of 11 agents implemented) |
| **2.0.0** | 2024-09-20 | Multi-brand support (Fabhalta, Remibrutinib) | ✅ Released |
| **1.0.0** | 2024-06-01 | Initial release (Kisqali only) | ✅ Released |

---

## Component Implementation Timeline

```
2024-06 (v1.0):    [Kisqali]─────────────[Base Analytics]
                        │                     │
2024-09 (v2.0):         ├──[Fabhalta]        │
                        ├──[Remibrutinib]    │
                        │                     │
2024-11 (v3.0):         │             [11 Agents: 3 implemented]
                        │             [V3 Schema: 14 tables]
                        │                     │
2024-12 (v4.0):         │             [18 Agents: 3 implemented]
                        │             [V4 Schema: 24+ tables]
                        │             [MLOps: 7 tools configured]
                        │                     │
2025-12 (v4.1):         │             [Documentation Audit]
                        │             [Implementation Tracking]
                        │             [tool_composer, digital_twin]
```

---

## Migration Guides

### Migrating from V3 to V4
1. **Database**: Run V4 migration scripts to add ml_* tables
2. **Configuration**: Update agent_config.yaml with Tier 0 agent configs
3. **Dependencies**: Update requirements.txt to include MLOps tools
4. **Environment**: Add MLOps tool credentials (.env update)
5. **Documentation**: Review new MLOps tools and Tier 0 agent specs

### Migrating from V2 to V3
1. **Database**: Run V3 migration scripts to add KPI gap tables
2. **Configuration**: Add Tier 1-5 agent configurations
3. **Code**: Integrate orchestrator and experiment_designer agents
4. **Testing**: Verify multi-tier agent coordination

---

## Deprecated Features

| Feature | Deprecated In | Removed In | Replacement |
|---------|---------------|------------|-------------|
| summary-mlops.md | v4.1.0 | v4.1.0 | summary-v4.md + mlops-tools.md |
| (none) | - | - | - |

---

## Breaking Changes

### V4.0.0
- **Database Schema**: Added 8 new ml_* tables (backward compatible)
- **Agent Configuration**: Expanded agent_config.yaml structure (backward compatible with existing agents)

### V3.0.0
- **Database Schema**: Added 6 KPI gap tables (backward compatible)
- **Agent Architecture**: Moved from single-agent to multi-tier (orchestrator handles routing)

### V2.0.0
- **Brand Field**: Made brand field required in patient_journeys table (breaking for older data without brand)

---

## Security Updates

| Date | CVE | Severity | Component | Fix |
|------|-----|----------|-----------|-----|
| - | - | - | - | No security updates yet |

---

## Performance Improvements

| Version | Component | Improvement | Impact |
|---------|-----------|-------------|--------|
| v4.2.0 | Causal Impact | Energy score-based estimator selection | Better causal estimate quality via multi-estimator evaluation |
| v4.0.0 | MLOps | Added Feast feature store configuration | Faster feature retrieval (pending implementation) |
| v3.0.0 | Database | Added ml_split_registry table | Prevents data leakage in train/test splits |
| v3.0.0 | Orchestrator | Multi-tier agent coordination | Better query routing and response quality |

---

## Known Issues

### Active Issues
1. **Tier 0 Agents Not Implemented**: All 7 Tier 0 agents are configuration-only (blocks MLOps integration)
2. **MLOps Tools Unverified**: Tool configurations exist but integrations untested
3. **SHAP Integration Incomplete**: src/mlops/shap_explainer_realtime.py exists but not fully integrated
4. **Causal Engine Incomplete**: Core causal inference module needs completion

### Resolved Issues
- ❌ **summary-mlops.md template confusion** → ✅ Resolved in v4.1.0 (file deleted)
- ❌ **Undocumented tool_composer module** → ✅ Resolved in v4.1.0 (documented in summary-v4.md)
- ❌ **Undocumented digital_twin module** → ✅ Resolved in v4.1.0 (documented in summary-v4.md)
- ❌ **KPI dictionary incomplete (only ~20 of 46 KPIs)** → ✅ Resolved in v4.1.0 (all 46 KPIs documented)
- ❌ **No self-improvement loop** → ✅ Resolved (RAGAS-Opik integration with rubric evaluator)
- ❌ **Single estimator fallback in causal_impact** → ✅ Resolved in v4.2 (energy score-based selection)

---

## Contributing

When adding entries to this changelog:

1. **Choose the right section**:
   - `Added` for new features
   - `Changed` for changes in existing functionality
   - `Deprecated` for soon-to-be removed features
   - `Removed` for now removed features
   - `Fixed` for any bug fixes
   - `Security` for vulnerability fixes

2. **Format**:
   ```markdown
   - **Component Name**: Brief description of change
   ```

3. **Reference issues/PRs** when applicable:
   ```markdown
   - **Agent System**: Fixed orchestrator routing bug (#123)
   ```

4. **Update version numbers** following semantic versioning:
   - MAJOR version for incompatible API changes
   - MINOR version for backward-compatible functionality additions
   - PATCH version for backward-compatible bug fixes

---

**Last Updated**: 2025-12-26
**Maintained By**: E2I Development Team
**Next Review**: 2026-01-26 (monthly cadence)
