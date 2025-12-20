# E2I Causal Analytics - Project Structure V4.1
## Natural Language Visualization + Self-Improving Agentic RAG
### 18-Agent 6-Tier Architecture • Tier 0 ML Foundation • Tri-Memory System • Causal Validation

---

## Revision Summary (V4.1)

This revision adds:

1. **Causal Validation Infrastructure** - 2 new tables for validating causal estimates:
   - `causal_validations` - DoWhy refutation test results (5 test types)
   - `expert_reviews` - Domain expert DAG approval tracking
   - Integrated with Causal Impact agent's 5-node workflow
   - New helper views: v_validation_summary, v_active_expert_approvals, v_blocked_estimates

2. **New Validation ENUMs** - Added to domain_vocabulary.yaml v3.1.0:
   - `refutation_test_types`: placebo_treatment, random_common_cause, data_subset, bootstrap, sensitivity_e_value
   - `validation_statuses`: passed, failed, warning, skipped
   - `gate_decisions`: proceed, review, block
   - `expert_review_types`: dag_approval, methodology_review, quarterly_audit, ad_hoc_validation

3. **Table Count Update** - 26 → 28 tables (added causal_validations, expert_reviews)

4. **Previous V4.0 Features Retained**:
   - 18-Agent 6-Tier Architecture (Tier 0 ML Foundation)
   - MLOps Integration (MLflow, Opik, Great Expectations, Feast, Optuna, SHAP, BentoML)
   - 8 ML Tables from V4.0
   - Tri-Memory Architecture (Working, Episodic, Procedural, Semantic)

---

## Project Structure

```
e2i-causal-analytics/
│
├── pyproject.toml                      # Project dependencies and config
├── .env.example                        # Environment variables template
├── .env                                # Local environment (gitignored)
├── README.md                           # Project documentation
├── Makefile                            # Common development commands
├── CLAUDE.md                           # Root-level agent instructions
│
├── docker/
│   ├── Dockerfile                      # Main application container
│   ├── Dockerfile.dev                  # Development container with hot reload
│   ├── docker-compose.yml              # Full stack local deployment
│   └── docker-compose.dev.yml          # Development stack
│
├── config/
│   ├── __init__.py
│   ├── settings.py                     # Pydantic settings (env-based config)
│   ├── agent_config.yaml               # Agent behavior configuration (18 agents)
│   ├── causal_config.yaml              # Causal inference settings
│   ├── thresholds.yaml                 # KPI and trigger thresholds
│   ├── kpi_definitions.yaml            # KPI metadata and calculations (46+ KPIs)
│   ├── visualization_rules.yaml        # Chart type selection rules
│   ├── compliance_rules.yaml           # Regulatory compliance checks
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  V4.1: Domain vocabulary with validation ENUMs
│   │  ═══════════════════════════════════════════════════════════════════
│   ├── domain_vocabulary.yaml          # Fixed entity vocabularies (V3.1.0)
│   │   # agents: 18 agents in 6 tiers
│   │   # brands: [Remibrutinib, Fabhalta, Kisqali]
│   │   # regions: [northeast, south, midwest, west]
│   │   # model_stages: [development, staging, shadow, production, ...]
│   │   # mlops_tools: [MLflow, Opik, Great Expectations, ...]
│   │   # memory_types: [working, episodic, procedural, semantic]
│   │   # V4.1 NEW: refutation_test_types, validation_statuses, gate_decisions
│   │
│   ├── ml_split_config.yaml            # ML split configuration defaults
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  V4 MLOps configuration files
│   │  ═══════════════════════════════════════════════════════════════════
│   ├── mlflow_config.yaml              # MLflow tracking server settings
│   ├── opik_config.yaml                # Opik observability settings
│   ├── feast_config.yaml               # Feature store configuration
│   └── great_expectations.yaml         # Data quality expectation suites
│
├── src/
│   ├── __init__.py
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  LAYER 1: CONVERSATIONAL INTERFACE (Domain-Specific NLP)
│   │  ═══════════════════════════════════════════════════════════════════
│   │  
│   │  Uses fixed vocabularies - NO medical NER required
│   │
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── query_processor.py          # Main NL query processing pipeline
│   │   ├── intent_classifier.py        # Query intent detection → agent routing
│   │   │   # V4: Routes to 18 agents across 6 tiers
│   │   │   # V4.1 NEW: VALIDATION intent for refutation queries
│   │   ├── entity_extractor.py         # Domain entity extraction (NO medical NER)
│   │   │   # V4.1 NEW: validation_id, review_id extraction
│   │   ├── query_rewriter.py           # Optimize queries for causal retrieval
│   │   ├── ambiguity_resolver.py       # Handle under-specified queries
│   │   │
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── query_models.py         # Pydantic models for parsed queries
│   │       ├── intent_models.py        # Intent classification schemas
│   │       │   # V4.1 NEW: VALIDATION intent type
│   │       └── entity_models.py        # Domain entity schemas
│   │
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  LAYER 2: CAUSAL REASONING & RAG (Knowledge Retrieval)
│   │  ═══════════════════════════════════════════════════════════════════
│   │
│   ├── causal_engine/
│   │   ├── __init__.py
│   │   ├── dag_builder.py              # NetworkX DAG construction
│   │   ├── effect_estimator.py         # DoWhy/EconML causal effect estimation
│   │   ├── path_analyzer.py            # Causal path tracing and analysis
│   │   ├── counterfactual.py           # What-if simulation engine
│   │   ├── heterogeneous_effects.py    # CATE estimation by segment
│   │   ├── validators.py               # Causal assumption checking
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  V4.1 ENHANCED: Refutation with validation persistence
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── refutation.py               # V4.1: RefutationRunner with DoWhy tests
│   │   │   # RefutationRunner: Executes 5 refutation tests
│   │   │   # run_suite(): placebo, common_cause, subset, bootstrap, sensitivity
│   │   │   # _aggregate_results(): Computes confidence score, gate decision
│   │   │   # Persists results to causal_validations table
│   │   │
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── causal_graph.py         # DAG data structures
│   │       ├── effect_models.py        # Treatment effect schemas
│   │       ├── pathway_models.py       # Causal pathway representations
│   │       │
│   │       │  ─────────────────────────────────────────────────────────
│   │       │  V4.1 NEW: Validation models
│   │       │  ─────────────────────────────────────────────────────────
│   │       └── validation_models.py    # V4.1: Validation result schemas
│   │           # RefutationResult: Single test result
│   │           # RefutationSuite: Aggregate of all tests
│   │           # ExpertReview: DAG approval tracking
│   │           # ValidationGate: Gate decision output
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── causal_rag.py               # CausalRAG: Graph-enhanced retrieval
│   │   │   # V4.1: Also retrieves from causal_validations for validation context
│   │   ├── retriever.py                # Hybrid retrieval (vector + sparse + graph)
│   │   ├── reranker.py                 # Cross-encoder reranking
│   │   ├── query_optimizer.py          # Query expansion with domain context
│   │   ├── insight_enricher.py         # LLM-based insight enrichment
│   │   ├── chunk_processor.py          # Semantic chunking for agent outputs
│   │   │
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── insight_models.py       # Insight and chunk schemas
│   │       └── retrieval_models.py     # Retrieval result schemas
│   │
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  LAYER 3: 18-AGENT 6-TIER ORCHESTRATION (V4.0)
│   │  ═══════════════════════════════════════════════════════════════════
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py               # Abstract base with Claude API & verification
│   │   ├── state.py                    # AgentState TypedDict definitions
│   │   ├── graph.py                    # LangGraph state graph construction
│   │   ├── registry.py                 # Agent registry with tier management
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  TIER 0: ML FOUNDATION (7 agents)
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── ml_foundation/
│   │   │   ├── __init__.py
│   │   │   ├── CLAUDE.md               # Tier 0 agent instructions
│   │   │   │
│   │   │   ├── scope_definer/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py            # Problem scope, success criteria
│   │   │   │   └── prompts.py
│   │   │   │
│   │   │   ├── data_preparer/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py            # Quality control, baseline metrics
│   │   │   │   └── prompts.py
│   │   │   │
│   │   │   ├── model_selector/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py            # Algorithm evaluation, selection
│   │   │   │   └── prompts.py
│   │   │   │
│   │   │   ├── model_trainer/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py            # Training pipeline, split enforcement
│   │   │   │   └── prompts.py
│   │   │   │
│   │   │   ├── feature_analyzer/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py            # SHAP values, feature importance (Hybrid)
│   │   │   │   └── prompts.py
│   │   │   │
│   │   │   ├── model_deployer/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py            # Model registry, deployments
│   │   │   │   └── prompts.py
│   │   │   │
│   │   │   └── observability_connector/
│   │   │       ├── __init__.py
│   │   │       ├── agent.py            # Opik integration, span emission
│   │   │       └── prompts.py
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  TIER 1: COORDINATION
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── orchestrator/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Orchestrator: Coordinates all agents
│   │   │   ├── router.py               # Dynamic agent routing logic
│   │   │   ├── planner.py              # Multi-step execution planning
│   │   │   └── synthesizer.py          # Response aggregation and synthesis
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  TIER 2: CAUSAL ANALYTICS (Core E2I Mission)
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── causal_impact/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Causal Impact Agent: Traces causal chains
│   │   │   │   # V4.1: 5-node workflow with validation gate
│   │   │   │   # GraphBuilder → Estimation → Refutation → Sensitivity → Interpretation
│   │   │   │   # Refutation node calls RefutationRunner.run_suite()
│   │   │   │   # Gate decision blocks Interpretation if 'block'
│   │   │   ├── chain_tracer.py         # Causal chain identification
│   │   │   ├── effect_narrator.py      # Natural language causal explanations
│   │   │   └── prompts.py              # Agent-specific prompts
│   │   │
│   │   ├── gap_analyzer/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Gap Analyzer: ROI opportunity detection
│   │   │   ├── gap_detector.py         # Gap identification algorithms
│   │   │   ├── roi_calculator.py       # ROI estimation logic
│   │   │   └── prompts.py
│   │   │
│   │   ├── heterogeneous_optimizer/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Heterogeneous Optimizer: Segment effects
│   │   │   ├── segment_analyzer.py     # CATE by segment analysis
│   │   │   └── prompts.py
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  TIER 3: MONITORING
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── drift_monitor/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Drift Monitor: PSI and degradation
│   │   │   ├── psi_calculator.py
│   │   │   ├── drift_detector.py
│   │   │   └── prompts.py
│   │   │
│   │   ├── experiment_designer/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Experiment Designer: A/B test design
│   │   │   ├── power_analyzer.py
│   │   │   ├── test_designer.py
│   │   │   └── prompts.py
│   │   │
│   │   ├── health_score/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Health Score: System health metrics
│   │   │   ├── pareto_scorer.py
│   │   │   ├── composite_calculator.py
│   │   │   └── prompts.py
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  TIER 4: ML PREDICTIONS
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── prediction_synthesizer/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Prediction Synthesizer: Ensemble ML
│   │   │   ├── ensemble_manager.py
│   │   │   ├── confidence_calibrator.py
│   │   │   └── prompts.py
│   │   │
│   │   ├── resource_optimizer/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Resource Optimizer: ROI allocation
│   │   │   ├── allocation_engine.py
│   │   │   ├── constraint_solver.py
│   │   │   └── prompts.py
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  TIER 5: SELF-IMPROVEMENT
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── explainer/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # Explainer: NL narrative generation
│   │   │   ├── narrative_builder.py
│   │   │   ├── viz_explainer.py
│   │   │   └── prompts.py
│   │   │
│   │   └── feedback_learner/
│   │       ├── __init__.py
│   │       ├── agent.py                # Feedback Learner: Self-improvement
│   │       ├── prompt_optimizer.py
│   │       ├── pattern_learner.py
│   │       └── prompts.py
│   │
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  LAYER 4: SELF-IMPROVEMENT & LEARNING
│   │  ═══════════════════════════════════════════════════════════════════
│   │
│   ├── learning/
│   │   ├── __init__.py
│   │   ├── feedback_collector.py
│   │   ├── feedback_processor.py
│   │   ├── quality_scorer.py
│   │   ├── importance_sampler.py
│   │   ├── ab_tester.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── feedback_models.py
│   │       └── optimization_models.py
│   │
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── verification_node.py
│   │   ├── confidence_scorer.py
│   │   ├── compliance_checker.py
│   │   ├── hallucination_detector.py
│   │   ├── causal_validity_checker.py  # V4.1: Uses causal_validations table
│   │   └── models/
│   │       ├── __init__.py
│   │       └── verification_models.py
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  V4: TRI-MEMORY ARCHITECTURE
│   │  ═══════════════════════════════════════════════════════════════════
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── CLAUDE.md
│   │   ├── working_memory.py           # Redis + LangGraph MemorySaver
│   │   ├── episodic_memory.py          # Supabase + pgvector
│   │   ├── procedural_memory.py        # Supabase + pgvector
│   │   ├── semantic_memory.py          # FalkorDB + Graphity
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── memory_models.py
│   │       ├── episodic_models.py
│   │       ├── procedural_models.py
│   │       └── semantic_models.py
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  V4: MLOPS INTEGRATIONS
│   │  ═══════════════════════════════════════════════════════════════════
│   │
│   ├── mlops/
│   │   ├── __init__.py
│   │   ├── CLAUDE.md
│   │   ├── mlflow_client.py            # Experiment tracking
│   │   ├── opik_connector.py           # LLM/agent observability
│   │   ├── great_expectations_validator.py  # Data quality
│   │   ├── feast_client.py             # Feature store
│   │   ├── optuna_tuner.py             # Hyperparameter optimization
│   │   ├── shap_explainer.py           # Model interpretability
│   │   ├── bentoml_service.py          # Model serving
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── mlflow_models.py
│   │       ├── opik_models.py
│   │       └── serving_models.py
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  VISUALIZATION ENGINE
│   │  ═══════════════════════════════════════════════════════════════════
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── chart_selector.py
│   │   ├── config_generator.py
│   │   ├── causal_viz.py
│   │   ├── comparison_viz.py
│   │   ├── trend_viz.py
│   │   ├── distribution_viz.py
│   │   ├── network_viz.py
│   │   ├── pharma_viz.py
│   │   ├── theme_manager.py
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  V4 ML-specific visualizations
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── ml_experiment_viz.py        # Experiment comparisons
│   │   ├── shap_viz.py                 # SHAP visualizations
│   │   ├── training_viz.py             # Training curves
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  V4.1 NEW: Validation visualizations
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── validation_viz.py           # V4.1: Refutation result charts
│   │   │   # refutation_summary_chart(): Bar chart of test results
│   │   │   # sensitivity_plot(): E-value sensitivity visualization
│   │   │   # gate_decision_badge(): Visual gate status indicator
│   │   │
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── chart_models.py
│   │       └── style_models.py
│   │
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  DATA LAYER (V4.1 - 28 Tables)
│   │  ═══════════════════════════════════════════════════════════════════
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── client.py                   # Supabase client singleton
│   │   ├── vector_store.py             # pgvector operations for insights
│   │   │
│   │   ├── migrations/                 # Database migrations
│   │   │   ├── __init__.py
│   │   │   ├── 001_initial_schema.sql
│   │   │   ├── 002_ml_split_tables.sql
│   │   │   ├── 003_core_tables.sql
│   │   │   ├── 004_views_and_functions.sql
│   │   │   ├── 005_indexes.sql
│   │   │   ├── 006_agent_enum_update.sql
│   │   │   ├── 007_mlops_tables.sql         # V4: 8 ML lifecycle tables
│   │   │   ├── 008_kpi_gap_tables.sql       # V3: 6 KPI gap tables
│   │   │   ├── 009_kpi_helper_views.sql     # V3: 8 KPI helper views
│   │   │   │
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   │  V4.1 NEW: Causal Validation Tables
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   └── 010_causal_validation_tables.sql  # V4.1: Validation infrastructure
│   │   │       # Creates ENUMs: refutation_test_type, validation_status,
│   │   │       #   gate_decision, expert_review_type
│   │   │       # Creates tables: causal_validations, expert_reviews
│   │   │       # Creates views: v_validation_summary, v_active_expert_approvals,
│   │   │       #   v_blocked_estimates
│   │   │       # Creates functions: is_dag_approved(), get_validation_gate()
│   │   │
│   │   ├── ml_split/
│   │   │   ├── __init__.py
│   │   │   ├── split_registry.py
│   │   │   ├── patient_assignments.py
│   │   │   ├── preprocessing_meta.py
│   │   │   ├── leakage_audit.py
│   │   │   └── split_helpers.py
│   │   │
│   │   ├── repositories/               # Data access layer
│   │   │   ├── __init__.py
│   │   │   ├── base_repository.py
│   │   │   │
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   │  Core data repositories
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   ├── hcp_profile.py
│   │   │   ├── patient_journey.py
│   │   │   ├── treatment_event.py
│   │   │   ├── ml_prediction.py
│   │   │   ├── trigger.py
│   │   │   ├── agent_activity.py
│   │   │   ├── business_metrics.py
│   │   │   ├── causal_path.py
│   │   │   │
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   │  V3: KPI Gap Table Repositories
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   ├── user_session.py
│   │   │   ├── data_source_tracking.py
│   │   │   ├── ml_annotation.py
│   │   │   ├── etl_pipeline_metric.py
│   │   │   ├── hcp_intent_survey.py
│   │   │   ├── reference_universe.py
│   │   │   ├── agent_registry.py
│   │   │   │
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   │  V4: ML Foundation Repositories
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   ├── ml_experiment.py
│   │   │   ├── ml_model_registry.py
│   │   │   ├── ml_training_run.py
│   │   │   ├── ml_feature_store.py
│   │   │   ├── ml_data_quality.py
│   │   │   ├── ml_shap_analysis.py
│   │   │   ├── ml_deployment.py
│   │   │   ├── ml_observability.py
│   │   │   │
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   │  V4.1 NEW: Validation Repositories
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   ├── causal_validation.py    # V4.1: causal_validations CRUD
│   │   │   │   # save_refutation_result(): Persist single test result
│   │   │   │   # save_validation_suite(): Persist full suite with gate decision
│   │   │   │   # get_validation_summary(): Aggregate by estimate_id
│   │   │   │   # get_blocked_estimates(): Find failed validations
│   │   │   ├── expert_review.py        # V4.1: expert_reviews CRUD
│   │   │   │   # save_review(): Persist expert approval
│   │   │   │   # get_dag_approval(): Check if DAG is approved
│   │   │   │   # get_pending_reviews(): Find DAGs awaiting approval
│   │   │   │
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   │  Learning & Feedback repositories
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   ├── feedback.py
│   │   │   ├── conversation.py
│   │   │   └── insight_embeddings.py
│   │   │
│   │   └── models/
│   │       ├── __init__.py
│   │       │
│   │       │  ─────────────────────────────────────────────────────────
│   │       │  Enum types (V4.1: Added validation ENUMs)
│   │       │  ─────────────────────────────────────────────────────────
│   │       ├── enums.py
│   │       │   # V4 ENUMs:
│   │       │   # AgentNameType: 18 agents
│   │       │   # AgentTierType: ml_foundation, coordination, causal_analytics, ...
│   │       │   # ModelStageType: development, staging, shadow, production, ...
│   │       │   # DQStatusType: passed, failed, warning, skipped
│   │       │   # DeploymentStatusType: pending, deploying, active, ...
│   │       │   #
│   │       │   # V4.1 NEW:
│   │       │   # RefutationTestType: placebo_treatment, random_common_cause,
│   │       │   #   data_subset, bootstrap, sensitivity_e_value
│   │       │   # ValidationStatus: passed, failed, warning, skipped
│   │       │   # GateDecision: proceed, review, block
│   │       │   # ExpertReviewType: dag_approval, methodology_review, ...
│   │       │
│   │       │  ─────────────────────────────────────────────────────────
│   │       │  Core & ML models
│   │       │  ─────────────────────────────────────────────────────────
│   │       ├── ml_split_models.py
│   │       ├── hcp_models.py
│   │       ├── patient_models.py
│   │       ├── treatment_models.py
│   │       ├── prediction_models.py
│   │       ├── trigger_models.py
│   │       ├── agent_models.py
│   │       ├── business_models.py
│   │       ├── causal_models.py
│   │       │
│   │       │  ─────────────────────────────────────────────────────────
│   │       │  V4.1 NEW: Validation models
│   │       │  ─────────────────────────────────────────────────────────
│   │       ├── validation_models.py    # V4.1: Validation Pydantic models
│   │       │   # CausalValidation: Single refutation test result
│   │       │   # ValidationSuite: Aggregate result with gate decision
│   │       │   # ExpertReview: DAG approval record
│   │       │
│   │       │  ─────────────────────────────────────────────────────────
│   │       │  V3: KPI Gap Table Models
│   │       │  ─────────────────────────────────────────────────────────
│   │       ├── user_session_models.py
│   │       ├── data_source_models.py
│   │       ├── annotation_models.py
│   │       ├── etl_pipeline_models.py
│   │       ├── intent_survey_models.py
│   │       ├── reference_universe_models.py
│   │       │
│   │       │  ─────────────────────────────────────────────────────────
│   │       │  V4: ML Foundation Models
│   │       │  ─────────────────────────────────────────────────────────
│   │       ├── ml_experiment_models.py
│   │       ├── ml_registry_models.py
│   │       ├── ml_training_models.py
│   │       ├── ml_feature_models.py
│   │       ├── ml_quality_models.py
│   │       ├── ml_shap_models.py
│   │       ├── ml_deployment_models.py
│   │       └── ml_observability_models.py
│   │
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  TRIGGERS & BUSINESS LOGIC
│   │  ═══════════════════════════════════════════════════════════════════
│   │
│   ├── triggers/
│   │   ├── __init__.py
│   │   ├── trigger_generator.py
│   │   ├── trigger_scorer.py
│   │   ├── trigger_router.py
│   │   ├── effectiveness_tracker.py
│   │   └── models/
│   │       ├── __init__.py
│   │       └── trigger_schemas.py
│   │
│   │
│   │  ═══════════════════════════════════════════════════════════════════
│   │  API LAYER
│   │  ═══════════════════════════════════════════════════════════════════
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   ├── agents.py               # V4: 18 agents
│   │   │   ├── insights.py
│   │   │   ├── visualizations.py
│   │   │   ├── splits.py
│   │   │   ├── kpis.py
│   │   │   │
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   │  V4 MLOps routes
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   ├── experiments.py          # /experiments endpoints
│   │   │   ├── models.py               # /models endpoints
│   │   │   ├── deployments.py          # /deployments endpoints
│   │   │   ├── features.py             # /features endpoints
│   │   │   ├── observability.py        # /observability endpoints
│   │   │   │
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   │  V4.1 NEW: Validation routes
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   ├── validation.py           # V4.1: /validation endpoints
│   │   │   │   # POST /validation/run-suite
│   │   │   │   # GET /validation/estimate/{id}
│   │   │   │   # GET /validation/blocked
│   │   │   │   # POST /validation/expert-review
│   │   │   │   # GET /validation/pending-reviews
│   │   │   │   # GET /validation/dag-approval/{hash}
│   │   │   │
│   │   │   └── admin.py
│   │   │
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── logging.py
│   │   │   └── split_context.py
│   │   │
│   │   └── deps.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── claude_client.py
│       ├── logging.py
│       ├── metrics.py
│       └── helpers.py
│
│
│  ═══════════════════════════════════════════════════════════════════════
│  FRONTEND (React + TypeScript)
│  ═══════════════════════════════════════════════════════════════════════
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── chat/
│       │   ├── visualizations/
│       │   ├── agents/
│       │   ├── splits/
│       │   ├── kpis/
│       │   │
│       │   │  ─────────────────────────────────────────────────────────
│       │   │  V4 ML components
│       │   │  ─────────────────────────────────────────────────────────
│       │   ├── mlops/
│       │   │   ├── ExperimentList.tsx
│       │   │   ├── ModelRegistry.tsx
│       │   │   ├── DeploymentStatus.tsx
│       │   │   └── SHAPDisplay.tsx
│       │   │
│       │   │  ─────────────────────────────────────────────────────────
│       │   │  V4.1 NEW: Validation components
│       │   │  ─────────────────────────────────────────────────────────
│       │   ├── validation/
│       │   │   ├── ValidationBadge.tsx     # Pass/fail indicator
│       │   │   ├── RefutationResults.tsx   # Test result display
│       │   │   ├── GateDecisionCard.tsx    # proceed/review/block visual
│       │   │   ├── ExpertReviewForm.tsx    # DAG approval UI
│       │   │   └── BlockedEstimatesTable.tsx
│       │   │
│       │   └── common/
│       │
│       ├── hooks/
│       │   ├── useChat.ts
│       │   ├── useAgent.ts
│       │   ├── useVisualization.ts
│       │   ├── useSplit.ts
│       │   ├── useKPI.ts
│       │   ├── useMLOps.ts             # V4: ML hooks
│       │   └── useValidation.ts        # V4.1: Validation hook
│       │
│       ├── store/
│       │   ├── chatSlice.ts
│       │   ├── userSlice.ts
│       │   ├── dashboardSlice.ts
│       │   ├── splitSlice.ts
│       │   ├── kpiSlice.ts
│       │   ├── mlopsSlice.ts           # V4: ML state
│       │   └── validationSlice.ts      # V4.1: Validation state
│       │
│       ├── types/
│       │   ├── api.ts
│       │   ├── chat.ts
│       │   ├── visualization.ts
│       │   ├── agent.ts                # 18 agent types
│       │   ├── splits.ts
│       │   ├── kpis.ts
│       │   ├── mlops.ts                # V4: ML types
│       │   └── validation.ts           # V4.1: Validation types
│       │       # RefutationTestType, ValidationStatus, GateDecision
│       │       # RefutationResult, ValidationSuite, ExpertReview
│       │
│       └── styles/
│
│
│  ═══════════════════════════════════════════════════════════════════════
│  TESTS
│  ═══════════════════════════════════════════════════════════════════════
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   │
│   ├── unit/
│   │   ├── test_nlp/
│   │   ├── test_causal_engine/
│   │   │   ├── test_effect_estimator.py
│   │   │   │
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   │  V4.1 NEW: Refutation tests
│   │   │   │  ─────────────────────────────────────────────────────────
│   │   │   └── test_refutation.py      # V4.1: RefutationRunner tests
│   │   │       # test_placebo_treatment()
│   │   │       # test_random_common_cause()
│   │   │       # test_data_subset()
│   │   │       # test_aggregate_results()
│   │   │       # test_gate_decision_logic()
│   │   │
│   │   ├── test_rag/
│   │   ├── test_agents/
│   │   ├── test_ml_split/
│   │   ├── test_kpis/
│   │   ├── test_mlops/                 # V4: MLOps tests
│   │   │
│   │   │  ─────────────────────────────────────────────────────────────
│   │   │  V4.1 NEW: Validation unit tests
│   │   │  ─────────────────────────────────────────────────────────────
│   │   ├── test_validation/
│   │   │   ├── test_refutation_runner.py
│   │   │   ├── test_validation_repository.py
│   │   │   └── test_expert_review.py
│   │   │
│   │   └── test_visualization/
│   │
│   ├── integration/
│   │   ├── test_chat_flow.py
│   │   ├── test_agent_coordination.py
│   │   ├── test_split_aware_queries.py
│   │   ├── test_kpi_views.py
│   │   ├── test_mlops_flow.py          # V4: ML pipeline tests
│   │   └── test_validation_gate.py     # V4.1: Validation gate integration
│   │
│   │  ─────────────────────────────────────────────────────────────────
│   │  V4.1 NEW: Synthetic benchmark tests
│   │  ─────────────────────────────────────────────────────────────────
│   ├── synthetic/
│   │   ├── __init__.py
│   │   ├── conftest.py                 # Synthetic data fixtures
│   │   │   # synthetic_simple_linear()
│   │   │   # synthetic_confounded()
│   │   │   # synthetic_heterogeneous()
│   │   ├── test_effect_recovery.py     # Ground truth recovery tests
│   │   └── test_refutation_accuracy.py # Refutation test accuracy
│   │
│   └── e2e/
│       ├── test_user_journey.py
│       └── test_leakage_prevention.py
│
│
│  ═══════════════════════════════════════════════════════════════════════
│  SCRIPTS & TOOLS
│  ═══════════════════════════════════════════════════════════════════════
│
└── scripts/
    ├── setup_db.py
    ├── seed_data.py
    ├── run_leakage_audit.py
    ├── generate_pilot_data.py
    ├── load_v3_data.py
    ├── export_splits.py
    ├── validate_kpi_coverage.py
    │
    │  ─────────────────────────────────────────────────────────────────
    │  V4 MLOps scripts
    │  ─────────────────────────────────────────────────────────────────
    ├── setup_mlflow.py
    ├── setup_opik.py
    ├── setup_feast.py
    ├── run_training_pipeline.py
    ├── deploy_model.py
    ├── run_data_quality.py
    ├── generate_shap_analysis.py
    │
    │  ─────────────────────────────────────────────────────────────────
    │  V4.1 NEW: Validation scripts
    │  ─────────────────────────────────────────────────────────────────
    ├── run_refutation_suite.py         # Run refutation tests on estimate
    ├── generate_synthetic_benchmarks.py # Generate synthetic test data
    └── export_validation_report.py     # Export validation results
```

---

## Key Changes from V4.0 to V4.1

### 1. Causal Validation Infrastructure

**New Tables (migration 010):**
```
causal_validations          # Refutation test results
├── validation_id (PK)
├── estimate_id (FK)        # Links to causal_paths
├── test_type               # ENUM: placebo, common_cause, subset, bootstrap, sensitivity
├── status                  # ENUM: passed, failed, warning, skipped
├── original_effect
├── refuted_effect
├── p_value
├── confidence_score
├── gate_decision           # ENUM: proceed, review, block
└── details_json

expert_reviews              # DAG approval tracking
├── review_id (PK)
├── review_type             # ENUM: dag_approval, methodology_review, ...
├── dag_version_hash
├── reviewer_id
├── approval_status
├── checklist_json
└── valid_until
```

### 2. New ENUMs in domain_vocabulary.yaml (v3.1.0)

```yaml
refutation_test_types:
  - placebo_treatment
  - random_common_cause
  - data_subset
  - bootstrap
  - sensitivity_e_value

validation_statuses:
  - passed
  - failed
  - warning
  - skipped

gate_decisions:
  - proceed
  - review
  - block

expert_review_types:
  - dag_approval
  - methodology_review
  - quarterly_audit
  - ad_hoc_validation
```

### 3. Integration with Causal Impact Agent

```
Causal Impact Agent (5-node workflow)
├── GraphBuilder ─────────────────────────── Expert Review check (v4.1)
│   └── Calls: is_dag_approved(dag_hash)
├── Estimation
├── Refutation ────────────────────────────── RefutationRunner.run_suite() (v4.1)
│   └── Writes to: causal_validations table
├── Sensitivity
└── Interpretation ────────────────────────── Gate decision check (v4.1)
    └── BLOCKED if gate_decision == 'block'
```

---

## Table Summary (V4.1: 28 Tables)

| Category | Tables | Count |
|----------|--------|-------|
| Core Data | patient_journeys, hcp_profiles, treatment_events, ml_predictions, triggers, agent_activities, business_metrics, causal_paths | 8 |
| ML Split Management | ml_split_registry, patient_split_assignments, preprocessing_metadata, leakage_audit_results | 4 |
| KPI Gap Tables (V3) | user_sessions, data_source_tracking, ml_annotations, etl_pipeline_metrics, hcp_intent_surveys, reference_universe | 6 |
| ML Foundation (V4) | ml_experiments, ml_model_registry, ml_training_runs, ml_feature_store, ml_data_quality_reports, ml_shap_analyses, ml_deployments, ml_observability_spans | 8 |
| **Validation (V4.1)** | **causal_validations, expert_reviews** | **2** |
| **Total** | | **28** |

---

## What the RAG Should Index

| Content Type | Source Table | Use Case |
|--------------|--------------|----------|
| Discovered causal relationships | `causal_paths` | "Why did performance drop?" |
| Agent analysis outputs | `agent_activities` | "What did the gap analyzer find?" |
| Business metric trends | `business_metrics` | "How is Midwest performing?" |
| ML experiments | `ml_experiments` | "What experiments have we run?" |
| SHAP analyses | `ml_shap_analyses` | "What drives this prediction?" |
| Model performance | `ml_model_registry` | "How is our model performing?" |
| **Validation results (V4.1)** | `causal_validations` | "Did the estimate pass refutation?" |
| **Expert reviews (V4.1)** | `expert_reviews` | "Is this DAG approved?" |

---

## Summary

This V4.1 structure adds:

1. **Causal validation infrastructure** - 2 new tables for refutation tests and expert reviews
2. **4 new ENUMs** - refutation_test_types, validation_statuses, gate_decisions, expert_review_types
3. **RefutationRunner integration** - Wired to Causal Impact agent's Node 3
4. **Validation API endpoints** - /validation/* routes
5. **Frontend validation components** - Badge, results display, gate decision UI
6. **Synthetic benchmark tests** - Ground truth recovery testing
7. **Validation visualizations** - Refutation summary charts, sensitivity plots

The system now has defensible causal validation for pharmaceutical regulatory compliance.
