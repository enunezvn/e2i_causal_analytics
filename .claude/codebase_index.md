# E2I Causal Analytics - Codebase Index

**Last Updated:** 2025-12-15
**Project Version:** V4.1
**Purpose:** Pharmaceutical causal inference system with 18-agent orchestration

---

## Quick Reference

### Project Type
Healthcare Engagement Intelligence (E2I) - Multi-agent causal analytics platform for pharmaceutical drug adoption analysis

### Key Brands Analyzed
- Remibrutinib (BTK inhibitor for CSU)
- Fabhalta (Factor B inhibitor for PNH)
- Kisqali (CDK4/6 inhibitor for breast cancer)

### Tech Stack
- **Backend:** Python 3.12+, LangGraph, FastAPI, Anthropic Claude
- **Database:** PostgreSQL/Supabase (28 tables), Redis, FalkorDB
- **ML/Causal:** DoWhy, EconML, MLflow, Opik, SHAP
- **Frontend:** React + TypeScript (planned)

---

## Architecture Overview

### 6-Tier, 18-Agent System

**TIER 0: ML FOUNDATION (7 agents)**
- scope_definer, data_preparer, feature_analyzer, model_selector, model_trainer, model_deployer, observability_connector

**TIER 1: COORDINATION (1 agent)**
- orchestrator (multi-agent routing)

**TIER 2: CAUSAL ANALYTICS (3 agents) ⭐ Core**
- causal_impact (effect estimation + validation)
- gap_analyzer (ROI opportunities)
- heterogeneous_optimizer (treatment effect heterogeneity)

**TIER 3: MONITORING (3 agents)**
- drift_monitor, experiment_designer, health_score

**TIER 4: ML PREDICTIONS (2 agents)**
- prediction_synthesizer, resource_optimizer

**TIER 5: SELF-IMPROVEMENT (2 agents)**
- explainer, feedback_learner

---

## Directory Structure

```
e2i_causal_analytics/
├── README.md                          # Project overview & quick start
├── pyproject.toml                     # Python project configuration
├── requirements.txt                   # Python dependencies
├── Makefile                           # Development commands
├── .env                               # Environment variables (gitignored)
├── .env.example                       # Environment template
├── .gitignore                         # Git ignore rules
│
├── config/                            # YAML Configurations (8 files)
│   ├── agent_config.yaml              # 18-agent definitions (982 lines)
│   ├── domain_vocabulary_v3.1.0.yaml  # Fixed entity vocabularies
│   ├── kpi_definitions.yaml           # 46+ KPI definitions
│   ├── alert_config.yaml              # Alert system (25+ types)
│   ├── confidence_logic.yaml          # 4-component confidence scoring
│   ├── experiment_lifecycle.yaml      # Experiment state machine
│   ├── filter_mapping.yaml            # Filter → Query Flow
│   └── agent_config_v3.yaml.backup    # Legacy backup
│
├── database/                          # SQL Schemas (28 tables)
│   ├── core/
│   │   └── e2i_ml_complete_v3_schema.sql  # 8 core data tables
│   ├── ml/
│   │   ├── mlops_tables.sql           # 8 ML foundation tables
│   │   ├── 010_causal_validation_tables.sql  # 2 validation tables
│   │   └── 012_data_sources.sql       # Data sources reference
│   ├── memory/
│   │   ├── 001_agentic_memory_schema_v1.3.sql  # 4 memory tables
│   │   ├── 001b_add_foreign_keys_v3.sql
│   │   └── 002_semantic_graph_schema.cypher   # FalkorDB schema
│   └── audit/
│       └── 011_audit_chain_tables.sql # Audit trail
│
├── data/
│   ├── synthetic/                     # Synthetic Data (18 JSON files)
│   │   ├── e2i_ml_v3_agent_activities.json
│   │   ├── e2i_ml_v3_business_metrics.json
│   │   ├── e2i_ml_v3_causal_paths.json
│   │   ├── e2i_ml_v3_patient_journeys.json
│   │   ├── e2i_ml_v3_hcp_profiles.json
│   │   ├── e2i_ml_v3_treatment_events.json
│   │   ├── e2i_ml_v3_ml_predictions.json
│   │   ├── e2i_ml_v3_triggers.json
│   │   ├── e2i_ml_v3_split_registry.json
│   │   ├── e2i_ml_v3_train.json
│   │   ├── e2i_ml_v3_validation.json
│   │   ├── e2i_ml_v3_test.json
│   │   ├── e2i_ml_v3_preprocessing_metadata.json
│   │   ├── e2i_ml_v3_user_sessions.json
│   │   ├── e2i_ml_v3_data_source_tracking.json
│   │   ├── e2i_ml_v3_ml_annotations.json
│   │   ├── e2i_ml_v3_etl_pipeline_metrics.json
│   │   ├── e2i_ml_v3_hcp_intent_surveys.json
│   │   ├── e2i_ml_v3_reference_universe.json
│   │   └── e2i_ml_v4_agent_activities.json  # V4 activities (3.8MB)
│   └── training/
│       └── e2i_corpus.txt             # fastText training corpus
│
├── src/                               # Main Source Code
│   ├── __init__.py
│   ├── nlp/                           # Natural Language Processing
│   │   ├── __init__.py
│   │   └── e2i_fasttext_trainer.py    # Typo-tolerant query matching
│   ├── agents/                        # 18 Agent Implementations
│   │   └── __init__.py
│   ├── memory/                        # Tri-Memory System
│   │   ├── __init__.py
│   │   ├── 004_cognitive_workflow.py  # LangGraph state machine
│   │   └── 006_memory_backends_v1_3.py # Redis/Supabase/FalkorDB
│   ├── causal/                        # Causal Inference Engine
│   │   └── __init__.py
│   ├── ml/                            # ML Operations & Data
│   │   ├── __init__.py
│   │   ├── data_generator.py          # Synthetic data generation
│   │   └── data_loader.py             # Bulk data loading
│   ├── api/                           # FastAPI Endpoints
│   │   └── __init__.py
│   └── utils/                         # Shared Utilities
│       └── __init__.py
│
├── tests/                             # Test Suite
│   ├── unit/
│   │   └── test_cognitive_simple.py
│   └── integration/
│
├── scripts/                           # Utility Scripts
│   └── validate_kpi_coverage.py
│
├── frontend/                          # UI/Dashboard
│   ├── E2I_Causal_Dashboard_V2.html
│   └── E2I_Causal_Dashboard_V3.html
│
├── docs/                              # Comprehensive Documentation
│   ├── e2i_nlv_project_structure_v4.1.md  # PRIMARY DOC (55KB)
│   ├── e2i_gap_analysis.md            # 14 gaps identified
│   ├── e2i_gap_todo.md                # Implementation tracking
│   ├── confidence_methodology.md       # Confidence scoring
│   ├── E2I_KPI_Verification_V3.md
│   ├── E2I_ML_V3_Package_README.md
│   ├── E2I_PRD_V4.1_Update_Summary.md
│   ├── README.md                      # Memory system setup
│   ├── e2i_query_flow.mermaid
│   ├── e2i_rag_nlp_library_analysis.md  # NLP library evaluation
│   ├── kpi_report.md
│   └── validate_kpi_coverage_README.md
│
├── docker/                            # Docker Configurations
│   └── (to be added)
│
├── archive/                           # Archived Files
│   └── (old versions)
│
├── .claude/                           # Claude Code Configuration
│   └── codebase_index.md              # This file
│
├── venv/                              # Virtual Environment (gitignored)
└── e2i_agentic_memory/                # Legacy folder (to be removed)
└── e2i_ml_compliant_data/             # Legacy folder (to be removed)
└── e2i_causalrag/                     # Legacy folder (to be removed)
```

---

## Database Schema (28 Tables)

### Core Data Tables (8)
Located: `database/core/e2i_ml_complete_v3_schema.sql`
- patient_journeys
- hcp_profiles
- treatment_events
- ml_predictions
- triggers
- agent_activities
- business_metrics
- causal_paths

### ML Foundation Tables (8)
Located: `database/ml/mlops_tables.sql`
- ml_experiments
- ml_model_registry
- ml_training_runs
- ml_feature_store
- ml_data_quality_reports
- ml_shap_analyses
- ml_deployments
- ml_observability_spans

### Memory Tables (4)
Located: `database/memory/001_agentic_memory_schema_v1.3.sql`
- episodic_memories (pgvector)
- procedural_memories (pgvector)
- semantic_memory_cache
- working_memory_sessions

### Causal Validation Tables (2)
Located: `database/ml/010_causal_validation_tables.sql`
- causal_validations (5 refutation test types)
- expert_reviews (DAG approval workflow)

### Supporting Tables (6)
- ml_split_registry
- patient_split_assignments
- preprocessing_metadata
- leakage_audit_results
- user_sessions
- data_source_tracking

---

## Tri-Memory Architecture

**SHORT-TERM (Working Memory)**
- Backend: Redis (port 6379) + LangGraph MemorySaver
- Stores: Session state, messages, evidence board
- TTL: 3600 seconds

**LONG-TERM:**

1. **EPISODIC** (What did I do?)
   - Backend: Supabase + pgvector
   - Stores: User queries, agent actions, events

2. **SEMANTIC** (What is the relationship?)
   - Backend: FalkorDB (port 6380) + Graphity
   - Stores: Entity nodes, relationships, causal chains

3. **PROCEDURAL** (How did I solve this?)
   - Backend: Supabase + pgvector
   - Stores: Tool sequences, query patterns

4. **SEMANTIC CACHE**
   - Backend: Supabase cache table
   - Stores: Hot graph triplets

**4-Phase Cognitive Cycle (LangGraph):**
Summarizer → Investigator → Agent → Reflector

---

## Key Implementation Files

### Configuration Files
- `config/agent_config.yaml` - 18-agent tier/capabilities/routing
- `config/domain_vocabulary_v3.1.0.yaml` - Vocabulary enums + validation types
- `config/kpi_definitions.yaml` - 46+ KPI metadata
- `config/alert_config.yaml` - Alert system (25+ alert types)
- `config/confidence_logic.yaml` - 4-component confidence scoring
- `config/experiment_lifecycle.yaml` - Experiment state machine
- `config/filter_mapping.yaml` - Filter → Query Flow mapping

### Database Schemas
- `database/core/e2i_ml_complete_v3_schema.sql` - Core 8 tables
- `database/ml/mlops_tables.sql` - ML Foundation 8 tables
- `database/ml/010_causal_validation_tables.sql` - Causal validation
- `database/ml/012_data_sources.sql` - Data sources reference
- `database/memory/001_agentic_memory_schema_v1.3.sql` - Memory tables
- `database/memory/002_semantic_graph_schema.cypher` - FalkorDB graph
- `database/audit/011_audit_chain_tables.sql` - Audit trail

### Python Core Modules
- `src/memory/004_cognitive_workflow.py` - LangGraph state machine
- `src/memory/006_memory_backends_v1_3.py` - Memory connectors
- `src/ml/data_generator.py` - Synthetic data generation
- `src/ml/data_loader.py` - Bulk data loading
- `src/nlp/e2i_fasttext_trainer.py` - Typo-tolerant query matching

### Documentation
- **PRIMARY:** `docs/e2i_nlv_project_structure_v4.1.md` (55KB, 925 lines)
- **SETUP:** `docs/README.md` (Memory system)
- **GAPS:** `docs/e2i_gap_analysis.md` - 14 gaps identified
- **TODO:** `docs/e2i_gap_todo.md` - Implementation tracking
- **NLP:** `docs/e2i_rag_nlp_library_analysis.md` - NLP evaluation

---

## Data Characteristics

### Synthetic Data Package
- **Volume:** ~200 patients, ~50 HCPs, ~30 dashboard users
- **Splits:** Train (60%), Validation (20%), Test (15%), Holdout (5%)
- **Patient-level isolation:** No cross-split contamination
- **Agent activity log:** 3.8MB (e2i_ml_v4_agent_activities.json)

### Reference Data
- **Brands:** Remibrutinib, Fabhalta, Kisqali
- **Regions:** Northeast, South, Midwest, West
- **Specialties:** Allergy, Hematology, Oncology, Cardiology, Rheumatology, Immunology
- **Event Types:** diagnosis, prescription, lab_test, imaging, follow_up, adverse_event

---

## Services & Ports

### Docker Services
- **Redis** (Working Memory): port 6379
- **FalkorDB** (Semantic Graph): port 6380
- **Supabase**: Cloud-hosted

### Environment Variables (.env)
- ANTHROPIC_API_KEY
- SUPABASE_URL
- SUPABASE_KEY
- REDIS_URL
- FALKORDB_URL
- OPIK_API_KEY
- MLFLOW_TRACKING_URI

---

## Development Setup

### Quick Start Commands

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
make install
# Or: pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Start services
make docker-up

# Generate data
make data-generate
```

### Available Make Commands

```bash
make help           # Show all commands
make install        # Install dependencies
make dev-install    # Install with dev tools
make test           # Run tests
make lint           # Check code quality
make format         # Format code
make clean          # Clean artifacts
make docker-up      # Start Docker services
make docker-down    # Stop Docker services
```

---

## Key Capabilities

### Causal Analytics
- DAG construction (NetworkX)
- Effect estimation (propensity score matching, DoWhy)
- 5 refutation tests with gate decisions
- Sensitivity analysis (E-value)
- Heterogeneous treatment effects (CATE)
- Counterfactual simulation

### ML Operations
- Experiment tracking (MLflow)
- Model registry & versioning
- Hyperparameter tuning (Optuna)
- Feature importance (SHAP)
- Data quality monitoring
- Deployment stages (dev → staging → shadow → production)
- LLM observability (Opik)

### Agent System
- Intent-based routing
- Multi-agent orchestration
- Parallel execution
- Response synthesis
- Confidence scoring
- Verification & compliance

### Query Robustness
- Typo normalization via fastText subword embeddings
- Abbreviation expansion (ROI, HCP, TRx, etc.)
- Fuzzy entity matching against domain vocabulary
- Cache invalidation with version-based keys
- 134 test cases for E2I domain terms

---

## Project Status

**Git Status:**
- Branch: main
- Commits: 1 (initial commit)
- Project restructured: 2025-12-15

**Development Stage:**
- Architecture: ✅ Complete (V4.1)
- Database Schema: ✅ Defined (28 tables)
- Configuration: ✅ In place
- Synthetic Data: ✅ Ready
- Memory System: ✅ Documented
- Project Structure: ✅ Reorganized
- Implementation: 🔄 In progress

---

## Next Steps

1. **Clean up legacy folders**
   - Remove: e2i_agentic_memory/, e2i_ml_compliant_data/, e2i_causalrag/

2. **Agent Implementation**
   - Implement 18 agents in src/agents/

3. **Memory System Integration**
   - Wire up tri-memory backends

4. **API Development**
   - Build FastAPI endpoints in src/api/

5. **Frontend Development**
   - Convert HTML mockups to React app

6. **Testing & Validation**
   - Expand test coverage

---

## Notes

- **Claude Model:** claude-sonnet-4-20250514
- **Extended Reasoning:** Available for deep agents
- **Compliance:** Built-in audit trail + expert review workflow
- **Split Enforcement:** Patient-level isolation prevents leakage
- **Documentation:** Comprehensive (120KB+ across all docs)
- **Structure:** Reorganized 2025-12-15 for production readiness

---

**Last Reorganization:** 2025-12-15
**Version:** 4.1.0
