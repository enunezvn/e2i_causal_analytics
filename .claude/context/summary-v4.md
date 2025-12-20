# Context Summaries V4

## Purpose
This file contains compressed summaries of architectural decisions and implementation details. Use these summaries to maintain cross-domain awareness without loading full specialist files.

**Update this file** after completing significant changes in any domain.

---

## System Overview (Updated: 2025-12-18)

E2I Causal Analytics is a **pharmaceutical commercial operations** platform (NOT clinical/medical) that:
- Answers natural language questions about business KPIs
- Performs causal inference on operational data
- Uses **18 AI agents organized in 6 tiers** (V4: added Tier 0 ML Foundation)
- Supports 3 brands: Remibrutinib, Fabhalta, Kisqali
- Integrates 7 MLOps tools for complete ML lifecycle

**Last Updated**: 2025-12-18

---

## Implementation Status (as of 2025-12-18)

**System Maturity**: Infrastructure Complete, Agent Implementation In Progress

### Fully Implemented (3 agents + 7 modules)

**Agents**:
- **orchestrator** (Tier 1) - src/agents/orchestrator/ - Query routing and multi-agent coordination
- **experiment_designer** (Tier 3) - src/agents/experiment_designer/ - A/B test design with digital twin simulation
- **tool_composer** (NEW, not in original spec) - src/agents/tool_composer/ - Multi-tool orchestration for complex queries

**Core Modules**:
- digital_twin (src/digital_twin/) - Patient journey simulation
- memory (src/memory/) - Tri-memory architecture (working, episodic, procedural, semantic)
- nlp (src/nlp/) - Query parsing (business entities only, NOT medical NER)
- api (src/api/) - FastAPI backend with WebSocket support
- utils (src/utils/) - Audit chain and utilities
- causal (src/causal/) - Partial: Core structures defined
- mlops (src/mlops/) - Partial: SHAP realtime explainer only

### Configuration Only (15 agents)

**IMPORTANT**: The following agents are fully configured in `config/agent_config.yaml` with complete specifications, database support, and specialist documentation, but **lack code implementation**:

**Tier 0 (ML Foundation)** - 0/7 implemented:
- scope_definer, data_preparer, feature_analyzer, model_selector, model_trainer, model_deployer, observability_connector

**Tier 2 (Causal Analytics)** - 0/3 implemented:
- causal_impact, gap_analyzer, heterogeneous_optimizer

**Tier 3 (Monitoring)** - 1/3 implemented:
- drift_monitor ❌, experiment_designer ✅, health_score ❌

**Tier 4 (ML Predictions)** - 0/2 implemented:
- prediction_synthesizer, resource_optimizer

**Tier 5 (Self-Improvement)** - 0/2 implemented:
- explainer, feedback_learner

### Infrastructure Status

| Component | Status |
|-----------|--------|
| Database Schema | ✅ 100% Complete (24+ tables) |
| MLOps Tools Config | ✅ 100% Complete (7 tools configured) |
| Agent Configs | ✅ 100% Complete (all 18 agents) |
| Specialist Docs | ✅ 100% Complete (all agents documented) |
| Agent Code | ⚠️ 17% Complete (3 of 18 agents) |
| MLOps Integration | ⚠️ 14% Complete (config only) |

**Overall System Completion**: ~65% (Infrastructure complete, agent implementations in progress)

**See** `.claude/context/implementation-status.md` for detailed roadmap and priorities.

---

## Domain Summaries

### NLP Layer
**Status**: Stable
**Key Decisions**:
- Uses domain vocabulary fuzzy matching (NOT medical NER)
- 5 intent types: CAUSAL, EXPLORATORY, COMPARATIVE, TREND, WHAT_IF
- Claude API as fallback for ambiguous queries
- Entities: brands, regions, KPIs, time_periods, hcp_ids

**Recent Changes**: None

### Causal Engine
**Status**: Stable
**Key Decisions**:
- DoWhy for causal model definition
- EconML for heterogeneous effects (CATE)
- NetworkX for DAG construction
- All estimates require refutation tests
- Assumptions must be documented

**Recent Changes**: None

### RAG System
**Status**: Stable
**Key Decisions**:
- Hybrid retrieval: dense (0.5) + sparse (0.3) + graph (0.2)
- Indexes ONLY operational data (NO medical literature)
- Sources: causal_paths, agent_activities, business_metrics, triggers
- Cross-encoder reranking
- Integrates with Feedback Learner for weight optimization

**Recent Changes**: Added Feedback Learner integration for DSPy optimization

### Agent Architecture
**Status**: V4 (18 agents, 6 tiers) - Specialist Files Complete
**Key Decisions**:
- **6-tier hierarchy** for routing priority (NEW: Tier 0 ML Foundation)
- All agents inherit from BaseAgent
- LangGraph for state management
- Agent registry in database for runtime routing
- Orchestrator never performs analysis, only coordinates
- Each agent has dedicated specialist file (CLAUDE.md) with LangGraph implementation
- **QC Gate** in data_preparer blocks downstream training on failure

**Tier Structure (V4)**:
```
Tier 0: scope_definer, data_preparer, model_selector, model_trainer, 
        feature_analyzer, model_deployer, observability_connector
Tier 1: orchestrator
Tier 2: causal_impact, gap_analyzer, heterogeneous_optimizer
Tier 3: experiment_designer, drift_monitor, health_score
Tier 4: prediction_synthesizer, resource_optimizer
Tier 5: explainer, feedback_learner
```

**Agent Types**:
- Standard: Linear flow, minimal/no LLM (13 agents)
- Hybrid: Computation + Deep reasoning (3 agents: Causal Impact, Experiment Designer, **Feature Analyzer**)
- Deep: Extended reasoning, async capable (2 agents: Explainer, Feedback Learner)

**Recent Changes (V4)**:
- Added Tier 0 ML Foundation with 7 new agents
- Total agents increased from 11 to 18
- Added 8 new ml_ database tables
- Integrated 7 MLOps tools
- Created CLAUDE.md files for all Tier 0 agents

### MLOps Integration (NEW in V4)
**Status**: Architecture Complete
**Key Decisions**:
- 7 MLOps tools integrated across Tier 0 agents
- Decoupled model serving (agents call endpoints via tools)
- MLflow for experiment tracking and model registry
- Opik for LLM/agent observability
- Great Expectations for data quality validation (QC Gate)
- Feast for feature store
- Optuna for hyperparameter optimization
- SHAP for model interpretability
- BentoML for model serving

**Tool-Agent Mapping**:
| Tool | Primary Agents |
|------|----------------|
| MLflow | model_trainer, model_selector, model_deployer |
| Opik | observability_connector, feature_analyzer |
| Great Expectations | data_preparer |
| Feast | data_preparer, model_trainer |
| Optuna | model_trainer |
| SHAP | feature_analyzer |
| BentoML | model_deployer |

**Recent Changes**: Initial integration in V4

### Database Schema
**Status**: V4 (ML-compliant, 8 new tables)
**Key Decisions**:
- Supabase (PostgreSQL + pgvector)
- Split-aware repositories (train/validation/test/holdout)
- Patient-based split assignment (hash-based)
- 6 tables in V3 for KPI coverage + **8 new ml_ tables in V4**
- 8 KPI helper views
- **18-agent enum** in agent_activity_type

**V4 New Tables (ML Foundation)**:
- ml_experiments (scope_definer)
- ml_data_quality_reports (data_preparer)
- ml_feature_store (data_preparer)
- ml_model_registry (model_selector, model_deployer)
- ml_training_runs (model_trainer)
- ml_shap_analyses (feature_analyzer)
- ml_deployments (model_deployer)
- ml_observability_spans (observability_connector)

**V3 Tables (unchanged)**:
- user_sessions (MAU/WAU/DAU)
- data_source_tracking (match rates)
- ml_annotations (label quality)
- etl_pipeline_metrics (time-to-release)
- hcp_intent_surveys (intent delta)
- reference_universe (coverage)

**Recent Changes**: V4 schema migration (migration 007) with 8 ml_ tables

### Memory Architecture (V4)
**Status**: Architecture Complete
**Key Decisions**:
- **Working Memory** (Redis): All 18 agents, active context (TTL: 24h)
- **Episodic Memory** (Supabase/pgvector): 7 agents (6 Tier 0 + drift_monitor + feedback_learner)
- **Procedural Memory** (Supabase/pgvector): 5 agents for successful patterns
- **Semantic Memory** (FalkorDB/Graphity): feature_analyzer, causal_impact, explainer

**Memory by Tier 0 Agent**:
| Agent | Working | Episodic | Procedural | Semantic |
|-------|---------|----------|------------|----------|
| scope_definer | ✓ | ✓ | ✓ | - |
| data_preparer | ✓ | ✓ | ✓ | - |
| model_selector | ✓ | ✓ | ✓ | - |
| model_trainer | ✓ | ✓ | ✓ | - |
| feature_analyzer | ✓ | ✓ | ✓ | ✓ |
| model_deployer | ✓ | ✓ | - | - |
| observability_connector | ✓ | - | - | - |

**Recent Changes**: Memory architecture defined for Tier 0

### API Layer
**Status**: Stable
**Key Decisions**:
- FastAPI with Pydantic v2
- WebSocket for streaming responses
- JWT auth via Supabase
- All endpoints under /api/v1/
- Response time target: <2s (95th percentile)

**Recent Changes**: None

### Frontend
**Status**: Stable
**Key Decisions**:
- React 18 + TypeScript 5
- Redux Toolkit for state
- D3.js for causal graph visualization
- Recharts for standard charts
- Agent badges with tier-based coloring
- AgentTierView component for **6-tier visualization** (V4)

**Recent Changes**: Updated agentSlice for 18-agent state

### Observability
**Status**: Architecture Complete (V4)
**Key Decisions**:
- **Opik** for span tracking across all 18 agents
- Latency histograms (p50, p95, p99) per agent
- Token usage tracking by model tier
- Error rate monitoring with category breakdown
- Fallback frequency metrics
- **observability_connector** agent for cross-cutting telemetry

**Recent Changes**: observability_connector agent added in V4

### Tool Composer (NEW)
**Status**: Implemented
**Key Decisions**:
- Multi-tool orchestration for complex queries requiring multiple tools
- NOT in original 18-agent specification (added during development)
- Pipeline: Decomposer → Planner → Composer → Executor → Synthesizer
- Tool Registry for dynamic tool discovery and composition
- Schema validation for tool inputs/outputs using Pydantic models
- Supports parallel and sequential tool execution
- Database support via tool_composer_tables.sql

**Components**:
- **Decomposer**: Breaks complex queries into atomic sub-tasks
- **Planner**: Determines optimal tool execution order and dependencies
- **Composer**: Combines multiple tools into execution graph
- **Executor**: Runs tool chains with error recovery
- **Synthesizer**: Aggregates multi-tool results into coherent response

**Use Cases**:
- Complex analytical queries spanning multiple domains
- Cross-domain data integration (e.g., causal + prediction + gap analysis)
- What-if scenarios requiring tool chaining

**Code Location**: src/agents/tool_composer/
**Database**: database/ml/013_tool_composer_tables.sql

**Recent Changes**: Initial implementation (2025-12)

### Digital Twin System (NEW)
**Status**: Implemented
**Key Decisions**:
- Patient journey simulation for what-if scenarios and counterfactual analysis
- Fidelity tracking to measure simulation accuracy vs. real-world outcomes
- Integration with experiment_designer for intervention simulation
- Twin generation from historical patient journey data
- Separate database schema for twin metadata and simulations

**Components**:
- **twin_generator.py**: Creates patient twins from historical data
- **simulation_engine.py**: Runs what-if scenarios on twins
- **fidelity_tracker.py**: Monitors twin-to-reality alignment
- **twin_repository.py**: Persists and retrieves twins

**Fidelity Metrics**:
- Trajectory similarity (Fréchet distance)
- Outcome alignment (binary classification metrics)
- Temporal correlation (time-series correlation)
- Feature distribution match (KL divergence)

**Use Cases**:
- Testing interventions before real-world deployment
- Causal experiment design (pre-validation of A/B tests)
- Rare event simulation (e.g., Fabhalta PNH patient scenarios)
- Counterfactual outcome estimation

**Code Location**: src/digital_twin/
**Database**: database/ml/012_digital_twin_tables.sql

**Integration Points**:
- **experiment_designer**: Uses twins to simulate interventions before trials
- **causal_impact**: Validates causal estimates via twin simulation
- **prediction_synthesizer**: Tests prediction models on synthetic twins

**Recent Changes**: Initial implementation (2025-12)

### Audit Chain (NEW)
**Status**: Database Schema Complete
**Key Decisions**:
- End-to-end audit trail for all agent activities and data transformations
- Immutable audit log for compliance and debugging
- Links agent executions to data lineage and model predictions

**Database**: database/audit/011_audit_chain_tables.sql

**Recent Changes**: Schema added (2025-12), agent integration pending

---

## Active Constraints

### ML Split Enforcement
- **Train**: 60% (development and training)
- **Validation**: 20% (hyperparameter tuning)
- **Test**: 15% (final evaluation - touched ONCE)
- **Holdout**: 5% (never used in development)
- **Rule**: Same patient always in same split
- **Preprocessing**: Fit on train ONLY, transform all splits

### QC Gate Enforcement (V4)
- **Location**: data_preparer agent
- **Tool**: Great Expectations
- **Behavior**: Training BLOCKED if QC fails
- **Severity**: "error" level blocks, "warning" level warns
- **Required**: model_trainer MUST verify QC passed before training

### Model Stage Progression (V4)
```
DEVELOPMENT → STAGING → SHADOW (24h min) → PRODUCTION → ARCHIVED
```
- No skipping stages
- Shadow mode required before production
- All success criteria must be met for production promotion

### KPI Coverage
- **Total KPIs**: 46 (100% calculable in V4)
- **Categories**: business, model_performance, data_quality, engagement, operational

### RAG Scope
- **Indexed**: causal_paths, agent_activities, business_metrics, triggers, conversations
- **NOT indexed**: clinical trials, medical literature, drug info, regulatory docs

### Agent Latency Budgets (V4)
| Tier | Budget | Agents |
|------|--------|--------|
| 0 | Variable | scope_definer (5s), data_preparer (60s), model_selector (120s), model_trainer (varies), feature_analyzer (120s), model_deployer (30s), observability_connector (100ms async) |
| 1 | <2s (strict) | orchestrator |
| 2 | <120s | causal_impact, gap_analyzer, heterogeneous_optimizer |
| 3 | <60s | experiment_designer, drift_monitor, health_score |
| 4 | <20s | prediction_synthesizer, resource_optimizer |
| 5 | No hard limit | explainer, feedback_learner |

---

## Integration Points

### Critical Interfaces
1. NLP → Orchestrator: `ParsedQuery`
2. RAG → Agents: `RetrievalContext`
3. Agents → Orchestrator: `AgentState`
4. API → Frontend: `ChatResponse` / WebSocket chunks
5. Repositories → All: Split-aware queries
6. Feedback Learner → RAG: Weight optimization (async)

**New in V4 (Tier 0 → Tier 1-5)**:
7. data_preparer → drift_monitor: Baseline metrics
8. model_trainer → prediction_synthesizer: Model artifacts
9. feature_analyzer → causal_impact: Feature relationships (semantic memory)
10. model_deployer → prediction_synthesizer: Endpoint URLs
11. observability_connector → health_score: Quality metrics

**New in V4.1 (Additional Modules)**:
12. tool_composer → orchestrator: Complex multi-tool queries
13. digital_twin → experiment_designer: Intervention simulation and validation
14. digital_twin → causal_impact: Counterfactual validation via simulation
15. digital_twin → prediction_synthesizer: Synthetic patient test data
16. audit_chain → All agents: Immutable audit trail for compliance
17. experiment_designer → digital_twin: Request simulation for what-if scenarios

### Data Flow (V4)

```
User Query
    → NLP (parse, classify intent)
    → RAG (retrieve context)
    → Orchestrator (route to agent)
    → Agent (analyze)
    → Orchestrator (synthesize)
    → API (format response)
    → Frontend (render)
    → Feedback Learner (async learn)

ML Training Flow (Tier 0):
    scope_definer (define problem)
    → data_preparer (validate, QC GATE)
    → model_selector (choose algorithm)
    → model_trainer (train model)
    → feature_analyzer (interpret SHAP)
    → model_deployer (deploy to production)
    ↓
    observability_connector (spans throughout)
```

### Tier 0 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TIER 0 ML PIPELINE                             │
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │  SCOPE   │──►│   DATA   │──►│  MODEL   │──►│  MODEL   │             │
│  │ DEFINER  │   │ PREPARER │   │ SELECTOR │   │ TRAINER  │             │
│  └──────────┘   └────┬─────┘   └──────────┘   └────┬─────┘             │
│                      │                              │                   │
│                      ▼                              ▼                   │
│                 ┌─────────┐                   ┌──────────┐              │
│                 │QC GATE  │                   │ FEATURE  │              │
│                 │(blocks) │                   │ ANALYZER │              │
│                 └─────────┘                   └────┬─────┘              │
│                                                    │                    │
│                                                    ▼                    │
│                                              ┌──────────┐               │
│                                              │  MODEL   │               │
│                                              │ DEPLOYER │               │
│                                              └──────────┘               │
│                                                                         │
│              OBSERVABILITY CONNECTOR (cross-cutting spans)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Open Decisions

### Pending
- [ ] Real-time drift alerting mechanism
- [ ] Feedback learner training schedule
- [ ] Health score threshold calibration
- [ ] Tier 0 agent implementation order
- [ ] Production deployment timeline

### Recently Resolved (V4)
- [x] V4 agent tier structure (6 tiers, 18 agents)
- [x] Tier 0 ML Foundation agent design (7 agents)
- [x] MLOps tool selection (7 tools)
- [x] Database schema extension (8 new ml_ tables)
- [x] QC gate enforcement pattern
- [x] Model stage progression rules
- [x] Memory architecture for Tier 0
- [x] CLAUDE.md files for all Tier 0 agents
- [x] Tier 0 contracts specification

### Previously Resolved (V3)
- [x] V3 agent tier structure (5 tiers)
- [x] KPI gap resolution (6 new tables)
- [x] Split assignment algorithm (hash-based)
- [x] Agent specialist file structure (individual files per agent)
- [x] Experiment Designer tier assignment (Tier 3: Design & Monitoring)

---

## Documentation Index (V4)

### Master Indexes
| Document | Purpose |
|----------|---------|
| AGENT-INDEX-V4.md | Master agent navigation (18 agents, 6 tiers) |
| SPECIALIST-INDEX-V4.md | Specialist file navigation |

### Contract Files
| Document | Purpose |
|----------|---------|
| tier0-contracts.md | ML Foundation agent contracts (NEW) |
| tier2-contracts.md | Causal inference agent contracts |
| tier3-contracts.md | Design & monitoring agent contracts |
| tier4-contracts.md | ML prediction agent contracts |
| tier5-contracts.md | Self-improvement agent contracts |

### Specialist Files (CLAUDE.md)
| Tier | Agents | Location |
|------|--------|----------|
| 0 | 7 + overview + mlops | specialists/ml_foundation/ |
| 1 | 1 | specialists/orchestrator-agent.md |
| 2 | 3 | specialists/*.md |
| 3 | 3 | specialists/*.md |
| 4 | 2 | specialists/*.md |
| 5 | 2 | specialists/*.md |

### Context Files
| Document | Purpose |
|----------|---------|
| summary-v4.md | This file - cross-domain context |
| implementation-status.md | Agent/module implementation tracking |
| experiment-history.md | Historical experiment outcomes |
| brand-context.md | Brand-specific context |
| kpi-dictionary.md | KPI definitions |
| mlops-tools.md | MLOps configuration |
| CONTEXT_AUDIT_REPORT.md | Context file audit findings |
| CHANGELOG.md | Version history and change tracking |
| CODE_REFERENCE.md | Cross-reference index linking docs to code |

### Specialist Files
| Document | Purpose |
|----------|---------|
| tool-composer.md | Tool Composer specialist instructions |
| digital-twin.md | Digital Twin system specialist instructions |

### Documentation Status
✅ **Complete** - All context files and specialist files created
- 9 context files in `.claude/context/`
- 2 new specialist files for tool_composer and digital_twin
- Cross-reference index for code navigation

---

## Update Log

| Date | Domain | Change | Impact |
|------|--------|--------|--------|
| 2025-12-18 | Docs | Created tool-composer.md specialist file | Complete specialist instructions for Tool Composer |
| 2025-12-18 | Docs | Created digital-twin.md specialist file | Complete specialist instructions for Digital Twin system |
| 2025-12-18 | Docs | Created CODE_REFERENCE.md | Cross-reference index linking all documentation to code locations |
| 2025-12-18 | Docs | Updated summary-v4.md documentation index | Removed "Missing Documentation" section - 100% complete |
| 2025-12-18 | Docs | Added Implementation Status section to summary-v4.md | Transparency on agent implementation progress |
| 2025-12-18 | Docs | Created implementation-status.md | Detailed implementation roadmap |
| 2025-12-18 | Docs | Created CONTEXT_AUDIT_REPORT.md | Context file audit findings |
| 2025-12-18 | Docs | Deleted summary-mlops.md (unmodified template) | Removed confusing template file |
| 2025-12-18 | Agents | Documented tool_composer agent (not in original spec) | Multi-tool orchestration |
| 2025-12-18 | System | Documented digital_twin system | Patient journey simulation |
| 2025-12-18 | Database | Documented audit_chain tables | Audit trail support |
| 2025-12-18 | Integration | Added 6 new integration points (V4.1) | tool_composer, digital_twin, audit_chain |
| 2025-12-08 | Agents | V4: Added Tier 0 ML Foundation (7 agents) | 18 total agents |
| 2025-12-08 | Database | V4: 8 new ml_ tables (migration 007) | ML lifecycle support |
| 2025-12-08 | MLOps | V4: Integrated 7 MLOps tools | Complete ML toolchain |
| 2025-12-08 | Docs | V4: CLAUDE.md files for Tier 0 | Agent implementation specs |
| 2025-12-08 | Docs | V4: tier0-contracts.md | Integration contracts |
| 2025-12-08 | Docs | V4: Updated indexes and summary | Navigation updated |
| 2025-12-04 | Agents | Agent specialist files created | Individual documentation |
| 2025-12-04 | Agents | AGENT-INDEX.md and SPECIALIST-INDEX.md added | Master indexes |
| 2025-12-04 | Agents | Standardized Experiment Designer to Tier 3 | Consistent tier assignment |

---

## How to Update This File

After completing a significant change:

1. Update the relevant **Domain Summary** section
2. Add entry to **Update Log**
3. If architecture changed, update **Integration Points**
4. If new constraints, add to **Active Constraints**
5. If decisions pending, add to **Open Decisions**

Keep summaries to 3-5 bullet points max per domain.

---

## Quick Reference

### Agent Count by Tier
| Tier | Name | Count |
|------|------|-------|
| 0 | ML Foundation | 7 |
| 1 | Orchestration | 1 |
| 2 | Causal Inference | 3 |
| 3 | Design & Monitoring | 3 |
| 4 | ML Predictions | 2 |
| 5 | Self-Improvement | 2 |
| **Total** | | **18** |

### Agent Types Distribution
| Type | Count | Examples |
|------|-------|----------|
| Standard | 13 | orchestrator, scope_definer, model_trainer |
| Hybrid | 3 | causal_impact, experiment_designer, feature_analyzer |
| Deep | 2 | explainer, feedback_learner |

### Database Tables by Version
| Version | Tables | Purpose |
|---------|--------|---------|
| V1-V2 | Core tables | Base functionality |
| V3 | +6 tables | KPI coverage |
| V4 | +8 ml_ tables | ML lifecycle |
