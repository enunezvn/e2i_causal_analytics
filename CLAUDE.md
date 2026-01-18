# E2I Causal Analytics - Claude Code Development Framework

**Project**: E2I Causal Analytics
**Type**: Complex ML/MLOps System with Multi-Agent Architecture
**Framework**: Unified Claude Code Framework v3.0 + E2I Extensions

---

## Project Overview

**E2I Causal Analytics** is a Natural Language Visualization platform with Self-Improving Agentic RAG for pharmaceutical commercial operations. It features an 18-agent tiered architecture (6 tiers) with 100% KPI coverage and ML-compliant schema.

**Domain**: Pharmaceutical commercial analytics (NOT clinical/medical)
**Brands**: Remibrutinib (CSU), Fabhalta (PNH), Kisqali (HR+/HER2- breast cancer)

---

## 🚨 ORCHESTRATION PROTOCOL (READ FIRST)

This project uses the **Claude Code Framework 4-Layer Orchestration Architecture** to prevent:
- Context Window Death Spiral (architectural drift after ~2,000 tokens)
- Permission Interrupt Cascade (fragmented context)
- Agent Collision Syndrome (incompatible implementations)

### Layer 1: Orchestrator (This File)
**CRITICAL**: This orchestrator NEVER writes code directly. It:
1. Decomposes tasks into specialist domains
2. Routes to appropriate specialist instructions
3. Validates integration contracts
4. Synthesizes multi-agent outputs

### Layer 2: Context Management (.claude/context/)
**Purpose**: Prevent context window bloat

- Load ONLY the specialist file relevant to current task
- Use context summaries for cross-domain awareness
- Compress implementation details into decisions
- Maintain domain boundaries

**E2I Context Files**:
- `summary-v4.md` - E2I project summary
- `brand-context.md` - Brand-specific information
- `kpi-dictionary.md` - KPI definitions
- `experiment-history.md` - Experiment tracking
- `mlops-tools.md` - MLOps stack information

### Layer 3: Specialist Execution (.claude/specialists/)
**Purpose**: Domain-specific expertise

- Each domain has isolated instructions
- Specialists own their bounded context
- Handoffs use structured protocols
- Prevent architectural drift

**E2I Specialists Available**:
- Agent specialists (Tiers 0-5)
- System specialists (NLP, Causal, RAG, API, Frontend, Database)
- ML foundation specialists
- MLOps integration specialists

### Layer 4: Integration Validation (.claude/contracts/)
**Purpose**: Ensure compatibility

- Contract definitions for interfaces
- Pre-commit validation rules
- Interface compatibility checks
- Quality gates

**E2I Contracts Available**:
- Base structures
- Orchestrator contracts
- Tier-specific contracts (Tiers 0-5)
- Integration contracts

---

## Task Decomposition Rules

When receiving a task, ALWAYS:

### 1. Classify the Task Domain(s)

#### E2I-Specific Domains

**System Layer Specialists**:
```
NLP Layer           → .claude/specialists/system/nlp.md
Causal Engine       → .claude/specialists/system/causal.md
RAG System          → .claude/specialists/system/rag.md
API/Backend         → .claude/specialists/system/api.md
Frontend            → .claude/specialists/system/frontend.md
Database/Schema     → .claude/specialists/system/database.md
Testing             → .claude/specialists/system/testing.md
DevOps              → .claude/specialists/system/devops.md
```

**Agent Specialists** (18 agents in 6 tiers):
```
Tier 0 (Foundation) → .claude/specialists/Agent_Specialists_Tier 0/
                    → 7 agents: scope_definer, data_preparer, feature_analyzer,
                      model_selector, model_trainer, model_deployer, observability_connector
Tier 1 (Orchestrator) → .claude/specialists/Agent_Specialists_Tiers 1-5/orchestrator-agent.md
                      → 2 agents: orchestrator, tool_composer
Tier 2 (Causal)     → .claude/specialists/Agent_Specialists_Tiers 1-5/causal-impact.md
                    → .claude/specialists/Agent_Specialists_Tiers 1-5/gap-analyzer.md
                    → .claude/specialists/Agent_Specialists_Tiers 1-5/heterogeneous-optimizer.md
                    → 3 agents: causal_impact, gap_analyzer, heterogeneous_optimizer
Tier 3 (Monitoring) → .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md
                    → .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md
                    → .claude/specialists/Agent_Specialists_Tiers 1-5/health-score.md
                    → 3 agents: drift_monitor, experiment_designer, health_score
Tier 4 (ML)         → .claude/specialists/Agent_Specialists_Tiers 1-5/prediction-synthesizer.md
                    → .claude/specialists/Agent_Specialists_Tiers 1-5/resource-optimizer.md
                    → 2 agents: prediction_synthesizer, resource_optimizer
Tier 5 (Learning)   → .claude/specialists/Agent_Specialists_Tiers 1-5/explainer.md
                    → .claude/specialists/Agent_Specialists_Tiers 1-5/feedback-learner.md
                    → 2 agents: explainer, feedback_learner
```

**ML Foundation**:
```
Data Preparation    → .claude/specialists/ml_foundation/data_preparer.md
Feature Engineering → .claude/specialists/ml_foundation/feature_engineer.md
Model Training      → .claude/specialists/ml_foundation/model_trainer.md
(See ml_foundation directory for complete list)
```

**Framework-Provided Specialists** (use when needed):
```
General ML Training  → .claude/specialists/model-training.md
Data Engineering     → .claude/specialists/data-engineering.md
Feature Engineering  → .claude/specialists/feature-engineering.md
Model Evaluation     → .claude/specialists/model-evaluation.md
MLOps Pipeline       → .claude/specialists/mlops-pipeline.md
```

### 2. Reference Agent Documentation

**Framework Agent Docs** (.claude/.agent_docs/):
- `anti-patterns.md` - Code smells to avoid
- `coding-patterns.md` - Best practices
- `error-handling.md` - Error handling conventions
- `testing-patterns.md` - Testing strategies
- `ml-patterns.md` - ML-specific patterns (data leakage, experiment tracking, etc.)
- `bug-investigation.md` - Debugging protocol
- `code-review-checklist.md` - Review checklist

**E2I Agent Index**:
- `AGENT-INDEX-V4.md` - Master agent index
- `SPECIALIST-INDEX-V4.md` - Specialist patterns and contracts

### 3. Check Cross-Domain Dependencies

Use `.claude/contracts/` to identify interfaces and dependencies

### 4. Execute in Waves

Max 2 domains per wave to manage context window

### 5. Validate Integration

Check contracts before completing

---

## Project Architecture Summary

```
e2i_causal_analytics/
├── .claude/                        # Claude Code Framework + E2I Extensions
│   ├── .agent_docs/               # Framework patterns
│   ├── agents/                    # BMAD planning agents (architect, PM, security)
│   ├── specialists/               # E2I + framework specialists
│   ├── contracts/                 # E2I integration contracts
│   ├── context/                   # E2I project context
│   ├── PRPs/                      # Product Requirements Plans
│   └── scripts/                   # Framework utility scripts
├── config/                         # YAML configurations (18 agents, 46 KPIs)
├── src/
│   ├── nlp/                       # Layer 1: Query processing (NO medical NER)
│   ├── causal_engine/             # Layer 2: DoWhy/EconML causal inference
│   ├── rag/                       # Layer 2: CausalRAG (operational insights only)
│   ├── agents/                    # Layer 3: 18 agents in 6 tiers
│   │   ├── tier_0/                # Tier 0: ML Foundation (7 agents)
│   │   ├── orchestrator/          # Tier 1: Coordination (2 agents)
│   │   ├── tool_composer/         # Tier 1: Tool orchestration
│   │   ├── causal_impact/         # Tier 2: Causal Analytics (3 agents)
│   │   ├── gap_analyzer/
│   │   ├── heterogeneous_optimizer/
│   │   ├── drift_monitor/         # Tier 3: Monitoring (3 agents)
│   │   ├── experiment_designer/
│   │   ├── health_score/
│   │   ├── prediction_synthesizer/ # Tier 4: ML Predictions (2 agents)
│   │   ├── resource_optimizer/
│   │   ├── explainer/             # Tier 5: Self-Improvement (2 agents)
│   │   └── feedback_learner/
│   ├── repositories/              # Data access layer (split-aware)
│   ├── api/                       # FastAPI backend
│   ├── digital_twin/              # Digital Twin simulation engine
│   ├── feature_store/             # Feature store client & retrieval
│   ├── memory/                    # Cognitive workflow & memory backends
│   ├── ml/                        # ML utilities (data generation, loading)
│   ├── mlops/                     # MLOps tooling (SHAP explainer)
│   ├── tool_registry/             # Agent tool registry
│   ├── utils/                     # Utilities (audit chain)
│   └── workers/                   # Background workers (Celery)
├── frontend/                       # React + TypeScript (root level)
├── tests/                         # unit/integration/e2e
├── scripts/                       # DB setup, data loading
├── docs/                          # Documentation
├── docker/                        # Docker configurations
├── database/                      # Database scripts
├── CLAUDE.md                      # This file
├── FRAMEWORK_README.md            # Framework documentation
└── README.md                      # Project README
```

---

## Agent Tier Reference

| Tier | Agent | Responsibility | Specialist File |
|------|-------|----------------|-----------------|
| 0 | Scope Definer | Define ML problem scope | See Agent_Specialists_Tier 0/ |
| 0 | Data Preparer | Data preparation & validation | See Agent_Specialists_Tier 0/ |
| 0 | Feature Analyzer | Feature engineering & selection | See Agent_Specialists_Tier 0/ |
| 0 | Model Selector | Model selection & benchmarking | See Agent_Specialists_Tier 0/ |
| 0 | Model Trainer | Model training & hyperparameter tuning | See Agent_Specialists_Tier 0/ |
| 0 | Model Deployer | Model deployment & versioning | See Agent_Specialists_Tier 0/ |
| 0 | Observability Connector | Connect MLflow, Opik, monitoring | See Agent_Specialists_Tier 0/ |
| 1 | Orchestrator | Coordinates all agents, routes queries | orchestrator-agent.md |
| 1 | Tool Composer | Multi-faceted query decomposition & tool orchestration | tool-composer-agent.md |
| 2 | Causal Impact | Traces causal chains, effect estimation | causal-impact.md |
| 2 | Gap Analyzer | ROI opportunity detection | gap-analyzer.md |
| 2 | Heterogeneous Optimizer | Segment-level CATE analysis | heterogeneous-optimizer.md |
| 3 | Drift Monitor | Data/model drift detection | drift-monitor.md |
| 3 | Experiment Designer | A/B test design with Digital Twin pre-screening | experiment-designer.md |
| 3 | Health Score | System health metrics | health-score.md |
| 4 | Prediction Synthesizer | ML prediction aggregation | prediction-synthesizer.md |
| 4 | Resource Optimizer | Resource allocation | resource-optimizer.md |
| 5 | Explainer | Natural language explanations | explainer.md |
| 5 | Feedback Learner | Self-improvement from feedback | feedback-learner.md |

---

## BMAD Planning Agents

In addition to the E2I domain-specific agents, the framework provides BMAD-inspired planning agents for development phases:

| Agent | Purpose | When to Use | Command |
|-------|---------|-------------|---------|
| **Architect** | System architecture and technical design | Design systems, make technology decisions, plan for scale, create architecture documentation | `/architect` |
| **Product Manager** | Product strategy and requirements | Define features, create PRDs, prioritize work, gather requirements | `/pm` |
| **Security** | Security review and threat modeling | Security assessment, OWASP compliance, threat analysis, vulnerability review | `/security` |
| **Code Reviewer** | Implementation quality review | Code review before commits, quality checks | (use code-reviewer agent) |
| **Documentation Expert** | Technical documentation | Create/update comprehensive documentation, API references | (use documentation-expert agent) |

**Key Features**:
- BMAD-inspired persona system for specialized expertise
- Integration with E2I conventions and specialists
- Structured workflows for planning and design
- Decision documentation templates stored in `.claude/PRPs/`

**Document Sharding**: For large planning documents (>20k tokens), use `/shard-doc` to split them into manageable sections, improving token efficiency by up to 90%.

---

## Technology Stack

### Backend & ML
- **Backend**: Python 3.11+, FastAPI, Pydantic
- **Causal Inference**: DoWhy, EconML, NetworkX
- **ML Framework**: LangGraph, Claude API
- **RAG**: Hybrid retrieval (vector + sparse + graph)
- **Database**: Supabase (PostgreSQL)
- **Experiment Tracking**: MLflow (framework integrated)

### Frontend
- **Frontend**: React, TypeScript, Redux Toolkit
- **Testing**: Pytest (backend), Vitest (frontend)

### Prompt Optimization
- **GEPA**: Generative Evolutionary Prompting with AI (replaced MIPROv2)
- **Module Location**: `src/optimization/gepa/`
- **Configuration**: `config/gepa_config.yaml`

### Infrastructure
- **Cloud Provider**: DigitalOcean
- **Droplet**: e2i-analytics-prod
- **Region**: NYC3 (New York)
- **Specs**: 4 vCPU, 16 GB RAM, 200 GB SSD (Ubuntu 24.04 LTS)
- **Public IP**: 138.197.4.36
- **Reference**: See `INFRASTRUCTURE.md` for full details

---

## Infrastructure Reference

This project is deployed on a DigitalOcean droplet. Key details:

| Property | Value |
|----------|-------|
| **Droplet Name** | e2i-analytics-prod |
| **Droplet ID** | 544907207 |
| **Public IPv4** | 138.197.4.36 |
| **Region** | NYC3 (New York) |
| **OS** | Ubuntu 24.04 LTS x64 |
| **Specs** | 4 vCPU, 8 GB RAM, 160 GB SSD |

### SSH Access
```bash
# Using configured SSH key (as non-root user)
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Or with SSH config alias
ssh e2i-prod
```

### Public Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://138.197.4.36/ | React dashboard (nginx proxy to port 5174) |
| **Backend API** | http://138.197.4.36:8000 | FastAPI backend |
| **API Docs** | http://138.197.4.36:8000/docs | Swagger/OpenAPI documentation |

### Virtual Environment (REQUIRED)

The production environment uses a Python virtual environment. **Always activate or reference the venv when running commands on the droplet.**

```bash
# Venv location
/opt/e2i_causal_analytics/venv/

# Activate venv on droplet
source /opt/e2i_causal_analytics/venv/bin/activate

# Or run commands directly with venv Python
/opt/e2i_causal_analytics/venv/bin/python <script.py>
/opt/e2i_causal_analytics/venv/bin/pytest <tests/>
```

### Common doctl Commands
```bash
# Authenticate (token in .env as DIGITALOCEAN_TOKEN)
source .env && doctl auth init --access-token "$DIGITALOCEAN_TOKEN"

# Droplet status
doctl compute droplet get 544907207

# Reboot/power cycle
doctl compute droplet-action reboot 544907207
doctl compute droplet-action power-off 544907207
doctl compute droplet-action power-on 544907207

# Create snapshot backup
doctl compute droplet-action snapshot 544907207 --snapshot-name "backup-$(date +%Y%m%d)"
```

**Full documentation**: See `INFRASTRUCTURE.md` for complete reference including firewall setup, SSH key management, and cost information.

### Remote Command Execution (via SSH)

Run commands on the droplet from your local machine using the production venv:

```bash
# Run tests on droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36 "cd /opt/e2i_causal_analytics && /opt/e2i_causal_analytics/venv/bin/pytest tests/unit/test_utils/ -v -n 4"

# Run Python commands
ssh -i ~/.ssh/replit enunez@138.197.4.36 "/opt/e2i_causal_analytics/venv/bin/python -c 'import src; print(src)'"
```

**Note**: `~/Projects/e2i_causal_analytics/` is for code syncing only. Always use `/opt/e2i_causal_analytics/` for execution.

### Droplet Service Discovery (IMPORTANT)

**ALWAYS check for running services before rebuilding or installing dependencies locally.** The production droplet typically has all services running via Docker.

```bash
# Check running containers and services
ssh -i ~/.ssh/replit enunez@138.197.4.36 "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"

# Check API health (primary backend on port 8000)
ssh -i ~/.ssh/replit enunez@138.197.4.36 "curl -s localhost:8000/health | python3 -m json.tool"

# Test agent endpoints via API (not direct imports)
ssh -i ~/.ssh/replit enunez@138.197.4.36 "curl -s -X POST localhost:8000/api/experiments/monitor -H 'Content-Type: application/json' -d '{\"check_all_active\": true}'"
```

**Key Services**:
| Service | Port | Purpose |
|---------|------|---------|
| E2I API | 8000 | Main FastAPI backend (uvicorn) |
| Opik Backend | 8001 | Agent observability API |
| MLflow | 5000 | Experiment tracking |
| Opik | 5173/8080 | Agent observability |
| Redis | 6382 | Working memory cache |
| FalkorDB | 6381 | Graph database |

**Do NOT**:
- Install dependencies (pip install) when testing - the venv is already configured
- Import agents directly in Python when testing - use the API endpoints
- Rebuild Docker containers unless specifically requested

**Project Location on Droplet**: `~/Projects/e2i_causal_analytics/`

---

## GEPA Prompt Optimization (V4.3)

The DSPy prompt optimization has been migrated from MIPROv2 to GEPA for 10%+ performance improvement:

### Architecture
```
src/optimization/gepa/
├── metrics/                    # Agent-specific GEPA metrics
│   ├── base.py                # E2IGEPAMetric protocol
│   ├── causal_impact_metric.py
│   ├── experiment_designer_metric.py
│   ├── feedback_learner_metric.py
│   └── standard_agent_metric.py
├── tools/causal_tools.py      # DoWhy/EconML tool definitions
├── integration/               # MLOps integrations
│   ├── mlflow_integration.py
│   ├── opik_integration.py
│   └── ragas_feedback.py
├── optimizer_setup.py         # Factory function
├── versioning.py              # Module versioning
└── ab_test.py                 # A/B testing
```

### Key Features
- **Budget Presets**: light (quick), medium (balanced), heavy (thorough)
- **Agent-specific Metrics**: Tailored evaluation for each agent tier
- **Tool Optimization**: Joint optimization of DoWhy/EconML tool selection
- **MLOps Integration**: MLflow logging, Opik tracing, RAGAS feedback

### Usage
```python
from src.optimization.gepa import create_optimizer_for_agent, get_metric_for_agent

# Create optimizer for specific agent
optimizer = create_optimizer_for_agent(
    agent_name="causal_impact",
    trainset=training_data,
    valset=validation_data,
    budget="medium",
)

# Get metric for agent
metric = get_metric_for_agent("causal_impact")
```

### Agents with GEPA Support
| Agent | Type | Metric Class | Status |
|-------|------|--------------|--------|
| Feedback Learner | Deep | FeedbackLearnerGEPAMetric | ✅ Primary |
| Causal Impact | Hybrid | CausalImpactGEPAMetric | ✅ Primary |
| Experiment Designer | Hybrid | ExperimentDesignerGEPAMetric | ✅ Primary |
| Standard Agents | Standard | StandardAgentGEPAMetric | ✅ Light |

### Migration Documentation
- Plan: `.claude/plans/E2I_GEPA_Migration_Plan.md`
- Schema: `database/ml/011_gepa_optimization_tables.sql`

### Opik Integration
GEPA uses the `GEPAOpikTracer` class for observability:
```python
from src.optimization.gepa.integration import GEPAOpikTracer

tracer = GEPAOpikTracer(project_name="gepa_optimization")
async with tracer.trace_run(agent_name="causal_impact", budget="medium") as ctx:
    # Optimization runs here
    ctx.log_generation(gen_num, best_score, candidates)
    ctx.log_optimization_complete(best_score, total_gens, total_calls, elapsed)
```

Local Opik instance: http://localhost:5173 (via nginx proxy to port 8080)

---

## Critical Constraints

### What This System IS:
- Pharmaceutical **commercial operations** analytics ✅
- Business KPIs: TRx, NRx, conversion rates, market share ✅
- HCP targeting and patient journey analysis ✅
- Causal inference on operational data ✅

### What This System IS NOT:
- Clinical decision support ❌
- Medical literature search ❌
- Drug safety monitoring ❌
- Patient treatment recommendations ❌

### RAG Indexes (Operational Only):
✅ `causal_paths` - Discovered causal relationships
✅ `agent_activities` - Agent analysis outputs
✅ `business_metrics` - Performance trends
✅ `triggers` - Trigger explanations
✅ `conversations` - Historical Q&A
❌ Clinical trials, medical literature, regulatory docs

### Data & ML Constraints:
- ✅ MUST prevent data leakage (see ml-patterns.md)
- ✅ MUST log all experiments with MLflow
- ✅ MUST validate model performance before deployment
- ✅ MUST monitor for data/model drift
- ❌ NEVER use medical NER (only business entity extraction)
- ❌ NEVER mix clinical and commercial data

---

## Handoff Protocol

When transitioning between specialists:

```yaml
handoff:
  from_specialist: <source>
  to_specialist: <target>
  context_summary: |
    <2-3 sentence summary of decisions made>
  contracts_affected:
    - <list relevant .claude/contracts/*.md>
  state_to_preserve:
    - <key variables/types/interfaces>
  validation_required:
    - <integration checks needed>
```

---

## Development Workflow

### Pattern-Driven Development

All code MUST follow:
1. **General patterns**: `.claude/.agent_docs/coding-patterns.md`
2. **Error handling**: `.claude/.agent_docs/error-handling.md`
3. **ML patterns**: `.claude/.agent_docs/ml-patterns.md`
4. **Testing patterns**: `.claude/.agent_docs/testing-patterns.md`

### ML-Specific Requirements

When working on ML components:
1. **Prevent data leakage** - See ml-patterns.md
2. **Track experiments** - Use MLflow for all model training
3. **Validate models** - Check against performance thresholds
4. **Test thoroughly** - Include data leakage tests, performance tests

### Code Review

All PRs use:
- `.claude/.agent_docs/code-review-checklist.md`
- Priority: Security → Data Leakage → Correctness → Performance

### Testing

**CRITICAL: Memory-Aware Test Execution**

This system has 217 test files with heavy ML imports (dspy, sklearn, econml, dowhy).
Running with too many parallel workers causes memory exhaustion and system freezes.

**Mandatory Settings** (enforced in `pyproject.toml`):
- **Max 4 workers**: `-n 4` (NOT `-n auto` which spawns 14 workers)
- **Scope-based distribution**: `--dist=loadscope` (groups tests by module)
- **30s timeout per test**: Prevents hanging tests

**Test Commands**:
```bash
make test        # Default: 4 workers, coverage (RECOMMENDED)
make test-fast   # 4 workers, no coverage (faster)
make test-seq    # Sequential run (low memory systems)
pytest tests/    # Uses pyproject.toml defaults automatically
```

**NEVER use**:
- `pytest -n auto` - Spawns 14 workers, exhausts 7.5GB RAM
- `pytest -n 8` or higher - Risk of memory exhaustion

**DSPy Tests**:
- All dspy test files have `@pytest.mark.xdist_group(name="dspy_integration")`
- This ensures they run on the same worker to prevent import race conditions

---

## Quick Commands

```bash
# Development
make dev              # Start dev environment
make lint             # Lint and format

# Testing (Memory-Safe - ALWAYS use these)
make test             # 4 workers + coverage (RECOMMENDED)
make test-fast        # 4 workers, no coverage (faster)
make test-seq         # Sequential (low memory systems)
# NEVER: pytest -n auto (causes memory exhaustion!)

# Database
python scripts/setup_db.py
python scripts/load_v3_data.py

# Validation
python scripts/validate_kpi_coverage.py
python scripts/run_leakage_audit.py

# BMAD Planning Agents
/architect            # Activate system architect for architecture planning
/pm                   # Activate product manager for PRD creation
/security             # Activate security specialist for security review
/shard-doc            # Split large documents for token efficiency
/framework-init       # Initialize framework for new projects

# Framework
# See FRAMEWORK_README.md for framework-specific commands
```

---

## Getting Started with a Task

### For E2I-Specific Tasks

1. **State the task clearly**
2. **I will classify domains** and load appropriate specialist(s)
   - Check E2I agent index first
   - Fall back to framework specialists if needed
3. **For multi-domain tasks**, I propose a wave-based execution plan
4. **Implementation** proceeds with context isolation
5. **Integration validation** before completion

### Example Task Routing

**E2I-Specific Tasks**:
- "Add a new KPI" → Database specialist + API specialist (Wave 1), Frontend specialist (Wave 2)
- "Fix causal chain tracing" → Causal specialist (system/causal.md)
- "New agent for compliance" → Agent specialist + contracts
- "Train churn model" → ML foundation specialist + model-training.md

**Framework-Handled Tasks**:
- "Add API authentication" → API specialist + security patterns
- "Improve error handling" → error-handling.md patterns
- "Set up CI/CD" → deployment conventions + DevOps specialist

**Hybrid Tasks**:
- "Deploy agent model to production" → E2I agent specialist + MLOps specialist + deployment patterns
- "Add experiment tracking" → E2I experiment specialist + ml-patterns.md

---

## Resources

### E2I-Specific
- **Context**: See `.claude/context/` for E2I project context
- **Contracts**: See `.claude/contracts/` for integration contracts
- **Specialists**: See `.claude/specialists/` for domain experts
- **Agent Index**: `.claude/specialists/AGENT-INDEX-V4.md`

### Framework Resources
- **Patterns**: See `.claude/.agent_docs/` for best practices
- **Framework Guide**: `FRAMEWORK_README.md`
- **Customization**: `CUSTOMIZATION_GUIDE.md` (if copied)
- **Quick Start**: `QUICKSTART.md` (if copied)

---

## Updates

This document combines:
- **Claude Code Framework v3.0** - Unified framework base with BMAD features
- **E2I Extensions** - Domain-specific specialists, contracts, context

**Last Updated**: 2025-12-27
**E2I Version**: v4.3 (GEPA Prompt Optimization)
**Framework Version**: v3.0 + BMAD (v2.1)

**Latest Changes (v4.3)**:
- Migrated DSPy optimizer from MIPROv2 to GEPA (10%+ performance improvement)
- Added `src/optimization/gepa/` module with agent-specific metrics
- Added GEPA Opik integration for optimization tracing (UUID v7 compatible)
- Added database schema `database/ml/011_gepa_optimization_tables.sql`
- Added 60+ GEPA-related tests across unit and integration
- Updated Feedback Learner with GEPA optimizer support

**Previous Changes (v4.2)**:
- Renamed `src/causal/` to `src/causal_engine/` (aligned with docs)
- Created `src/rag/` with CausalRAG implementation structure
- Created `src/repositories/` with split-aware data access layer
- Documented additional directories: digital_twin, feature_store, memory, ml, mlops, tool_registry, utils, workers
- Updated architecture diagram to reflect complete project structure

**Previous Changes (v4.1)**:
- Added BMAD planning agents: `/architect`, `/pm`, `/security`
- Added document sharding utility: `/shard-doc`
- Added framework initialization: `/framework-init`
- Added `.claude/PRPs/` directory for Product Requirements Plans
- Added `.claude/scripts/` directory for framework utilities

---

## Quick Decision Guide

**What kind of task do you have?**

### E2I Agent/System Task
✅ Use E2I specialists in `.claude/specialists/`
- Agent-specific tasks → Agent specialists (Tiers 0-5)
- System-specific tasks → System specialists (NLP, Causal, RAG, etc.)
- Reference E2I contracts and context

### General Software Task
✅ Use framework patterns in `.claude/.agent_docs/`
- API development → coding-patterns.md + API conventions
- Error handling → error-handling.md
- Testing → testing-patterns.md

### ML/MLOps Task
✅ Use both E2I and framework ML resources
- Data pipelines → E2I ml_foundation + framework data-engineering.md
- Model training → E2I ml_foundation + framework model-training.md
- Experiment tracking → ml-patterns.md
- Always prevent data leakage → ml-patterns.md

### Multi-Domain Task
✅ Use wave-based execution
- Decompose into specialist domains
- Execute in waves (max 2 domains per wave)
- Validate contracts between waves
- Synthesize results
