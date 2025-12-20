# Technical Architecture (High-Level)

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                       │
│  React Frontend + Natural Language Query Interface               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                     FastAPI Backend (REST API)                    │
│  Authentication, Rate Limiting, Request Routing                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                  Orchestrator (Tier 1 Agent)                      │
│  4-Stage Query Classification → Agent Routing → Synthesis         │
└──────┬───────────────┬───────────────┬──────────────────────────┘
       │               │               │
       │               │               │
   ┌───▼───┐      ┌───▼───┐      ┌───▼────┐
   │Causal │      │  ML   │      │Monitor │
   │Agents │      │Agents │      │Agents  │
   │Tier 2 │      │Tier 4 │      │Tier 3  │
   └───┬───┘      └───┬───┘      └───┬────┘
       │              │              │
       └──────┬───────┴──────┬───────┘
              │              │
         ┌────▼────┐    ┌────▼────┐
         │Causal   │    │ MLOps   │
         │Engine   │    │ Stack   │
         │DoWhy    │    │ MLflow  │
         │EconML   │    │ SHAP    │
         └────┬────┘    └────┬────┘
              │              │
         ┌────▼─────────────▼────┐
         │   Data & Memory Layer │
         │ Supabase │ Redis │ FalkorDB
         │ (Postgres + pgvector) │
         └────────────────────────┘
```

## Data Architecture

**Databases**:
- **Supabase (PostgreSQL)**: 37 tables across 5 categories
  - Core data (patient journeys, HCP profiles, treatment events)
  - ML metadata (experiments, models, deployments)
  - Memory (episodic, procedural)
  - Causal validation audit
  - Digital twin models

- **Redis**: Working memory, session state, caching
- **FalkorDB (Neo4j-compatible)**: Semantic knowledge graph

**Data Flow**:
```
Data Sources → Data Preparer Agent → Validated Tables
→ Feature Analyzer → Feature Store (Feast)
→ Model Trainer → Model Registry (MLflow)
→ Prediction/Causal Analysis → Results Cache
→ User Query → Retrieved Results
```

## Integration Points

**External Integrations**:
- **Anthropic Claude API**: LLM inference for agents
- **MLflow**: Experiment tracking, model registry
- **Opik**: Observability and monitoring
- **Great Expectations**: Data validation
- **Optuna**: Hyperparameter optimization
- **BentoML**: Model serving (future deployment)

**Internal Integrations**:
- All agents communicate via LangGraph orchestration
- Shared memory layer (Redis, Supabase, FalkorDB)
- Event-driven architecture for drift detection, alerts

## Deployment Architecture

**Current State**:
- **Development**: Docker Compose (Redis, FalkorDB)
- **API**: Uvicorn ASGI server
- **Database**: Supabase cloud-hosted

**Production-Ready**:
- **Container Orchestration**: Kubernetes (recommended)
- **API Gateway**: Load balancer with rate limiting
- **Database**: Supabase Production tier with read replicas
- **Caching**: Redis cluster (3+ nodes)
- **Monitoring**: Prometheus + Grafana

---
