# Architecture Decision Records

This directory is the platform's decision log. Each record captures one decision: the context that forced it, what was decided, and the consequences we accepted.

## Conventions

- **Numbering** continues the embedded series in [`docs/ARCHITECTURE.md` §8](../ARCHITECTURE.md#8-architecture-decision-records) (ADR-001–008 live there; ADR-009 onward live here as standalone files).
- **File naming**: `adr-NNN-<short-slug>.md`. One decision per file.
- **Status** is one of Proposed / Accepted / Amended / Superseded. Records are append-only — a reversed decision gets a new record superseding the old one, not an edit that rewrites history.
- Cite the PRs and migrations that implemented the decision so the record stays verifiable against git.

## Index

### Embedded set (in ARCHITECTURE.md §8)

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](../ARCHITECTURE.md#adr-001-6-tier-agent-architecture) | 6-Tier Agent Architecture | Accepted (v3.0) |
| [ADR-002](../ARCHITECTURE.md#adr-002-single-droplet-deployment) | Single-Droplet Deployment | Accepted (v4.2) |
| [ADR-003](../ARCHITECTURE.md#adr-003-tri-memory-architecture) | Tri-Memory Architecture | Accepted (v3.0) |
| [ADR-004](../ARCHITECTURE.md#adr-004-networkx--dowhy--econml--causalml-for-causal-inference) | NetworkX + DoWhy + EconML + CausalML for Causal Inference | Accepted (v3.0), refined v4.1/v4.2 |
| [ADR-005](../ARCHITECTURE.md#adr-005-hybrid-rag-with-three-backends) | Hybrid RAG with Three Backends | Accepted (v4.0) |
| [ADR-006](../ARCHITECTURE.md#adr-006-docker-compose-over-kubernetes) | Docker Compose Over Kubernetes | Accepted (v4.2) |
| [ADR-007](../ARCHITECTURE.md#adr-007-supabase-jwt-for-authentication) | Supabase JWT for Authentication | Accepted (v4.1) |
| [ADR-008](../ARCHITECTURE.md#adr-008-prometheus--grafana--loki-for-observability) | Prometheus + Grafana + Loki for Observability | Accepted (v4.2); **amended July 2026** (Opik stopped May 2026 → `llm_usage_events`) |

### Standalone records (this directory)

| ADR | Title | Date | Status |
|-----|-------|------|--------|
| [ADR-009](adr-009-llm-factory-tiers-model-refresh.md) | Central LLM factory with model tiers; July 2026 model refresh | 2026-07-18 | Accepted |
| [ADR-010](adr-010-dspy-terra-scoped-anthropic-flip.md) | DSPy default → `openai/gpt-5.6-terra`; scoped Anthropic flip for factory lanes | 2026-07-18 | Accepted |
| [ADR-011](adr-011-feature-importance-covariate-group-estimand.md) | Feature-importance stability gate certifies the displayed covariate-group ranking | 2026-07-18 | Accepted |
| [ADR-012](adr-012-rct-ancova-efficiency-adjustment.md) | RCT questions use baseline-ANCOVA efficiency adjustment, not confounder machinery | 2026-07-13 | Accepted |

### Pre-series records

| Record | Date | Notes |
|--------|------|-------|
| [M5 backend-orphans triage](m5-backend-orphans-triage-20260608.md) | 2026-06-08 | One-off triage decision predating this series; kept under its original name |
