# Architecture Decision Records

This directory is the platform's decision log. Each record captures one decision: the context that forced it, what was decided, and the consequences we accepted.

## Conventions

- **Numbering** continues the embedded series in [`docs/ARCHITECTURE.md` §8](../ARCHITECTURE.md#8-architecture-decision-records) (ADR-001–008 live there; ADR-009 onward live here as standalone files). **Next free number: ADR-018.**
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
| [ADR-008](../ARCHITECTURE.md#adr-008-prometheus--grafana--loki-for-observability) | Prometheus + Grafana + Loki for Observability | Accepted (v4.2); **amended July 2026** (Opik stopped May 2026 → `llm_usage_events`); **amended August 2026** (#1806 observability behind the `monitoring` compose profile — an unmanaged service is not an outage; #1807 maintenance-freshness alarm) |

### Standalone records (this directory)

| ADR | Title | Date | Status |
|-----|-------|------|--------|
| [ADR-009](adr-009-llm-factory-tiers-model-refresh.md) | Central LLM factory with model tiers; July 2026 model refresh | 2026-07-18 | Accepted |
| [ADR-010](adr-010-dspy-terra-scoped-anthropic-flip.md) | DSPy default → `openai/gpt-5.6-terra`; scoped Anthropic flip for factory lanes | 2026-07-18 | Accepted (deferred A/B since executed — see note) |
| [ADR-011](adr-011-feature-importance-covariate-group-estimand.md) | Feature-importance stability gate certifies the displayed covariate-group ranking | 2026-07-18 | Accepted |
| [ADR-012](adr-012-rct-ancova-efficiency-adjustment.md) | RCT questions use baseline-ANCOVA efficiency adjustment, not confounder machinery | 2026-07-13 | Accepted |
| [ADR-013](adr-013-sampling-aware-performance-trend.md) | Model-performance trend is classified against sampling noise, never a fixed ±5% fold rule | 2026-09-07 | Accepted |
| [ADR-014](adr-014-copilot-pill-capability-catalog.md) | Copilot suggestion pills come from a code-derived capability catalog plus a deterministic validator | 2026-09-06 | Accepted |
| [ADR-015](adr-015-agent-run-outcome-error-row.md) | An agent run has failed only when a node left a `<node>_error` audit row | 2026-09-06 | Accepted |
| [ADR-016](adr-016-cooperative-cancel-heartbeat-orphan-repair.md) | Long causal-discovery jobs are cancelled cooperatively and orphan-detected by heartbeat | 2026-09-05 | Accepted |
| [ADR-017](adr-017-discovery-corroboration-and-honest-provenance.md) | Discovery corroboration is bootstrap edge stability; diagnostics annotate, never gate; a prior-asserted DAG is never reported as discovered | 2026-09-02 | Accepted |

**ADR-010 follow-up (recorded 2026-09-07).** The record deferred the DSPy lane's provider question to a golden-set A/B. That A/B **was executed**, 2026-07-18/19: 30 real production queries plus disproof and intent-coverage queries, all through production code paths in the `e2i_api` container with production keys. Both candidates **failed** the pre-registered gates (`claude-sonnet-5` on RAGAS faithfulness and e2e p50 latency; `claude-haiku-4-5` on one gate) → **data-driven NO-FLIP**. `DSPY_LM_MODEL` stays `openai/gpt-5.6-terra`, exactly as ADR-010 decided; the record needs no amendment. Report: [`docs/reports/dspy_lane_ab_20260718.md`](../reports/dspy_lane_ab_20260718.md); harness `scripts/run_dspy_lane_ab.py`.

### Pre-series records

| Record | Date | Notes |
|--------|------|-------|
| [M5 backend-orphans triage](m5-backend-orphans-triage-20260608.md) | 2026-06-08 | One-off triage decision predating this series; kept under its original name |

## Operational decisions (no ADR)

Decisions that are load-bearing and easy to undo by accident, but too narrow for a record of their own. They are listed here so the omission is **deliberate**: each one lives as a comment or a guard test at the site named, which is the authority — this table is only the index.

| Decision | PR | Where it lives |
|---|---|---|
| A required CI context is made **conditional, never path-filtered** — a path filter makes the context never report, so a required check waits forever | #1445 | `.github/workflows/backend-tests.yml`, `.github/workflows/tier1-5-test.yml`; guard `tests/integration/test_tier1_5_workflow_alarm_only.py` |
| Gunicorn runs with **preload on** (`GUNICORN_PRELOAD` defaults true; kill switch in the host `.env`) — the master imports once pre-fork | #1589 | `docker/docker-compose.yml` (comment at the env block), `config/gunicorn.conf.py` |
| `causal_impact` heavy compute runs in a **dedicated bounded pool sized per process**, with the compute budget re-checked on the worker thread | #1606 | `src/api/dependencies/compute.py`; guard `tests/unit/test_agents/test_causal_impact/test_bounded_agent_compute_1601.py` |
| Frontend **mutations never retry** (app-wide react-query mutation retry 1 → 0) — a replayed non-idempotent mutation is worse than a visible failure | #1852 | `frontend/src/lib/query-client.ts` (+ `query-client.test.ts`) |
| FalkorDB persists to the **`/data` volume explicitly** — the image's `run.sh` otherwise defaults `--dir` to a container-local path despite declaring `VOLUME /data` | #1759 | `docker/docker-compose.yml`, guard test `tests/unit/test_docker/test_compose_falkordb_data_volume_1758.py` |
| Celery **beat state persists on a volume** and daily entries run on wall-clock crontabs, not interval timers | #1653 | `docker/docker-compose.yml`; guard `tests/unit/test_workers/test_beat_daily_wallclock_1645.py` |
| Security findings on an **unused surface are allowlisted with the reason**, not force-upgraded (PYSEC-2026-3716, `datasets` path traversal) | #1731 | `.github/workflows/security.yml` |
| **RAGAS posture** — see below | #1491–#1494 (origin #504) | `.github/workflows/ragas-evaluation.yml`, `.github/workflows/ragas-smoke.yml` |

### RAGAS posture (three jobs, three different questions)

Recorded here rather than as an ADR because the split is documented at length in the workflow headers, which are the authority.

1. **Fixture regression (`ragas-evaluation.yml`) is a judge-drift sentinel and is MANUAL-ONLY.** It never invokes the RAG pipeline: every golden-set sample keeps its hardcoded `answer` and its `retrieved_contexts` is byte-identical to its reference `contexts`, so context precision/recall are 1.0 by construction and faithfulness scores the fixture author's prose. Because the input is frozen, a score move means the **judge stack** moved — a real signal, but not product quality. It is manual because the binding constraint is the **CI OpenAI key's throughput** (a 30-sample run takes ~96 min; raising concurrency trips ragas's per-job timeouts and collapses scores to 0.0 rather than going faster). Restoring the automatic triggers is the right move only on a higher-tier key.
2. **A key-free dependency smoke (`ragas-smoke.yml`) gates every PR that touches the eval stack.** Making the full eval manual removed the one automatic signal that the RAGAS dependency tree still *imports* — the exact path that broke silently for five days in #491, with the evaluator degrading to plausible-looking heuristic fallback scores. The smoke runs the real import sequence plus golden-set integrity with no gpt-4o calls and no secret.
3. **Production quality is measured by the real-pipeline gate, which fails loud.** `scripts/replay_golden_set.py` → `scripts/run_real_pipeline_ragas.py --fail-on-threshold` judges genuinely generated answers over genuinely retrieved contexts, run on demand from a host that can reach the pipeline (GitHub's runners cannot), at n≈10–15 per #504's throughput constraint.
