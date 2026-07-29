# CohortProfiler Agent - Contract Validation Report

**Agent**: Cohort Profiler
**Tier**: 0 (population profiling for chat)
**Agent Type**: Standard fast-path (pure async compute, **no graph / no LLM**)
**Contract**: *No tier-doc exists.* Implementation is the sole source of truth (agent + `__init__` docstrings, router/pattern_selector/dispatcher comments).

**Date**: 2026-07-29
**Version**: 1.0
**Status**: ✅ Chat-routable · Reuses live KPI path · Fails closed on genuine empty

> Produced from the 2026-07-29 contract-review findings (GitHub issue #1344). `cohort_profiler` is a newer agent with no `.claude/contracts` entry — every claim below is grounded directly in code at the cited `file:line`.

---

## What This Agent Is

The **chat companion** to `cohort_constructor`. Where `cohort_constructor` *materializes* an eligible patient list (a real `DataFrame` + audit trail) for the ML pipeline and therefore **cannot** run from a free-text chat query, `cohort_profiler` answers the *chat* form — "size / define a cohort of ... patients" — with **REAL DB-backed per-segment prescribing counts**. It never fabricates.

**Evidence**: `src/agents/cohort_profiler/agent.py:1-25` (module docstring), `src/agents/cohort_profiler/__init__.py:1-11`.

Three concepts kept deliberately separate (agent.py:18-24):
- population **profiling** (this agent),
- cohort **materialization** (`cohort_constructor`, ML pipeline),
- **explanation** of upstream analysis (`explainer`).

---

## Contract Compliance Summary

| Category | Status | Evidence |
|----------|--------|----------|
| **Chat routable** | ✅ | `INTENT_TO_AGENTS['cohort_definition']` → `cohort_profiler` (`router.py:176-184`) |
| **Classifier route** | ✅ | `Domain.COHORT_DEFINITION` → `cohort_profiler` (`pattern_selector.py:34-38`) |
| **Input resolver** | ✅ | `_resolve_cohort_profiler_input` grounds optional brand; never fails closed at resolve time (`dispatcher.py:1216-1234`, registered `dispatcher.py:1251`) |
| **Fail-closed on empty** | ✅ | Member of `_FAIL_CLOSED_ON_FAILED_STATUS` (`dispatcher.py:1285-1299`); `status="failed"` only on genuine empty |
| **Dispatch method** | ✅ | Default `analyze` (Tier-0 fall-through; **NOT** in `AGENT_METHOD_MAP`, `_agent_method_map.py:64-69`) |
| **Response field** | ✅ | `AGENT_RESPONSE_FIELDS['cohort_profiler'] = ['narrative']` (`_agent_method_map.py:208-209`) |
| **Real data (no fabrication)** | ✅ | Reuses `get_kpi_calculator().calculate` (WS3-BI-006 NRx); `n/a` on missing, fail-closed on total empty (`agent.py:82-116, 136-159`) |
| **No graph / no LLM** | ✅ | Pure async computation, health_score-style fast path (`agent.py:62-65`) |

---

## Dispatch Contract

**Routing** (`src/agents/orchestrator/nodes/router.py:176-184`):

```
INTENT_TO_AGENTS['cohort_definition'] → AgentDispatch(
    agent_name   = "cohort_profiler",
    priority     = "critical",
    timeout_ms   = 30000,          # ≤8 sequential DB-backed KPI calls per brand
    fallback_agent = None,          # no fallback: real data or honest fail-closed
)
```

`fallback_agent=None` is intentional: an `explainer` fallback would only re-fail with nothing to explain. `cohort_constructor` was the *original* target of this intent and dead-ended (it fails closed from chat, and its explainer fallback then also failed closed) — verified by container replay, per the router comment (`router.py:169-175`).

**4-stage active classifier** (`src/agents/orchestrator/classifier/pattern_selector.py:34-38`): `Domain.COHORT_DEFINITION` maps to `cohort_profiler`.

**Input resolver** (`src/agents/orchestrator/nodes/dispatcher.py:1216-1234`): `_resolve_cohort_profiler_input` extracts an optional brand (`_extract_brand_region`, `parsed_query.entities` → `user_context`) and **always returns inputs** — it never fails closed at resolve time, because profiling always has real data to report. Registered in `INPUT_RESOLVERS` at `dispatcher.py:1251`.

**Method spec**: dispatches via the default `analyze` method (Tier-0 fall-through). `cohort_profiler` is **intentionally NOT** in `AGENT_METHOD_MAP` — that map is the Tier 1–5 contract pinned to 13 agents by `test_agent_registry_consistency` (`_agent_method_map.py:64-69`). Timeout comes from the router `AgentDispatch.timeout_ms`; the narrative surfaces via `AGENT_RESPONSE_FIELDS` (`_agent_method_map.py:208-209, 242`).

**Fail-closed contract**: `cohort_profiler` IS in `_FAIL_CLOSED_ON_FAILED_STATUS` (`dispatcher.py:1285-1299`). It emits `status="failed"` **only** on a genuine empty/error state (no prescribing population, calculator unavailable); the dispatcher then fails the dispatch rather than laundering an empty profile into a transport-level success.

---

## Compute Contract (`analyze`, no graph / no LLM)

**Implementation**: `src/agents/cohort_profiler/agent.py:68-159`.

Single dispatch, pure async computation over the shared KPI calculator. Named `analyze` (not `run`) precisely because the Tier-0 fall-through dispatch expects `analyze` (agent.py:70-75).

### Flow

```
analyze(agent_input)
  → _canonical_brand(brand)            # mis-cased → canonical, else None
  → brands = [brand] or all SUPPORTED_BRANDS   # no brand named ⇒ profile all
  → get_kpi_calculator()               # ImportError/init failure ⇒ fail closed
  → for each brand: _profile_brand(...)
  → if no profiles: fail closed (genuine empty)
  → _render(profiles) → markdown narrative
```

### Per-brand profile (`_profile_brand`, agent.py:146-159)

Each brand runs **≤8 sequential KPI-calculator calls**, each via `_value` → `asyncio.to_thread(calculator.calculate, "WS3-BI-006", context=...)`:

| # | Context | Meaning |
|---|---------|---------|
| 1 | `{"brand": b}` | Headline new-Rx count (population size). `None`/0 ⇒ brand skipped honestly |
| 2–4 | `{"brand": b, "segment": <tier>}` | Disease-severity tiers: `low_severity`, `medium_severity`, `high_severity` |
| 5–8 | `{"brand": b, "therapy_line": <line>}` | Line of therapy: `0`, `1`, `2`, `3` prior lines |

**KPI identity**: `_NRX_KPI_ID = "WS3-BI-006"` (new prescriptions), reusing the **mig-105** `_segment` / `_line` registry variants (PR #1208) — the exact path the CopilotKit chat KPI tool and `/api/kpis` breakdown use, so reported numbers match the live chat UI (`agent.py:40-56, 119-124`).

**Supported brands** (case-SENSITIVE exact match `brand::text = $1`): `("Remibrutinib", "Fabhalta", "Kisqali")` (`agent.py:35-38`).

### Output (on success, agent.py:106-116)

```python
{
  "status": "completed",
  "narrative": <markdown>,                # ← surfaced via AGENT_RESPONSE_FIELDS
  "cohort_profile": {"segment_axis": "severity+line_of_therapy", "brands": [...]},
  "confidence": 0.9,
  "recommendations": [ "Materialize the eligible patient list ... via the cohort "
                       "pipeline (scope_definer → cohort_constructor) ..." ],
}
```

The narrative is REAL per-segment counts (`_render`, agent.py:161-186); missing values render as `n/a`, never a fabricated number (`_fmt`, agent.py:188-192).

### Fail-closed (agent.py:96-103, 194-196)

If every requested brand returns no prescribing population — a genuine empty, not a zero to narrate — `_failed(...)` returns `{"status": "failed", "errors": [...], "narrative": ""}`, and the dispatcher fails the dispatch closed (no values fabricated).

---

## Scope

**Covers**: how many patients are in the Remibrutinib/Fabhalta/Kisqali prescribing population; size/define the cohort for a brand; break down a brand's population by disease-severity tier or by line of therapy; the same across all supported brands when none is named.

**Does NOT cover**: materializing the actual eligible patient rows/IDs with FDA-EMA criteria + audit trail for the ML pipeline (→ `cohort_constructor`); narrative explanation of an upstream analysis (→ `explainer`); arbitrary KPI/metric questions beyond cohort sizing (→ domain KPI agents / chat KPI tool).

The two segment axes are **exactly two** (disease-severity tier AND line-of-therapy); it reports population **sizes**, not a patient list.

---

## Test Coverage

`tests/unit/test_agents/test_orchestrator/test_cohort_profiler.py`:

| Test | Verifies |
|------|----------|
| `test_analyze_returns_real_severity_and_line_breakdown` | Real per-segment severity + line breakdown |
| `test_analyze_canonicalizes_brand_casing` | Mis-cased brand → canonical casing |
| `test_analyze_fails_closed_when_no_population` | Genuine empty ⇒ `status="failed"`, no fabrication |
| `test_resolver_grounds_brand_and_never_fails_closed` | `_resolve_cohort_profiler_input` grounds brand, never fails at resolve time |
| `test_classifier_routes_cohort_definition_to_profiler` | `Domain.COHORT_DEFINITION` → `cohort_profiler` |

---

## Verification Discrepancies

1. **No `.claude/contracts` tier doc and no prior CONTRACT_VALIDATION.md exist** for `cohort_profiler`. Contract intent lives only in the agent + `__init__` docstrings and the router/pattern_selector/dispatcher comments — implementation is the sole source of truth. This document is the first written record; it is grounded in code, not distilled from memory.

---

## Conclusion

`cohort_profiler` is the **Tier-0 chat companion** to `cohort_constructor`: a pure async, no-graph, no-LLM fast-path agent that answers free-text cohort-sizing questions with REAL DB-backed per-segment prescribing counts (new-Rx headline + disease-severity tier + line-of-therapy breakdowns), for one brand or all supported brands, via ≤8 sequential KPI-calculator calls per brand over the mig-105 WS3-BI-006 path — so its numbers match the live chat UI. It is fully chat-routable (`INTENT_TO_AGENTS['cohort_definition']`, `Domain.COHORT_DEFINITION`), grounds an optional brand at resolve time without ever failing closed there, and fails closed only on a genuine empty (no prescribing population / calculator unavailable), never laundering an empty profile into a success.
