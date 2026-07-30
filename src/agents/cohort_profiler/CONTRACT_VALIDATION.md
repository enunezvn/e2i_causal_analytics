# CohortProfiler Agent - Contract Validation Report

**Agent**: Cohort Profiler
**Tier**: 0 (population profiling for chat)
**Agent Type**: Standard fast-path (pure async compute, **no graph / no LLM**)
**Contract**: `.claude/contracts/tier0/cohort_profiler.md` (created 2026-07-30, #1356 — kept in sync with the benchmark registry `scripts/benchmarks/routing/data/agent_contracts.json`).

**Date**: 2026-07-29 (extension addendum 2026-07-30)
**Version**: 1.1
**Status**: ✅ Chat-routable · Reuses live KPI path · Fails closed on genuine empty · #1356 ask-binding + HCP cohorts

> Originally produced from the 2026-07-29 contract-review findings (GitHub issue #1344), when no `.claude/contracts` entry existed. Updated 2026-07-30 to the post-#1356 implementation: the runtime contract now lives at `.claude/contracts/tier0/cohort_profiler.md`, and every claim below is grounded in the CURRENT code (symbol references; line numbers only where stable).

---

## What This Agent Is

The **chat companion** to `cohort_constructor`. Where `cohort_constructor` *materializes* an eligible patient list (a real `DataFrame` + audit trail) for the ML pipeline and therefore **cannot** run from a free-text chat query, `cohort_profiler` answers the *chat* form — "size / define a cohort of ... patients" — with **REAL DB-backed per-segment prescribing counts**. It never fabricates.

Since #1356 it also binds the ask's parameters (brand, servable criteria, KPI threshold, window) and serves **HCP-entity cohorts with quantitative KPI thresholds** — see the extension addendum below.

**Evidence**: `src/agents/cohort_profiler/agent.py` module docstring, `src/agents/cohort_profiler/__init__.py`.

Three concepts kept deliberately separate (agent.py module docstring, "Design rationale"):
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
| **Real data (no fabrication)** | ✅ | Reuses `get_kpi_calculator().calculate` (WS3-BI-006 NRx) on the criteria-less path; mig-117 `kpi_query` allowlist RPC on the criteria/HCP paths; `n/a` on missing, fail-closed on genuine empty (`agent.py::_profile_patients_legacy`, `_value`, `_rpc_rows`) |
| **No graph / no LLM** | ✅ | Pure async computation, health_score-style fast path (`agent.py::CohortProfilerAgent.__init__`) |
| **Ask binding (#1356)** | ✅ | `ask.py::parse_cohort_ask` → brand / criteria / threshold / window bind into the query; per-criterion Applied / NOT-applied accounting; unservable-only asks fail closed (`agent.py::_analyze_patients`, `_analyze_hcp`) |

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

**Implementation**: `src/agents/cohort_profiler/agent.py` (`analyze`, `_analyze_patients`, `_analyze_hcp`) + `src/agents/cohort_profiler/ask.py` (`parse_cohort_ask`).

Single dispatch, pure async computation. Named `analyze` (not `run`) precisely because the Tier-0 fall-through dispatch expects `analyze` (see `analyze` docstring).

### Flow (post-#1356)

```
analyze(agent_input)
  → parse_cohort_ask(query, brand_hint, today)   # entity, brand, criteria, threshold, window
  → entity == "hcp" ? _analyze_hcp(ask) : _analyze_patients(ask)

_analyze_patients(ask)
  → threshold on a patient ask ⇒ unservable pseudo-criterion (never silently dropped)
  → unservable-only ask (no brand, no servable criterion) ⇒ fail closed with guidance
  → servable criteria (age bounds) ⇒ _profile_patients_with_criteria (mig-117 RPC,
      params [brand, min_age_exclusive, max_age_exclusive], grouped both axes)
  → else ⇒ _profile_patients_legacy: brands = [brand] or all SUPPORTED_BRANDS,
      get_kpi_calculator(), per brand _profile_brand(...), no profiles ⇒ fail closed
  → narrative + per-criterion "Criteria accounting" (Applied / NOT applied / no other)

_analyze_hcp(ask)
  → unservable threshold metric (e.g. NRx) ⇒ fail closed with guidance
  → recognized criteria re-tagged unservable (_hcp_unservable); criteria-only ask
      (no brand/threshold/explicit window) ⇒ fail closed with guidance
  → window = ask.window or disclosed trailing-90-days (inclusive-today) default
  → mig-117 RPC cohort_profiler_hcp_trx_cohort, params
      [brand, window_start, window_end_exclusive, trx_floor_exclusive]
  → zero matches: threshold-free probe distinguishes honest zero (completed)
      from genuine empty (fail closed)
  → narrative: cohort size + specialty / priority-tier tables (+ accounting when
      criteria were recognized)
```

### Per-brand profile (`_profile_brand`, criteria-less legacy path)

Each brand runs **≤8 sequential KPI-calculator calls**, each via `_value` → `asyncio.to_thread(calculator.calculate, "WS3-BI-006", context=...)`:

| # | Context | Meaning |
|---|---------|---------|
| 1 | `{"brand": b}` | Headline new-Rx count (population size). `None`/0 ⇒ brand skipped honestly |
| 2–4 | `{"brand": b, "segment": <tier>}` | Disease-severity tiers: `low_severity`, `medium_severity`, `high_severity` |
| 5–8 | `{"brand": b, "therapy_line": <line>}` | Line of therapy: `0`, `1`, `2`, `3` prior lines |

**KPI identity**: `_NRX_KPI_ID = "WS3-BI-006"` (new prescriptions), reusing the **mig-105** `_segment` / `_line` registry variants (PR #1208) — the exact path the CopilotKit chat KPI tool and `/api/kpis` breakdown use, so reported numbers match the live chat UI (`agent.py::_NRX_KPI_ID`, `_get_calculator`).

**Supported brands** (case-SENSITIVE exact match `brand::text = $1`): `("Remibrutinib", "Fabhalta", "Kisqali")` (`agent.py::SUPPORTED_BRANDS`).

### Output (on success, patient paths)

```python
{
  "status": "completed",
  "narrative": <markdown>,                # ← surfaced via AGENT_RESPONSE_FIELDS
  "cohort_profile": {"segment_axis": "severity+line_of_therapy", "brands": [...],
                     "criteria_applied": [...], "criteria_not_applied": [...]},
  "confidence": 0.9,
  "recommendations": [ "Materialize the eligible patient list ... via the cohort "
                       "pipeline (scope_definer → cohort_constructor) ..." ],
}
```

HCP asks return `cohort_profile{entity:"hcp", segment_axis:"specialty+priority_tier", brand, window{...}, threshold{...}, cohort_size, specialty{}, priority_tier{}, trx_total, trx_max, criteria_not_applied[]}` — mirroring the patient shape.

The narrative is REAL per-segment counts (`_render` / `_render_hcp`); missing values render as `n/a`, never a fabricated number (`_fmt`).

### Fail-closed (`_failed`)

Genuine empty (no prescribing population / query layer unavailable), an ask whose recognized parameters are ALL unservable, or an unservable threshold metric — `_failed(...)` returns `{"status": "failed", "errors": [...], "narrative": ""}`, and the dispatcher fails the dispatch closed (no values fabricated). An HCP zero-match over a NONZERO prescribing base is NOT a failure: it completes honestly with cohort_size 0.

---

## Scope

**Covers**: how many patients are in the Remibrutinib/Fabhalta/Kisqali prescribing population; size/define the cohort for a brand (brand binds from the query text or an indication mention when no structured entity was grounded); break down a brand's population by disease-severity tier or by line of therapy; the same across all supported brands when none is named; patient cohorts with servable inclusion criteria bound in (age-at-diagnosis bounds) plus per-criterion Applied / NOT-applied accounting; HCP-entity cohorts with quantitative TRx thresholds over an explicit (or disclosed-default) time window, with specialty / priority-tier breakdowns.

**Does NOT cover**: materializing the actual eligible patient rows/IDs with FDA-EMA criteria + audit trail for the ML pipeline (→ `cohort_constructor`); narrative explanation of an upstream analysis (→ `explainer`); arbitrary KPI/metric questions beyond cohort sizing (→ domain KPI agents / chat KPI tool); criteria the data model cannot serve (diagnosis-year — zero diagnosis events, no diagnosis-date column; per-patient KPI thresholds; age criteria on HCP cohorts; NRx thresholds) — each is NAMED as not-applied with guidance, and an ask consisting only of unservable criteria fails closed; adoption-propensity ranking (#1356 part 3, blocked on #1354).

Patient segment axes are disease-severity tier AND line-of-therapy; HCP segment axes are specialty AND priority tier. It reports population/cohort **sizes**, not a patient or HCP contact list.

---

## Test Coverage

`tests/unit/test_agents/test_orchestrator/test_cohort_profiler.py` (pre-#1356 contract, unchanged):

| Test | Verifies |
|------|----------|
| `test_analyze_returns_real_severity_and_line_breakdown` | Real per-segment severity + line breakdown |
| `test_analyze_canonicalizes_brand_casing` | Mis-cased brand → canonical casing |
| `test_analyze_fails_closed_when_no_population` | Genuine empty ⇒ `status="failed"`, no fabrication |
| `test_resolver_grounds_brand_and_never_fails_closed` | `_resolve_cohort_profiler_input` grounds brand, never fails at resolve time |
| `test_classifier_routes_cohort_definition_to_profiler` | `Domain.COHORT_DEFINITION` → `cohort_profiler` |

`tests/unit/test_agents/test_orchestrator/test_cohort_profiler_extend.py` (#1356 + codex iter-1): brand/indication binding, age-vs-threshold disambiguation, criteria binding + honest NOT-applied accounting on BOTH paths, patient-threshold disclosure + fail-closed legs, HCP aggregation/threshold/window (incl. exact inclusive-day counts), zero-vs-empty distinction, cache-identity (two asks never share a payload).

`tests/unit/test_kpi/test_mig117_registry_presence.py`: migration-117 drift-lock (4 statement ids, synthetic twin discipline, substrate + param binding, `_profiler_query_id` flag behavior).

---

## Verification Discrepancies

1. **Resolved 2026-07-30 (#1356)**: the original 2026-07-29 finding — no `.claude/contracts` tier doc existed and implementation was the sole source of truth — is closed. The runtime contract lives at `.claude/contracts/tier0/cohort_profiler.md`, kept in sync with the benchmark registry (`scripts/benchmarks/routing/data/agent_contracts.json`).

---

## Conclusion

`cohort_profiler` is the **Tier-0 chat companion** to `cohort_constructor`: a pure async, no-graph, no-LLM fast-path agent that answers free-text cohort questions with REAL DB-backed counts and BINDS the ask's parameters. Patient asks: new-Rx headline + severity-tier + line-of-therapy breakdowns for one brand or all supported brands (criteria-less asks stay on the mig-105 WS3-BI-006 calculator path so numbers match the live chat UI; servable criteria route through the mig-117 allowlist statement). HCP asks: per-HCP TRx aggregation over an explicit window with a threshold filter, specialty + priority-tier breakdowns, on the platform TRx-KPI substrate. Every recognized-but-unservable parameter is NAMED in a per-criterion accounting — nothing is silently dropped — and an ask made only of unservable parameters fails closed with guidance. It is fully chat-routable (`INTENT_TO_AGENTS['cohort_definition']`, `Domain.COHORT_DEFINITION`), grounds an optional brand at resolve time without ever failing closed there, and never launders an empty profile into a success.

---

## #1356 Extension Addendum (2026-07-30)

Implements parts 1 + 2 of the user-ratified 2026-07-29 `extend:cohort_profiler` ruling (#1337 verdict review; part 3 — adoption-propensity ranking — remains **blocked on #1354** and is NOT implemented). Empirical basis: benchmark q11 returned a canned all-brands profile ignoring brand + criteria; q15 returned the byte-identical payload in 26.4ms (the context-keyed Redis KPI cache serving two asks that had collapsed to the same parameterless call set).

**What changed** (evidence: `src/agents/cohort_profiler/ask.py`, `agent.py`, `database/migrations/117_cohort_profiler_ask_bound_queries.sql`):

1. **Ask parsing** (`ask.py::parse_cohort_ask`): entity type (patient vs HCP), brand (resolver hint → query text → indication mention; exactly-one rule), age criteria (servable — `patient_journeys.age_at_diagnosis` populated on all rows, verified READ-ONLY 2026-07-30), diagnosis-year (recognized, UNSERVABLE — zero `diagnosis` events, no diagnosis-date column), TRx threshold, explicit time window ("last quarter" → calendar dates).
2. **Criteria-bound patient path**: brand + age bounds bind into the allowlisted `cohort_profiler_patient_criteria_profile` statement (mig-117, `kpi_query` RPC, params `[brand, min_age_exclusive, max_age_exclusive]`), grouped by `(segment_assignment, prior_therapy_lines)`. The narrative carries a per-criterion **Criteria accounting** (Applied / NOT applied with guidance / "No other criteria were applied"). Criteria-less asks keep the mig-105 KPI-calculator path untouched. An ask whose recognized parameters are ALL unservable (and no brand) fails closed with guidance.
3. **HCP-entity path**: `cohort_profiler_hcp_trx_cohort` (mig-117, params `[brand, window_start, window_end_exclusive, trx_floor_exclusive]`) aggregates per-HCP TRx on the platform TRx-KPI substrate (`treatment_events` prescription rows, `hcp_id IS NOT NULL`) joined to `hcp_profiles` for specialty / priority-tier axes. Zero matches over a nonzero base (threshold-free probe) = honest completed zero; empty base = fail-closed. Undeclared window → trailing 90 days, disclosed; undeclared threshold → all prescribing HCPs, disclosed; NRx thresholds honestly refused.
4. **Cache identity**: every ask parameter now reaches the data layer (KPI-cache context on the calculator path; positional RPC params on the mig-117 path), so two different asks can never share a cached payload.

**Tests**: `tests/unit/test_agents/test_orchestrator/test_cohort_profiler_extend.py` (brand/criteria binding, honest fail-closed, HCP aggregation + threshold + window, zero-vs-empty, cache identity), `tests/unit/test_kpi/test_mig117_registry_presence.py` (migration drift-lock). Prior tests unchanged and green.

**Verification discrepancy resolved**: the runtime contract now lives at `.claude/contracts/tier0/cohort_profiler.md`, kept in sync with the benchmark registry (`scripts/benchmarks/routing/data/agent_contracts.json`).

### Codex iter-1 hardening (2026-07-30, red-first)

1. **Patient-path KPI threshold** (HIGH): a threshold on a patient-entity ask ("patients with >50 TRx") is now an unservable pseudo-criterion — disclosed in the NOT-applied accounting with re-ask guidance, or fail-closed when it is the only thing the ask pinned down. Never silently dropped.
2. **HCP-path criteria** (HIGH): recognized age / diagnosis-year criteria on an HCP ask are re-tagged unservable (`_hcp_unservable`) and surface in `criteria_not_applied` + the narrative accounting; a criteria-only HCP ask (no brand/threshold/explicit window) fails closed with guidance.
3. **Window off-by-one** (MEDIUM): "last N days" and the disclosed 90-day default now use inclusive-today semantics — exactly N dates in `[today-(N-1), today+1)` — so the math matches the words.
4. **This document** (MEDIUM): consolidated to a single current-version report; superseded 2026-07-29 claims removed, evidence refs converted to stable symbol references.
