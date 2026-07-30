# cohort_profiler — Tier-0 Runtime Contract

**Agent**: Cohort Profiler (chat companion to `cohort_constructor`)
**Tier**: 0 (population profiling for chat)
**Type**: Standard fast-path — pure async compute, **no graph / no LLM**
**Created**: 2026-07-30 (#1356 — first `.claude/contracts` entry for this agent; prior contract intent lived only in code docstrings and `src/agents/cohort_profiler/CONTRACT_VALIDATION.md`)
**Extend ruling**: `extend:cohort_profiler`, user-ratified 2026-07-29 (#1337 Step 0 verdict review); parts 1 + 2 implemented in #1356. Part 3 (adoption-propensity ranking) is **blocked on #1354** model promotion and is NOT in force.

> Kept in sync with the benchmark contract registry
> (`scripts/benchmarks/routing/data/agent_contracts.json`, the SSOT for the
> 14-agent registry) — change BOTH together.

---

## Purpose

Answers the *chat* form of a cohort question with REAL DB-backed counts, never
fabricating. Where `cohort_constructor` materializes an eligible patient list
(DataFrame + audit trail) for the ML pipeline and cannot run from free text,
`cohort_profiler` sizes and segments the population — and, since #1356, binds
the ask's parameters into the profile query instead of returning a canned
profile.

## Covers

- Patient prescribing-population sizing for one brand or all brands
  (new-Rx headline + disease-severity tier + line-of-therapy breakdowns).
- **Brand binding** (#1356 part 1): the brand binds from the resolver-grounded
  entity, else the query text, else an indication mention
  (CSU → Remibrutinib, PNH → Fabhalta, breast cancer/HR+/HER2 → Kisqali).
  Exactly-one-brand rule: two different brands named ⇒ honest all-brands scope.
- **Criteria binding** (#1356 part 1): servable inclusion criteria bind into
  the profile query — age-at-diagnosis bounds ("adults over 18") via the
  allowlisted `cohort_profiler_patient_criteria_profile` statement
  (migration 117). Every recognized criterion is accounted for in the answer:
  **Applied** (named with its bound value) or **NOT applied** (named with
  guidance on why the data model cannot serve it).
- **HCP-entity cohorts with quantitative KPI thresholds** (#1356 part 2):
  "HCPs who prescribed more than 50 TRx last quarter" → per-HCP TRx
  aggregation over an explicit half-open window with a strict threshold
  filter, via the allowlisted `cohort_profiler_hcp_trx_cohort` statement
  (migration 117). Substrate = `treatment_events` prescription rows — the SAME
  substrate as the platform TRx KPI (`business_impact_trx`) — joined to
  `hcp_profiles` for the segment axes. Returns cohort size + specialty and
  priority-tier breakdowns, mirroring the patient-profile shape. A zero-match
  cohort over a NONZERO prescribing base (verified by a threshold-free probe)
  is an honest completed answer, not a failure. No named window ⇒ trailing
  90 days (inclusive-today: exactly 90 dates in `[today-89, today+1)`),
  explicitly disclosed. No threshold ⇒ all prescribing HCPs, explicitly
  disclosed. All named windows use inclusive-today semantics ("last N days"
  = exactly N dates).

## Does NOT cover

- Materializing eligible patient rows/IDs with FDA-EMA criteria + audit trail
  → `cohort_constructor` (ML pipeline).
- Narrative explanation of upstream analysis → `explainer`.
- Arbitrary KPI/metric questions beyond cohort sizing → domain KPI agents /
  chat KPI tool.
- Criteria the data model cannot serve — e.g. diagnosis-year filters:
  `treatment_events` has ZERO `diagnosis` events and no diagnosis-date column
  exists (`patient_journeys.journey_start_date` is only a documented proxy,
  migration 044 note). These are named honestly as NOT applied; if **nothing**
  in the ask can be served (no brand, no servable criterion), the agent FAILS
  CLOSED with guidance rather than answer a different question.
- NRx thresholds on the HCP path (recognized, honestly refused with guidance;
  TRx only today).
- KPI thresholds on the PATIENT path ("patients with >50 TRx") — recognized
  and NEVER silently dropped: disclosed as NOT applied (with re-ask-as-HCP
  guidance), or fail-closed when the threshold is the only thing the ask
  pinned down.
- Patient-attribute criteria on the HCP path (age bounds, diagnosis-year) —
  recognized, re-tagged unservable, disclosed in `criteria_not_applied` + the
  narrative accounting; an HCP ask whose ONLY specifics are such criteria
  (no brand, no threshold, no explicit window) fails closed with guidance.
- Adoption-propensity ("model-scored high-value") HCP ranking — #1356 part 3,
  blocked on #1354 promotion of `hcp_adoption_{kisqali,fabhalta,remibrutinib}`.

## Dispatch

- `INTENT_TO_AGENTS['cohort_definition']` → `cohort_profiler`, priority
  `critical`, `timeout_ms=30000`, `fallback_agent=None`
  (`src/agents/orchestrator/nodes/router.py:169-184`).
- 4-stage classifier: `Domain.COHORT_DEFINITION` → `cohort_profiler`
  (`pattern_selector.py:34-38`).
- Resolver `_resolve_cohort_profiler_input` grounds an optional brand and
  NEVER fails closed at resolve time (`dispatcher.py:1216-1234`); the agent
  parses the raw query text itself as fallback (#1356 — the q11/q15 surfaces
  carried no grounded entities).
- IS in `_FAIL_CLOSED_ON_FAILED_STATUS` (`dispatcher.py:1285-1299`):
  `status="failed"` fails the dispatch — genuine empty, unservable-only ask,
  or query-layer failure is never laundered into success.
- Dispatches via default `analyze` (Tier-0 fall-through; intentionally NOT in
  `AGENT_METHOD_MAP`); narrative surfaces via `AGENT_RESPONSE_FIELDS`.

## Data access

All DB reads go through vetted read-only statements: the mig-105 KPI-calculator
path (criteria-less patient asks — numbers in lock-step with the live chat UI)
and the migration-044 `kpi_query` allowlist RPC for the #1356 statements
(migration 117: `cohort_profiler_hcp_trx_cohort`,
`cohort_profiler_patient_criteria_profile`, each with an `_include_synthetic`
twin selected by `_profiler_query_id()` under the showcase flag — the additive
variant idiom, deliberately absent from `SYNTHETIC_TWINNED_QUERY_IDS`). The
agent never sends raw SQL.

## Cache identity (#1356 part "cache keying")

The pre-#1356 q11/q15 byte-identical 26.4ms repeat was the context-keyed Redis
KPI cache serving two asks that had collapsed to the SAME parameterless call
set. The contract now: every ask parameter (brand, age bounds, entity type,
window, threshold) binds into the data-layer calls — KPI-cache contexts on the
calculator path, positional RPC params on the mig-117 path — so two different
asks can never share a cached payload.

## Output shape

Patient: `{status, narrative, cohort_profile{segment_axis, brands[],
criteria_applied[], criteria_not_applied[]}, confidence, recommendations}`.
HCP: `{status, narrative, cohort_profile{entity:"hcp", segment_axis,
brand, window{start,end_exclusive,label,explicit}, threshold{metric,
min_exclusive,stated}, cohort_size, specialty{}, priority_tier{}, trx_total,
trx_max, criteria_not_applied[]}, confidence, recommendations}`.
Fail-closed: `{status:"failed", errors:[{error}], narrative:""}`.

## Sources

`src/agents/cohort_profiler/agent.py`, `src/agents/cohort_profiler/ask.py`,
`src/agents/cohort_profiler/__init__.py`,
`database/migrations/117_cohort_profiler_ask_bound_queries.sql`,
`src/agents/cohort_profiler/CONTRACT_VALIDATION.md`,
tests: `tests/unit/test_agents/test_orchestrator/test_cohort_profiler.py`,
`tests/unit/test_agents/test_orchestrator/test_cohort_profiler_extend.py`,
`tests/unit/test_kpi/test_mig117_registry_presence.py`.
