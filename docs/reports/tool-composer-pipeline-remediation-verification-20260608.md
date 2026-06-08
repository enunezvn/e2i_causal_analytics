# Tool Composer Pipeline — Remediation Verification & Completion

**Date:** 2026-06-08
**Author:** Claude (Opus 4.8) — faithful-execution verification + completion, no mocks
**Component:** `src/agents/tool_composer/` (+ `src/tool_registry/tools/causal_discovery.py`, `src/agents/orchestrator/nodes/dispatcher.py`)
**Predecessor:** `docs/reports/tool-composer-pipeline-audit-20260607.md` (findings F1–F7)
**Canonical query under test:** *"What drove Kisqali conversion in the Northeast, and which segments respond best?"*

---

## Executive summary

The 2026-06-07 audit found the Tool Composer **non-functional for its flagship use case** (0/4 tools succeeded, confidence 0.3). Two waves of work followed: the 4-shard remediation (PRs #774–#777) and two tool follow-ups (PRs #780/#781). This report **verifies that work against the real pipeline** and **completes the residual gaps**.

**Verification verdict (before this work):** real progress — the pipeline went from *total failure* to *real data flowing, real tools running, honest partial answers* — but **F1 was only partially met** (≈1/4 tools; the causal + comparative core still failed) and **three findings were still open**: F6(b) semantic binding, F7 data-contract unification, and the second (orchestrator) half of F2(a). Plus a **new, data-bound finding** the audit never checked.

**Completion (this work):** three disjoint shards built in parallel + one integration robustness fix, all proven with **faithful real runs (real `.env` LLM, real Supabase, real DoWhy/EconML — no mocks)**:

| Finding | Audit severity | Status now | Proof |
|---|---|---|---|
| F1 — flagship query answerable | CRITICAL | **Plumbing CLOSED**; query is partly *data-bound* (see below) | deterministic gate recovers finite ATE; real run binds only real columns |
| F2 — data delivered (chat + orchestrator) | CRITICAL | **CLOSED (both entry points)** | chat: 255-row frame flows; orchestrator: dispatcher now delivers it |
| F3 — fabricated tools | HIGH | **CLOSED** (was closed by R1/#778) | census 6/6 fail-closed or real |
| F4 — crash vs fail-close | MEDIUM | **CLOSED** (R1) | `cohort_statistics` real stats / `RuntimeError` |
| F5 — dependency cascade | MEDIUM | **CLOSED** (R2) | step skips "dependency unmet" |
| F6 — column/entity binding | HIGH | **CLOSED** | profiles + semantic prompt + fail-fast enforcement; real LLM binds only real columns |
| F7 — incompatible data contracts | LOW/MED | **CLOSED** | discover_dag/rank_drivers consume the shared frame; sparse-column robust |
| Rec#6 — functional gate | PROCESS | **HARDENED** | deterministic finite-ATE gate + new sparse-column regression |

**Honest residual (data, not code):** for the *exact* canonical query, "conversion" maps to `treatment_initiated`, which is **94.7% positive** on the real Kisqali/Northeast cohort — a near-constant outcome on which any causal estimate is degenerate. The pipeline now binds real columns, delivers real data, and runs the real causal engine; it produces a *meaningful* ATE on a **well-posed** outcome (proven: ATE 1.166), and honestly returns degenerate/partial results when the chosen outcome is near-constant. Making the *flagship query specifically* yield a strong causal answer is an outcome-definition / cohort question, not a remaining code defect.

---

## Methodology (cheapest-disproof, faithful environment)

Per `CLAUDE.md` (CHEAPEST-DISPROOF-FIRST, REASON-BEFORE-RULES, anti-mocking), every claim below is from a **faithful real run**, not from `grep` or the test harness:

1. Ran the **real production entry point** (`chatbot_tools.tool_composer_tool`) on the canonical query with the live `.env` LLM and real Supabase.
2. Probed the **real data layer** (`resolve_cohort_frame`) and the **real causal engine** directly.
3. Ran a fresh **fabricated-tool census** (each tool, two inputs).
4. Read the **orchestrator dispatcher** source to find the second, unwired entry point.
5. Built the completion in **isolated worktrees** (one per shard), each TDD red-first with faithful tests, then integrated and re-ran the unified faithful E2E.

Repro scripts: `docs/reports/tool-composer-audit-20260607-repro/verify_remediation_e2e.py` (+ the original `e2e_faithful.py` / `census_and_withdata.py`).

---

## Part 1 — Verification of the prior remediation (the real runs)

### F2 data delivery — REAL (chat path)
`resolve_cohort_frame("Kisqali","Northeast")` returns a real **255-row × 52-col** frame carrying the causal variables (`engagement_score`, `treatment_initiated`, `disease_severity`, `academic_hcp`, `age_at_diagnosis`, `days_to_treatment`). The chat tool threads it into `context["estimation_data"]`; the executor auto-injects it into the causal tools. Not papered-over.

### F3 fabrication — CLOSED
Census of the 6 originally-fabricated tools (two inputs each):

| Tool | Result |
|------|--------|
| `psi_calculator` | **fail-closed** (`RuntimeError` requires real DataFrame) |
| `refutation_runner` | **fail-closed** (no more fabricated `gate_decision="proceed"`) |
| `distribution_comparator` | **fail-closed** |
| `cohort_statistics` | **fail-closed** w/o data; **real stats** with data |
| `cohort_validator` | **fail-closed** (hardcoded 0.95/0.92 gone) |
| `sensitivity_analyzer` | **real** (E-value 1.16 vs 6.06e+39 — varies with input) |
| `cohort_builder` | **real** (distinct patient IDs per cohort) |
| `power_calculator` | **real** (n=3135 vs n=195) |

### F4 / F5 — CLOSED
`cohort_statistics` computes real stats (size 255) with data and `RuntimeError`s without — no `AttributeError`. A failed upstream step now yields `Skipping step … dependency unmet` (graceful `SKIPPED`), not a `None.get()` crash.

### F1 / F6 / F7 — the residual (before this work)
The live run on the canonical query produced ~1/4 tools succeeding: `cohort_statistics` returned real descriptive stats, but
- `cate_analyzer` fail-closed: the LLM bound `outcome='conversion_rate'` (no such column) and `treatment='Kisqali'` (a brand *value*) — **F6 not enforced**;
- `discover_dag` raised a pydantic `ValidationError` — **F7 data-contract still split**;
- segment ranking was skipped (correct cascade).

### F2(a) second entry point — OPEN (before this work)
Two **live** entry points exist: the chat tool (fixed) and the **orchestrator** (`intent_classifier` → `router multi_faceted` → `dispatcher._prepare_agent_input` → `agent.run`). The dispatcher threaded only `query/user_context/parameters({})/session_id/parsed_query` — **no data** — and `grep` found zero cohort-resolution under `src/agents/orchestrator/`. So multi-faceted queries via the orchestrator delivered no data.

### New finding — the canonical query is partly DATA-bound
Variance check on the real cohort: `treatment_initiated` ("conversion") is **94.7% positive** (mean 0.950, ~12 negatives in 255); `is_churned`/`adherence_rate` ~94% null; `refill_count` 100% null. A causal estimate of "what drove conversion" is statistically degenerate on this outcome **regardless of binding**. Good predictors *do* exist (`engagement_score`, `academic_hcp` 34/66, `disease_severity`, `days_to_treatment`, `age_at_diagnosis`), so the data supports causal analysis on a **well-posed** outcome.

---

## Part 2 — The cheapest-disproof gate (run before building)

Before building F6(b), the gating assumption — *"with correct bindings on a well-posed outcome, the real causal engine produces a real result on this data"* — was tested directly:

```
causal_effect_estimator(
    treatment="academic_hcp", outcome="days_to_treatment",
    confounders=["age_at_diagnosis","disease_severity"], estimation_data=<real frame>)
-> ATE = 1.166, CI [1.165, 1.167], p = 0.001, n = 228, method = backdoor.linear_regression
```

**Survived.** The build was therefore justified: the gap was column/contract wiring, not the engine.

---

## Part 3 — Completion (three parallel shards + one integration fix)

Built in isolated worktrees off `main`, merged into `feat/tool-composer-functional-completion` (disjoint files, clean 3-way merge).

### F6(b) — planner semantic/entity binding + enforcement
`src/agents/tool_composer/{planner,composer}.py`, `tool_registrations.py`.
- New `composer._extract_column_profiles`: per-column dtype-family (binary / numeric / categorical), cardinality, and value lists for low-cardinality columns — threaded into `planner.plan(...)`.
- Planner prompt now instructs the LLM to bind treatment→binary/low-card, outcome→numeric, segments→low-card categoricals, map business terms to real columns, and never use a brand/region *value* as a column.
- **Enforcement** (not just warn): unresolvable column-typed args fail fast with an "unbound column" reason at plan time (`PlanningError`), so a bad binding never reaches a tool.
- Fixed a real bug: the `cate_analyzer` registration pointed at a mismatched `CATEInput` model; replaced with a truthful `CateAnalyzerInput(treatment, outcome, segments: List[str])`.
- **Faithful real-LLM proof:** on the canonical query the planner now binds **only real columns** (`treatment_initiated`/`engagement_score`/`age_group`/`gender`…) — zero invented columns, zero brand-as-column.

### F7 — unify the data contract
`src/tool_registry/tools/causal_discovery.py`, `src/agents/tool_composer/executor.py`.
- `discover_dag`/`rank_drivers` now accept the shared real DataFrame via the standard `estimation_data` kwarg (real SHAP derived via RandomForest + TreeExplainer), keeping back-compat with an explicit `data: Dict[str,List]`.
- Executor Gate-1 refined (`_is_explicit_dataframe_input`): the planner's broken `{col:'$ref'}` dict no longer blocks frame injection; a genuine frame / valid dict still wins.
- **Integration robustness fix (this report):** `_numeric_frame` previously required complete (non-NaN) rows across **all** numeric columns. On the real 52-col frame (with ~94%-null columns) that produced an **empty frame** ("cannot run on an empty frame"). Fixed to drop overly-sparse columns (non-null fraction < 0.5, surfaced via WARNING) **before** the complete-case filter, and fail-closed if <2 dense columns remain.
- **Faithful proof:** on the real 52-col frame, `_numeric_frame` drops the 9 sparse columns → keeps **7 dense / 228 rows** → `discover_dag` **succeeds** (was: empty-frame error).

### F2(a) — wire the orchestrator entry point
`src/agents/orchestrator/nodes/dispatcher.py`.
- For the `tool_composer` agent **only**, the dispatcher extracts brand/region from `parsed_query.entities` (user_context fallback), calls `resolve_cohort_frame`, and threads the frame as `input_data["data"]` (which `agent.run` normalizes to `estimation_data`). Best-effort: None/exception → proceed without data (tools fail-closed honestly). Scoped so other agents are untouched.
- **Faithful proof (real Supabase):** the dispatcher path with brand=Kisqali/region=Northeast now delivers a real `(255, 52)` frame (`brand=['Kisqali']`, `region=['northeast']`).

### Rec#6 — hardened functional gate
- The deterministic stub-planner gate (`tests/integration/test_tool_composer_functional_e2e.py`) already asserts a **finite ATE** (not merely `tools_succeeded>0`) — verified passing on integrated code (4 passed, 2 real-LLM opt-ins skipped).
- Added a **sparse-column regression** (`test_numeric_frame_drops_sparse_columns_before_complete_case`, `…_fails_closed_when_too_few_dense_columns`) pinning the integration fix so the empty-frame bug cannot return.
- Real-LLM variants remain manual-only (`E2I_RUN_REAL_LLM_E2E=1`) so nondeterministic planning JSON can't flake CI (#504 precedent).

---

## Part 4 — Unified faithful evidence (integrated code)

- **Data contract:** real 52-col frame → `discover_dag` success (was: ValidationError / empty-frame).
- **Binding:** real-LLM planner binds only real columns (was: `conversion_rate` / `Kisqali`).
- **Causal engine:** finite ATE 1.166 on a well-posed outcome (deterministic gate green).
- **Orchestrator:** dispatcher delivers the real 255-row frame (was: no data).
- **Tests:** F7 contract+regression **25 passed**; dataframe-autoinject green; functional gate **4 passed/2 skipped**; tool_composer unit suite green.

---

## What remains (honest)

1. **Outcome definition for the flagship query.** "Conversion" = `treatment_initiated` is 94.7% positive on this cohort → degenerate. To get a strong causal answer to the *literal* Kisqali query, either reframe the outcome (e.g. time-to-therapy, adherence on a denser cohort) or widen the cohort. This is a product/data decision; the pipeline now surfaces it honestly instead of fabricating.
2. **Degenerate-target avoidance is a soft nudge, not enforced.** F6(b) guarantees *real-column* bindings; it does not *force* the LLM away from a near-constant column (enforcing that risks false fail-fasts on legitimately-skewed columns). The column profile exposes distributions so the LLM can avoid them.
3. **`rank_drivers` on weakly-connected DAGs.** Pre-existing `DriverRanker` constraints (isolated-node / single-feature) live in `src/causal_engine/discovery/driver_ranker.py` (outside this work); data now flows correctly to the engine.

---

## Key evidence index

`cohort_resolution.resolve_cohort_frame` (255×52) · `causal_effect_estimator` ATE 1.166 · census 6/6 · `dispatcher._prepare_agent_input` (F2a delivery) · `planner._enforce_column_bindings` / `_extract_column_profiles` (F6b) · `causal_discovery._numeric_frame` sparse-drop (F7) · `executor._is_explicit_dataframe_input` (F7 Gate-1) · `tests/integration/test_tool_composer_functional_e2e.py` (Rec#6) · `tests/unit/test_tool_registry/test_causal_discovery_f7_contract.py` (25 tests).
