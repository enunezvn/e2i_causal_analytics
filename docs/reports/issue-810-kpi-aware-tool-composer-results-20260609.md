# Issue #810 — KPI-aware Tool Composer: results

**Date:** 2026-06-09
**Branch:** `feat/tc-810-kpi-aware` (off `main` `e8a905fe`)
**Scope:** make the Tool Composer recognize defined KPIs and resolve the KPI's real causal substrate, so *"what drove `<brand>` conversion in `<region>`, and which segments respond best?"* returns a real causal driver + segment analysis. No deploy.

## Problem
"Conversion" is the defined **Conversion Rate KPI (WS3-BI-009)** — *% of triggers resulting in a prescription within 30 days* (`triggers ⋈ treatment_events`), not `patient_journeys.treatment_initiated` (degenerate 94.7%). The composer was KPI-unaware: it always loaded the patient-clinical cohort, so the causal core could never bind the KPI outcome.

## Cheapest-disproofs (faithful, real Supabase — run before building)
- Per-trigger conversion is **non-degenerate**: Kisqali+NE = 916 triggers, rate **0.067** (KPI target 0.08).
- **Data-driven across the whole domain** (NOT hardcoded to Kisqali/NE): brand ∈ {Fabhalta, Kisqali, Remibrutinib}, region ∈ {midwest, northeast, south, west} — all 12 brand×region cells compute (rates 0.049–0.072).
- Real DoWhy recovers a finite ATE on the trigger grain with a **binary** treatment: `causal_effect_estimator(accepted→converted)` → ATE −0.0152, p=0.001, n=916; `cate_analyzer` by `delivery_channel` → real per-channel CATEs (rep_alert +0.029, email +0.012 respond best).
- Schema reality: `triggers.brand_id` is all `'UNKNOWN'` → brand resolved from the **prescription** (`treatment_events.brand`); region from `triggers.hcp_id ⋈ hcp_profiles.geographic_region`.

## What was built
1. **`src/services/kpi_resolution.py`** (new): `recognize_kpi(query)` (registry-driven, 46 KPIs + alias map); `resolve_kpi_frame(kpi, brand, region)` → materializes the KPI substrate. Conversion builder = `triggers ⋈ treatment_events` per the authoritative SQL, with the `converted` outcome + driver columns. **Brand/region are parameters matched case-insensitively against the actual distinct data values** — no hardcoded brand/region list; unrecognized → fail-closed `None`. Per-KPI builder registry (`_BUILDERS`) is the extension point; unbuilt KPIs return `None` (honest), never fabricate.
2. **Caller wiring** (both live entry points, mirroring F2a/Rec#1a): `chatbot_tools.tool_composer_tool` and `orchestrator/nodes/dispatcher.py` recognize the KPI and thread the KPI frame + `kpi_outcome` into context/input; `tool_composer/agent.py` normalizes `kpi_outcome`.
3. **Planner** (`planner.py`): `outcome_hint` (prompt + deterministic override of `outcome`/`target` to the KPI column) + `_apply_treatment_guard` (override a categorical treatment to a binary/numeric driver — DoWhy/CATE need a usable treatment). Gated on KPI queries.
4. **Deterministic KPI causal plan** (`composer._build_kpi_causal_plan`): for a KPI causal query, build a known-good plan — `causal_effect_estimator` (ATE) + `discover_dag → rank_drivers` (driver importance) + `cate_analyzer` (segment heterogeneity) — over the KPI substrate. Treatment = best binary/numeric driver; segments = **low-cardinality** categoricals (high-card ID columns excluded); outcome = the KPI. Falls back to LLM planning when no usable treatment. This replaces unreliable free-form LLM planning (which picked descriptive tools / categorical treatments / passed string cohort args that cascade-failed).
5. **`rank_drivers` robustness** (`causal_discovery.py`): when `discover_dag` finds no edges, return a real **predictive-only SHAP ranking** instead of erroring on the empty graph (no-silent-cap note in `errors`).

## Faithful end-to-end proof (real `.env` LLM + real Supabase)
Production path `tool_composer_tool` on the canonical query → **success, confidence 0.85**:
- `kpi_ate`: causal_effect_estimator → **ATE −0.0152, CI [−0.0162,−0.0142], p=0.001, n=916**
- `kpi_cate`: cate_analyzer → real per-segment CATEs (`next_best_action +0.46`, `adherence_risk +0.017`, `alert −0.077`, …) — *which segments respond best*
- `kpi_rank`: predictive SHAP ranking (confidence_score #1) via the empty-DAG fallback
- synthesized answer: a real driver + segment narrative grounded in the above.

Before this change the same query returned ~1/4 tools / NaN CATE / "unable to analyze" because it ran on the wrong (patient_journeys) substrate.

## Tests (no mocking of logic)
- `test_kpi_resolution.py`: recognition (dynamic), pure outcome/assembly logic on real-shaped frames, fail-closed, + **faithful real-Supabase** frames across all 3 brands (non-degenerate).
- `test_planner_kpi_outcome_hint.py`: outcome-hint + treatment-guard (override categorical→binary; keep valid; never clobber `$step`).
- `test_composer_kpi_plan.py`: deterministic plan binds treatment/outcome, excludes high-card IDs from segments, falls back without a usable treatment.
- `test_causal_discovery_f7_contract.py`: rank_drivers empty-DAG predictive-only fallback.
- Targeted #810 suites: **63 passed**; lint clean.

## Codex review (gpt-5.5, read-only) — ralph-loop, 3 iterations
Codex confirmed all three iter-1 MED findings resolved and the core brand/region resolution dynamic (no allow-list). Each iteration's in-scope findings were verified against source and fixed:

- **MED-1 (resolved):** `resolve_kpi_frame` documents that infra errors propagate; both callers (chatbot_tools, dispatcher) catch/log/proceed (mirrors `cohort_resolution`).
- **MED-2 (resolved):** silent `_MAX_ROWS` truncation → `KpiFrame.is_truncated` set, logged in chatbot, and **threaded through the dispatcher** (iter-2 follow-up: `_resolve_tool_composer_data` now returns `(frame, kpi_outcome, is_truncated)` and sets `agent_input["kpi_truncated"]`; TDD guards in `test_dispatcher_tool_composer_data.py`). The brand distinct-value scan is the trickiest: `treatment_events.brand` is a **PG enum (`brand_type`)** so an indexed `ILIKE` is impossible (`operator does not exist: brand_type ~~* unknown`, verified live) — `_resolve_brand_canonical` scans distinct enum values, matches case-insensitively, and returns `(canonical, scan_truncated)` folded into `is_truncated`; never a silent fail-closed (TDD guard `test_brand_scan_truncation_is_not_silent`).
- **MED-3 (resolved):** `_apply_outcome_hint` hard no-ops when `available_columns is None`.
- **MED-B / anti-mocking (resolved):** the empty-DAG predictive-only fallback emitted causal-looking zeros (`causal_score=0.0`, `rank_correlation=0.0`). Those causal fields are now `Optional` and set to **`None`** (not `0`) so an *absent* causal estimate can never be read as a real zero-effect value; F7 contract test asserts `None`. Verified mypy-clean and all consumers (dict-`.get` / the separate `DriverRanker`) still green.
- **LOW (resolved):** all tool-facing brand/region enumerations (Field descriptions, `Args` docstrings, `tool_registrations.py`, and `examples` region values) softened to "resolved case-insensitively against the actual data values" / real example regions; removed the factually-wrong `US, EU, APAC` region notes. Core resolution stays dynamic.

### Out-of-scope HIGH (deferred with reasoning — REASON-BEFORE-RULES)
Codex iter-3 raised a HIGH on `dispatcher._mock_agent_execution` (`dispatcher.py:529`) — the unregistered-agent path returns canned narratives with plausible values (`ATE=0.12`, `15%`). **This is deliberately NOT fixed in this PR**, for reasons verified against source:
- It is **pre-existing** (platform-genesis commit `3e1c70cf`, 2025-12-20), unrelated to #810's KPI-aware routing, and untouched by this work.
- It is a **documented test / degraded-mode scaffold**: it only runs when `agent_name not in self.agents`. The primary prod orchestrator path (`cognitive.py:97`) builds `registry = create_agent_registry(...)` and passes it, so registered agents never hit the mock — it is a fallback "only used when the factory fails to instantiate any agents."
- Codex flagged it by pattern-matching the `ATE=0.12` detection-signal **without** verifying reachability (the exact trap CLAUDE.md's REASON-BEFORE-RULES warns against: "a grep is a snapshot of now").
- A hasty fail-closed change would risk the **intentionally registry-less** RAG fallback paths (`rag/causal_rag.py:205`, `rag/cognitive_rag_dspy.py:1256/1290` pass `agent_registry or {}`), which deserve their own investigation.

**Recommendation:** treat the registry-less mock-fallback as a separate hardening task (own issue) — audit which entry points run registry-less, then either fail-closed or gate behind an explicit dev flag. Not bundled here to keep #810 scoped.

## Honest notes
- The conversion causal signal in this data is small (ATE ≈ −0.015) — an honest finding (trigger acceptance weakly affects conversion), not a defect.
- Only Conversion Rate (WS3-BI-009) has a substrate builder today; other KPIs fail closed to the cohort path (clear extension point).
- `treatment_events.brand` is a PG enum — distinct brands are inherently bounded (3 today); the truncation contract is a correctness guard for the general case, not a live risk at current scale.
