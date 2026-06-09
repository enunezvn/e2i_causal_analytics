# Issue #810 — KPI-aware Tool Composer data resolution

## Problem (corrected)
"Conversion" is the defined **Conversion_Rate KPI (WS3-BI-009)**: *% of triggers resulting in a prescription within 30 days* (`triggers ⋈ treatment_events`), not `patient_journeys.treatment_initiated` (degenerate 94.7%). The Tool Composer is KPI-unaware: `cohort_resolution.resolve_cohort_frame` always loads `patient_journeys`, so "what drove Kisqali conversion" can never bind the real KPI outcome.

## Cheapest-disproof (PASSED, faithful real Supabase)
- Per-trigger conversion (prescription within [ts, ts+30d]) is non-degenerate: **Kisqali+NE = 916 triggers, rate 0.067** (61 converted; KPI target 0.08).
- Real DoWhy recovers a finite ATE on the trigger grain: `causal_effect_estimator(accepted→converted)` → ATE 0.0045, CI [0.0035,0.0055], p=0.001, n=4356.
- Schema facts: `triggers.brand_id` is all `'UNKNOWN'` (cannot filter triggers by brand) → brand comes from the **prescription** (`treatment_events.brand`); region from `triggers.hcp_id ⋈ hcp_profiles.geographic_region`.

## Design
New service **`src/services/kpi_resolution.py`**:
- `recognize_kpi(query) -> KPIMetadata | None` — registry-driven (`src/kpi/registry.py`, 46 KPIs) + small alias map (conversion→WS3-BI-009, trx→WS3-BI-005, …). Returns the matched KPI or None.
- `KpiFrame` — `frame`, `outcome_column`, `driver_columns`, `kpi_id`, `kpi_name`.
- `_compute_conversion_outcome(triggers_df, events_df, window_days=30)` — **pure** function (real logic, unit-tested on real-shaped frames, NO mocks): per-trigger `converted` = ∃ prescription for the patient within the window.
- `_build_conversion_frame(brand, region, *, supabase_client, window_days=30)` — real DB fetch (triggers + hcp_profiles region join + brand-filtered prescriptions) → `_compute_conversion_outcome` → `KpiFrame` (drivers: trigger_type, delivery_channel, priority, confidence_score, lead_time_days, acceptance_status). Fail-closed (None) on no data / unrecognized region.
- `_BUILDERS: dict[kpi_id, builder]` — extension point. WS3-BI-009 implemented now; unbuilt KPIs return None with a logged "no substrate builder for KPI X yet" (honest, never fabricated).
- `resolve_kpi_frame(kpi, brand, region, ...) -> KpiFrame | None` — dispatch.

Wiring (mirror the F2a / Rec#1a pattern, both live entry points):
- `chatbot_tools.tool_composer_tool`: recognize_kpi(query) → if KPI+builder, resolve_kpi_frame → `context["estimation_data"] = kpi.frame`, `context["kpi_outcome"] = outcome_column`, `context["kpi_name"]`; else existing cohort path. Best-effort (fail-closed).
- `orchestrator/nodes/dispatcher.py`: same for the `tool_composer` agent; thread via `input_data["data"]` + `input_data["kpi_outcome"]`.
- `tool_composer/agent.py`: normalize `input_data["kpi_outcome"]` → `merged_context`.
- `tool_composer/composer.py` + `planner.py`: thread an optional **outcome hint** (`context["kpi_outcome"]`) into the planner prompt so the causal `outcome` binds to the KPI outcome column (not an invented one). F6b enforcement still guarantees real-column bindings.

## Tests (TDD red-first, no mocking)
- `tests/unit/test_services/test_kpi_resolution.py`: recognize_kpi (conversion→WS3-BI-009; unknown→None); `_compute_conversion_outcome` real logic on real-shaped frames (within/outside window, multi-event); fail-closed on empty; builder-registry extension contract.
- Faithful integration (real Supabase): `resolve_kpi_frame("Conversion Rate","Kisqali","Northeast")` → real frame, outcome in (0,1) non-degenerate, expected driver columns.
- Planner outcome-hint (deterministic stub): with `kpi_outcome`, the plan binds `outcome` to the hint column.
- Faithful real-LLM E2E (gated `E2I_RUN_REAL_LLM_E2E=1`): canonical query → KPI frame → finite non-degenerate ATE + driver ranking on Conversion_Rate.

## Discipline
Worktree-isolated, memory-watched, CI batched at the end, no deploy. codex:codex-rescue review → fixed point.
