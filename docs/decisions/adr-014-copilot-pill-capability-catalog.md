# ADR-014: Copilot suggestion pills come from a code-derived capability catalog plus a deterministic validator

**Date**: 2026-09-06 | **Status**: Accepted | **Implemented by**: PR #1900 (catalog + validator); refinement arc PRs #1903, #1908, #1909, #1910, #1912, #1913, #1915, #1919

## Context

`POST /api/chat/suggestions` generates the sidebar's suggestion pills — the questions an analyst clicks. A pill the assistant cannot answer is a defect: the click ends in "no data" or an invented answer.

Measured on a live sample 2026-09-05 (46 calls, 92 pills; `docs/demos/results/2026-09-05_pill_suggestions_review/`), **42%** of live pills asked for analyses no bound tool serves — SHAP recomputation, territory detail, per-patient predictions, trends of causal-registry outcomes — rising to **63%** on pages that publish a page summary. Two root causes: the pill prompt described the assistant's abilities in **one prose sentence**, and a context asymmetry (the pill generator saw page summaries the agent never received). A faithful prototype whose prompt carried a catalog derived from code and data scored **9%**.

## Decision

Two independent mechanisms, because neither alone is sufficient — the prompt reduces the rate, the validator bounds the tail.

1. **The prompt's capability description is a CATALOG built from code and data, never transcribed prose** (`src/services/chat_capability_catalog.py`, the #1638 roster pattern). Every list-shaped field is derived: KPI names and windowability from the KPI registry, trend coverage from `v_kpi_history_coverage`, axis-capable KPIs from `SEGMENTED_KPI_QUERY_FAMILIES`, causal outcomes from the causal-path registry, the agent roster from the factory. The only hand-written text is the axis/composition **rules** (`AXIS_RULES`) and the `NEVER_BLOCK` list, and the axis rules are guarded by a test against `kpi_calculate_tool`'s signature so prose cannot drift from the tool.
2. **A narrow deterministic post-filter runs after generation.** `filter_unsupported_pills` drops the pill families measured unanswerable, each with a named rule. It is deterministic regex/vocabulary matching over the catalog — not a second LLM judgement — so a drop is reproducible and reviewable.
3. **Every drop is logged at INFO with its rule.** That log, not a test fixture, is the production measurement of how often the prompt still proposes an unanswerable pill. The rate is the gate the refinement PRs are certified against.
4. **The route fails loud, never degrades.** No pills parsed, or every pill dropped → **502**, never a silent fallback to invented or stale suggestions.
5. **A degraded catalog carries the last-good lists forward and stays visible.** Each DB loader has a 5 s budget and the two run sequentially, so one stalled connection cannot hold a pill request for the client's full connect+read timeouts. A field that fails to load is named in `degraded`, the previous catalog's good value for that field is reused, and the TTL drops from 600 s to 60 s so the refresh retries soon without hammering a down database. The catalog is a lazy process-wide single-flight cache — no startup hook (CI runs `TestClient` lifespans on a 30 s thread timeout).

## Consequences

- (+) Measured at #1900's certification: **1 NO in 30 kept pills (3.3%)** on the pages that publish page context, from 42% before. Each later PR in the arc is certified against that ≤3.3% gate on a live probe against the merged container.
- (+) The catalog moves when the platform moves: a new KPI, a new history-coverage row or a new causal outcome changes the prompt with no prose edit.
- (−) A prompt change regenerates **almost every pill**, so a before/after comparison is not paired. Certification needs SAME/NEW tagging, an interim probe, and — for a rule that should make a family disappear — certifying by **absence** of the target title rather than by a diff.
- (−) The validator is regex over natural language and therefore has both tails: a dropped answerable ask costs the analyst a good question, so every rule ships with KEEP fixtures pinning the boundary alongside the DROP fixtures. Documented residual misses are recorded per PR rather than patched blindly.
- (−) The filter can only drop, never rewrite. A family that keeps recurring is fixed at the prompt or, where the tool itself was wrong, in the tool (#1913 gated `kpi_calculate_tool`'s patient axis).

## References

- `src/services/chat_capability_catalog.py` — `build_capability_catalog`, `render_catalog_block`, `catalog_rules`, `filter_unsupported_pills`, `AXIS_RULES`, `NEVER_BLOCK`
- `src/api/routes/chat.py` — `_SYSTEM_PROMPT` template, `route_hint`, the drop log and the 502 paths
- `docs/superpowers/plans/2026-09-05-copilot-pill-capability-catalog.md` — the 2026-09-05 live sample and the prototype measurement
