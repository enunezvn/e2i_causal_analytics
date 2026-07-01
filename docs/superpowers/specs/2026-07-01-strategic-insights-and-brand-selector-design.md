# Design — Strategic Insights (5 pages) + Brand Selector Fix

**Date:** 2026-07-01
**Branch:** `worktree-feat+strategic-insights-brand-selector`
**Status:** Approved (design), pending spec review

## 1. Goals

Two independent asks from the frontend review of https://eznomics.site/:

1. **Brand selector overflow** — on the Home page, the brand menu shows "All Brands (Combined Portfolio)"; the "(Combined Portfolio)" parenthetical spills outside the 180px box. Remove the indication that "all brands == Combined Portfolio" (which also fixes the overflow).
2. **Agentic strategic-insights explanation** on five pages that currently have none as a first-class card:
   - `/knowledge-graph`
   - `/causal-analysis`
   - `/predictive-analytics`
   - `/model-performance`
   - `/resource-optimization` (already has post-run `optimization_summary` + `recommendations`; needs consistent surfacing)

## 2. Non-goals

- No change to the semantic value of the "All" brand option (`value: 'All'` → no-filter). Label only.
- No new generic/shared DSPy signature — insights are bespoke per page (user decision).
- No auto-generation on load for the always-on pages — on-demand button + cache (user decision).
- No fabricated data anywhere. Every insight is grounded in the page's real data with an honest deterministic fallback when the LLM is unavailable.
- No CI run per slice — CI is batched once at the very end (OOM/cost discipline).

## 3. Grounded findings (from code, not assumption)

### 3.1 Brand selector
- `frontend/src/pages/Home.tsx:115` — `BRANDS` array hardcodes `{ value: 'All', label: 'All Brands', indication: 'Combined Portfolio', color: 'bg-slate-500' }`.
- Dropdown renders `{brand.label}` + `({brand.indication})` (~`Home.tsx:889-892`) inside a `flex items-center gap-2` row with no truncation, in a `SelectTrigger className="w-[180px]"` — hence the overflow of the long parenthetical.
- **"Combined Portfolio" has zero data semantics.** Only the value `'All'` is consumed (state init `Home.tsx:399`; API guards `!== 'All'` at `463`, `490`, `569`; card visibility `981`; header text `1026`). Removing the label breaks nothing downstream.
- Test touching indication labels: `frontend/src/pages/Home.test.tsx:360` (asserts CSU/PNH/HR+HER2- BC; not "Combined Portfolio" specifically — verify during red-first).

### 3.2 Reference strategic-insights pattern (SegmentAnalysis, end-to-end)
- Frontend: plain `<Card>` rendering `analysisResult.strategic_interpretation` (string) at `frontend/src/pages/SegmentAnalysis.tsx:1081-1095` — inline, not yet a shared component.
- Hook/API: `useRunSegmentAnalysisAndWait` (`frontend/src/hooks/api/use-segments.ts:203`) → `runSegmentAnalysisAndWait` (`frontend/src/api/segments.ts:230`) → `POST /segments/analyze`.
- Backend: `src/api/routes/segments.py` route → `_execute_segment_analysis` → LangGraph `create_heterogeneous_optimizer_graph()` → `profile_generator` node → DSPy `CATEInterpretationSignature` + `generate_cate_interpretation()` in `src/agents/heterogeneous_optimizer/dspy_integration.py`.
- LM: OpenAI-only via `src/optimization/dspy_lm.py:ensure_dspy_configured()`; blocking DSPy call wrapped in `asyncio.to_thread`; **honest factual fallback** when no LM (never fabricates).

### 3.3 Existing insight assets to reuse (avoid duplication)
- Shared insight components dir: `frontend/src/components/insights/` (has `ExecutiveAIBrief`, `PriorityActionsROI`, etc. — none is a generic strategic-narrative card yet).
- Rich discrete-insight card exists: `frontend/src/components/visualizations/agents/AgentInsightCard.tsx` (opportunity/warning/recommendation items with confidence/evidence/feedback). Too heavy for a strategic narrative → we add a focused `StrategicInsightCard` instead.
- Existing insights routes: `src/api/routes/executive_insights.py`, `src/api/routes/explain.py`.
- **Per-agent DSPy interpretation signatures already exist:**
  | Page | Existing signature | Plan |
  |---|---|---|
  | resource-optimization | `OptimizationSummarySignature`, `AllocationRecommendationSignature` (`src/agents/resource_optimizer/dspy_integration.py:107,123`) | already LLM-grounded — surface via shared card |
  | causal-analysis | `CausalInterpretationSignature` (`src/agents/causal_impact/dspy_integration.py:262`) | reuse/adapt to discovered-effects leaderboard |
  | predictive-analytics | `PredictionInterpretationSignature` (`src/agents/prediction_synthesizer/dspy_integration.py:204`) | reuse/adapt to cohort + SHAP |
  | model-performance | none (only `health_score`/`drift_monitor`, not model-accuracy) | **new** `ModelPerformanceInsightSignature` |
  | knowledge-graph | none (`CausalGraphSignature` builds DAGs, doesn't interpret the KG) | **new** `KnowledgeGraphInsightSignature` |

  Net: **2 new signatures + 2 reuse/adapt + 1 already-done**.

### 3.4 Current state of the 5 pages (real data available for grounding)
| Page | File | Run action? | Real grounding data |
|---|---|---|---|
| knowledge-graph | `frontend/src/pages/KnowledgeGraph.tsx` | no (always-on) | `useNodes`/`useRelationships` — node/edge counts, causal paths, confidence |
| causal-analysis | `frontend/src/pages/CausalAnalysis.tsx` | yes (discover effects) | `DiscoveredEffect[]` — treatment→outcome, ATE, CI, estimator, gate status |
| predictive-analytics | `frontend/src/pages/PredictiveAnalytics.tsx` | yes (score cohort) | score distribution, top-N targets, SHAP drivers |
| model-performance | `frontend/src/pages/ModelPerformance.tsx` | no (always-on) | accuracy trend/current/baseline, confusion→P/R/F1, AUC, alerts |
| resource-optimization | `frontend/src/pages/ResourceOptimization.tsx` | yes (run) | already emits `optimization_summary` + `recommendations[]` |

## 4. Design

### 4.1 Part A — Brand selector fix (`frontend/src/pages/Home.tsx`)
- Set `BRANDS[0].indication = ''` (the `value: 'All'` entry).
- Render the `({brand.indication})` span only when `brand.indication` is truthy → "All Brands" shows with no parenthetical; specific brands keep their indication.
- Add defensive `min-w-0` on the flex row and `truncate` on the label span so any future long label truncates instead of spilling.
- Acceptance: All option shows "All Brands" only; no overflow; specific brands unchanged; no backend/behavior change.

### 4.2 Part B1 — Shared frontend component
`frontend/src/components/insights/StrategicInsightCard.tsx` (+ barrel export in `frontend/src/components/insights/index.ts`).

Props:
- `title?` (default "Strategic Interpretation"), `description?`
- `insight?: string` — the narrative (rendered `whitespace-pre-line`)
- `groundingChips?: { label: string; value: string }[]` — the real numbers the insight was grounded in (e.g. "ATE +2.3pp", "AUC 0.82")
- `isLoading?: boolean`, `error?: string | null`
- `onGenerate?: () => void` — when provided and no insight yet, render a "Generate strategic insight" button (button pages)
- `generatedAt?: string`, `isFallback?: boolean` (badge: "factual summary — LLM unavailable"), `provenance?: string`

States: empty→(button | hint), loading skeleton, error, insight + grounding chips + provenance/fallback badge.

### 4.3 Part B2 — Backend signatures, endpoints, grounding, caching
- **Route module:** new module `src/api/routes/insights_strategic.py`, registered in the API router alongside the existing `executive_insights.py`/`explain.py` insights routes (keeps these 5 endpoints cohesive and avoids bloating `executive_insights.py`). Per-page endpoints:
  - `POST /insights/knowledge-graph`
  - `POST /insights/model-performance`
  - `POST /insights/causal-discovery`
  - `POST /insights/predictive-cohort`
  - `POST /insights/resource-optimization`
- **Grounding (real data, never client-fabricated):**
  - Always-on pages (KG, model-perf): endpoint **derives grounding server-side** from the same repositories the page reads (graph repo / model-perf metrics). Client cannot inject fake numbers.
  - Run pages (causal, predictive, resource): client passes the run result it already received *from the backend* (real by construction); endpoint validates shape before interpreting.
- **Signatures:**
  - Reuse/adapt `CausalInterpretationSignature`, `PredictionInterpretationSignature`; surface resource-opt's existing summary/recommendations.
  - New `KnowledgeGraphInsightSignature` (inputs: scope, node counts by type, top hubs by degree, key CAUSES chains, edge-confidence summary; outputs: interpretation + key takeaways).
  - New `ModelPerformanceInsightSignature` (inputs: model, current vs baseline accuracy + trend, precision/recall/F1 from confusion, AUC, active alerts; outputs: health diagnostic + recommended action).
  - Each signature paired with a **deterministic fallback builder** that composes a factual summary from the same real numbers.
- **LM invocation:** `ensure_dspy_configured()` + `dspy.ChainOfThought(sig)` inside `asyncio.to_thread`; OpenAI-only.
- **Response shape (uniform):** `{ insight: str, grounding: {label,value}[], is_fallback: bool, generated_at: str, provenance: str }`.
- **Caching:** Redis, key `insight:{page}:{scope}:{hash(grounding_inputs)}`, ~1h TTL (mirrors the SHAP-explainer cache). Button re-clicks / revisits with identical inputs return cached instantly.

### 4.4 Part B3 — Per-page wiring & placement
- Per-page hook (`useKnowledgeGraphInsight`, `useModelPerformanceInsight`, `useCausalDiscoveryInsight`, `usePredictiveCohortInsight`, `useResourceOptimizationInsight`) + API client fn.
- `StrategicInsightCard` placement:
  - knowledge-graph → below stats cards, above graph viz (button-triggered).
  - causal-analysis → above the leaderboard table (auto-generated from the discovered effects as part of the discover run — run page, no separate button).
  - predictive-analytics → between model summary card and results grid (after cohort scored).
  - model-performance → above the metric KPI cards (button-triggered).
  - resource-optimization → in the results section, surfacing existing `optimization_summary`/`recommendations` through the shared card for visual consistency.

## 5. Testing strategy (TDD red-first · no mocking · real results)

- **Backend (pytest, red-first):**
  - Each fallback builder: deterministic, grounded — asserts output numbers are the real inputs, no invention.
  - Each endpoint contract: uniform response shape; with **no OpenAI key configured (CI default) → `is_fallback: true` with real numbers**; cache hit/miss behavior.
  - CI has no OpenAI key → CI exercises the **real grounded-fallback path** (real computation, not a mock).
  - **Live LLM path verified manually on the droplet against real data** (faithful env) — consistent with the "don't burn CI on OpenAI throughput" lesson and cheapest-disproof discipline.
- **Frontend (Vitest, red-first):**
  - `StrategicInsightCard`: renders narrative, loading skeleton, error, fallback badge, grounding chips; "Generate" button calls `onGenerate`; hides button once insight present.
  - Each page: mounts the card and calls its hook with the right params; button pages start empty with a generate button.

## 6. Rollout / ops

- **Worktree isolation:** `worktree-feat+strategic-insights-brand-selector` (already created).
- **Order (CI once, at the end):** (1) brand-selector fix → (2) shared `StrategicInsightCard` + backend scaffold/response shape → (3) wire resource/causal/predictive (existing signatures) → (4) add KG + model-perf new signatures and wire → (5) batched CI + PR.
- **Convergence:** ralph-loop to iterate to green; `codex:codex-rescue` to fixed point on any stuck diagnosis.
- **Memory-safety:** scope mypy to changed files only on the droplet; **CI is the type/test arbiter**; no whole-tree `mypy src/` on the box (known ~1.6 GiB spike). Prefer targeted pytest locally.
- **Merge:** no squash (preserve history); `git config --global http.https://github.com.proxy ""` before push; PR at the end.

## 7. Risks & mitigations

- **Reuse-vs-adapt on causal/predictive signatures:** the existing signatures may expect different inputs than the page's real data shape. Mitigation: red-first endpoint tests force reading each signature's I/O before wiring; adapt inputs, don't fabricate.
- **Latency on button click (LLM call):** mitigated by Redis cache + `asyncio.to_thread`; fallback path is instant.
- **Resource-opt double-surfacing:** ensure we surface the *existing* summary/recommendations, not a redundant new signature (avoids drift between two narratives).
- **KG grounding cost:** deriving top hubs / causal chains server-side must be a cheap read (bounded query), not a full graph recompute.

## 8. Acceptance criteria

- Home: "All Brands" option renders with no "(Combined Portfolio)" and no overflow; specific brands still show their indication; no behavior change; tests green.
- Each of the 5 pages renders a `StrategicInsightCard` in the specified placement.
- With no LLM key: every insight endpoint returns `is_fallback: true` with a real, grounded factual summary (no fabrication); pages render it with the fallback badge.
- With LLM configured (manual droplet verify): each page shows a tailored, grounded narrative sourced from that page's real data.
- No new generic signature; resource-opt reuses existing summary; KG + model-perf have new bespoke signatures.
- CI (batched, at the end): green — backend tests, frontend tests, type check.
