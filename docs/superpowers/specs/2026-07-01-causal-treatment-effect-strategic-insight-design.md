# Design — Treatment-effect strategic insight (causal-analysis page)

**Date:** 2026-07-01
**Branch:** `feat/causal-treatment-effect-insight`
**Status:** Approved (design), pending implementation plan

## Problem

On `/causal-analysis`, the **"Treatment effects"** tab lets an analyst run a real
DoWhy+EconML fit for one (cohort, brand) cell and shows the raw readout
(ATE / 95% CI / p-value / n). Unlike the **"Validated effects"** (leaderboard)
tab — which carries a `StrategicInsightCard` grounded in the discovered-effects
leaderboard via `POST /api/insights/causal-discovery` — the Treatment-effects
tab has **no** agentic interpretation of the estimate. The user wants a
strategic insight there, consistent with the per-page insight pattern shipped in
PR #1110.

This work also ships alongside a **client-timeout fix** (already staged on this
branch): the treatment-effect fit legitimately takes ~40s (measured for
`hcp_adoption`/`Remibrutinib`, EconML `linear_dml`, n=5000), but the frontend
axios client capped at 30s and aborted first ("timeout of 30000ms exceeded").
The fix forwards a per-call 95s timeout for `getTreatmentEffects` (backend budget
is 90s; nginx allows 120–300s) and corrects the "~5-30s" UI copy to "~10-90s".
The insight cannot be exercised end-to-end until the estimate itself returns, so
the two changes belong together.

## Goal

Add a **6th** strategic-insight endpoint + module — `treatment-effect` —
mirroring the existing `causal-discovery` insight, and render it on the
Treatment-effects tab, **auto-generating** once an estimate lands.

Non-goals (YAGNI): no new insight for the other tabs; no change to the
DoWhy+EconML fit itself; no batching/polling redesign of the estimate endpoint.

## Chosen approach

**Dedicated `treatment-effect` insight** (not reuse of `causal-discovery`). The
`causal-discovery` signature interprets a *ranked leaderboard with gate statuses*;
feeding it a single ATE would mis-frame the grounding. A dedicated signature
grounded in one de-confounded ATE (with CI/p/n and the confounder set) reads
correctly and matches the one-module-per-page convention in `src/insights/`.

## Backend

### `src/insights/treatment_effect.py` (new — mirrors `causal_discovery.py`)

Four parts, same shape as every other insight module:

1. **`TreatmentEffectInsightSignature`** (`dspy.Signature`, guarded by the same
   `try/except ImportError` → `DSPY_AVAILABLE` pattern). Interprets ONE
   de-confounded ATE for a brand analyst, STRICTLY grounded in the provided
   numbers. Must:
   - State magnitude & direction of the effect.
   - Judge **actionability**: is the 95% CI entirely above/below 0 (robust) or
     does it straddle 0 (not distinguishable from no effect)? Use the p-value
     and n as supporting evidence.
   - Name the treatment/outcome and the confounders adjusted for.
   - Include an explicit **robustness caveat**: this is a single model-based
     estimate; refutation tests were NOT run (mirrors the endpoint's existing
     `_ROBUSTNESS_UNVALIDATED_WARNING`). Never invent numbers or claim
     refutation was performed.
   - Inputs: `scope` (cohort + brand), `estimate` (ATE [CI], p, n,
     estimator), `design` (treatment → outcome, confounders).
   - Outputs: `interpretation` (str), `key_takeaways` (list, 3–5 grounded).

2. **`build_grounding(cohort, brand, treatment_var, outcome_var, confounders,
   ate, ci_lower, ci_upper, p_value, n, estimator)`** → dict with the signature
   input strings + a `grounding` chip list:
   - `ATE` = `{ate:+.4f}`
   - `95% CI` = `[{lo:+.4f}, {hi:+.4f}]` or `—` when CI is absent (DoWhy fallback)
   - `p` = `< 0.001` / `{p:.3f}` / `—`
   - `n` = `{n}`
   - A derived `ci_excludes_zero` boolean fed into the `estimate` string so both
     the LLM and the fallback speak to significance consistently.

3. **`_fallback(g)`** → deterministic factual summary (`is_fallback=True`) when
   the LLM is unavailable — states the ATE, CI, significance verdict, n, and the
   robustness caveat. No fabricated interpretation.

4. **`generate_insight(g)`** → `run_signature(...)`; on `None`/empty
   `interpretation`, return `_fallback(g)`. Same control flow as
   `causal_discovery.generate_insight`.

Export `treatment_effect` from `src/insights/__init__.py`.

### `src/api/routes/insights_strategic.py`

- **`TreatmentEffectInsightRequest(BaseModel)`**: `cohort: str`, `brand: str`,
  `treatment_var: str`, `outcome_var: str`,
  `confounders: list[str] = []`, `ate: float`, `ci_lower: float | None = None`,
  `ci_upper: float | None = None`, `p_value: float | None = None`, `n: int`,
  `estimator: str | None = None`.
- **`@router.post("/treatment-effect", response_model=StrategicInsightResponse)`**
  `async def treatment_effect_insight(req, user=Depends(require_analyst))`:
  `build_grounding(...)` → `cache_key("treatment-effect", f"{cohort}/{brand}",
  {"ate": round(ate, 4), "n": n})` → `cache_get` / `asyncio.to_thread(
  treatment_effect.generate_insight, g)` + `cache_set` → `_finalize(payload,
  provenance="Live DoWhy+EconML treatment-effect fit")`. Reuses existing
  `require_analyst`, `StrategicInsightResponse`, `_finalize`, and the existing
  `"Strategic Insights"` tag (no new openapi tag).

No new router registration is needed (endpoint added to the already-registered
`insights_strategic_router`). **Verify** the `test_sentinel_external_unreachable`
slim-app drift guard: if it enumerates registered route paths, add
`/api/insights/treatment-effect` there.

## Frontend

### `frontend/src/api/insights.ts`
- `TreatmentEffectInsightRequest` type (mirror the backend model).
- `getTreatmentEffectInsight = (r) => post<StrategicInsightResponse,
  TreatmentEffectInsightRequest>(\`${BASE}/treatment-effect\`, r)`.

### `frontend/src/hooks/api/use-insights.ts` (+ `hooks/api/index.ts`)
- `useTreatmentEffectInsight()` — `useMutation({ mutationFn:
  getTreatmentEffectInsight })`, mirroring `useCausalDiscoveryInsight`.

### `frontend/src/pages/CausalAnalysis.tsx`
- Inside the **`treatment-effects`** `TabsContent`, below the ATE readout, render
  `<StrategicInsightCard>` bound to the mutation's `isPending` / `error` /
  `data` (`insight`, `key_takeaways`, `grounding`, `is_fallback`, `provenance`,
  `generated_at`), with `onGenerate` for manual re-generate.
- **Auto-generate**: a `useEffect` keyed on the estimate identity
  (`${teData.cohort}-${teData.brand}-${teData.ate}`) fires the mutation once per
  distinct result when `teData` is present and not errored. Guard against
  re-firing on unrelated re-renders by tracking the last-fired key in a ref.
- Card only appears once an estimate has been run (i.e. `teData` exists);
  hidden in the empty/prompt state.

## Data flow

```
[Treatment-effects tab] run estimate
  → GET /api/causal/treatment-effects  (~40s, real ATE/CI/p/n)      [existing]
  → teData lands
  → useEffect(key=cohort-brand-ate) fires useTreatmentEffectInsight
  → POST /api/insights/treatment-effect { cohort, brand, ate, ci, p, n, ... }
  → build_grounding → redis cache → DSPy signature | deterministic fallback
  → StrategicInsightCard renders interpretation + takeaways + grounding chips
```

## Error handling / honesty

- LLM unavailable (no `OPENAI_API_KEY`) → `is_fallback=True` factual summary;
  never fabricated. Consistent with the other five insights.
- Grounding uses ONLY the numbers the estimate returned; CI-straddles-0 and
  p/n are surfaced so the insight cannot over-claim significance.
- The robustness caveat (refutation not run) is mandatory in both the signed and
  fallback text — mirrors the endpoint's `_ROBUSTNESS_UNVALIDATED_WARNING`.
- Insight fetch failure is non-blocking: the ATE readout still shows; the card
  shows its error state.

## Testing

- **Backend**: route test for `/api/insights/treatment-effect` — LLM-off →
  `is_fallback=True`, response echoes the grounded ATE/CI/p/n, provenance set,
  200 (never 500). A CI-straddles-0 case asserts the "not distinguishable from
  no effect" verdict in the fallback text.
- **Frontend**: `insights.test.ts` — `getTreatmentEffectInsight` POSTs to
  `/insights/treatment-effect` with the request body; hook test mirrors
  `useCausalDiscoveryInsight`. A `CausalAnalysis` test asserts the card
  auto-generates once when `teData` lands and does not re-fire on unrelated
  re-renders.

## CI gotchas to pre-empt (from prior insight rounds)

- `ruff format --check` and import sort (I001) on the new/edited Python files.
- mypy: `float | None` fields, `list[str]` defaults via `Field(default_factory)`.
- `test_sentinel_external_unreachable` slim-app drift guard — add the new route
  path if the guard enumerates paths.
- Existing `"Strategic Insights"` tag reused → no openapi tag registration churn.

## Out of scope

- Optimizing the ~40s fit (separate follow-up if desired).
- Async job/polling redesign of the estimate endpoint.
- Insights for other tabs (Estimators / History).
