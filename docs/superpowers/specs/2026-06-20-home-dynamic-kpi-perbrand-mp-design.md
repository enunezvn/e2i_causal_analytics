# Home Dashboard — Dynamic KPI Visibility + Per-Brand Model Performance — Design

**Date:** 2026-06-20
**Branch:** `feat/home-dynamic-kpi-perbrand-mp` (stacked on `fix/home-dashboard-ux` / PR #1062)
**Status:** Design — awaiting user review before plan + build.

---

## Problem (from the dashboard review)

On `https://eznomics.site/` Home:

1. **"Model Accuracy" tile is invariably 77%** — it is bound to `WS1-MP-001` (ROC-AUC), which reads `ml_predictions.model_auc` corpus-wide with **no brand filter** (`Home.tsx:457`, `model_performance.py:_calc_roc_auc`). Same number for every brand.
2. **KPI grid cards look hardcoded / not brand-reactive** — the grid renders every catalog KPI; many show "Not yet computed"; the visible set does not change with the brand selector.
3. **"Not yet computed" cards make no sense** — the gold standard should surface what is calculable; null cards should not be advertised.

User's directive (2026-06-20):
- (a) **Confirmed**: dynamically show only KPIs that are calculable and have data; grid becomes brand-reactive; surface calculable brand-specific KPIs; small "N of M" note.
- (b) **Fix** the model-performance KPIs; the Model Accuracy tile shows a **per-brand average** and **says it is an average**; the per-model breakdown is available when you drill into the model-performance page.
- (c) Group 4 (#577) is **separate work** → filed as **issue #1064** (root cause: synthetic-exclusion zeroes the eligible cohort; needs a manual DB change or real-data seed). Out of scope here.

Tile metric decision (user-selected): show the **real accuracy** metric (per-brand average ≈ 70%), matching the label, not ROC-AUC.

---

## Verified data facts (read-only probes against the deployed Supabase, 2026-06-20)

- **12 gold-standard models**, named `{cohort}_{brand}_goldstd_lr_v1`, `stage='staging'`, `is_synthetic=false`:
  - cohorts: `initiation`, `persistence`, `discontinuation`, `hcp_adoption`
  - brands: `fabhalta`, `kisqali`, `remibrutinib`
- Each has `ml_performance_metrics` rows at `source='holdout'` with `metric_name` ∈ **{accuracy, precision, recall, f1, auc_roc}**, plus `holdout_curve` rows (confusion_matrix, roc_curve) populated earlier this session, plus `backtest_wf` trend rows.
- **No** PR-AUC, Brier, calibration, fairness, or recall@K exist for these models (`ml_model_registry.pr_auc/brier_score` are NULL for gold-standard rows). So those KPIs **cannot** be made per-brand from this source.
- The registry is otherwise dominated by a **720-model synthetic hyperparameter sweep** (`synth_<brand>_exp_NNNN_model_1`) — explicitly **not** used here (averaging a sweep would be a misleading number).

Per-brand average **accuracy** (the tile) and **ROC-AUC** / **F1** (two grid KPIs), computed from the holdout rows:

| brand | n_models | avg accuracy | avg ROC-AUC | avg F1 |
|---|---|---|---|---|
| Fabhalta | 4 | 0.710 | 0.761 | 0.623 |
| Kisqali | 4 | 0.700 | 0.750 | 0.615 |
| Remibrutinib | 4 | 0.703 | 0.748 | 0.604 |
| All (12) | 12 | 0.704 | 0.753 | 0.614 |

These vary per brand — modestly, because cohort dominates variance, but they are real and not invariant.

---

## Architecture

Two coordinated pieces, no DB migration required.

### A. Backend — per-brand gold-standard aggregate (service + endpoint + calculator rewire)

Reuses the existing **service-layer** read pattern (`PerformanceTracker` → `PerformanceMetricRepository`, the same path that already serves `/performance/{id}/confusion` and `/roc`). No new `kpi_query` allowlist statement, so **no migration** (important — `deploy.yml` skips migrations).

1. **`PerformanceTracker.get_brand_goldstd_summary(brand)`** (new, `src/services/performance_tracking.py`)
   - Enumerate the brand's gold-standard staging models: `ml_model_registry` where `model_name ILIKE '%_<brand>_goldstd_lr_v1'`, `stage='staging'`, `is_synthetic=false`. When `brand` is `None`/`"all"`, take all 12.
   - For each model, read its `source='holdout'` scalar metrics (accuracy, precision, recall, f1, auc_roc) from `ml_performance_metrics` (reuse the repository read used by the existing holdout readers).
   - Return `{ brand, n_models, model_versions: [...], accuracy, precision, recall, f1, auc_roc, is_synthetic_cohort: true }` where each metric is the **mean over models that have it** (and `null` if none do). Return `None` when no models / no holdout metrics (honest empty).
   - `is_synthetic_cohort=true`: the gold-standard models are `is_synthetic=false` in the registry but the eval cohort is synthetic demo data, so the dashboard must preserve its synthetic disclosure banner. Carried so the tile feeds `isSyntheticKpis` honestly.

2. **`GET /api/monitoring/performance/brand-summary?brand=<brand|all>`** (new endpoint, `src/api/routes/monitoring.py`)
   - Response model `BrandPerformanceSummaryResponse { brand, available: bool, n_models, accuracy, precision, recall, f1, roc_auc, is_synthetic_cohort }`.
   - `available=false` (honest empty) when the summary is `None`; never fabricates. Mirrors the `available`-guard shape of the confusion/ROC endpoints.

3. **`ModelPerformanceCalculator` rewire** (`src/kpi/calculators/model_performance.py`)
   - `_calc_roc_auc` (WS1-MP-001) and `_calc_f1_score` (WS1-MP-003): a shared helper reads `get_brand_goldstd_summary(context.get("brand"))` and returns the brand's avg `auc_roc` / `f1`. This makes both **per-brand** and computes F1 (today null).
   - Fallbacks preserved: if the gold-standard summary is unavailable, `_calc_roc_auc` falls back to its existing corpus `ml_predictions` SQL leg then MLflow; `_calc_f1_score` falls back to MLflow (its current behavior). Never fabricates.
   - The other five MLflow-only KPIs (WS1-MP-002 PR-AUC, -004 Recall@K, -005 Brier, -006 Calibration, -008 Fairness) are **unchanged** — they have no per-brand gold-standard source, stay fail-closed/null, and are hidden by Part B. Docstrings updated to record why.
   - **Cache correctness:** the per-KPI value/batch cache key MUST include `brand` (and region). Verify the value/batch endpoint already scopes its cache by context; if not, add brand to the key. (Without this, WS1-MP-001 would cache one brand's value and serve it to all — re-introducing the invariance bug at the cache layer.)

### B. Frontend — dynamic visibility + per-brand tile (`frontend/src/pages/Home.tsx` + hooks)

4. **Dynamic visibility (rule A).**
   - Derive `computedKPIs` = `effectiveKPIs` filtered to those whose batch result `hasValue` (`r != null && r.value != null && !r.error`) — the same `hasValue` test already at `Home.tsx:1048`.
   - Drive the **grid**, the **tabs** (`kpiCategories`, `Home.tsx:611`), the **filtered list** (`filteredKPIs`, `:632`), and the **summary counts** (`summaryStats`, `:638`) from `computedKPIs` so null cards disappear from the grid, the tabs, and the counts **together** (no orphaned empty tab, no count drift).
   - **Loading state:** while the batch values are in flight, show a loading skeleton — not an empty flash (the catalog/`effectiveKPIs` resolves before batch values arrive).
   - **Brand reactivity:** because the batch is re-fetched per `{brand, region}` context, the computed (visible) set changes with the brand selector — KPIs that only compute for some brands appear/disappear; brand-specific KPIs appear when a brand is chosen.
   - **Surface brand-specific:** remove `'brand_specific'` from `HIDDEN_HOME_WORKSTREAMS` (`Home.tsx:174`) so calculable brand KPIs (`BR-002…005` compute live) show; `BR-001` stays hidden because it is null (#1064).
   - **"N of M" affordance:** show e.g. *"Showing N of M defined KPIs"* (scoped to the brand) so hiding is transparent, not silent truncation.
   - **Demo-mode fallback unchanged:** when the API is offline (`effectiveKPIs` falls back to `SAMPLE_KPIS`), behavior is unchanged and the existing demo badge still announces it.

5. **"Model Accuracy" tile** (`Home.tsx:~939`)
   - Replace `useKPIValue('WS1-MP-001', …)` with a new `useBrandModelSummary(brand)` hook → the brand-summary endpoint.
   - Show `accuracy` as a percent; subtitle/label **"avg of N models"** so it is explicitly an average; keep the drill-down to `/model-performance` (where per-model accuracy lives). Honest loading + empty (`available=false` → "—", never fabricated).
   - Keep the page synthetic-disclosure banner correct: feed `is_synthetic_cohort` into `isSyntheticKpis` (replacing the old `rocAucResult?.data_source` signal at `Home.tsx:602`).

6. **New FE hook + typed client** — `useBrandModelSummary(brand)` in `frontend/src/hooks/api/` calling `GET /api/monitoring/performance/brand-summary`, validated by a zod schema (and the generated OpenAPI client if the repo regenerates it).

---

## Data flow

```
brand selector
 ├─ tile  ──> GET /performance/brand-summary?brand=X ──> avg accuracy ("avg of N models")
 └─ grid  ──> POST /api/kpis/batch {context:{brand,region}}
                 └─ ModelPerformanceCalculator
                      ├─ WS1-MP-001/003 ──> get_brand_goldstd_summary(brand) (per-brand auc/f1)
                      └─ others ──> fail-closed null
                 └─ FE filters to computedKPIs (hasValue) ──> grid + tabs + counts (brand-reactive)
```

## Error handling / honesty

- Honest empty everywhere: `available=false` / null metric → hidden grid card or "—" tile, **never** a fabricated 0/0.77/0.5.
- No new synthetic values fabricated; gold-standard holdout figures are real eval outputs over the (clearly-disclosed) synthetic cohort.
- Anti-mocking: the summary averages only metrics that exist; `n_models` reflects models that actually have holdout data.

## Testing

- **Backend:** `get_brand_goldstd_summary` (per-brand filter, all-12, averaging, empty→None, partial-metric averaging); endpoint `available` true/false; calculator WS1-MP-001/003 per-brand value + fallback when summary absent + cache key includes brand.
- **Frontend:** dynamic visibility (null cards hidden; tabs/counts derive from computed set; brand switch changes the visible set; "N of M" text); brand-specific KPIs surface; tile shows per-brand accuracy + "avg of N"; honest loading/empty; demo-mode fallback intact.

## Out of scope (tracked separately)

- **#1064** — Group 4 / #577 Patient Touch Rate + BR-001 (synthetic-exclusion).
- Computing the other synthetic-excluded KPIs via `*_include_synthetic` allowlist twins (per-KPI backend work).
- Per-brand PR-AUC / Brier / Recall@K / Calibration / Fairness (no gold-standard source; would need new eval outputs).

## File map

- Create: `src/services/performance_tracking.py` (method), `src/api/routes/monitoring.py` (endpoint + response model), `frontend/src/hooks/api/use-model-summary.ts` (hook + schema).
- Modify: `src/kpi/calculators/model_performance.py` (WS1-MP-001/003 + helper), `frontend/src/pages/Home.tsx` (visibility, tabs, counts, tile, unhide brand_specific), `frontend/src/pages/Home.test.tsx`.
- Tests: `tests/unit/test_services/test_performance_tracking*.py`, `tests/unit/test_api/test_monitoring*.py`, `tests/unit/test_kpi/test_model_performance*.py`, `frontend/src/pages/Home.test.tsx`.
