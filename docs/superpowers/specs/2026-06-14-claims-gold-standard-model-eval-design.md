# Claims Gold-Standard Model Evaluation — Design Spec

**Date:** 2026-06-14
**Status:** Approved design (pending spec review)
**Author:** Claude (brainstormed with user)

---

## 1. Problem & motivation

The platform has **722 registered models** across 4 cohorts, but the
`ml_performance_metrics` table is **empty (0 rows)**. As a result the
Model-Performance page, Monitoring page, and the Time-Series page's
"Model performance" mode render no data — there is no real model-performance
signal anywhere in the product.

**Root cause (verified, not assumed):** nothing scores the registered models
against a held-out labelled dataset and records the results over time.
`ml_performance_metrics` is written only at runtime by
`src/services/performance_tracking.py::record_performance` (→
`PerformanceMetricRepository`, `drift_monitoring.py:935`), which has never been
driven with real evaluation data. The Time-Series trend endpoint
(`src/api/routes/monitoring.py:1115 get_performance_trend`) reads exactly that
empty table, so the page is empty regardless of which model is selected.

A frontend "default model" alone therefore **cannot** populate the page — the
binding constraint is missing evaluation data. This spec defines the
evaluation capability that produces that data **honestly** (real predictions on
real held-out labels), which in turn populates the existing pages.

### What this is NOT
- Not real-world claims data — **synthetic only** (Optum/real RWD explicitly out
  of scope, per user direction).
- Not new data generation — we **reuse the existing synthetic holdouts**.
  New/regenerated data is a **fallback only if a champion fails its performance
  threshold**.
- Not a frontend redesign — the only frontend change is setting a real default
  model on the Time-Series page once data exists.

---

## 2. Existing assets (verified in prod, 2026-06-14)

The gold-standard substrate already exists and is synthetic + longitudinal:

| Asset | Detail |
|---|---|
| `patient_journeys` | 25,000 synthetic patients; longitudinal (`journey_start_date`/`journey_end_date`, **37-month span 2023-06 → 2026-06**); ground-truth labels: `treatment_initiated` (8,750⁺/16,250⁻), `persistent_180d`, `discontinued_180d`, `is_churned`, `adherence_rate`, `treatment_arm`; `data_split`, `split_config_id` |
| `treatment_events` | 94,796 longitudinal events (`drug_ndc`, `icd/cpt/loinc_codes`, `treatment_response`, `outcome_indicator`, `sequence_number`) |
| Split registry | `ml_split_registry`, `ml_patient_split_assignments`, `ml_cohort_definitions`, `ml_patient_cohort_assignments` — the authoritative holdout designations |
| Splits (by brand) | Fabhalta holdout 4,993 · Kisqali 5,143 · Remibrutinib 5,075 (+ train/validation/test); **holdout total 15,211** |
| `ml_model_registry` | 722 models (722 distinct names, **only 2 distinct versions**) |
| `ml_experiments` | 1,053 rows; targets: `csu_treatment_initiation` (121), `pnh_persistence` (120), `kisqali_dx_adoption` (120); carries acceptance thresholds (`minimum_auc`, `minimum_precision_at_k`, `maximum_fpr`) — **not** measured metrics |
| `business_metrics` | 21,808 rows, 163 distinct dates — the *working* time-series; proves the longitudinal-synthetic premise |

### Reuse points (don't rebuild)
- **Writer:** `performance_tracking.record_performance` → `ml_performance_metrics`.
- **Reader:** `monitoring.py get_performance_trend` (already wired to the
  Time-Series page via `usePerformanceTrend`).
- **Champion resolution + artifact loading:** the prediction-synthesizer go-live
  built `MLModelRegistryRepository.get_models_for_target` and
  `LiveChampionModelRegistry` (lazy, fail-closed champion loading with
  `artifact_path`). The harness reuses these to load loadable champions.

---

## 3. Cohorts

Four cohorts, each a binary/most-likely classification task with a label that
already exists in `patient_journeys`:

| Cohort | Predicts | Ground-truth label | Example target/brand |
|---|---|---|---|
| Initiation | will start treatment | `treatment_initiated` | `csu_treatment_initiation` / Remibrutinib |
| Discontinuation | will discontinue ≤180d | `discontinued_180d` | (CSU/other) |
| Persistence | persists ≥180d | `persistent_180d` | `pnh_persistence` / Fabhalta |
| HCP adoption | HCP adopts/dx-adoption | adoption label | `kisqali_dx_adoption` / Kisqali |

Holdout membership for each cohort is read from the **split registry**
(`ml_patient_split_assignments` / `ml_split_registry`) — **never re-split** in the
harness. (Planning step: confirm the exact cohort→label mapping and the
discontinuation cohort's registry entry.)

---

## 4. Design

### 4.1 Components (each independently testable)

1. **CohortHoldoutResolver** — given a cohort, returns its holdout patient set +
   the cohort's label column, by reading the split registry. No re-splitting.
   *In:* cohort id. *Out:* `(patient_ids[], label_name, split_version)`.
2. **HoldoutFeatureLoader** — assembles each holdout patient's feature vector in
   the champion's `feature_columns` order, via the existing feature pipeline
   (`feature_values` / feature contract). Fail-closed on missing features.
3. **ChampionLoader** — wraps `LiveChampionModelRegistry` /
   `get_models_for_target` to load the loadable champion for a cohort (fail-closed
   if no champion or no `artifact_path`).
4. **Scorer** — runs real `model.predict`/`predict_proba` and computes metrics
   (AUC-ROC, accuracy, precision, recall, F1, precision@k) vs the real labels.
   Pure function: `(y_true, y_score) → {metric: value}`.
5. **MetricRecorder** — writes points via `record_performance` →
   `ml_performance_metrics`, stamped with `model_version` + `split_version` +
   a `source` tag (`backtest` | `scheduled` | `event`). Idempotent.
6. **BacktestRunner** — orchestrates the historical trend (§4.2).
7. **EvalTrigger** — event-driven + scheduled entry points (§4.3).

### 4.2 Temporal strategy — the trend ("points now")

Performance is bucketed by **journey-start month** across the holdout, scoring
the champion on each monthly patient *vintage*:

- Yields **~37 real monthly points per cohort, per metric**, available
  immediately (~135 holdout patients/cohort/month — adequate for a trend).
- **Optional 3-month rolling window** (~400 patients/window) to smooth variance
  while keeping ~37 points.
- Recorded with `measured_at` = the vintage month; `metric_name` ∈ {auc_roc,
  accuracy, precision, recall, f1}. The page's metric selector exposes each.
- **Honesty label:** series is "champion performance by patient-journey vintage,
  backtested on the frozen holdout" — explicitly distinguished from live
  production drift. Forward re-evals (§4.3) are the live points.

### 4.3 Ongoing evaluation

- **Primary — event-driven:** on champion promotion for a cohort, run a fresh
  eval and append one point (the only thing that genuinely moves the trend, since
  holdout + champion are otherwise frozen).
- **Backstop — monthly scheduled**, with **dedupe**: record a new point only when
  `model_version` or `split_version` changed since the last point. Prevents a
  misleading flat line of identical re-computations.

### 4.4 Data flow / payoff

```
holdout (registry) ─► HoldoutFeatureLoader ─► ChampionLoader.predict
        ─► Scorer ─► MetricRecorder ─► ml_performance_metrics
        ─► get_performance_trend (existing) ─► usePerformanceTrend ─► Time-Series page
```

Once metrics land, set `frontend/src/pages/TimeSeries.tsx` `DEFAULT_MODEL_ID` to a
cohort champion handle (the field stays editable). This **closes the original
"populate time-series with a default model" request with real data**, and the
same data feeds Model-Performance / Monitoring.

### 4.5 Error handling (fail-closed, never fabricate)
- No champion / no `artifact_path` → skip cohort, log, record nothing.
- Empty or under-sized monthly bucket (< n_min) → skip that point, log.
- Feature assembly or predict failure → skip, log; never substitute synthetic
  scores.
- Recorder is idempotent on `(model_id, metric_name, measured_at, source)`.

### 4.6 Honesty guardrails
- Every point = real `model.predict` on real holdout features vs real labels.
- No interpolation, no duplication, no carried-forward identical points.
- `n_min` per bucket enforced; buckets below it are omitted (not filled).
- All rows tagged synthetic; backtest vs live clearly separated by `source`.

### 4.7 Fallback (per user direction)
Only **if a champion fails its acceptance threshold** (`ml_experiments`
`minimum_auc` / `minimum_precision_at_k` / `maximum_fpr`) do we escalate to
generating new/improved synthetic data or retraining — tracked as a separate
effort, not part of this harness's happy path.

---

## 5. Phasing

- **Phase 1 — Harness + backtest backfill.** Components 1–6 + BacktestRunner;
  populate `ml_performance_metrics` with the 37-month trend for each cohort
  champion. *Deliverable: Time-Series Performance mode renders a real trend.*
- **Phase 2 — Ongoing eval.** EvalTrigger: event-driven on promotion + monthly
  dedup backstop.
- **Phase 3 — Frontend default + monitoring wiring.** Set `DEFAULT_MODEL_ID`;
  confirm Model-Performance / Monitoring populate.

---

## 6. Testing
- **Unit:** Scorer metric math; month-bucketing + rolling window; dedupe logic;
  `n_min` enforcement; recorder idempotency.
- **Integration (faithful):** run the harness on a real holdout subset for one
  cohort → assert real metrics land in `ml_performance_metrics` and reproduce
  digit-exact on re-run.
- **E2E:** with metrics present, the trend endpoint returns history and the
  Time-Series page renders the trend + "Trend Summary" for the default model.

---

## 7. Acceptance criteria
1. For each cohort with a loadable champion, `ml_performance_metrics` holds
   ≥ (number of qualifying months) real monthly points per recorded metric.
2. `get_performance_trend` returns non-empty history for each cohort champion.
3. Time-Series Performance mode renders a real trend + populated "Trend Summary"
   / KPI tiles for the default model.
4. Re-running the backtest reproduces identical values (idempotent, no dup rows).
5. No fabricated/interpolated points; backtest vs live distinguishable by `source`.
6. Champions failing acceptance thresholds are flagged (not silently shown).

---

## 8. Open questions / verify during planning
- Exact cohort→label mapping and the **discontinuation** cohort's registry entry.
- Exact `ml_performance_metrics` columns (`model_id` FK, `metric_name`,
  `metric_value`, `measured_at`, + any `source`/version columns — may need a
  light migration to carry `source`/`split_version`).
- Holdout **feature availability**: can every holdout patient's feature vector be
  assembled in each champion's `feature_columns`? (Ties to `feature_values` /
  feature contract.)
- **Champion existence per cohort**: confirm a loadable champion (with
  `artifact_path`) exists for all four cohorts; if not, that cohort is Phase-1
  blocked until one is registered (reuse prediction-synthesizer deploy path).
- Whether MLflow (`mlflow_experiment_id`) holds real historical training-eval
  metrics worth ingesting as additional real points (nice-to-have).

---

## 9. Risks
- **Champion artifacts not loadable** for some cohorts → those cohorts can't be
  evaluated until artifacts are registered (known gap from prediction-synthesizer
  work). Mitigation: Phase 1 covers cohorts that have loadable champions; others
  follow.
- **Per-bucket sample size** for monthly AUC (~135) is modest → variance.
  Mitigation: 3-month rolling option; enforce `n_min`.
- **Misread as live drift** → mitigated by explicit `source`/labeling.
