# Claims Gold-Standard Model Evaluation — Design Spec (v2 · Path A)

**Date:** 2026-06-14
**Status:** Approved direction (Path A); pending spec review → implementation plan
**Author:** Claude (brainstormed + 4-probe investigation with user)

---

## 0. Revision note — why v2

v1's happy path ("score the existing champions against the registry holdouts; 37-month
backtest; 4 cohorts") was **disproven by a 4-probe read-only investigation** (2026-06-14,
each verified in prod with file:line evidence):

1. **Split registry is EMPTY** — `ml_cohort_definitions`, `ml_patient_split_assignments`,
   `ml_patient_cohort_assignments` = 0 rows; `patient_journeys.split_config_id` all NULL.
   Holdout is **global** via `patient_journeys.data_split='holdout'`. The "registry split
   JSON" is one **orphaned/stale** `ml_split_registry` row, referenced by nothing.
2. **Holdout = 3 months, not 37** — the split is **chronological** (`base.py:283
   _assign_splits`, 60/20/15/5 by row-share; holdout = newest 5% = 2026-04→06). The 37-month
   span is the *full* table; using train/val/test months as "eval" would leak.
3. **Only 1 of 4 cohorts had a loadable model** — `csu_treatment_initiation` (AUC 0.83,
   artifact verified). Persistence + HCP-adoption: champions but **0 artifacts**.
   Discontinuation: **no target/model**. Of 722 registry rows, only 2 have a real artifact.
4. **The killer:** the one loadable champion's 60 features are synthetic CSU-**generator**
   columns with **ZERO overlap** with the holdout patients' data (`patient_journeys` /
   `treatment_events` / `feature_values`). The model and the gold-standard were built in
   **different feature universes** — it cannot honestly score the holdout.

**Decision (user, 2026-06-14): Path A** — engineer features from the gold-standard data,
**train cohort models that consume them**, register loadable artifacts, then evaluate
(walk-forward + holdout). This makes model and eval-set share one feature space, so the
metrics are honest.

---

## 1. Problem

`ml_performance_metrics` is empty (0 rows, verified) → Model-Performance / Monitoring /
Time-Series "Model performance" mode are blank. The deeper cause (from the investigation):
**no model exists that consumes the gold-standard's feature space**, so there is nothing to
score the held-out claims against. Path A builds that model + the eval that produces real,
leakage-free performance metrics over time.

### Non-goals
- Real (Optum) claims data — **synthetic only**.
- Reusing the mismatched CSU demo champion to score the holdout (feature universes differ).
- A fabricated/zero-filled trend (the v1 frozen-holdout backtest is abandoned).

---

## 2. Verified assets (corrected, prod 2026-06-14)

| Asset | Detail |
|---|---|
| `patient_journeys` | 25,000 synthetic patients; labels `treatment_initiated` (8,750⁺), `persistent_180d`, `discontinued_180d` (= **1 − persistent**, exact mirror); **global chronological split** train 6,192 / val 2,144 / test 1,453 / **holdout 15,211** (newest 5% ⇒ 2026-04→06). `is_churned` is **100% NULL** (drop). Journey span 2023-06→2026-06 (37 mo, full table). |
| `treatment_events` | 94,796 longitudinal events (NDC, ICD/CPT/LOINC, `treatment_response`, `sequence_number`, `event_date`) — source for windowed aggregates. |
| `feature_values` | 49,504 rows but only **15 distinct features** (8 patient-keyed, 4 hcp-keyed, 3 brand/region); patient coverage ~65% (any) / ~20% (per-feature). Reader: `src/feature_store/retrieval.py:141 get_historical_features`. |
| `hcp_profiles` | HCP-adoption grain: label `adoption_category` (ADOPTER/NON_ADOPTER), 5,000 rows, **no split column** — needs a designated holdout (later phase). |
| `ml_model_registry` | 722 rows, mostly `is_synthetic=true` metadata-only (NULL artifact). Only CSU has artifacts. Cols incl. `model_name, model_version, stage(enum), is_champion, artifact_path, is_synthetic`; **no `feature_columns` col** (lives in the pickle's `feature_names_in_`); target via `experiment_id→ml_experiments.prediction_target`. |
| `ml_performance_metrics` | **0 rows.** Schema already has `metric_name, metric_value, measured_at, model_id(FK), source(default 'mlflow'), data_split, metadata(jsonb), sample_size, ci_*, …`. **No migration needed to store** `source`/`split_version` (latter → `metadata`). |

### Reuse points (with required extensions)
- **Writer:** `performance_tracking.record_performance` (`:81`) → `PerformanceMetricRepository.record_metrics` (`drift_monitoring.py:915`). **Must extend** to accept `measured_at` (currently hardcoded `now()` — blocks dated points), `source`, and `split_version`-in-metadata. Additive, no migration.
- **Reader:** `monitoring.py:1115 get_performance_trend` + `get_metric_trend`. **Must raise** the `days` cap (`le=90`) and `PerformanceTrackingConfig.trend_window_days` (30) so multi-month history isn't filtered out; add a longer frontend range option.
- **Registration/loading:** `prediction_synthesizer_deploy.py` (train→serialize→register→manifest), `MLModelRegistryRepository.get_models_for_target` (`ml_experiment.py:859`), `LiveChampionModelRegistry` (`registry_adapter.py`), `InProcessModelClient` (`inproc_model_client.py:65`). **Must parametrize** the deploy CLI (currently hardcoded to `csu_treatment_initiation`) per cohort + feed it gold-standard features.
- **Feature vectorizer:** `predictions.py:119 _vectorize_feature_dict` (dict→ordered row, fail-closed on missing). Reusable. The "patient_id → feature dict" builder **does not exist — we build it** (see §5.1).
- **Leakage validation:** `src/data/feature_contract.py` (`knowable_at`/window) — use at feature-design time.
- **Provenance trap:** `cohort_resolution.py apply_provenance_filter` default-EXCLUDES `is_synthetic` → all reads must pass `include_synthetic=True`.

---

## 3. Cohorts (corrected)

| Cohort | Label | Grain | Notes |
|---|---|---|---|
| Initiation | `patient_journeys.treatment_initiated` | patient | Phase-1 anchor |
| Persistence | `patient_journeys.persistent_180d` | patient | |
| Discontinuation | `patient_journeys.discontinued_180d` | patient | = **1 − persistence**; one model serves both label views (don't train twice) |
| HCP adoption | `hcp_profiles.adoption_category` | **HCP** | different grain, no split — later phase |

Authoritative mapping: `src/services/cohort_resolution.py:259-275` (`_PJ_COHORTS`), `:355-423`
(HCP). Holdout read from `patient_journeys.data_split='holdout'`, partitioned by `brand`
(initiation→Remibrutinib, persistence→Fabhalta, hcp-adoption→Kisqali) since there is no
per-cohort registry split.

---

## 4. Design

### 4.1 FeatureBuilder (the missing middle)
Build a leakage-safe feature vector for any patient_id from the gold-standard, identical for
train and eval:
- **Sources:** `patient_journeys` static/clinical cols (demographics, diagnosis, severity,
  biomarkers, payer), windowed `treatment_events` aggregates (counts/recency *before the
  prediction anchor*), and the 8 patient-keyed `feature_values` (imputed + missingness flag).
- **Leakage-safety:** only features knowable at the cohort's prediction anchor (e.g.
  pre-initiation). Validate each with `feature_contract.py` `knowable_at`. NO outcome-derived
  or post-anchor data.
- **Output:** a fixed, versioned `feature_columns` list (the model's `feature_names_in_`),
  reused verbatim at eval. Pure, batchable, unit-tested.

### 4.2 Trainer + Registrar (per cohort)
- Train on the **train split** (chronological), tune on **validation**; start with the
  project's LR pattern (calibrated), keep the algorithm pluggable.
- Serialize with `feature_names_in_` = the §4.1 columns; register a **loadable** artifact via
  the parametrized deploy path (real `artifact_path`, `is_synthetic=False`, manifest entry,
  `stage='production'`, champion flag). Honors the `tr_single_champion` trigger.

### 4.3 Walk-forward (rolling-origin) backtest — the honest trend
For each month M in the journey timeline: train on patients with `journey_start < M`
(expanding window, min-train floor), evaluate **strictly out-of-sample** on month M's
patients, record real metrics at `measured_at = M`. → up to ~**37 leakage-free monthly
points** per cohort per metric, immediately. Guards: `min_train_n`, `n_min` per eval month
(skip thin months, e.g. the ~40-patient 2026-04 bucket), log skips. This replaces v1's
disproven frozen-holdout backtest. (Compute is trivial — LR on ≤25k rows × ~37 fits.)

### 4.4 Headline champion + holdout
Final champion = trained on train+val, evaluated on **test + holdout** (the registry's
designated newest-5%) for the headline `current_value`/`baseline_value`. This is the
"deployed model on freshest held-out data" number.

### 4.5 MetricRecorder (writer extension)
Record via the extended writer: `(model_id, metric_name ∈ {accuracy,precision,recall,f1,
auc_roc}, metric_value, measured_at, source ∈ {backtest_wf,holdout,scheduled,event},
sample_size, metadata{split_version})`. **Metric-name alignment:** page queries `f1`; writer
emits `f1_score` — reconcile to the page's strings. **Idempotency:** delete-by-`source`
(+`model_id`+`split_version`)-then-insert (no unique key exists; avoids dup rows on re-run).

### 4.6 Read-path + frontend
Raise trend route `days` cap + `trend_window_days`; add a multi-year frontend range; set
`TimeSeries.tsx DEFAULT_MODEL_ID` to a cohort champion handle that resolves. Result: Model-
Performance mode renders the real walk-forward trend + headline tiles; same data feeds
Model-Performance / Monitoring pages.

### 4.7 Ongoing eval
Event-driven on (re)training/promotion (+1 real point — now meaningful, since Path A
produces new champions) + monthly dedup backstop (record only on model/data version change).

### 4.8 HCP-adoption (later phase)
Designate a holdout on `hcp_profiles`; FeatureBuilder variant on HCP features
(`peer_influence_score`, `influence_network_size`, the 4 hcp `feature_values`); same
train→register→eval loop on the HCP grain.

---

## 5. Honesty guardrails
- Walk-forward = strictly out-of-sample (train < M, eval = M) → no leakage; leakage-safe
  features validated via `feature_contract`.
- Real `predict` on real engineered features vs real labels; **no zero-fill scoring**
  (`InProcessModelClient` silently 0-fills missing features → the FeatureBuilder must
  guarantee all `feature_columns` present, else fail closed).
- `n_min`/`min_train_n` enforced; thin months skipped + logged, never interpolated.
- All rows tagged synthetic; `include_synthetic=True` on every gold-standard read; `source`
  distinguishes walk-forward vs holdout vs live.

---

## 6. Components (each isolated + unit-testable)
1. `CohortSpec` — cohort → (label, grain, brand, anchor) from `_PJ_COHORTS`.
2. `FeatureBuilder` — patient_id(s) → leakage-safe feature matrix (§4.1).
3. `CohortTrainer` — fit + calibrate on a split; returns model + `feature_names_in_`.
4. `ArtifactRegistrar` — serialize + register loadable champion (parametrized deploy path).
5. `Scorer` — `(y_true, y_score) → metrics` (aligned names).
6. `WalkForwardRunner` — orchestrates §4.3.
7. `MetricRecorder` — extended writer + idempotent delete-then-insert (§4.5).
8. `EvalTrigger` — event-driven + scheduled (§4.7).

---

## 7. Phasing
- **P1 — Initiation end-to-end:** FeatureBuilder + Trainer + Registrar + WalkForwardRunner +
  holdout + MetricRecorder + read-path/frontend fixes → Time-Series Performance mode renders
  a real initiation trend. *(Proves the whole loop on real data.)*
- **P2 — Persistence (+ discontinuation as its complement view).**
- **P3 — HCP-adoption** (HCP grain + holdout designation).
- **P4 — Ongoing eval** (event-driven + monthly dedup) + champion default wiring across pages.

---

## 8. Testing
- **Unit:** FeatureBuilder leakage-safety + determinism + no-missing-columns; Scorer math;
  walk-forward windowing + `n_min`/`min_train_n` skips; recorder idempotency + name alignment.
- **Integration (faithful):** P1 initiation — train on train split, walk-forward over the
  real timeline, assert real monotone-ish AUC points land in `ml_performance_metrics` and
  reproduce on re-run (idempotent).
- **E2E:** with rows present + read-path cap raised, `get_performance_trend` returns
  multi-month history and the page renders the trend + tiles for the default model.

---

## 9. Acceptance criteria
1. A loadable, **non-synthetic** champion exists for initiation with a real `artifact_path`
   whose `feature_names_in_` are the gold-standard-derived features.
2. `ml_performance_metrics` holds ≥ (qualifying months) real walk-forward points per metric +
   a holdout headline point — all reproducible, no dup rows.
3. Every point is strictly out-of-sample (no leakage); thin months skipped, not filled.
4. `get_performance_trend` returns non-empty multi-month history; Time-Series Performance
   mode renders the real trend + populated tiles for the default initiation model.
5. P2/P3 extend the same loop without rework.

---

## 10. Risks
- **Feature coverage** (15 `feature_values`, ~65%): mitigate with `patient_journeys`/
  `treatment_events`-derived features + imputation+flags; FeatureBuilder guarantees column
  completeness or fails closed.
- **Walk-forward class balance** in thin/early months → `n_min` + min-positive guards.
- **Leakage** is the chief honesty risk → `feature_contract` validation + strict train<M<eval.
- **Registration parity:** the deploy CLI is CSU-hardcoded → parametrization must preserve the
  `get_models_for_target` contract (`stage='production'` ∧ `artifact_path` ∧ `¬is_synthetic`).
</content>
