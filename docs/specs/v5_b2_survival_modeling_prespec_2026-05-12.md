# v5 Gate B2 — Cox + RSF Survival Modeling Pre-Spec

**Status**: ACTIVE pre-spec
**Created**: 2026-05-12
**Branch**: `v5-b2-survival-modeling`
**Anchor commit**: this memo committed BEFORE running `measure_b2_cindex_contrast.py`. Anti-HARKing discipline mirrored from v5 B3 (`docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md`).

## 1. Hypothesis

If the underlying outcome on a cohort is genuinely time-to-event rather than binary (i.e., among `treatment_initiated == 1` patients, the *time* to first biologic prescription varies meaningfully), then a survival model (Cox proportional-hazards, Random Survival Forest) should achieve a Harrell concordance index `C` strictly greater than the binary logistic-regression val_AUC by at least `0.03` on that cohort.

Disease-agnostic rationale: survival regression uses partial-likelihood / ranking-based losses that incorporate event-time ordering. When the binary classifier wastes information by collapsing time to an indicator, survival reclaims that information.

## 2. Cohort feasibility audit (PHASE 1 finding)

Empirical inspection of `data/rwd/optum/initiation/e2i_ml_v3_treatment_events.parquet` + `data/rwd/csu/e2i_ml_v3_treatment_events.json` performed 2026-05-12 (commit-anchor for this memo).

| Cohort | n_patients | n_positives | post-index rx events | unique patients w/ rx | time-to-event feasibility |
|---|---|---|---|---|---|
| **CSU** | 9607 | 1743 | 9223 / 10000 | 1743 (100% of positives) | **REAL** — derive T from first post-index rx `days_from_diagnosis` |
| **Optum initiation** | 1294 | 37 | 0 / 22 | 6 (16% of positives) | **DATA-FIDELITY-BOUND** — no usable post-index rx; fall back to artificial 180d horizon |

**Implication**: the v5 plan §B2 phrasing assumed Optum was primary because its binary AUC sits in the marginal [0.62, 0.68] band. The actual data carries time-to-event signal only on CSU. **Primary cohort for B2 acceptance is CSU**; Optum is secondary and is expected to NULL because survival collapses to binary at a fixed administrative censoring horizon.

This is a load-bearing audit finding — committed in this memo BEFORE running the model so the choice of primary cohort cannot be HARKed post-hoc.

## 3. Candidate models

| Model | Library | Reason |
|---|---|---|
| Cox proportional-hazards | `sksurv.linear_model.CoxPHSurvivalAnalysis` | Linear parametric baseline; matches binary LR as the survival analogue |
| Random Survival Forest | `sksurv.ensemble.RandomSurvivalForest` | Non-parametric, captures interactions; survival analogue of RF binary classifier |

Both via scikit-survival 0.24.1 (pinned in `pyproject.toml` for sklearn 1.6 / econml<1.7 compatibility). Lifelines 0.30.3 is installed but not used in the model contrast — reserved for Kaplan-Meier diagnostic plots if needed.

## 4. Derivation of survival target

For each cohort independently:

### CSU
- `event` = `treatment_initiated` (1 = initiated biologic, 0 = no biologic during follow-up).
- `time` = first post-index rx event `days_from_diagnosis` if `event == 1`; else administrative censoring at the maximum observed follow-up window per cohort split (compute from `journey_duration_days` 95th percentile, capped at 365d).
- Cohort-scoped derivation lives in `src/agents/ml_foundation/model_trainer/nodes/survival_model.py::derive_survival_target` — pure helper, manifest-aware, returns `(time, event)` numpy arrays.

### Optum initiation
- `event` = `treatment_initiated`.
- `time` = 180d administrative censoring for ALL patients (since no usable post-index rx event date is available). This is mathematically equivalent to a logistic binary regression when paired with proportional-hazards loss; expected NULL by construction. Documented here so the NULL is anchored pre-hoc, not retrofit.

## 5. Manifest declarations

The derived columns `survival_time_days` (continuous) + `survival_event` (binary) are **derived outputs**, not features. They are NOT added to the feature manifest. They are declared in the model_trainer state contract:

```python
# In src/agents/ml_foundation/model_trainer/state.py
survival_time_days: Optional[np.ndarray] = None
survival_event: Optional[np.ndarray] = None
enable_survival_modeling: bool = False  # default off; opt-in via runner flag
```

Default `enable_survival_modeling=False` preserves existing binary-classifier pipeline behavior. Opt-in for B2 measurement.

## 6. Acceptance threshold (locked here BEFORE measurement)

### Per v5 plan §B2

Net effect on at least one cohort:
- **IMPROVEMENT**: `c_index_survival - val_auc_binary >= 0.03` (closes B2 with positive evidence).
- **NULL**: `|c_index_survival - val_auc_binary| < 0.03` on every cohort (closes B2 with documented null).
- **REGRESSION**: `c_index_survival - val_auc_binary <= -0.03` on every cohort (documented null with regression flag).

### B2-specific refinement

Because the Optum survival framing is **degenerate by construction** (constant administrative censoring → equivalent to binary), the Optum result is non-load-bearing. **CSU is the only cohort that can produce an IMPROVEMENT verdict**. If CSU yields delta < 0.03, B2 closes NULL on both cohorts.

### Why >= 0.03 (not 0.02 like B3)

B3 was a same-class-of-model contrast (logistic vs. logistic + 4 features). B2 is a model-class change (binary classifier vs. survival regressor) with strictly more parameters per feature (RSF) or comparable (Cox). Lifting the threshold to 0.03 controls for the extra capacity — a 0.01-0.02 lift on CSU at AUC 0.91 is within noise of the same-architecture binary LR.

## 7. Methodology

### Baseline arm
- Logistic regression with `class_weight="balanced"`, `max_iter=2000`.
- Manifest-filtered pre-anchor features (production-parity filter via `_filter_to_manifest_safe`, identical to B3 contrast script).
- 5-fold stratified CV; same seed as engineered arm; report mean + std.
- Target: binary `treatment_initiated`.

### Survival arm
- Same feature surface (manifest-filtered pre-anchor numeric features).
- 5-fold stratified CV on the binary event indicator (to keep fold composition comparable; the survival model uses `(time, event)` jointly within each fold).
- Per fold:
  - Cox: `CoxPHSurvivalAnalysis(alpha=1e-3)` to regularize against collinearity (CSU surface is collinear per B3 finding).
  - RSF: `RandomSurvivalForest(n_estimators=100, min_samples_leaf=15, n_jobs=-1, random_state=seed)`.
- Metric: Harrell concordance index via `sksurv.metrics.concordance_index_censored` on the validation fold.

### Pre-processing
- `SimpleImputer(strategy="median", keep_empty_features=True)` (matches B3 M3 fix).
- `StandardScaler()` for Cox (linear model needs scaling); RSF gets unscaled (tree-based).
- Pipelines fit on train fold, applied to val fold.

## 8. Reproducibility

- Seed: 42 (matches B3).
- Folds: 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on event indicator.
- Model hyperparameters frozen in this section. No tuning.

## 9. Decision rule (locked here BEFORE measurement)

After running `scripts/measure_b2_cindex_contrast.py`:

1. **Report** per-cohort `c_cox - auc`, `c_rsf - auc` deltas in `docs/calibration/b2_cindex_contrast_20260512.json`.
2. **Best survival model per cohort**: `c_best_per_cohort = max(c_cox, c_rsf)`.
3. **Verdict per cohort**:
   - IMPROVEMENT if `c_best - auc >= 0.03`.
   - REGRESSION if `c_best - auc <= -0.03`.
   - NULL otherwise.
4. **B2 closure**:
   - IMPROVEMENT on CSU: **POSITIVE CLOSURE** — survival framing adds material discrimination signal; production wiring becomes a v5+ workstream (NOT a B2 deliverable; B2 is the empirical demonstration only).
   - NULL or REGRESSION on CSU: **NULL CLOSURE** — survival framing does not add signal at current CSU feature surface; B2 closes per pre-spec §6 + v5 §4 risk register.
   - Optum result is non-load-bearing per §6.

5. **No threshold shopping**. The 0.03 threshold is locked here.

## 10. Files (planned, will be reified by ralph-loop iterations)

- `src/agents/ml_foundation/model_trainer/nodes/survival_model.py` (NEW; ~250 LOC) — pure helper `fit_cox_rsf(X, time, event, seed) -> Dict[str, BaseEstimator]` + LangGraph node `survival_model_node(state) -> Dict[str, Any]` (replay-safe per B3 H3 lesson — returns mutated state in patch, no in-place mutation).
- `src/agents/ml_foundation/model_trainer/nodes/__init__.py` (export).
- `src/agents/ml_foundation/model_trainer/state.py` (3 new fields).
- `scripts/audit_b2_survival_target_feasibility.py` (NEW; reproducibility-anchor for §2 audit table).
- `scripts/measure_b2_cindex_contrast.py` (NEW; CV contrast script).
- `tests/unit/test_model_trainer/test_survival_model.py` (NEW; ~15-25 tests).
- `docs/calibration/b2_cindex_contrast_20260512.json` (artifact).

## 11. Codex review pattern (mirror B3)

- **Pass-1**: codex:codex-rescue with full diff + acceptance criteria; expected findings categorized HIGH / MEDIUM / LOW.
- **Pass-2**: codex:codex-rescue on the post-pass-1 state; verify all pass-1 findings closed; surface any new defects.
- **Per-fix per-commit**: each codex finding gets its own commit (matches B3 commits `a96465bf` + `792506e4`).

## 12. What this memo is NOT

- NOT positive-evidence for production survival deployment. Even an IMPROVEMENT verdict on CSU is a single-cohort empirical demonstration; production deployment requires a separate workstream (calibration of survival probabilities, manifest expansion, regulatory artifact alignment).
- NOT an HBLP-coupled experiment. Survival modeling is orthogonal to leakage classification; the manifest-filtered pre-anchor feature surface is shared with the binary baseline so the contrast is solely about model class.
- NOT a sample-size workaround for Optum. Optum's data-fidelity bound is documented in §2; B2 does not pretend to solve it.

---

## 13. Measurement section (PRE-RUN — will be amended with empirical results)

This section is intentionally empty at memo-creation time. Results from `scripts/measure_b2_cindex_contrast.py` will be appended as §13.1 (CSU) + §13.2 (Optum) + §13.3 (verdict).
