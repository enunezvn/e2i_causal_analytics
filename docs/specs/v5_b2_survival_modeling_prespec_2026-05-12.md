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

**Coverage threshold rationale (M5 codex pass-1)**: REAL/DATA-FIDELITY-BOUND verdict pivots on `post_index_rx_coverage_of_positives >= 0.10`. Below 10%, survival imputation noise (fallback to journey_duration censoring time for unmatched positives) dominates the event-time signal, and the cohort effectively reduces to a binary-with-fixed-horizon problem. The 10% floor preserves ≥175 informative event-time observations on CSU (1743 positives × 10%) which is the minimum for stable Cox partial-likelihood fitting.

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
# In src/agents/ml_foundation/model_trainer/state.py — 5 fields total.
enable_survival_modeling: bool = False  # gate; default off
survival_time_days: Optional[np.ndarray] = None  # float days
survival_event: Optional[np.ndarray] = None  # bool
survival_manifest_source: Optional[str] = None  # echoes manifest_source
survival_target_error: Optional[str] = None  # set if derivation raised
```

(L1 codex pass-1: this list shows all 5 fields. The §10 planned-files line
below is rounded up to 5 as well.)

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
- `src/agents/ml_foundation/model_trainer/state.py` (5 new fields).
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

## 13. Measurement section (post-run, 2026-05-12)

Run: `PYTHONPATH=. python scripts/measure_b2_cindex_contrast.py` at branch HEAD after impl commits. Artifact: `docs/calibration/b2_cindex_contrast_20260512.json`. Audit anchor: `docs/calibration/b2_survival_target_audit_20260512.json`.

### 13.1 CSU (primary cohort per §2)

| Metric | Value |
|---|---|
| n_rows | 9607 |
| n_events | 1743 |
| event time median | 214d |
| censored time median | 365d (capped) |
| binary_auc (5-fold CV mean) | **0.9057** ± 0.0021 |
| cox_cindex (5-fold CV mean) | 0.8790 ± 0.0040 |
| rsf_cindex (5-fold CV mean) | 0.8721 ± 0.0029 |
| Δ cox (cox - auc) | **-0.0267** |
| Δ rsf (rsf - auc) | -0.0336 |
| Δ best | -0.0267 |
| **Verdict** | **NULL** (|delta| < 0.03 threshold) |

### 13.2 Optum initiation (secondary per §2)

| Metric | Value |
|---|---|
| n_rows | 1294 |
| n_events | 37 |
| event time median | 180d (constant administrative censoring) |
| censored time median | 180d (same) |
| binary_auc (5-fold CV mean) | **0.6782** ± 0.0485 |
| cox_cindex (5-fold CV mean) | 0.6800 ± 0.0617 |
| rsf_cindex (5-fold CV mean) | NaN (5/5 folds: ValueError "constant-time horizon"; see §13.4) |
| Δ cox | **+0.0018** |
| Δ rsf | inapplicable |
| Δ best | +0.0018 |
| **Verdict** | **NULL** (|delta| < 0.03 threshold) |

### 13.3 B2 closure verdict

**NULL CLOSURE** per pre-spec §9 decision rule + v5 plan §4 risk register: every cohort produced |delta| < 0.03. No threshold shopping; this verdict was reachable under any cohort prioritization (CSU primary or Optum primary).

### 13.4 Reasoning about the NULL

**CSU**: cox_c = 0.879 is materially below binary_auc = 0.906. The base feature surface already saturates the discrimination signal at AUC ≈ 0.91 (in-distribution; production parity = 0.66 because the production pipeline drops the engagement_score leak). Survival framing trades some discrimination capacity for time-to-event modeling, which on a saturated binary surface yields a small REGRESSION (delta = -0.027, still well within ±0.03 null band). The architectural reading: on a strong-binary surface, survival framing cannot lift discrimination further; it produces equivalent ranking at best. RSF underperforms Cox slightly (-0.007) — consistent with the surface being approximately linear in the surviving pre-anchor features.

**Optum**: cox_c = 0.680 ≈ binary_auc = 0.678 (delta = +0.0018, sample-size-bounded by n_events = 37). RSF could not fit due to the all-constant 180d time horizon — pre-spec §4 documented this as expected. The Optum result is **non-load-bearing** per §6: no possible empirical outcome on this cohort would have closed B2 IMPROVEMENT, because the survival framing IS the binary framing when administrative censoring is constant.

### 13.5 Disease-agnostic finding

For B2 to lift discrimination materially, the data needs:
1. A genuine time-to-event signal (CSU has it; Optum at current ingest does not — backlog item to ingest continuous `time_to_initiation`).
2. A feature surface NOT already saturated at AUC ≈ 0.9 — i.e., a real-world cohort where binary AUC sits in the marginal [0.62, 0.68] band. Optum is the right cohort topologically; it's just the data-fidelity bound that blocks B2 acceptance.

This rhymes with B3's NULL finding (`docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md` §10.3): on the current Optum surface, n_events = 37 makes mean shift of any feature/model intervention indistinguishable from binomial fold variance (±0.05).

### 13.6 Forward-looking implications (not B2 deliverables)

- v4 backlog #32 (Optum cohort growth) becomes load-bearing for any future survival re-run.
- v4 backlog #34 (Optum survival-target re-ingest with continuous `time_to_initiation`) is the prerequisite for Optum RSF being non-degenerate.
- Cox is a viable model class for Optum at the current cohort size if RSF is dropped — but it does not lift discrimination over binary LR per the empirical contrast.
- The implementation surface (CoxPHSurvivalAnalysis + RandomSurvivalForest wired through the model_trainer state) is **kept** as engineering infrastructure regardless of the NULL verdict — future cohort-growth or data-fidelity-improvement workstreams can fire B2 contrast without re-implementing.

### 13.7 Closure checklist (per pre-spec §1)

- [x] Pre-spec memo committed BEFORE measurement (commit `52a2aa16`).
- [x] Cohort feasibility audit published as artifact (b2_survival_target_audit_20260512.json).
- [x] Survival target derivation declared in state (model_trainer/state.py, 5 new fields).
- [x] Pure-helper + LangGraph node split (replay-safe per B3 H3 lesson; verified by `test_node_does_not_mutate_state_in_place`).
- [x] 5-fold CV c-index contrast script wired (scripts/measure_b2_cindex_contrast.py).
- [x] 25 unit tests passing.
- [x] Empirical result documented above with verdict locked at pre-spec threshold.
- [ ] Codex pass-1 review (next phase).
- [ ] Codex pass-2 review (after pass-1 fixes).
- [ ] CI green.

