# Model Success Criteria & QC Gates (Tier-0)

**Status:** production default since PR #641 (2026-06-02).
**Scope:** how a Tier-0 binary-classification model is judged pass/fail, plus the data-quality
(QC) gate that runs before training.

This is the reader-facing companion to the code in
`src/agents/ml_foundation/scope_definer/nodes/criteria_validator.py` (success criteria),
`src/agents/ml_foundation/model_trainer/nodes/evaluator.py` (gate evaluation + calibration),
and `src/agents/ml_foundation/data_preparer/nodes/qc_threshold.py` /
`qc_remediation.py` (the QC gate). When this doc and the code disagree, **the code wins** —
the values below are transcribed from it, but it is the source of truth.

---

## 1. The flag: `ADAPTIVE_CRITERIA` (default `true`)

`_adaptive_criteria_enabled()` reads `os.getenv("ADAPTIVE_CRITERIA", "true")`
(`criteria_validator.py`; there is no `is_adaptive_criteria_on()`). The env var is read
**fresh per call** so a test or run can flip it without re-import.

> **Caveat — the flag does not reach the containers.** `ADAPTIVE_CRITERIA` is read
> by code but is **not** in the `x-common-env` whitelist in
> `docker/docker-compose.yml` (`grep -c ADAPTIVE_CRITERIA docker/docker-compose.yml`
> → 0), so setting it in the host `.env` is inert inside Docker and the in-code
> default (`true`) governs. The rollback below works for local runs and CI;
> flipping it in production needs a compose change.

- **Default (`true`) — v3 "Option C" adaptive criteria.** Every binary-classification
  model-quality evaluation is governed by the regime/N/baseline-keyed v3 contract (§2).
- **Opt-out (`ADAPTIVE_CRITERIA=false`) — fixed Apr-26 baseline.** The legacy fixed bars:

  | gate | value |
  |---|---|
  | `minimum_auc` | 0.75 |
  | `minimum_precision` | 0.70 |
  | `minimum_recall` | 0.65 |
  | `minimum_f1` | 0.70 |

  (`_BINARY_CLASSIFICATION_DEFAULTS` in `criteria_validator.py`.) This is the **rollback
  path** — set the env var to `false` to revert the whole platform to fixed thresholds.

> **Why the change (Van Calster et al. 2025, *Lancet Digital Health*):** fixed precision/F1
> bars are prevalence-dependent and penalise well-calibrated models on imbalanced cohorts.
> v3 drops `minimum_precision` and `minimum_f1` (`_V3_DEPRECATED_FIXED_KEYS`) and replaces
> them with decision-analytic and calibration gates.

---

## 2. The v3 adaptive contract

`adaptive_success_criteria(n_samples, prevalence, baseline_auc, feature_count, regime,
deployment_intent, n_train, n_val, n_test)` returns `(thresholds, skipped)`
(`criteria_validator.py`). The thresholds are keyed on the run's **regime**
(`default` / `clean` / `adverse`; `None` is treated as `clean`), the
**`deployment_intent`** (§2.0), the training-split size `N`, the positive-class
prevalence, the stratified-dummy `baseline_auc`, the post-preprocessing
`feature_count`, and — for the cap widening in §2.3 — the split sizes.

**Skipped criteria are ABSENT from the dict, never present with a `None` value.** The list of
skipped names is stored on `success_criteria['_adaptive_skipped']`; the evaluator records
`met=None` for those names from that explicit list — not from a None-value heuristic (this
closes a config-typo silent-skip vulnerability).

### 2.0 The `deployment_intent` axis (`clinical` | `commercial`)

**Orthogonal to `regime`** (added 2026-06-07). It recalibrates the bar to the *use
case* rather than the data difficulty, and is set by `scripts/run_tier0_test.py
--deployment-intent` / `scripts/run_optum_tier0_test.py --deployment-intent`,
defaulting to `clinical`. `_normalize_deployment_intent()` coerces anything
unrecognised back to `clinical`, so the default **never silently loosens**.

| Gate | `clinical` (default) | `commercial` |
|---|---|---|
| `minimum_auc` | `max(0.75, baseline+0.20)`; `max(0.70, baseline+0.15)` on `adverse`; **skipped** on the `default` regime | `max(0.60, baseline+0.05)`; `max(0.58, baseline+0.03)` on `adverse`. **Never skipped** — a targeting model with no discrimination floor is meaningless |
| `minimum_recall` | 0.65 (0.50 on `adverse` / prevalence < 0.05) | 0.50 always |
| `minimum_mcc` | regime-keyed 0.45 / 0.35 / 0.20 | 0.10 — a weak guard; MCC deflates further under commercial targeting, so AUC, lift and net benefit are the load-bearing gates |
| `minimum_lift_over_baseline` | 0.10 | 0.08 |
| `_adaptive_p_t` (NB gate) | regime-keyed (§2.1) | 0.05, i.e. c_FP : c_FN ≈ 1 : 19 |

Floors live in `_INTENT_AUC_PARAMS`, `_COMMERCIAL_LIFT_FLOOR` and `_COMMERCIAL_P_T`.
The commercial AUC floor is the "minimum useful discrimination for
targeting/propensity" range (Hosmer & Lemeshow 2013), not a relaxation of the
clinical bar; the commercial `p_t` is an **economic-cost** parameter — the model
still has to clear NB > 0 at it on its merits.

> **Do not confuse the commercial floor with a runner flag.** The adaptive
> commercial floor is `max(0.60, baseline+0.05)`. The `0.65` that appears in the
> Optum docs is `scripts/run_optum_tier0_test.py --min-auc`, that runner's own
> step gate — a different mechanism. See `docs/OPTUM_MART_CONVERSION.md`.

### Active gates (transcribed from `adaptive_success_criteria` in `criteria_validator.py`)

The columns below are the **`clinical`** intent; for `commercial` read §2.0 first.

| Gate | `default` | `clean` (and `None`) | `adverse` / `prevalence < 0.05` | Notes |
|---|---|---|---|---|
| `minimum_auc` | *skipped* | `max(0.75, baseline+0.20)` | `max(0.70, baseline+0.15)` | default is a rubric-stress regime; AUC relocated to a regime-keyed expectation |
| `minimum_recall` | 0.65 | 0.65 | 0.50 | looser for low-prevalence |
| `minimum_net_benefit_at_p_t` | 0.0 | 0.0 | 0.0 | DCA gate; **NB>0 ⇔ precision > p_t**. Always fires; the regime cost ratio enters via `_adaptive_p_t` (§2.1) |
| `minimum_mcc` | 0.35 | 0.45 | 0.20 | replaces F1 (Chicco-Jurman 2020) |
| `maximum_calibration_slope_deviation` | 0.15 | 0.15 | 0.15 | regime-independent **floor**; slope ∈ [0.85, 1.15]. Widened on small test splits — §2.3 |
| `maximum_calibration_intercept_magnitude` | 0.30 | 0.30 | 0.30 | regime-independent **floor** (van Calster 2019 "moderate calibration"). Widened on small test splits — §2.3 |
| `minimum_lift_over_baseline` | 0.10 *or skipped* | 0.10 *or skipped* | 0.10 *or skipped* | skipped when the AUC SE proxy (`2·SE` at AUC=0.5) ≥ 0.10 — too noisy to be stable |
| `maximum_calibration_error` (ECE) | 0.05 if `N≥1000` else 0.10 | same | same | tighter for larger N; a **floor**, widened on small test splits — §2.3 |
| `maximum_train_val_delta` | step on `feature_count/N` | same | same | overfit gate **floor**, see §2.2 and §2.3 |

### 2.1 The net-benefit gate and `_adaptive_p_t`

The NB threshold is fixed at `0.0`; the regime-specific **threshold probability** `p_t` is
recorded on `success_criteria['_adaptive_p_t']` for audit and consumed by the evaluator's NB
grid (`_V3_REGIME_P_T` in `criteria_validator.py`; commercial intent overrides it with `_COMMERCIAL_P_T = 0.05` — §2.0):

| regime | `p_t` | implied FP:FN cost ratio | rationale |
|---|---|---|---|
| `adverse` | 0.05 | ≈ 19:1 | rare-responder; missing a responder is far costlier than a false alert |
| `default` | 0.20 | ≈ 4:1 | rubric-stress |
| `clean` | 0.30 | ≈ 7:3 | RWD-like |

(Cost ratio `c_FP/c_FN = (1−p_t)/p_t`; Vickers 2019 calibration.) Because NB>0 ⇔ precision >
`p_t`, at adverse `p_t=0.05` the gate equates to precision > 0.05 — any non-degenerate
classifier clears it; the gate's job is to kill *degenerate* models, not to impose a fixed
precision bar.

### 2.2 The overfit gate (`maximum_train_val_delta`)

A feature-density step function on `fpr = feature_count / n_samples`:

| `fpr` | `maximum_train_val_delta` |
|---|---|
| ≤ 1/50 | 0.03 |
| ≤ 1/30 | 0.05 |
| ≤ 1/15 | 0.07 |
| otherwise | 0.10 |

This gate is **calibration-invariant** — post-hoc calibration cannot close a train/val
*ranking* gap; only more data can. This is why the `clean` synthetic regime uses 4000 rows
(see `docs/SYNTHETIC_DATA.md` and `_REGIME_N_SAMPLES` in `scripts/run_tier0_test.py`).

### 2.3 The calibration and overfit caps are FLOORS, widened for small splits (#866)

Every cap in §2 and in the overfit table is a **floor, not a fixed value**. Below a
~1000-row anchor the published bands sit *inside* their own sampling noise, so a
statistically clean model fails on noise alone (measured on the known-clean
synthetic-CSU gold-standard run, 2026-06-10: clean train→val deltas 0.0397 / 0.0317
against the fixed 0.03 cap). The validator therefore widens them:

| Cap | Widening rule | Hard ceiling |
|---|---|---|
| `maximum_calibration_slope_deviation` | `0.15 × √(1000 / n_test)` | `0.30` (2× the van Calster band) |
| `maximum_calibration_intercept_magnitude` | `0.30 × √(1000 / n_test)` | `0.60` |
| `maximum_calibration_error` (ECE) | `max(base, mean + k·SD)` of the per-bin binomial noise floor at `B = 10` bins on `n_test` | `0.15` |
| `maximum_train_val_delta` | the feature-density tier stays as the floor; widened by the delta's own Hanley-McNeil SE over `n_train` / `n_val` | `0.10` — the loosest historical `fpr` tier, so never looser than pre-#866 |

Two guards keep this from becoming a loophole (codex R1 HIGH, PR #865):

- **Fail-closed below `_MIN_EVAL_SUPPORT = 100` evaluation rows** — no widening at
  all. `split_enforcer` admits splits as small as 10 rows, where unbounded SE
  scaling produced a delta cap of ≈ 0.35 and a slope cap of ≈ 0.47, wide enough to
  deploy a severely overfit model.
- **Clamped to the hard ceilings above it**, per the table.

Constants: `_CALIBRATION_ANCHOR_N`, `_MIN_EVAL_SUPPORT`, `_MAX_SLOPE_DEV_CAP`,
`_MAX_INTERCEPT_CAP`, `_MAX_ECE_CAP`, `_MAX_TRAIN_VAL_DELTA_CAP` in
`criteria_validator.py`.

---

## 3. Deploy-calibrated artifact contract (#640 / #633)

The v3 calibration gates (`slope_deviation`, `intercept_magnitude`, ECE) must judge the
probabilities that are **actually deployed**, not the raw tree's under-confident scores.
So when post-hoc calibration is applied, the **calibrated** model becomes the deployed/
checkpointed artifact:

- `evaluator.py:1100` — `deployed_model = calibrated_model`.
- `evaluator.py:1172` — `inner_test_metrics["deployed_model_is_calibrated"] = True`.

The calibration gates then evaluate the calibrated probabilities, so a green calibration gate
reflects what ships. Among models tied on discrimination (AUC), selection prefers the
better-calibrated candidate (lower `calibration_slope_deviation`) — this is the #640
calibration-aware tiebreak; see `evaluator.py` `calibration_slope_deviation` handling
(≈ lines 1114, 2123).

> ⚠️ **Faithfulness caveat (from project history):** calibration/overfit metrics computed on
> a local AVX2 machine do **not** match CI's AVX512 numbers. Treat the CI slow-tests
> `Synthetic Regime E2E` run as the faithful arbiter for these gates; never rebaseline gate
> numbers from a local run.

---

## 4. The QC gate (`overall_score`, default 0.80)

Before training, the data_preparer QC gate blocks a cohort whose weighted `overall_score`
falls below a minimum bar. That bar is resolved through a single source of truth —
`resolve_qc_min_overall_score` (`qc_threshold.py`) — used at **three** enforcement points:
`quality_checker.run_quality_checks`, `graph.finalize_output`, and the model_trainer
`check_qc_gate`.

**Resolution precedence (first valid match wins):**

1. Per-run override on agent state: `qc_min_overall_score` (caller / `PipelineConfig`).
2. Per-cohort override: `scope_spec["qc_min_overall_score"]`.
3. Ops/CI env override: `QC_MIN_OVERALL_SCORE`.
4. Default `0.80` (`DEFAULT_QC_MIN_OVERALL_SCORE`) — **unchanged baseline**.

Every candidate is parsed defensively and clamped to `[0.0, 1.0]`; a missing / `None` /
non-numeric / out-of-range override is **ignored** (falls through to the next source). A
malformed override can never silently *lower* the gate or crash it.

> **Why not regime-keyed?** (deferred with reasoning, not an invented formula) — the `regime`
> signal lives in scope_definer state, not in `ScopeSpecSchema`, so it does not reach the
> data_preparer QC gate. Keying the QC bar on N/prevalence instead would invent a formula
> without basis and risk silently moving the bar for small-N/rare real cohorts. When a
> principled regime signal is threaded into `scope_spec`, this resolver is the single place to
> add a branch — additively, since callers already opt in explicitly.

### 4.1 All-null columns are NOT imputed (#630/#631)

When QC remediation hits an **all-null column**, it does **not** placeholder-impute it. It
logs `SKIPPED impute on <column>: all-null column, requires investigation` and leaves the
column untouched so the QC completeness dimension keeps blocking
(the `SKIPPED impute on <column>` branch in `qc_remediation.py`). Rationale: a placeholder fill could pass QC and reach training
with a silently-fabricated feature. A cohort missing a required feature entirely is meant to
fail QC, not be papered over.

---

## 5. Quick reference

| I want to… | Do this |
|---|---|
| Revert to fixed AUC/precision/recall/F1 bars | `ADAPTIVE_CRITERIA=false` — local/CI only; **not forwarded by compose**, so in Docker it needs a compose change (§1) |
| Judge an HCP-targeting / propensity model | `--deployment-intent commercial` (§2.0): AUC floor `max(0.60, baseline+0.05)`, recall 0.50, MCC 0.10, lift 0.08, `p_t` 0.05 |
| Relax the QC bar for one cohort | set `scope_spec["qc_min_overall_score"]` (or `QC_MIN_OVERALL_SCORE` env) — cannot go below safe parsing; default 0.80 |
| Understand why a model failed on a balanced cohort | check MCC (0.45 clean) and calibration slope/intercept — not precision/F1 (dropped in v3) |
| Understand why `clean` uses 4000 rows | the calibration-invariant `maximum_train_val_delta` overfit gate — data quantity, not calibration, closes it |
| Trust a local calibration number | don't — use the CI AVX512 slow-tests run |

**Related:** `docs/SYNTHETIC_DATA.md` (regimes & N), `docs/data/03-ML-PIPELINE-SCHEMA.md`
(`ml_experiments` schema), `docs/data/08-LEAKAGE-DETECTION-CONTRACT.md` (leakage gate, when
created), `docs/reports/DOCUMENTATION_UPDATE_INDEX_20260603.md` (this doc = item C1).
