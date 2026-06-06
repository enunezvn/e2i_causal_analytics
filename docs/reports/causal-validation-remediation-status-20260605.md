# Causal Validation Pipeline — Remediation Status Audit

**Date:** 2026-06-05
**Audits:** `docs/reports/causal-validation-pipeline-review-20260605.md` (the 67-finding review)
**Against:** `main` @ `2b7ffb7f` (== `origin/main`, deployed) — contains all 12 remediation PRs **#712–#723** (P1–P12), confirmed ancestors.
**Method:** 80-agent verify→adversarial-refute workflow (one agent reads the *current* code at each finding's location, traces the prod call path for fail-open/wiring findings; a second agent independently re-reads to refute the verdict). Dispatcher independently re-verified the H6 overturn. **Read-only — nothing changed.**

> Headline: **Both CRITICAL and 10 of 12 HIGH are fully closed.** The two remaining HIGHs (H6, H11) are *partially* closed with a real prod-reachable residual. The bulk of still-open work is **MEDIUM fail-open / reachability findings that had no dedicated remediation phase**, plus LOW hygiene and one follow-up. The dominant remaining risk is the same theme the review named #1: **fail-open validation at the multi-library pipeline seam.**

---

## 1. Status tally (40 audited findings/clusters)

| Status | Count | IDs |
|---|---|---|
| **ADDRESSED** | 19 | C1, C2, H1, H2, H3, H4, H5, H7, H8, H9, H10, H12, IV4, M-stat3, M-stat5, M-stat6, + FU-memhooks, FU-instrument, FU-prioritizer |
| **PARTIAL** | 8 | **H6, H11**, M-stat1, M-stat4, M-fo3, M-reach1, M-reach2, L-cluster-A |
| **NOT_ADDRESSED** | 13 | M-stat2, M-fo1, M-fo2, M-fo4, M-fo5, M-reach3, M-est1, M-est2, M-est3, M-gb1, M-gb2, L-cluster-B, FU-hasher |

Adversarial pass overturned **two** first-pass verdicts: **H6** ADDRESSED→PARTIAL (dispatcher-confirmed) and **M-reach2** ADDRESSED→PARTIAL (benign — see §5).

---

## 2. Fully closed — CRITICAL + 10 HIGH (no action)

| ID | Was | Verified fix (current code) |
|---|---|---|
| **C1** | `/hierarchical/analyze` fabricated `np.random` data → COMPLETED CATEs | No `np.random` anywhere in `causal.py`; non-demo path fail-closes via `_resolve_hierarchical_dataframe` → 503 (no real backend) / 400 (missing cols); `demo_mode` gate emits pinned zeros + `is_demo=True` + do-not-use warning. Tests assert new contract incl. `"np.random" not in getsource`. |
| **C2** | LIML κ eigenproblem args swapped (κ≈0.26) | `liml.py:309` → `eigvalsh(WtM0W, WtMzW)`, `min` now ≥1. |
| **H1** | Refutation fails open → `completed`+full confidence | Router diverts refutation-error to error path; `_build_output` keys status on `refutation_ran && !gate_blocked`, not ATE-presence. |
| **H2** | REVIEW band surfaced as `passed`/robust | `refutation_passed = proceed-only`; distinct `needs_review`/`gate_decision`/`review_caveat`; interpretation downgrades `overall_robust=False`; docstrings reconciled. (Deeper expert-review *workflow* = M-reach1, partial.) |
| **H3** | E-value used raw ATE as standardized | Effect standardized by outcome SD before RR in **both** engines. |
| **H4** | Hausman used residual not coefficient variances | Coefficient-variance Hausman + Sargan covariate projection. |
| **H5** | Qini normalized area by a height | Qini/AUUC normalize by **area** (`np.trapezoid`); honest estimand labels. |
| **H7** | LiNGAM adjacency untransposed (edges reversed) | Transpose applied; `get_mixing_matrix` kept on raw structural B; orientation test added. |
| **H8** | `consensus_confidence` mislabeled `library_agreement_score` | Real sign-concordance agreement surfaced as `library_agreement_score`; `consensus_confidence` kept separate. |
| **H9** | DoWhy hardcoded `confidence=1.0` dominated consensus | Inverse-variance consensus; DoWhy SE gated to `linear_regression`. |
| **H10** | `effect_size==0.0` dropped to NULL | `... is not None else None`. |
| **H12** | `require_dag`/`min_algorithm_agreement` inert knobs | Inert knobs **deleted** (+ docstring claims). |
| **IV4** | κ test too loose to catch C2 | Over-identified, genuinely-endogenous regression test added. |
| M-stat3 | AUUC height-as-area | Normalizes by area. |
| M-stat5 | Sargan ignored covariates | Covariate projection added. |
| M-stat6 | Ensemble confidence ÷ failed algos | Divides by converged count. |

---

## 3. PENDING inventory

Legend — **Reach:** PROD = user-reachable now · DARK = code exists but not reachable on current prod config · TEST = test-only.

### 3a. Partial HIGHs (real prod-reachable residual — finish the started work)

| ID | Sev | Reach | Residual (current code) |
|---|---|---|---|
| **H6** | HIGH | PROD | Engine bridge fixed (`analyzer.py:642` uses `cate_se`), but **two un-patched callers still feed raw dispersion as the SE**: `src/api/routes/causal.py:454` and `src/agents/heterogeneous_optimizer/nodes/hierarchical_analyzer.py:532` both `ate_std=seg.cate_std or 0.01` → aggregate CIs ~√n too wide, I²/τ² wrong, on the live `POST /hierarchical/analyze` route and the heterogeneous_optimizer node. **Fix:** propagate `cate_se` onto these two callers (fall back to `cate_std` only when None), or have them consume the engine's already-fixed `result.nested_ci`. |
| **H11** | HIGH | PROD | `StoreResult{persisted,degraded}` + ERROR-tagged metric now exist and the in-engine `store()` uses them, **but the production caller can't see them**: `log_validation_outcome` (`validation_outcome_store.py:993`) returns the bare id and `refutation.py:667` logs success unconditionally. **Fix:** surface `StoreResult` through `log_validation_outcome`; have `refutation.py` branch on `persisted/degraded` (warn / skip Feedback-Learner signal when degraded). |

### 3b. MEDIUM — fail-open / reachability (no dedicated phase; mostly untouched)

| ID | Reach | Residual |
|---|---|---|
| **M-reach3** | PROD | `/causal/sequential` + `/causal/parallel` **never run refutation** and responses carry **no "unvalidated" flag** — 2 of 3 user-facing estimate paths return an ATE with zero robustness validation. `executors/dowhy.py` still hardcodes `"refutation_results": {}` ("Empty until C-2/C-6 wires refutation as a pipeline stage"). **Single biggest remaining fail-open.** |
| **M-fo2** | PROD | Cyclic (non-DAG) NetworkX graph does **not** block estimation in the pipeline path — only a 50% confidence haircut (`orchestrator.py:266`, `executors/networkx.py:137`, `sequential.py:255`). |
| **M-fo1** | — | `PipelineValidator`/`StageValidators` (refutation-failure / negative-CATE / identification gates) **never invoked** by any pipeline; P11 added an honest docstring only. Wire-or-delete. |
| **M-fo4** | PROD | PC/GES/FCI wrappers swallow `_graph_to_adjacency` parse exceptions with `pass` and still return `converged=True` with an empty edge list — a failed parse is indistinguishable from a legitimately edge-free discovery. |
| **M-fo5** | TEST | `split_validator.py` temporal-overlap is `severity="warning"` (never flips `is_valid`); leakage-detector exceptions swallowed into a warning. |
| **M-gb2** | PROD | Adjustment-set finder (`graph_builder.py:398-419`) treats **any** non-endpoint node on a backdoor path as blocking — no collider/d-separation check → conditions on colliders (M-bias), biased backdoor adjustment on **every** graph build on the live agent path. |
| **M-gb1** | DARK | `_run_discovery` reads `data_cache.get("data")` but the populated key is `"estimation_data"` → auto-discovery silently skipped (raises→swallowed). Dark today (`auto_discover` never True in prod) but a latent wiring bug. |
| **M-reach1** | PROD | `ExpertReviewGate` is now *invoked* on REVIEW, but always **repository-less** (`refutation.py:434` → no-arg gate; no `ExpertReviewRepository` is constructed anywhere in `src/`) so it self-bypasses (`PROCEED/is_approved=True`). The H2 user-signal carries via flags/caveat; the actual human-review queue/block does not exist. Wire a repo, or formally accept flag-only and delete the gate. |
| **M-reach2** | TEST | Benign. Synthetic-validator cluster genuinely unreachable from prod (behavior fine); only a method-level docstring was relabeled, not the package-level docstrings. Doc nit. |

### 3c. MEDIUM — statistical-gate sharpening

| ID | Reach | Residual |
|---|---|---|
| **M-stat1** | PROD | Wrong-CI-bound fixed (now `min(abs(lo),abs(hi))`), but **null-crossing guard still missing in both engines** (`sensitivity.py:49`, `refutation_runner.py:1232`): a CI straddling 0 (e.g. (-0.3,0.5)) yields E-value>1 → falsely "robust." Should collapse to 1.0 when `sign(lo)≠sign(hi)`. |
| **M-stat2** | PROD | E-value **gate** decides off the **point-estimate** E-value; the conservative CI-bound `e_value_ci` is computed (`refutation_runner.py:1283`, `sensitivity.py`) but never compared to a threshold. |
| **M-est3** | PROD | `max_acceptable_energy_score` is **warn-only** (`estimator_selector.py:1240`): a high (unreliable) energy score still selects the estimator whose ATE flows downstream on the causal_impact path; no `requires_review`/block. |
| **M-est2** | PROD | ~55% of the energy score (treatment_balance 0.35 + propensity 0.20) is estimator-invariant; outcome_fit RMSE capped at 1.0 (`score_calculator.py:403`) collapses all poor estimators to one score. (Selection *direction* is correct; discrimination is weak.) |
| **M-stat4** | PROD | Uplift `_calculate_ate` honest label exists **only in the internal docstring**; the consumer-facing `UpliftResult` docstring (`base.py:97`) still says "Average Treatment Effect/on Treated/on Control" and `to_dict()` emits bare `ate/att/atc` with no provenance — and `att==atc` (both are `mean(predicted scores)`). Carry the label on the result object, or compute real ATT/ATC. |

### 3d. LOW + follow-up

| ID | Reach | Residual |
|---|---|---|
| **L-cluster-A** | mixed | 3 of 4 remain: dead `np.cumsum` (`uplift/metrics.py:149-150`); parallel fail-fast cancels remaining libraries on first failure (`parallel.py:269`, mitigated by `fail_fast=False` default); expert-review pending lookup omits brand (`expert_review_gate.py:175`, `repositories/expert_review.py:348`) → cross-brand pending review can gate. (1 of 4 fixed incidentally by P4.) |
| **L-cluster-B** | mixed | All 5 remain: `query_failures` LIMIT-before-JSONB-filter (`validation_outcome_store.py:566`); SKIPPED refuters pad confidence at 0.5 (`refutation_runner.py:1376`); misleading `PROCEED` enum comment (`refutation_runner.py:122`); synthetic `refutation_pass_rate=1.0` when DoWhy unavailable; estimation partial-success edge skips refutation (`graph.py`). |
| **M-est1** | TEST | `_estimate_ate_simple` still returns a Pearson correlation as the ATE (docstring relabeled only); test-only reachability. |
| **FU-hasher** | PROD | `discovery/hasher.py:79` gates `np.round` behind `dtype in [float64,float32]`, so an **object-dtype** DataFrame (mixed categorical+float — the common pharma case) bypasses the 8-decimal rounding contract → unstable `DiscoveryCache` keys. P11's "cache integrity" touched `cache.py` (corrupt-entry handling), not `hasher.py`. |

### 3e. Follow-up files CLEARED (no fix needed — note the nuance)

`FU-instrument` and `FU-prioritizer` returned **ADDRESSED meaning "the critic's concern does not hold / no independent defect found,"** not "we fixed it":
- `instrument_analyzer` runs **2SLS, not LIML** (no κ field) so C2 cannot reach it; the only ranking-gating consumer (`prioritizer._apply_instrument_availability_bonus`) reads **only** `instrument_strength` + `first_stage_f_stat` (sound Staiger-Stock F≥10), **never** any `hausman_*` key, so H4 can't corrupt ranking. Bonus-only/asymmetric, fail-closed below n-floor. Reachable via the gap_analyzer graph.
- `FU-memhooks` is **genuinely fixed** at producer + persistence layers (REVIEW → `refutation_passed=False`, semantic/RAG write gated on `is_proceed_validated`) — though the causal_impact memory_hooks currently has **no caller in `src/`**, so the RAG-amplification was latent, not active.

---

## 4. Recommended execution sequence

Mirrors the team's proven method (one worktree/branch per phase, red-first TDD + codex-rescue, serial deploy). Ordered by present-harm × reachability, with file-locality respected to keep branches coherent.

### R1 — Close the two partial HIGHs *(HIGH severity, small effort, finishes started work → best ROI)*
- **H6:** propagate `cate_se` (true SE) onto `routes/causal.py:454` and `heterogeneous_optimizer/nodes/hierarchical_analyzer.py:532` (or consume `result.nested_ci`).
- **H11:** thread `StoreResult` through `log_validation_outcome` → `refutation.py`; branch on `persisted/degraded`.
- *Test:* assert aggregate CI width is SE-based (not dispersion) on the API route; assert a degraded write flips a caller-visible flag.

### R2 — Pipeline robustness seam *(the dominant remaining fail-open; same theme as review #1)*
- **M-reach3:** wire `RefutationRunner` as a sequential/parallel pipeline stage, **or** stamp `robustness_validation_performed=false` + warning on `SequentialPipelineResponse`/`ParallelPipelineResponse`.
- **M-fo2:** hard-block (or REVIEW) estimation on a cyclic NetworkX graph instead of a soft 0.5 penalty.
- **M-fo1:** wire `PipelineValidator.validate_full_pipeline` into `Sequential/Parallel.execute()`, **or** delete per its own documented intent.
- *Files:* `pipeline/executors/dowhy.py`, `orchestrator.py`, `sequential.py`, `parallel.py`, `validators.py`, `routes/causal.py` response models.

### R3 — Statistical-gate sharpening
- **M-stat1:** null-crossing guard (E-value→1.0 when CI straddles 0) in both engines.
- **M-stat2:** gate on the conservative `e_value_ci`, not the point estimate, in both engines.
- **M-est3:** make `max_acceptable_energy_score` a hard REVIEW/block.
- **M-est2:** uncap/rescale `outcome_fit` RMSE (saturating-monotone, e.g. tanh) and/or reweight toward estimator-variant components.
- *Files:* `nodes/sensitivity.py`, `refutation_runner.py`, `energy_score/*`.

### R4 — Live agent-graph causal-correctness
- **M-gb2:** collider-aware adjustment-set finder (delegate to networkx d-separation / `is_valid_adjustment_set`).
- **M-fo3:** `_build_output` consults `sensitivity_error` to downgrade status/needs_review/confidence; narrative flags a failed sensitivity analysis instead of reporting a defaulted E-value.
- **M-gb1:** read `estimation_data` (not `data`) in `_run_discovery` (latent; cheap).
- **M-stat4:** carry honest labeling/`data_provenance` on `UpliftResult`/`to_dict()` (or compute real ATT/ATC).
- *Files:* `nodes/graph_builder.py`, `agent.py`, `nodes/interpretation.py`, `uplift/base.py`.

### R5 — Wire-or-delete decisions + latent + hygiene *(needs product decisions)*
- **M-reach1:** construct an `ExpertReviewRepository` + thread a repo-backed `ExpertReviewGate` into the refutation node (so REVIEW can actually queue/block), **or** formally accept flag-only and delete the gate.
- **M-fo4:** discovery wrappers surface parse failure as `converged=False`/error.
- **FU-hasher:** dtype-aware rounding for object-dtype DataFrames in `discovery/hasher.py`.
- **M-est1 / M-fo5 / M-reach2:** synthetic-validator cluster (test-only) — wire-or-delete or formally defer.
- **L-cluster-A / L-cluster-B:** hygiene + minor fail-opens.

---

## 5. Honesty / method notes
- Verdicts are against the *current source*, not commit messages (several "fixes" were docstring-only relabels caught by the adversarial pass: M-est1, M-stat4, M-fo1, M-reach2).
- For every fail-open/wiring finding the agents traced the actual prod call path; "ADDRESSED" required reachable corrected code. Two ADDRESSED→PARTIAL overturns (H6 dispatcher-confirmed; M-reach2 benign).
- Reachability caveats are explicit (PROD/DARK/TEST) so effort isn't spent hardening dark paths before the prod-reachable ones.
