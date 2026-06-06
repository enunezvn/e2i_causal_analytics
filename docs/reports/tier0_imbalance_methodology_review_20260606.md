# Class-imbalance methodology review — is it sound, and what to do (2026-06-06)

## Question
Tier-0 models keep failing the QC/deployer gate and the failure is being
attributed to **class imbalance**. Is the current imbalance methodology sound?
If sound, can it be enhanced; if not, what should we do instead?

## Sources reviewed
1. **ULB fraud-detection-handbook, Ch.6 "Imbalanced Learning"** (cost-sensitive,
   resampling, ensembles; toy + simulated + real-world).
2. **"Separability in Class Imbalance"** (Chawla / Daily-Dose-of-Data-Science).
3. **The Optum-mart disproof** — `tier0_optum_mart_initiation_events_disproof_20260606.md`
   (full 787K population, real split: AUC ~0.68, PR-AUC ~0.029 ≈ 2× the 1.41%
   baseline; sampling/scaling/regularization all measured null → feature-bound).
4. **The codebase** — `src/agents/ml_foundation/model_trainer/` (detect_class_imbalance,
   apply_resampling, hyperparameter_tuner, evaluator, graph).

## Bottom line
The imbalance-handling **code is largely sound** — in several respects better
than typical production systems. But it is aimed at the **wrong problem**. The
models are not failing because of class imbalance; they are failing because the
classes are **not separable in the current feature space** — a feature/signal
problem. All four sources agree that imbalance techniques cannot fix that, and
tend to make the **PR-AUC the gate reads slightly worse** while inflating
balanced accuracy into a mirage of improvement.

## Evidence — four independent directions converge
| direction | finding |
|---|---|
| Disproof (real 787K) | feature-bound ceiling AUC ~0.68 / PR-AUC ~0.029; sampling, scaling, regularization all null |
| ULB handbook | resampling/cost-sensitive improve AUC/balanced-acc but are "usually detrimental to Average Precision"; best AP = no rebalancing; weighted-XGBoost optimal class weight = 1 |
| Separability notebook | imbalance only hurts when classes OVERLAP; separable classes train fine at any ratio |
| Reproduction (below) | every imbalance treatment degrades PR-AUC while inflating balanced accuracy; separability — not balance — is the lever |

## Reproduction (mechanism-faithful; real data isn't committed so it can't run here)
`docs/reports/imbalance_separability_repro/imbalance_repro.py` (logistic-link
DGP at 1.41% prevalence, 64 features, calibrated to the disproof's AUC≈0.68
ceiling; split-then-resample-train-only; lbfgs — which the disproof proved
matches the thrashing saga's AUC in ~20 iters). It also drives the codebase's
**own** `detect_class_imbalance` + `apply_resampling` nodes (faithful, not mocked).

Faithfulness check vs the real-data disproof:

| metric | disproof (real 787K) | reproduction (signal=0.2) |
|---|---|---|
| prevalence | 1.41% | 1.41% |
| baseline AUC | 0.676 | 0.689 |
| baseline PR-AUC | 0.0294 | 0.0333 |
| PR-AUC lift | ~2.1× | 2.36× |

Treatment comparison at the feature-bound ceiling (test PR-AUC; lift over the 1.41% baseline):

```
treatment                         AUC-ROC   PR-AUC   lift  bal_acc
LR baseline                        0.6894   0.0333   2.36   0.500
LR class_weight=balanced           0.6860   0.0320   2.27   0.637   <- PR-AUC down, bal_acc up
LR + SMOTE 1:1                     0.6875   0.0315   2.23   0.634   <- PR-AUC down
LR + SMOTE 0.5 + cw (combined)     0.6877   0.0315   2.23   0.633   <- the OLD extreme.non_tree default
LR + RandomUnderSample             0.6716   0.0292   2.06   0.624   <- worst
XGB baseline                       0.6526   0.0255   1.81   0.500
XGB scale_pos_weight=IR            0.6274   0.0233   1.65   0.555   <- PR-AUC down
XGB + SMOTE 1:1                    0.5986   0.0203   1.44   0.546   <- worst

codebase's OWN nodes:  LR -> strategy=combined  PR-AUC=0.0315 (< baseline 0.0333)
                       XGB -> strategy=class_weight PR-AUC=0.0233 (< baseline 0.0255)

SEPARABLE contrast (same 1.41% prevalence, NO imbalance handling):
LR baseline                        0.9693   0.4862  34.3    <- separability, not balance, is the lever
```

Every imbalance treatment — including the pipeline's own chosen strategies —
left PR-AUC flat or **degraded** it while raising balanced accuracy. Flip to a
*separable* feature space at the **same** 1.41% prevalence and PR-AUC jumps to
0.49 (34× lift) with **zero** rebalancing.

## Is the methodology sound?
**Imbalance machinery — yes, with one wart.** Strengths well-aligned with the handbook:
- HPO optimizes `average_precision` (PR-AUC), not AUC, at severe/extreme imbalance
  (`hyperparameter_tuner._get_default_metric`) — the handbook's #1 prescription.
- Tree models → native `class_weight`, never SMOTE; resampling is train-only with
  explicit no-leakage discipline.
- Deterministic, config-driven strategy matrix; PR-AUC / MCC / net-benefit / prevalence-aware gates.
- The wart (now fixed, below): non-tree extreme imbalance picked `combined`
  (SMOTE→0.5 + class_weight), which measured at/below the no-resampling PR-AUC baseline.

**Diagnosis/framing — not sound.** "Failed because of class imbalance"
misattributes the cause. The gate fail-closing these models is working
*correctly* — it is refusing a feature-bound model (exactly the disproof's
conclusion). No amount of imbalance handling clears it, because the gate reads
PR-AUC, which imbalance handling does not lift.

## What changed in this PR
1. **Separability diagnostic** (`feature_ceiling_diagnostic` node) — advisory; runs
   on the preprocessed train before resampling, reports native AUC / PR-AUC-lift /
   `feature_bound|intermediate|separable` so a feature ceiling is named instead of
   misread as imbalance. Does not alter control flow.
2. **Re-tuned non-tree resampling** — severe + extreme `non_tree` now use
   `class_weight` (cost-sensitive) instead of SMOTE / `combined`. These are exactly
   the regimes where the HPO objective is already PR-AUC, and where synthetic
   oversampling is empirically detrimental to it.
3. **lbfgs/saga runtime fix** — `reconcile_lr_solver` picks the fastest valid solver
   for the chosen penalty (lbfgs for l2/None, saga for l1). Identical AUC, ~20 iters
   vs 1000; `_LR_FIXED_PARAMS=saga` retained as the l1-safe floor (#232 preserved).

## Recommendation (prioritized)
1. **Richer pre-index features — the only measured lever.** Utilization
   trajectories, prior-Rx sequences, lab/dx recency from raw claims, unlinked
   HCP/market signal. Demographics+comorbidity cap at AUC ~0.68.
2. Use the new `feature_ceiling_label` to stop prescribing resampling for
   feature-bound cohorts.
3. Keep gating on PR-AUC / net-benefit, never balanced accuracy, at low prevalence.
4. The runtime fix makes a full-population HPO tractable on the prod box.

## Artifacts
- Reproduction: `docs/reports/imbalance_separability_repro/imbalance_repro.py`
  (+ `imbalance_repro_output.txt`, `README.md`).
- Code: `feature_ceiling_diagnostic.py`, `config/imbalance_strategy.yaml`,
  `src/mlops/lr_solver_policy.py`, plus wiring + tests.

*Generated 2026-06-06. Branch `claude/serene-rubin-ktLAs`.*
