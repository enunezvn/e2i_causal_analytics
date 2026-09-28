You are auditing a statistical root-cause analysis. Try hard to FALSIFY its conclusion. Work read-only.

Do NOT run mypy (or any type checker) on this machine; do not run the whole pytest tree; any pytest must use `-n 0`.
Do not touch the database; everything you need is in files. Do not run heavy model fits (the box is memory-constrained); cheap pandas/numpy on the CSVs is fine.

## Question
The Digital Twin's estimator (src/digital_twin/effect/cohort_causal_estimator.py estimate_cohort_effect: CausalForestDML, treatment binarized at median, X = region+specialty one-hot, W = market_share + log1p(triggers_total_count), DR interval cf.ate_ +- z*cf.ate_stderr_) reports rep_training_score (planted beta = 0 on hcp_brand_adoption.adopted) as significant on Remibrutinib: +0.047, CI (+0.008,+0.086). Fabhalta +0.006 and Kisqali -0.011 cover 0. Is this a defect (substrate DGP, adjustment set, estimator/interval) or chance?

## Mechanics (read these)
- adopted DGP: src/ml/synthetic/generators/hcp_adoption_artifact.py::_compute_adoption (centrality, specialty affinity, treatment_arm, channel_shift, logit noise, then Bernoulli with rng.random).
- Re-plant that produced the live labels: scripts/backfill_hcp_treatment_arm.py::derive, seed 427 (owner-executed 2026-09-23).
- Channel values: scripts/backfill_segment_engagement.py::generate_dgp (each channel = f(market_share, log volume, region) + independent child-RNG noise, row level), collapsed per (hcp,brand) by src/data/per_hcp_cohort_collapse.py.
- Analysis dir: /home/enunez/Projects/e2i_causal_analytics/.worktrees/t2-evidence/docs/demos/results/2026-09-28_remi_null_caveat  (remi_null_probe.py, bern_resid_all_channels.py, summarize.py, and their CSV outputs, run.log, summary.txt). Earlier MC: docs/demos/results/2026-09-23_ab_reload_interval/interval_mc.py + interval_mc.csv (it re-drew adopted with derive seeds while holding rollups/centrality fixed).

## Measurements (all on the twin's own load_cohort_frame frame, read-only)
1. derive(seed=427) reproduces live adopted and treatment_arm 100% on all joined rows (3354/3404/3378).
2. Decomposition (decompose.csv): adjusting for the estimator's X+W by OLS, the Remi null-tbin contrast in adopted is +0.0466 (z 2.79). Split: p_true = sigmoid(realized logit) contributes +0.0021 (z 0.23) -- this contains centrality, affinity, channel shift (other 7 channels), treatment_arm term and logit noise; the Bernoulli residual adopted - p_true contributes +0.0445 (z 3.28). Fab/Kis: bern_resid -0.010/-0.017. Across all 24 channel x brand cells, bern_resid z has mean +0.18, sd 1.14, max |z| 3.28 (the Remi null).
3. Correlations (null_tbin_correlations.csv): raw r of Remi null tbin with other tbins 0.07-0.12, with centrality -0.01, treatment_arm -0.01; partial r given X+W all |r| < 0.04; similar in Fab/Kis.
4. Augmented-W refits (refit_augmented_w.csv): Remi null default +0.047 (excl 0); +other 7 tbins +0.034 (-0.001,+0.069); +DGP drivers (centrality_z, affinity, treatment_arm) +0.052 (excl 0); +both +0.043 (+0.014,+0.073). Planted channels, mean error default +0.003, mean |err| 0.0215 default -> 0.010 with +both; mean SE 0.027 -> 0.018; coverage 1.0 / 0.95.
5. Placebo (rep_training permuted within region, real adopted): exclusion rate Remi 6/100, Fab 2/40, Kis 3/40; SE/empSD 1.04/1.03/0.96.
6. Fresh-seed null (100 new derive seeds, design fixed, arm+noise+coin flips re-drawn): Remi mean +0.0009 (SE 0.0018), empSD 0.0179, excl0 3/100; seed 427's +0.0472 is z 2.59 vs that distribution; 0/100 seeds >= it, 1/100 with |ate| >= it.

## Draft conclusion
NOT A DEFECT (calibrated chance). The +0.047 lives entirely in the final Bernoulli coin flips of seed 427 (brand_rng uniforms), which are independent of rep_training by construction (separate RNG streams, separate scripts). Explanations B (omitted DGP confounders) and C (leakage from correlated planted channels) are refuted: the structural part p_true carries +0.002; adding the drivers and/or other channels to W leaves +0.043; the systematic bias across 100 seeds is +0.001 +- 0.002. The interval is calibrated (placebo 6%, seeds 3%, SE/empSD ~1). The seed-427 realization is a ~2.6-SD tail event (about 1-in-100 per brand, about 3% for "any of 3 brands"). Planted channels are not biased (mean err +0.003); augmenting W with the other channels' tbins is a PRECISION gain (SE -35%), not a bias correction.
Recommendation (not implemented): do not change the estimator or re-draw the seed (re-seeding to make the null look clean is p-hacking). Correct the harness wording in scripts/verify_adoption_channel_recovery.py KNOWN_NULL_BIAS ("known per-brand bias") because the data show it is not a bias but a realized draw, and keep the existing 2-of-3 null gate. Optionally evaluate adding the other planted channels to W as a precision improvement, as a separate decision.

## Your task
Try to falsify this. In particular: (a) is the decomposition valid (is bern_resid truly independent of the null tbin by construction -- check RNG streams, ordering, any shared seed between derive's brand_rng and backfill_segment_engagement's rng; check whether seed 427 was SELECTED in a way that could correlate with rep_training)? (b) does the fresh-seed experiment actually measure what is claimed? (c) is the "+0.05 before planting" in scripts/verify_adoption_channel_recovery.py consistent with this story? (d) is the recommended fix functional or cosmetic? (e) anything missed (collapse, n_metric_rows, region mix, the ETL null rows filtered out)?

If a recommendation solves a labeling problem instead of a functional problem, flag it as HIGH finding. If a recommendation preserves code without investigating intent (PR history, linked issues, user-requested functionality), flag it as HIGH finding. If a recommendation deletes code without verifying intent, flag it as HIGH finding. Audit the question being asked, not just the answer given.

End with one line: VERDICT: AGREE | AGREE-WITH-CHANGES | DISAGREE, then numbered findings with severity.
