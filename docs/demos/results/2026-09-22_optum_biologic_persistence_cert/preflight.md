# Lane A pre-flight: the causal_impact agent on the real Optum biologic-persistence frame

**Verdict: PASS (suggestive, host process).** After two engine fixes the full-graph agent run
(Auto estimator, discovery off, full cohort n = 15,209 × 77 resolved covariates) completes in
**267 s** against the 900 s hard cap, on all three pre-flighted outcomes; both earlier runs had
timed out at 900 s. Read as: the run shape is now viable; the decisive measurement is the live
API cert after deploy (plan Task 12) — this is the host venv, not the `e2i_api` container.

## Runs (in-process, `preflight_agent.py`, run from the worktree root)

| outcome | n | k | estimator | wall s | max RSS | status | ATE | 95 % CI | p | naive ATE | refutation | gate | E-value | dag_source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `persistent_at_180d_g28` (primary) — BEFORE fixes, Auto | 15,209 | 76 | — | 900.5 | 1.04 GB | **failed** (timeout at the refutation reconstruction) | — | — | — | — | — | — | — | — |
| `persistent_at_180d_g28` — BEFORE fixes, forced LinearDML | 15,209 | 77 | — | 900.4 | 1.42 GB | **failed** (same place) | — | — | — | — | — | — | — | — |
| `persistent_at_180d_g28` — AFTER fixes, Auto (run alone) | 15,209 | 77 | LinearDML (energy tie → confounding-robust tie-break) | **266.9** | 1.06 GB | completed | **+0.0337** | [0.0169, 0.0505] | 8.4e-5 | +0.0100 | 4/5 passed (bootstrap: effect moved 32 %) | proceed | 1.26 | domain_knowledge |
| `persistent_at_180d_g28` — AFTER fixes, Auto, write-free re-run | 15,209 | 77 | LinearDML | 280.0 | 1.05 GB | completed | +0.0337 (identical to the digit) | [0.0169, 0.0505] | 8.4e-5 | +0.0100 | 4/5 | proceed | 1.26 | domain_knowledge |
| `discontinued_180d` — AFTER fixes, Auto | 15,209 | 77 | LinearDML | 262.8 | 1.10 GB | completed | +0.0052 | [-0.0072, 0.0176] | 0.41 | +0.0161 | 4/5 (unobserved_common_cause) | proceed | 1.28 | domain_knowledge |
| `biologic_switch_180d_flag` — AFTER fixes, Auto | 15,209 | 77 | LinearDML | 216.0 | 1.11 GB | completed | +0.0031 | [-0.0014, 0.0077] | 0.18 | +0.0028 | 4/5 (unobserved_common_cause) | proceed | 1.98 | domain_knowledge |

The two secondary outcomes are reported by the agent as null findings ("CI includes zero at
n = 15,209"). The shipped `persistent_at_180d` was NOT pre-flighted (days-supply artefact — see the
spec); it runs in Task 12 only alongside the sweep. Payloads: `preflight_<outcome>.json`
(the two timeout payloads are kept: `preflight_persistent_at_180d_g28_LinearDML.json` and the
first `preflight_persistent_at_180d_g28.json` was overwritten by the completed run — its numbers are
in the row above and in the plan's Task 9 amendment).

## Where the 900 s went, and the two fixes (both red-first, both measured)

Standalone attribution of the pre-fix budget on this frame (`timing_probe*.py/json`,
`identify_probe*.py/txt`, `identify_equivalence.py/json`):

| piece | measured | mechanism |
|---|---|---|
| bare econml `LinearDML` fit, production RF nuisances | 13.1 s (ATE +0.0335) | — |
| `LinearDMLWrapper.fit` | 605 s | of which `LogisticRegressionCV(cv=3, max_iter=500)` on the UNSCALED design **441.4 s** (5.8 s after `StandardScaler`, 76×); `_honest_ate_ci` 0.4 s; design rank 61 of 77 (econml's "Co-variance matrix is underdetermined" warning) |
| refutation node `_build_dowhy_estimate` | 478 s | `identify_effect` **467 s** (probe log: 05:13:48 → 05:21:35, "Max number of iterations 100000 reached"), econml fit ~11 s |

- **Fix 1 — `refutation.py`: `identify_effect(..., optimize_backdoor=True)`.** dowhy 0.14's default
  identifier accepts the full common-cause set on its first candidate, then repeats the search as a
  minimal-set search from the smallest subset upward; when the minimal valid set IS the full set (a
  data-built graph) that pass burns its 100,000-iteration cap of d-separation checks — k=8: 0.05 s,
  k=12: 0.65 s, k=17: 16.8 s (the cap), k=77: 467 s. The path-based search returns the same adjustment
  set at every k measured and a byte-identical LinearDML estimate at k=8 and k=12
  (`identify_equivalence.json`); on the real frame it takes 0.01 s (`identify_probe_output.txt`).
  Test: `tests/unit/test_agents/test_causal_impact/test_refutation_identify_budget.py` (red: the
  enumerator ran twice; green after).
- **Fix 2 — `nuisance_config.propensity_model()`** (`StandardScaler → LogisticRegressionCV`) at the
  9 wrapper sites in `estimator_selector.py` / `dml_learner.py`. The propensity feeds only the energy
  score (and `XLearner`'s CATE, not in the default chain). Test:
  `tests/unit/test_causal_engine/test_energy_score/test_propensity_scale_invariance.py` (red: all 400
  scores moved with the units; green after; `OrthoForestWrapper` excluded — it cannot fit any frame
  today, pre-existing `max_depth=None` TypeError). Served-output check on the LIVE synthetic datasets
  through the route's own loader and the production selector (`tournament_compare.txt`,
  `tournament_before/after.json`, paired on one loaded frame for nba_triggers in
  `tournament_paired_nba.txt/json`): 14 pairs, **0 winner flips, ATEs identical on every pair,
  max |Δ energy score| 0.020** (one pair; all others ≤ 0.003).

Post-fix node timeline of the primary run (from the run's own INFO log; the graph itself started at
~07:19:58 after imports):

```
2026-09-22 07:21:19,792 INFO src.agents.causal_impact.nodes.estimation: Using energy score selection (path=energy_score, explicit_method=None, strategy=best_energy)
2026-09-22 07:21:19,838 INFO src.causal_engine.energy_score.estimator_selector: Energy-score tournament on stratified subsample: 5000/15209 rows (cap=5000); winner will be refit on the full 
2026-09-22 07:21:51,897 INFO src.causal_engine.energy_score.estimator_selector: Refitting tournament winner linear_dml on the full frame (15209 rows)...
2026-09-22 07:22:08,850 INFO src.agents.causal_impact.nodes.refutation: Running refutation suite for treatment_dupixent → persistent_at_180d_g28 (ATE=0.0337, CI=[0.0169, 0.0505])
2026-09-22 07:22:08,965 INFO src.agents.causal_impact.nodes.refutation: Refutation subsampled estimation data 15209 -> 5000 rows (#1419); reported ATE/CI remain the full-frame fit.
2026-09-22 07:24:13,677 INFO src.causal_engine.refutation_runner: Refutation suite completed: 4/5 passed, confidence=0.95, gate=proceed
2026-09-22 07:24:13,887 INFO src.agents.causal_impact.nodes.refutation: Persisted 6 validation records under estimate 201cb9d7-c9d4-5a6f-9a49-6e0fd952408d (causal_impact_query).
2026-09-22 07:24:14,063 INFO src.causal_engine.validation_outcome_store: Stored validation outcome 3e22a727-7727-4140-9daf-270834433b27: passed
2026-09-22 07:24:14,063 INFO src.agents.causal_impact.nodes.refutation: Logged validation outcome 3e22a727-7727-4140-9daf-270834433b27 for Feedback Learner: passed
2026-09-22 07:24:14,064 INFO src.agents.causal_impact.nodes.refutation: Refutation PASSED: confidence=0.95, tests_passed=4/5
2026-09-22 07:24:17,343 INFO src.agents.causal_impact.mlflow_tracker: Logged causal impact analysis to MLflow: ATE=0.033677220325547375, confidence=0.00
```

## Caveats (read before citing the numbers)

- **Host process, not the container**: the venv on the droplet host; `e2i_api` runs the same code
  under gunicorn with `_AGENT_HARD_TIMEOUT_S = 900` and a bounded compute executor. 267 s leaves a
  3.4× margin, but the container run in Task 12 is the measurement that counts.
- **The design is rank-deficient** (61 of 77): econml warns "Co-variance matrix is underdetermined.
  Inference will be invalid!" on the final-stage covariance. The reported CI comes from
  `_honest_ate_ci` (econml `ate_inference`); treat the interval as the estimator's, with that warning
  attached. The 16 exactly-collinear columns are one-hot dummies and score/band pairs — a follow-up
  is to drop redundant dummies at load time.
- **Leaderboard cap**: `_DISCOVERY_ROW_CAP = 5,000 < 15,209`, so a discovery run would see a stratified
  subsample; the agent path used here loads the full cohort (`limit=20000`). The energy-score
  tournament itself ran on its own 5,000-row stratified subsample and refit the winner on the full frame
  (#1392), and the refutation suite reconstructed on a 5,000-row subsample (#1419) — both are the
  production shapes.
- **Refutation**: 4/5 on every outcome. The primary fails `bootstrap` (effect moved 32 % across
  bootstrap resamples, "medium" instability); the two null secondaries fail `unobserved_common_cause`.
  Gate = proceed on all three.
- **`bootstrap` and lbfgs**: no ConvergenceWarnings remain after the scaler; the pre-fix runs emitted
  them from the unscaled propensity fit.
- **Prod writes — an incident, disclosed.** The plan claimed the pre-flight "runs WITHOUT `.env` so it
  writes nothing to prod tables". That was FALSE and untested: `src/ml/data_loader.py` calls
  `load_dotenv()` lazily and, because the worktree sits inside the main checkout, finds the main
  checkout's `.env` (prod Supabase URL + service-role key). The FIRST run that reached the refutation
  node (07:24 UTC, the 266.9 s row) persisted **6 rows to `causal_validations`**
  (estimate_id `201cb9d7-c9d4-5a6f-9a49-6e0fd952408d`, source `causal_impact_query`), **1 row to
  `validation_outcomes`** (outcome_id `3e22a727-7727-4140-9daf-270834433b27`, estimate_id
  `c44cc1c9-bb39-44e9-b210-6a064c0f686a`) and **one MLflow run** (`faef563d27c44517b8c403820e2627f9`,
  experiment 219 `e2i_causal/causal_impact/default`, run name `analysis_20260922_072414`). They were
  NOT deleted — that is the owner's call; the cleanup SQL is in the PR description. `preflight_agent.py`
  now blanks the Supabase env before importing `src` and points MLflow at a scratch file store; the
  three later runs logged "Validation persistence unavailable (no Supabase config)" and the prod row
  counts were identical before and after them (3,265 / 458).
- **Run from the worktree root.** From the evidence dir the script imports `src` from the main checkout
  (editable `.pth`); its own assert catches it.
- The `.log` files are gitignored; the status lines are the table above and the timeline is pasted in.
