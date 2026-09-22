# Lane A pre-flight: the causal_impact agent on the real Optum biologic-persistence frame

**Verdict: PASS (suggestive, host process).** After the engine fixes below the full-graph agent
run (Auto estimator, discovery off, full cohort n = 15,209, **61 resolved covariates after the
load-time collinearity prune**) completes in **258 s / 221 s / 185 s** on the three pre-flighted
outcomes against the 900 s hard cap, selects LinearDML on each, and serves a CI from a full-rank
final stage (a warned fit is now refused, so a completed run IS a clean served fit); both original
runs had timed out at 900 s. Read as: the run shape is viable; the decisive measurement is the live
API cert after deploy (plan Task 12) — this is the host venv, not the `e2i_api` container.

## Runs (in-process, `preflight_agent.py`, run from the worktree root)

| outcome | n | k | estimator | wall s | max RSS | status | ATE | 95 % CI | p | naive ATE | refutation | gate | E-value | dag_source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `persistent_at_180d_g28` (primary) — BEFORE any fix, Auto | 15,209 | 76 | — | 900.5 | 1.04 GB | **failed** (timeout at the refutation reconstruction) | — | — | — | — | — | — | — | — |
| `persistent_at_180d_g28` — BEFORE any fix, forced LinearDML | 15,209 | 77 | — | 900.4 | 1.42 GB | **failed** (same place) | — | — | — | — | — | — | — | — |
| `persistent_at_180d_g28` — after fixes 1+2 (round 1), Auto, run alone | 15,209 | 77 | LinearDML | 266.9 | 1.06 GB | completed, but the served CI came from a WARNED final stage (rank 61/77) — superseded | +0.0337 | [0.0169, 0.0505] | 8.4e-5 | +0.0100 | 4/5 | proceed | 1.26 | domain_knowledge |
| **`persistent_at_180d_g28` — FINAL code (fixes 1–4), Auto** | 15,209 | **61** | LinearDML (energy tie → confounding-robust tie-break) | **257.6** | 1.04 GB | **completed** | **+0.0336** | **[0.0168, 0.0504]** | 8.9e-5 | +0.0100 | 4/5 (bootstrap: effect moved 32 %) | proceed | 1.27 | domain_knowledge |
| **`discontinued_180d` — FINAL code, Auto** | 15,209 | 61 | LinearDML | 221.3 | 1.02 GB | completed | +0.0050 | [-0.0074, 0.0174] | 0.43 | +0.0161 | 4/5 (unobserved_common_cause) | proceed | 1.27 | domain_knowledge |
| **`biologic_switch_180d_flag` — FINAL code, Auto** | 15,209 | 61 | LinearDML | 185.3 | 1.04 GB | completed | +0.0034 | [-0.0012, 0.0079] | 0.15 | +0.0028 | 4/5 (unobserved_common_cause) | proceed | 2.04 | domain_knowledge |

Superseded intermediate runs (kept as evidence of what each fix changed, payloads in the session
scratchpad, status lines in the plan's Task 9 amendment): with the wrappers refusing every warned
fit but the pre-flight still bypassing the loader prune (k=77), LinearDML/DRLearner were refused and
the tournament fell to CausalForestDML — 601 s / 645 s / 646 s, refutation budget exhausted on two
of three (`status=failed`); with the prune in the path (k=61) but the refusal still unconditional,
the primary completed in 180 s while the two secondaries' 5,000-row tournament subsamples were
rank 61 of 62 (no constant column — rare dummies can be exactly dependent inside a subsample) and
again fell to CausalForest (575 s failed / 664 s). The served-fit-only refusal (fix 4) resolved it.

The two secondary outcomes are reported by the agent as null findings ("CI includes zero at
n = 15,209"). The shipped `persistent_at_180d` was NOT pre-flighted (days-supply artefact — see the
spec); it runs in Task 12 only alongside the sweep. Payloads: `preflight_<outcome>.json`
(the forced-LinearDML timeout payload is kept as `preflight_persistent_at_180d_g28_LinearDML.json`;
each outcome's `preflight_<outcome>.json` is the FINAL-code run — earlier runs' numbers are in the
table above and the plan's Task 9 amendment).

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
  `tournament_before/after.json`; then PAIRED — each pair loaded once and run with the old and the
  new propensity model — for all 14 in `tournament_paired_all.txt/json`): **14/14 same winner,
  14/14 identical ATE and CI, 14/14 identical `exceeded_max_energy_score` / `requires_review`,
  max |Δ energy score| 0.020** (one pair; all others ≤ 0.0013); every synthetic design is full rank.

- **Fix 3 — `loaders.py` `_prune_exactly_collinear` + the shared post-fetch resolver** (codex r2
  HIGH). The real design is rank 61 of 77: 16 columns are exact linear combinations of earlier
  registry columns (Elixhauser flags duplicating Charlson flags, a risk band implied by its score,
  payer dummies implied by a coarser payer axis — `collinearity_probe.json`). econml's final stage
  warned "Co-variance matrix is underdetermined. Inference will be invalid!" and the round-1 run
  served that CI. Measured: the pruned 61-column fit gives ATE 0.03353 (was 0.03353) and SE 0.00858
  (was 0.00855) with no warning — the redundant columns carry no information. The loader now prunes
  once, on the full frame, in the RESOLVED order (numerics, then dummies; earlier wins; Gram–Schmidt vs
  the intercept on the reference-subtracted, power-of-two-scaled column, residual ≤ max(n,k)·eps of the
  centered norm; skipped at n < k+1 — see Rounds 3–4 below), so
  estimation and the refutation rebuild see the same design; every
  synthetic served pair is full rank so it is a no-op there. `_resolve_agent_estimation_frame` is the
  loader's whole post-fetch path and `preflight_agent.py` now calls it — its own copy of that path had
  silently skipped the prune (a run through a re-implementation is not a run through the served path).
- **Fix 4 — the wrappers refuse a warned final stage, on SERVED fits only** (codex r2 HIGH).
  `LinearDMLWrapper`, `DRLearnerWrapper` (and `dml_learner` via the shared helper) refuse a fit whose
  statsmodels final stage warned, so a warned CI can never be reported: the tournament skips the
  estimator, a forced run fails closed on the missing CI. A subsampled tournament fit only ranks —
  its CI is never served (#1392) — so there the wrappers keep the point estimate with NO interval;
  needed because two outcomes' 5,000-row subsamples are rank 61 of 62 although the full frame is
  full rank. Tests: `test_invalid_inference_rejected.py` (refusal, unserved-keeps-point-estimate,
  selector served-flag contract).
- **Fix 1 addendum — the empty adjustment set keeps the default identifier** (codex r2 HIGH): the
  path-based search returns no set for `common_causes=[]` and DoWhy's estimate carried
  `value=None` silently (the negative-control suite caught it); `optimize_backdoor` is now gated on
  a non-empty set and the identifier test pins both identifiers to the same set and the same estimate
  on empty / numeric / categorical-expanded / continuous-treatment shapes.
- **Round 3 (codex r3 → REVISE: 1 HIGH + 3 MED, all fixed in `fabbec0c9`).** (1) HIGH — the prune
  criterion compared the residual with the RAW column norm, so a genuinely varying large-offset column
  was dropped (`1e10 + arange(100)`: residual / raw norm 2.9e-9 < 1e-8). The residual is now compared
  with the CENTERED norm (translation- and unit-invariant); a column counts as constant only at machine
  rounding (centered / raw ≤ 1e-12); the skip is `n < k+1` (was `≤`, off by one); and the documented
  order is the resolved one (numerics first, then dummies in their categoricals' registry order) — the
  design is NOT reordered, because every forest fit subsamples features by column index. Re-run on the
  real frame: the SAME 16 columns are dropped (k = 61), so the numbers in the table stand unchanged.
  Tests added: large-offset kept, unit-invariant decision, near-collinear (1e-4 noise) kept, `n == k+1`
  runs / `n < k+1` skips, constant dropped, dummy-after-numeric order. (2) MED — after a subsampled
  tournament, a winner whose SERVED full-frame refit was refused ended the selection; Auto now refits
  the next ranked successful candidate until a served fit succeeds (refused candidates stay visible as
  failed results; a forced estimator never subsamples and still fails closed). (3) MED — the identifier
  test now asserts the rebuilt model's treatment IS the median split, and a LinearDML rebuild asserts
  its effect modifiers are the encoded common causes. (4) MED — `load_optum_causal_cohort.load_frame`
  refuses an empty export or one without both arms BEFORE any write (VERIFIED must not be printable on
  such a table); the real-DB gate asserts both arms live.
- **Round 4 (codex r4 → REVISE: 1 HIGH + 1 MED, both fixed).** (1) HIGH — the Round 3 criterion was
  still not translation- or scale-invariant in two branches: the "constant" test (centered/raw ≤ 1e-12)
  dropped `1e14 + arange(100)` as constant, `np.linalg.norm` underflowed at a 1e-200 scale, and the
  fixed 1e-8 tolerance dropped an independent component at 5e-9 relative — which econml's own rank
  check (`np.linalg.lstsq(rcond=None)`, `rank < df`, i.e. max(n,k)·eps ≈ 3.4e-12 at this n) would
  NOT call underdetermined, so the prune was ~3,000× looser than the check it exists to satisfy. Now:
  constant = exact represented equality; the first value is subtracted before centering (exact for
  close values, so an offset cannot leak rounding into the variation); the column is scaled by a power
  of two (exact); tolerance = max(n,k)·eps (`_collinearity_rel_tol`). Measured on the real frame
  (`prune_tolerance_probe.py/.json`): the SAME 16 columns in the same order (k = 61); the largest
  dropped residual ratio is 2.6e-15 against a tolerance of 3.4e-12; the smallest kept ratio is 0.047
  (`elx_liver_disease`). Codex's three probes pass, plus an exact duplicate with a 1e14 offset (the
  old code dropped BOTH columns; now only the later one). Four tests added. (2) MED — after the
  next-candidate fallback, the refused winner's failed entry carried no tournament score, so
  `energy_scores` and the gap started at the served candidate and the reason claimed it had the lowest
  score. Now the refused candidate keeps its tournament `energy_score_result` (`success=False` says it
  was not served), `energy_scores` covers every scored candidate, the gap is the tournament's, and the
  reason states which winner was refused, why, and what was served instead; `exceeded_max_energy_score`
  / `requires_review` are judged on the SERVED estimator's score (asserted).
  Found by this round's wider run, not by codex: the r3 fallback had silently broken the #1392 test
  `test_winner_full_frame_refit_failure_fails_closed` (it pinned the OLD mechanism — fail closed even
  with an honest candidate left; the energy-score directory had last been run BEFORE `fabbec0c9`).
  Its intent (never serve a subsample fit) survives: the test now asserts the served fit is the next
  candidate's FULL-FRAME refit, and a sibling asserts fail-closed when EVERY refit fails.

Final-code node timeline of the primary run (from the run's own INFO log; the graph itself started
at ~09:22:30 after imports):

```
2026-09-22 09:23:38,791 INFO src.agents.causal_impact.nodes.estimation: Using energy score selection (path=energy_score, explicit_method=None, strategy=best_energy)
2026-09-22 09:23:38,912 INFO src.causal_engine.energy_score.estimator_selector: Energy-score tournament on stratified subsample: 5000/15209 rows (cap=5000); winner will be refit on the full 
2026-09-22 09:24:28,385 INFO src.causal_engine.energy_score.estimator_selector: Refitting tournament winner linear_dml on the full frame (15209 rows)...
2026-09-22 09:24:45,669 INFO src.agents.causal_impact.nodes.refutation: Running refutation suite for treatment_dupixent → persistent_at_180d_g28 (ATE=0.0336, CI=[0.0168, 0.0504])
2026-09-22 09:24:45,903 INFO src.agents.causal_impact.nodes.refutation: Refutation subsampled estimation data 15209 -> 5000 rows (#1419); reported ATE/CI remain the full-frame fit.
2026-09-22 09:26:59,134 INFO src.causal_engine.refutation_runner: Refutation suite completed: 4/5 passed, confidence=0.95, gate=proceed
2026-09-22 09:26:59,135 INFO src.agents.causal_impact.nodes.refutation: Logged validation outcome ae8f7569-cbca-41cf-91e9-29d6beefc052 for Feedback Learner: passed
2026-09-22 09:26:59,135 INFO src.agents.causal_impact.nodes.refutation: Refutation PASSED: confidence=0.95, tests_passed=4/5
2026-09-22 09:27:00,061 INFO src.agents.causal_impact.mlflow_tracker: Logged causal impact analysis to MLflow: ATE=0.03361622617921824, confidence=0.00
```

## Caveats (read before citing the numbers)

- **Host process, not the container**: the venv on the droplet host; `e2i_api` runs the same code
  under gunicorn with `_AGENT_HARD_TIMEOUT_S = 900` and a bounded compute executor. 267 s leaves a
  3.4× margin, but the container run in Task 12 is the measurement that counts.
- **The served fit is full rank; the refutation REFITS are not always.** The 16 exactly collinear
  columns are pruned at load, and a served fit that still warned would be refused (fix 4). The
  refutation suite reconstructs and refits on 5,000-row subsamples through DoWhy (not the wrappers);
  those refits emit econml's "underdetermined" warning on this frame (26–96 occurrences per run in the
  logs). They are point estimates compared against the reported ATE — no CI from them is served — and
  the ATE of an exactly collinear design is estimable (measured: 0.03353 vs 0.03353 above). Not
  touched in this lane; noted for the refutation rebuild.
- **Rare dummies**: six dummies have 15–56 supporting rows of 15,209 (`health_exchange_flag`,
  `cci_severe_liver`, `cci_paraplegia`, `payer_product=IND`, `elx_lymphoma`, `cci_metastatic_cancer`);
  they are why subsamples can lose rank without any column being constant.
- **Loader row order**: `_load_agent_estimation_frame` reads with `.limit(limit)` and no `ORDER BY`
  (`loaders.py`). This dataset loads its whole table (15,209 < 20,000) so its frame is deterministic;
  a dataset larger than its limit (nba_triggers at 5,000) serves a different row subset per request —
  which is why the served-pair comparison below is PAIRED on one loaded frame. A stable ordering
  changes which rows those datasets serve, so it is filed as a follow-up rather than slipped in here.
- **Leaderboard cap**: `_DISCOVERY_ROW_CAP = 5,000 < 15,209`, so a discovery run would see a stratified
  subsample; the agent path used here loads the full cohort (`limit=20000`). The energy-score
  tournament itself ran on its own 5,000-row stratified subsample and refit the winner on the full frame
  (#1392), and the refutation suite reconstructed on a 5,000-row subsample (#1419) — both are the
  production shapes.
- **Refutation**: 4/5 on every outcome. The primary fails `bootstrap` (effect moved 32 % across
  bootstrap resamples, "medium" instability); the two null secondaries fail `unobserved_common_cause`.
  Gate = proceed on all three.
- **lbfgs**: no ConvergenceWarnings remain after the scaler; the pre-fix runs emitted them from the
  unscaled propensity fit.
- **Prod writes — an incident, disclosed.** The plan claimed the pre-flight "runs WITHOUT `.env` so it
  writes nothing to prod tables". That was FALSE and untested: `src/ml/data_loader.py` calls
  `load_dotenv()` lazily and, because the worktree sits inside the main checkout, finds the main
  checkout's `.env` (prod Supabase URL + service-role key). The FIRST run that reached the refutation
  node (07:24 UTC, the 266.9 s row) persisted **6 rows to `causal_validations`**
  (estimate_id `201cb9d7-c9d4-5a6f-9a49-6e0fd952408d`, source `causal_impact_query`), **1 row to
  `validation_outcomes`** (outcome_id `3e22a727-7727-4140-9daf-270834433b27`, estimate_id
  `c44cc1c9-bb39-44e9-b210-6a064c0f686a`) and **one MLflow run** (`faef563d27c44517b8c403820e2627f9`,
  experiment 219 `e2i_causal/causal_impact/default`, run name `analysis_20260922_072414`). They were
  deleted at ~08:05 UTC after the owner's explicit approval ("clean up SQL approved"), inside a
  guarded transaction (DELETE 6, DELETE 1; 3,259 / 457 rows remain); the MLflow run was left (not SQL). `preflight_agent.py`
  now blanks the Supabase env before importing `src` and points MLflow at a scratch file store; the
  three later runs logged "Validation persistence unavailable (no Supabase config)" and the prod row
  counts were identical before and after them (3,265 / 458).
- **Run from the worktree root.** From the evidence dir the script imports `src` from the main checkout
  (editable `.pth`); its own assert catches it.
- The `.log` files are gitignored; the status lines are the table above and the timeline is pasted in.
- **Pre-existing on this box, not from this lane**: the wide run of
  `tests/unit/test_agents/test_causal_impact/` aborts at
  `test_refutation.py::TestRefutationNode::test_run_all_refutation_tests` (pytest-timeout 30 s, also
  90 s, inside an econml GRF fit). Reproduced at the pre-session commit `4ba8372b8` in a throwaway
  worktree; CI's sharded `test_agents` lane is the arbiter for that directory.
