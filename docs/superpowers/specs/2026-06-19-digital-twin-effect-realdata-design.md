# Digital Twin Effect Model — Real-Data Design (convergence working doc)

Status: DRAFT under ralph-loop + codex:codex-rescue convergence. NOT committed.
Goal (user): a solution that **works with real data** — no hardcoded per-intervention
magnitudes, nothing laundered through a synthetic frame. The same code path must produce
honest answers on synthetic-gold today and on RWD tomorrow.

Convergence status: codex round-1 returned 3 High + 3 Medium + 1 sound. ALL verified against
source (see §8). This revision resolves them and PIVOTS the primary direction.

## 0. Rejected approaches
- **Synthetic per-intervention tiers** (user-rejected): fabricates magnitudes, does not
  generalize to real data.
- **Direction 1, twin-counterfactual (REJECTED after codex round-1, verified):** the active
  HCP twin is NOT a causal object. Its training target is a **random-weighted sum of
  standardized numeric features + noise** (`src/digital_twin/training_data.py:150-161`,
  weights `rng.uniform(0.2,1.0)`), and the features (digital_engagement_score,
  interaction_frequency, peer_influence_score) are drawn as random uniforms
  (`training_data.py:87-90`). A do-intervention on the twin would read off a *random
  regression coefficient*, not a causal effect. The twin is also not exposed as an
  outcome scorer — `_load_trained_generator` returns a generator and production discards its
  baseline regressor and fits a separate uplift model (`src/api/routes/digital_twin.py:910-911`,
  `src/digital_twin/simulation_engine.py:205-211`). So Direction 1 is both invalid and
  unavailable. Dropped.

## 1. Faithful measurement (prod Supabase, 2026-06-19, read-only)

Substrate: `business_metrics` per_hcp_rollup is **100% synthetic-gold** (`is_synthetic=True`),
~4067 Kisqali / 3932 Fabhalta / 4029 Remibrutinib rows. No RWD connected → "works with real
data" = build the estimation path to consume whatever cohort is wired, and run it on
synthetic-gold now.

Recorded `twin_simulations` (6 rows; drive the recommendation): 3 cohort digital_engagement
(ate ~0.39-0.40, CI width ~0.003), 2 synthetic-fallback digital_engagement + 1 email (ate
~0.15, CI width ~0.007-0.009). ALL deploy.

Recomputed `region_standardized_ate`: digital_engagement +0.378/+0.389/+0.388 (Fab/Kis/Remi,
brand-invariant); call_frequency +0.149/+0.162/+0.155. Honest two-proportion 95% CI width
~0.056-0.062 (recorded CI is 7-20x too tight).

## 2. The synthetic-gold DGP — VERIFIED ground truth (the key to the whole design)

`scripts/backfill_segment_engagement.py` plants the cohort effect with a documented,
recoverable DGP:

- **engagement_score -> conversion_rate IS causal**, region-heterogeneous:
  `TRUE_CATE_BY_REGION = {Northeast 0.45, West 0.30, South 0.18, Midwest 0.08}`
  (`backfill_segment_engagement.py:161`), planted as `conversion = baseline + tau*t_bin + noise`
  (`:313-314`). Population ATE ~0.25 (count-weighted; recovery probe at `:423`).
- **Designed confounders** (this is what makes it a real causal problem):
  `_OUT_BETA_MARKET=0.80` (market_share -> conversion, STRONG), `_OUT_BETA_VOLUME=0.06`
  (log1p(total_rx_count)), `_OUT_REGION_BASELINE` (region intercept, NOT the CATE)
  (`:186-189`). Treatment `engagement` is itself a sigmoid of confounded drivers (`:293`).
- **call_frequency is NOT in the causal path** — explicit: a Poisson engagement-linked
  *exposure correlate* (`:80`, `:318`). Toggling it has no causal effect on conversion.

Consequences, verified:
1. The ~0.39 region-only estimate is **inflated** because it omits `market_share` (the strong
   confounder, beta=0.80). A proper estimate that adjusts for {region, market_share,
   total_rx_count} recovers the true ~0.25 population ATE and the per-region CATE.
2. `call_frequency_increase` is **not identified** — its ~0.15 is pure confounding (call_freq
   correlates with the causal engagement). It must NOT report a confident uplift.
3. The 4 non-cohort interventions (email, speaker, samples, peer) have **no treatment column**
   in the cohort and no causal wiring -> not identified.

## 3. Diagnosis — why every sim says DEPLOY
A. **Fabricated magnitude (4 of 6):** no treatment column -> constant 0.15 via
   SyntheticEffectDataProvider -> DEPLOY. Silent mock.
B. **Confounded magnitude (engagement):** region-ONLY g-formula omits market_share -> ~0.39
   vs ~0.25 true; carried through a synthetic injected-effect frame
   (`provider.py:267-271`) so CI + heterogeneity are synthetic artifacts (7-20x too tight).
C. **Non-causal magnitude (calls):** confounded ~0.15 reported as if causal.
D. Even an honest CI keeps every reported magnitude > min_effect=0.05 -> still DEPLOY. The
   binding problem is magnitude **validity / identification**, not CI width alone.

## 4. Design principles
1. Estimate magnitude, uncertainty, and heterogeneity from the connected data over a
   defensible **pre-treatment** adjustment set. Never launder through a synthetic frame.
2. **Identification gate.** An intervention is estimated ONLY if it maps to a manipulable
   treatment that is a CAUSE of the outcome in the connected data. Otherwise return honest
   "no effect data" — never a constant, never a confounded proxy.
3. Real uncertainty: CI from the estimator's own sampling variability (bootstrap over rows /
   analytic SE).
4. Substrate-agnostic: identical code on synthetic-gold and RWD; only provenance differs.

## 5. Proposed design (primary: direct cohort estimation — Direction 2)

Estimate each identified intervention's effect **directly on the connected cohort** with a
defensible adjustment set; no synthetic-frame handoff.

**Per-intervention identification (data-driven, from the verified DGP):**

| intervention | treatment col | identified? | behavior |
|---|---|---|---|
| digital_engagement | engagement_score | YES (causal in DGP) | estimate (below) |
| call_frequency_increase | call_frequency | NO (explicit non-causal correlate) | effect_basis=unavailable |
| email_campaign / speaker_program_invitation / sample_distribution / peer_influence_activation | (none) | NO (no treatment col) | effect_basis=unavailable |

**Estimator for identified interventions (engagement):**
- Adjustment set = the designed pre-treatment confounders present in the cohort:
  {region, market_share, total_rx_count}. (Discovered from data/columns, not hardcoded to
  the synthetic DGP — on RWD the adjustment set is whatever pre-treatment confounders the
  connected cohort exposes; document the selection rule.)
- Method: regression-adjustment / doubly-robust (AIPW) or DML on the binary treatment
  (t = engagement above the pre-registered threshold), outcome conversion_rate. This recovers
  the population ATE and, with a region interaction / per-stratum fit, the region CATE.
- CI: nonparametric bootstrap over cohort rows (or the AIPW analytic SE). Honest width.
- Replace `CohortEffectDataProvider`'s `SyntheticEffectDataProvider(true_ate=ate)` handoff
  (`provider.py:267-271`) with a real EffectEstimate built from this estimator; keep the
  twin population only for reporting per-twin heterogeneity IF a valid CATE model exists,
  else report ATE + honest CI without fake per-twin spread.

**Pre-registered contrast (resolves Medium-4):** mirror the DGP estimand — t = 1 if
engagement_score above its cohort median (the DGP binarizes engagement), else use P75 vs P25
of the continuous intensity with clipping. ONE fixed contrast per intervention, defined in
code, never chosen by the UI/request.

**Honest-unavailable wiring (resolves Medium-5):** add `effect_basis="unavailable"` /
`available_for_effect=false` to the intervention-types contract
(`digital_twin.py:730-737`); FE shows an explicit unavailable state (or hides those
interventions from the simulate menu). Unmapped/non-identified interventions return this
state up-front rather than running and 422-ing. NOTE (codex round-2): the current FE filters
the menu on **trained-model** availability only (`SimulationPanel.tsx:138`,
`frontend/src/types/digital-twin.ts:117-123`), so a NEW `available_for_effect` flag + FE
state is required — effect-availability is distinct from model-availability.

**Recommendations vary — honestly:** engagement (genuinely causal, pop ATE ~0.25) DEPLOYs at
the population level; Midwest CATE 0.08 with an honest CI straddles 0.05 -> REFINE at that
stratum. The other 5 interventions are **unavailable**, not a fabricated DEPLOY. So the page
stops showing 6/6 DEPLOY: it shows engagement DEPLOY (+ possible REFINE strata) and 5
unavailable.

## 6. Validation / test plan (red-first)
- **Recover-known-effect:** on the cohort DGP, the estimator recovers `TRUE_CATE_BY_REGION`
  (0.45/0.30/0.18/0.08) within tolerance and population ATE ~0.25. Use the
  **backfill-specific** recovery probe (`backfill_segment_engagement.py:421-423`, which knows
  the planted tau) as the acceptance gate. NOTE (codex round-2, verified): the generic
  `src/ml/synthetic/dgp/recovery_probe.py` is **patient-frame specific**
  (`_COVARS=["disease_severity","academic_hcp"]`, PatientGenerator) — reference it only as a
  pattern, not as the direct runner for the HCP cohort.
- **De-confounding:** estimate WITHOUT market_share recovers ~0.39 (the inflated number),
  WITH it recovers ~0.25 — locks in that the adjustment set matters.
- **Honest CI width:** CI width within an order of magnitude of the analytic two-proportion SE
  for n (catches the 7-20x understatement).
- **No synthetic-frame laundering:** assert the cohort path does NOT instantiate
  SyntheticEffectDataProvider with an injected true_ate.
- **Honest-unavailable:** call_frequency_increase + the 4 non-cohort interventions return
  effect_basis=unavailable (no number, no 0.15, no 422-on-submit).
- **No all-DEPLOY by construction:** across the catalog, DEPLOY is not the only reachable
  outcome.
- **Tests to update (contract change — currently encode the behavior we are breaking):**
  `tests/unit/test_digital_twin/effect/test_provider.py:10-23` (asserts true_ate=0.15 for
  email), `tests/unit/test_digital_twin/test_simulation_engine.py:221-232` (email sim
  completes with synthetic uplift), `tests/unit/test_digital_twin/effect/test_cohort_provider.py:84-90`
  (cohort ATE injected as synthetic ground truth),
  `tests/unit/test_digital_twin/test_engine_real_effect.py:98` and
  `tests/unit/test_digital_twin/test_simulation_engine.py:232` (assert
  `data_provenance == "synthetic_uplift_v1"` — break when the path emits real-estimate
  provenance). [test_engine_real_effect:98 surfaced reconciling the peer review; the peer's
  cited `test_digital_twin_concurrency.py:475` is HALLUCINATED — that file does not exist.]

## 7. Open questions (remaining after round-2)
1. Estimator choice: RESOLVED (codex round-2, verified). **EconML 0.16.0 is installed**
   (`pyproject.toml:39`) and already used for CATE (`src/agents/heterogeneous_optimizer/nodes/cate_estimator.py:155`
   -> `CausalForestDML`; DRLearner importable). Use **CausalForestDML** for region CATE or
   **DRLearner** (doubly-robust) for the headline ATE; plain regression-adjustment is the
   realistic fallback. No new dependency.
2. Report granularity (open): population ATE for the headline recommendation + per-region CATE
   in the drilldown, or per-brand ATE? Brand x region cells may be thin — check n per cell
   before committing to per-cell CATE.
3. RWD adjustment-set selection rule (open): how to choose pre-treatment confounders
   generically when the connected cohort schema differs from synthetic-gold (avoid
   colliders/mediators like nrx/trx/conversion outcomes). Needs a documented allowlist or a
   pre-treatment tag — flagged as the one item to keep explicit at implementation time.

## 8. Codex round-1 findings — verification + resolution (verify-before-accept)
- **High-1** twin not a counterfactual scorer -> VERIFIED (`digital_twin.py:910-911`,
  `simulation_engine.py:205-211`). Resolved: Direction 1 dropped (§0).
- **High-2** engagement/calls not valid do-interventions -> VERIFIED
  (`training_data.py:150-161` random target; `backfill_segment_engagement.py:80,318` call non-causal;
  `hcp_generator.py:104-136` peer exogenous). Resolved: identification gate (§5); calls + peer
  -> unavailable. REFINEMENT: engagement IS causal in the *cohort* DGP, so it stays identified.
- **High-3** cohort estimate launders associational gap through synthetic frame -> VERIFIED
  (`provider.py:184-218,267-271`). Resolved: direct estimation, adjustment set incl
  market_share, no synthetic handoff (§5).
- **Medium-4** contrast underspecified -> Resolved: pre-registered contrast (§5).
- **Medium-5** honest-unavailable not wired -> Resolved: effect_basis=unavailable contract (§5).
- **Medium-6** tests encode old behavior -> Resolved: named in §6 for red-first update.
- **Low (sound)** CI-only fix insufficient -> incorporated (§3D, magnitude validity first).

### Codex round-2 (re-review of the revised doc) — VERIFIED
Result: **no remaining High/Medium findings.** All sections confirmed sound. Two factual
refinements, both verified against source and folded in:
- Generic `recovery_probe.py` is patient-frame specific -> use the backfill-specific probe as
  the acceptance gate (§6). VERIFIED (`recovery_probe.py:1,20`).
- EconML 0.16.0 installed + already used for CATE -> estimator feasible (§7). VERIFIED
  (`.venv` import OK; `cate_estimator.py:155`).
- FE filters on model-availability only -> needs a distinct `available_for_effect` flag (§5).
  VERIFIED (`SimulationPanel.tsx:138`).
Fixed point reached: a fresh codex review of the revised design returns no new valid
High/Medium findings.

### Peer review (a parallel inline Claude review) — reconciled against source
A second, independent review arrived after convergence. Reconciliation (all verified):
- **Corroborates** the Direction-1 rejection and findings High-1/Medium-5/High-3/CI-artifact.
- **Its two novel HIGHs (engagement has no causal DGP; recover-known-effect unwritable) are
  REJECTED as stated** — they never read `scripts/backfill_segment_engagement.py`, where
  engagement->conversion IS causal (`TRUE_CATE_BY_REGION`, verified). They are true only for
  the twin-training-frame path, which is exactly why Direction 1 was dropped.
- **Its Finding 7 is HALLUCINATED** (`test_digital_twin_concurrency.py:475` — file does not
  exist). Rejected. Verifying it surfaced the REAL provenance-asserting tests now in §6.
- **Valuable contribution — alternative/complementary validation substrate:**
  `src/ml/synthetic/generators/hcp_adoption_artifact.py` is a real causal cohort
  (exogenous centrality -> confounded `treatment_arm` -> `adopted`) with a **stored per-HCP
  `cate_estimate`**. This is a STRONGER recover-known-effect gate than the region-level backfill
  tau (per-HCP truth, not just region tau), and `treatment_arm` is a cleanly designed
  manipulable treatment. CONSIDER at implementation: validate the DML estimator against the
  stored per-HCP `cate_estimate` here, in addition to recovering `TRUE_CATE_BY_REGION` on the
  business_metrics cohort. (Does not change the primary path — the live effect provider reads
  business_metrics — but adds a sharper validation signal.)

## 9. Scope / non-goals
- No deploy (held). Implementation lands as a separate PR after convergence.
- FE PR #1048 (brand filter, result title, dedup view) is independent and already up.
