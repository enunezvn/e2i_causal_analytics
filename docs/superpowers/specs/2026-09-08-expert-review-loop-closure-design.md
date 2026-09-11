# Expert-review loop closure: visible review state, real non-critical evidence, guarded promotion

**Date:** 2026-09-08
**Status:** approved in conversation 2026-09-08 (approach 2 of 3); spec under user review
**Branch:** `claude/lane1-expert-review-loop`
**Evidence:** live database and Redis measurements taken 2026-09-08 on the droplet; offline
replication script and outputs in the session scratchpad (`review_disproof/`), figures
copied into §2 below.

## 1. Problem

PRs #1983, #1984 and #1985 (issues #1971, #1972, #1974) shipped the machinery of the
expert-review loop: a rejection probe on every band, an enforcement switch
(`CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL`, default off), one schema-owned definition of an
active approval, and a durable record of every discovered DAG in `public.discovered_dags`.
Nothing user-facing consumes it:

- The API already returns `refutation.expert_review_id`, `refutation.expert_review_decision`
  and `discovered_dag_id` on every agent run. The hand-written frontend types and the
  drill-down (`frontend/src/types/causal.ts`, `components/causal/CausalAnalysisDetail.tsx`)
  ignore all three, so an analyst cannot see that a run's structure is pending review, was
  rejected, or has a durable DAG record.
- The queue page (`frontend/src/pages/ExpertReviews.tsx`) lists only pending rows, has no
  brand filter although the API supports one, reads the summary with no error path (a 503
  silently drops the count badges), and generates the advisory assessment only on a second
  click per row. 39 pending rows carry a DAG snapshot and context; 3 carry an assessment.
- `CausalPathRepository.set_validation_status` conditions the promote only on the current
  status, so a rejection committed between the node's probe and its status write is not seen.
- The enforcement switch acts on REVIEW bands only, and the live gate has never produced one.

## 2. Measured findings that shape the design

Live agent runs on record (`causal_validations`, `estimate_source = causal_impact_query`):

| Measurement | Value |
|---|---|
| Runs on record | 96 |
| PROCEED / REVIEW / BLOCK | 49 / 0 / 47 |
| Confidence values observed | 1.000 or 0.867 (PROCEED); 0.667 or 0.333 (BLOCK) |
| PROCEED runs with a sensitivity WARNING | 34 of 49 |
| data_subset SKIPPED | 96 of 96 (DoWhy result lacks `subset_effects`) |
| bootstrap SKIPPED | 93 of 96 (lacks `bootstrap_estimates`); 3 budget-skipped |
| Expert reviews pending / rejected / approved | 39 / 1 / 0 |

Why REVIEW never occurs. The gate (`RefutationRunner._determine_gate_decision`) is BLOCK on
any critical failure, PROCEED at confidence ≥ 0.70, REVIEW at ≥ 0.50, else BLOCK. Confidence
is a weighted mean over non-skipped tests (critical tests 0.25 each, non-critical 0.125 each;
PASSED 1.0, WARNING 0.6, FAILED 0.0). With only the three critical tests scoring, the reachable
values without a critical failure are 1.0 and 0.867, both PROCEED. REVIEW requires the two
non-critical tests to score and fail: with a sensitivity WARNING the sum is
0.65 + 0.125·(s_subset + s_bootstrap), which is under 0.70 only when both are FAILED.

Offline replication (venv, same DoWhy 0.14 / EconML 0.16 / NumPy as the container; 1,500-row
frames; LinearDML with the production RandomForest nuisances, discrete treatment, seed 42):

| Pair | Subset effects (5 × 80%) | Coverage vs reported CI | Bootstrap 95% CI (20) | Width ratio vs reported CI |
|---|---|---|---|---|
| all brands, treatment_arm → persistent_180d (the live probe; reported CI [0.019, 0.152]) | 0.050–0.123 | 1.00 | [0.021, 0.156] | 1.01 |
| Remibrutinib, treatment_arm → treatment_initiated (wrapper CI [0.117, 0.236]) | 0.156–0.180 | 1.00 | [0.117, 0.212] | 0.81 |

Costs measured: subset loop 5.2 s vs production's discarded call 5.3 s; bootstrap loop 37.5 s
vs 38.4 s. The evidence is computed today and thrown away.

Threshold defect found. `PASS_THRESHOLDS["bootstrap_ci_ratio"]` is `{pass: 0.50, warning:
0.75}` with `ratio = bootstrap_width / original_width`, while its comment says "must not be
> 50% wider than original" (i.e. pass ≤ 1.5). Under the code's values both pairs above FAIL;
under the comment's intent both PASS. A bootstrap interval as wide as the analytic one is the
expected outcome for a stable estimate, so the code's values would fail nearly every run the
moment real evidence flows.

Reference interval. The runner receives `original_ci` from the estimation node (the reported
interval, `ate_inference(X).conf_int_mean()`). The reconstruction's own interval is unusable
(standard error 4.9 vs the reported 0.034 on the same pair) and is never used by the runner.
That stays so.

Consequences accepted by the user (2026-09-08): REVIEW becomes *reachable* for genuinely
unstable estimates but will be rare; the live impact run in §7 measures how rare. Any change
to the band semantics (for example treating a sensitivity WARNING as REVIEW, which would have
moved 34 of 49 historical PROCEED runs) is a separate decision taken with that measurement in
hand, not part of this lane.

## 3. Goals and non-goals

Goals

1. An analyst can see, on the drill-down of any run, the review state of its DAG structure,
   who rejected it and why, the durable discovered-DAG record id, and reach the review.
2. A reviewer can work the queue by brand, sees an honest error when the summary is
   unavailable, and gets the advisory assessment without a second click.
3. The two non-critical refutation tests produce real distributional evidence at no extra
   compute, with thresholds that mean what their comments say.
4. A promote can never land over a rejection committed after the probe.
5. `docs/lineage/causal_dag_lineage.html` describes the shipped system.
6. The whole loop is exercised live: approve, reject, re-run, and (if a REVIEW band appears)
   the switch.

Non-goals

- Changing the confidence weights or the 0.70 / 0.50 band edges.
- A writer for `driver_rankings` / `feature_rankings` (measured in #1984: that data never
  reaches the causal_impact path).
- Sweeping `expired` into the table (owner decision recorded in #1972).
- Bulk approve/reject.

## 4. Design

### 4.1 Engine: real non-critical evidence (`src/causal_engine/refutation_runner.py`)

`_run_data_subset_test` and `_run_bootstrap_test` stop calling
`causal_model.refute_estimate(...)` and discarding the result. Each runs its own resample loop
with the same public estimator calls DoWhy 0.14's `_refute_once` uses:

```
new_estimator = estimate.estimator.get_new_estimator_object(identified_estimand)
new_estimator.fit(new_data, effect_modifier_names=estimate.estimator._effect_modifier_names,
                  **getattr(new_estimator, "_fit_params", {}))
value = new_estimator.estimate_effect(new_data, control_value=estimate.control_value,
                  treatment_value=estimate.treatment_value,
                  target_units=estimate.estimator._target_units).value
```

- Data subset: `data.sample(frac=subset_fraction, random_state=rng)` per resample,
  `num_subsets` resamples. Coverage = share of resample effects inside `original_ci`
  (existing `_calculate_ci_coverage`). Thresholds unchanged (pass ≥ 0.80, warning ≥ 0.70).
- Bootstrap: `sklearn.utils.resample(data, n_samples=len(data), random_state=rng)` per
  resample, `num_bootstraps` resamples, no confounder noise (DoWhy's default bootstrap refuter
  perturbs chosen variables with `noise = 0.1`; that is a measurement-error refuter, not a
  variance check, and the test's question is variance. The decision and the reason are
  recorded in the method docstring). Bootstrap CI = 2.5th and 97.5th percentiles;
  `ci_ratio = bootstrap_width / original_width`. Thresholds become
  `{pass: 1.50, warning: 1.75}`; the old values and the measured reason are recorded next to
  the constant.
- p-value: `dowhy.causal_refuter.test_significance(estimate, np.array(effects))["p_value"]`,
  so `_require_p_value`'s contract (a real p-value from real refuter output) is kept.
- Seeding: a `numpy.random.default_rng(seed)` derived from `estimate_id` when present, so a
  re-run of the same estimate reproduces its evidence; otherwise unseeded, as today.
- Deadline: the loop checks the runner's cooperative deadline between resamples. If it stops
  early with at least `MIN_RESAMPLES` completed (3 for subsets, 10 for bootstrap), the test
  scores on what completed and `details` records `resamples_completed` / `resamples_requested`.
  Below the minimum it returns SKIPPED with reason `time_budget`, exactly as the #1419 policy
  does today for a test that never started.
- `details` carries the per-resample effects (`subset_effects` / `bootstrap_effects`) so the
  persisted `causal_validations.details_json` row holds the distribution it judged.
- The reconstruction (`nodes/refutation.py::_reconstruct_dowhy_artifacts`) and the reference
  interval are unchanged.

Expected effect on live bands: none on stable estimates (both replicated pairs PASS both
tests; with a sensitivity WARNING the run scores 0.90, PROCEED). REVIEW becomes possible when
both non-critical tests fail.

### 4.2 Backend: review lookup route (`src/api/routes/expert_review.py`)

`GET /expert-reviews/{review_id}` (operator role, declared after `/pending` and `/summary` so
FastAPI's first-match routing cannot shadow them):

- Response `ExpertReviewDetailResponse`: `review` (a new `ReviewRecord` schema: every
  `PendingReviewItem` field plus `approval_status`, `reviewer_name`, `reviewer_email`,
  `approved_at`, `resolved_at`, `valid_from`, `valid_until`, `concerns_raised`, `conditions`,
  `comments_json`, `supersedes_review_id`) and `history: List[ReviewRecord]` — every row
  sharing the DAG hash and brand, newest first, expired included
  (`get_reviews_for_dag(include_expired=True)`), which is the same read the gate's rejection
  probe performs.
- Resolution provenance (Task 12 whole-diff fold, codex HIGH F1): the resolve route records the
  authenticated operator as the resolver (`reviewer_name` from the profile name, else email,
  else id; `reviewer_email`) and `submit_review` stamps `resolved_at` for both statuses
  (migration 136, no backfill); `reviewer_id` stays the requester breadcrumb the gate wrote.
- 404 when the id does not exist; 503 through `_store_unavailable` on a store failure.
- Any docstring or schema change here regenerates `frontend/src/types/generated/api.ts`
  (`make generate-types`); the union of this lane's changes is regenerated once, at the end.

### 4.3 Backend: guarded promote (migration 134 + repository + node)

`database/migrations/134_guarded_causal_path_promote.sql` adds

```
public.promote_causal_path_guarded(
    p_path_id text, p_new_status text, p_allowed_current text[],
    p_dag_version_hash text, p_brand text) RETURNS jsonb
```

One `UPDATE public.causal_paths SET validation_status = p_new_status WHERE path_id = p_path_id
AND validation_status = ANY(p_allowed_current) AND NOT public.dag_structure_rejected(
p_dag_version_hash, p_brand)`, returning `{"moved": 0|1, "rejected": bool}` — `moved` is the row
count and `rejected` is `dag_structure_rejected(hash, brand)` re-evaluated only when nothing moved, so the
caller can log "rejected, not moved" distinctly from "current status not in `p_allowed_current`" (a bare
row count cannot tell the two apart; the migration's own DO block reads `->> 'moved'`).
`dag_structure_rejected` is a STABLE
SQL function encoding the gate's chronology rule exactly as `ExpertReviewGate._latest_adjudication`
does: take the newest row whose `approval_status <> 'pending'` for the hash (and brand when
`p_brand` is not null, else any brand — the Python reader filters by brand only when given);
the structure is rejected when that row is `rejected` and no `pending` row is newer than it.
A null hash means "unchecked" and the function returns false, matching the node, which
already refuses to move a path when its own probe was `unchecked` or `unknown`.

SECURITY INVOKER, `SET search_path = public`, `REVOKE ALL ... FROM PUBLIC, anon, authenticated`,
`GRANT EXECUTE ... TO service_role`, and an asserting `DO` block that checks the privileges
(precedent: `database/ml/036`). Rehearsed on the live database inside `BEGIN ... ROLLBACK`
twice for idempotency before it ships.

`CausalPathRepository.set_validation_status` gains keyword-only `dag_version_hash` and
`brand`; when a hash is given it calls the RPC, else the existing plain update (kept for the
no-hash callers and tests). `RefutationNode._persist_suite_and_promote` passes the run's
`dag_version_hash` and brand. The trigger from migration 119 stays the second line of
defence for `validated`.

### 4.4 Frontend: drill-down review state

- `frontend/src/types/causal.ts`: `RefutationSummary` gains `expert_review_id?: string | null`
  and `expert_review_decision?: string | null`; `AgentCausalAnalysisResponse` gains
  `discovered_dag_id?: string | null`. These mirror fields already present in the generated
  types.
- `CausalAnalysisDetail.tsx` renders a "Review status" block under the gate badges whenever
  `expert_review_decision` or `discovered_dag_id` is present:
  - decision label and one-line meaning (`pending_review`: "structure queued for expert
    review"; `rejected`: "a reviewer rejected this structure", with reviewer and reason from
    the run's warnings when present — the `Estimate withheld` halt line is shown for any decision
    that carries one (rejection, the approval-enforcement switch, or the route's fallback), since
    the run's warnings are rendered nowhere else in the drill-down; `proceed` /
    `renewal_required`: "structure approved",
    with the renewal note; `blocked` / `unavailable`: honest text from the schema description);
  - a link to `/expert-reviews?review=<expert_review_id>` when an id exists;
  - the discovered DAG record id in full (the exact lineage handle), with a copy affordance and
    the sentence "durable discovery record" (no route exists for it yet).
- Nothing is inferred: absent fields render nothing.

### 4.5 Frontend: queue page

- `?review=<id>` opens a "Linked review" card above the queue using a new
  `useExpertReview(id)` hook over the new route: status, brand, treatment → outcome, DAG
  snapshot, the recorded reviewer, decision time and comments, validity, and the same-hash
  history table. Only RECORDED provenance is rendered: the reviewer is `reviewer_name`, else
  `reviewer_email`; the decision time is `resolved_at`, else `approved_at`; each reads
  `not recorded` when absent (never `reviewer_id`, which holds the requester, and never
  `created_at`). A pending linked review offers the existing resolve form in place.
- Brand filter: the page reads the global brand filter (`useE2IFilters`, the same SSOT the
  Causal Analysis page uses, #1752) and passes it to `usePendingReviews` and
  `useReviewSummary`. "All" sends no brand, which is the only way the four brand-less rows
  are visible; the page says so in the card description.
- Summary error: `useReviewSummary` `isError` renders a `WarningBanner` in place of the badges.
- Assessment: expanding a row with no cached assessment fires the assessment mutation once
  (keyed by review id, StrictMode-safe). A "Prepare assessments" button walks the visible rows
  lacking a cache one at a time, shows `k / n`, is cancellable, and stops on the first error
  with the error shown. No new backend endpoint; each call is cached server-side.
- `queryKeys.expertReviews.detail(id)` added; resolve and assessment mutations also invalidate
  it.

### 4.6 Lineage document (`docs/lineage/causal_dag_lineage.html`)

Rewritten sections, each verified against the merged tree at write time:

- §3.3 "Where the DAG goes": `public.discovered_dags` / `discovered_edges` /
  `discovery_algorithm_runs` written by `record_discovered_dag` when discovery ran; failures
  visible in `warnings`; `discovered_dag_id` on the response.
- §4.3 "Expert review": probe on every band; enforcement switch semantics and its live
  inertness (0 REVIEW in 96 runs) with the band arithmetic; approval structural; the new
  lookup route and the UI's review-status surface.
- §4.4 "Registry promotion": the guarded promote and the chronology rule.
- §4.2 "Refutation gate": real non-critical evidence, corrected bootstrap thresholds, the
  measured band incidence table.
- §4.8 "Gaps register": remove the entries this lane and the September cluster closed
  (discovery tables, hard block, can_use_estimate), add "REVIEW is rare by design" with the
  arithmetic, keep "expiry is never swept".
- Map: a `discovered_dags` box on the outcome side and the switch on the governance side.
- Every `file:line` anchor re-resolved by matching the referenced line's text against the
  merged tree; the pinned commit in the nav note updated. Last commit of the PR.

## 5. Error handling

- Engine: any exception inside a resample loop raises `RefutationError` (fail-closed, as
  today for the DoWhy path); a deadline stop is not an error.
- Route: 404 / 503 as in §4.2; never an empty body with 200.
- RPC: raises on a malformed status; the node's existing `except` logs and leaves the status
  unchanged, evidence persisted.
- Frontend: every new fetch has loading, error and empty states; the linked-review card for a
  404 says the review no longer exists and keeps the queue usable.

## 6. Testing

Red-first per seam, in the lane's worktree; whole-tree checks run in CI.

- Runner: a `TestRealNonCriticalEvidence` class with a fake estimator object exposing the four
  public calls above: distribution present in `details`; coverage and ratio thresholds at the
  boundaries (0.79 / 0.80, 1.50 / 1.51, 1.75 / 1.76); deadline stop above and below the
  minimum; seeded reproducibility; p-value comes from `test_significance`. The existing
  "skipped rather than fabricate" tests are inverted deliberately with the reason in the test
  name. Confidence arithmetic test: sensitivity WARNING + both FAILED → REVIEW, WARNING +
  both PASSED → PROCEED at 0.90.
- Repository / node: `set_validation_status` routes to the RPC when a hash is given and to the
  plain update otherwise; the node passes the run's hash and brand; a fake RPC returning 0
  produces the existing "no transition" log path.
- Migration: a SQL rehearsal script (BEGIN / apply twice / asserts / ROLLBACK) run on the live
  database before merge; unit test that the migration file exists and names the function
  (pattern of `test_lineage_residue_1975_1979.py`).
- Route: 200 with history, 404, 503; declared-order test that `/pending` and `/summary`
  still resolve.
- Frontend: `CausalAnalysisDetail.test.tsx` cases for each decision value and for absent
  fields; `ExpertReviews.test.tsx` cases for the linked card (pending / rejected / 404), the
  brand filter passing through both hooks, the summary error banner, auto-assessment firing
  once, and the prefetch walking rows and stopping on error; `use-expert-review.test.ts` for
  the new hook and invalidations.
- Generated types: `make generate-types` on the union of changes; `verify-types` must be clean.

## 7. Live verification plan

Runs on the droplet, in this order; every step records its evidence in
`docs/demos/results/<run-date>_expert_review_loop/` (dated on the day the runs happen) (job ids, review ids, response excerpts).

1. **Baseline before merge (current image).** Start the Causal Analysis discovery job for
   Remibrutinib at patient grain over its 12 registry questions (`copay_support` → adherent /
   low_gap / persistent; `psp_enrolled` → adherent / persistent; `rep_detailing_high` and
   `sample_dropped` and `trigger_accepted` → treatment_initiated; `treatment_arm` →
   discontinued / persistent / treatment_initiated; `urticaria_severity_uas7` → persistent).
   Record per question: band, confidence, per-test statuses, review decision.
2. **Merge, deploy, certify content.** Container image tag, markers for the new runner code,
   the new route in the OpenAPI document, migration 134 applied, the RPC's privileges.
3. **Impact run (new image).** Same 12 questions. Compare bands and per-test statuses with
   step 1; count REVIEW rows; record data_subset / bootstrap statuses and their distributions.
4. **Approve.** Approve review `4eab7033-7422-422d-83f6-659c9c3b9987` (treatment_arm →
   persistent_180d, no brand) through the UI's linked-review card, then re-run that exact
   request (`dataset patient_journeys`, `limit 1500`, no brand). Expect `expert_review_decision
   = proceed`, `expert_review_id` = the approved row, gate still BLOCK, the caveat naming the
   approval, and the drill-down showing "structure approved".
5. **Reject.** Reject one other synthetic pending row (chosen at step 4 from the brand-less
   set), re-run its pair. Expect status `failed`, the halt message naming the reviewer and
   review id, no new pending row, the drill-down showing the rejection with its link.
6. **Switch.** If step 3 produced at least one REVIEW row: add
   `CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL=true` to the root `.env` (the compose project reads it
   through the `docker/.env` symlink), `docker compose -f docker/docker-compose.yml up -d
   --no-deps api`, confirm the container env, re-run that question. Expect status `failed`,
   `current_phase awaiting_expert_review`, the message naming the review id and the resolve
   endpoint. Then the owner decides: leave on, or remove the line and recreate the container.
   If step 3 produced no REVIEW row: the switch is demonstrated by the integration test that
   drives the node with a REVIEW suite, and the band-semantics question goes to the owner with
   the step-3 data.
7. **Record.** Certification comments on the PR, `docs/demos/results/...`, memory, handoff.

Writes performed by this plan are ordinary product writes (job store, `causal_validations`,
`discovered_dags`, two `expert_reviews` resolutions) authorised on 2026-09-08.

## 8. Rollout notes

- One PR from `claude/lane1-expert-review-loop`, merged with `--merge`. Touches `src/**`,
  `frontend/**` and `database/**`, so the merge deploys.
- Migration 134 applies through `scripts/run_migrations.sh` in the deploy; until it applies,
  the node's RPC call fails and the existing `except` leaves the status unchanged with a
  warning — the same behaviour as any promote failure today.
- Runner threshold change is not a migration; existing `causal_validations` rows keep their
  recorded statuses. No backfill.

## 9. Open decisions (owner)

- Band semantics after the impact run (§2, §7 step 6). Not in this lane.
- Whether to keep the switch on if step 6 runs.
- Filing issues for the two findings surfaced here and not fixed here: REVIEW unreachable by
  construction (with the arithmetic), and the reconstruction's own interval being unusable
  (documented, harmless while unused).
- Whether to open the CausalPFN trial (§11) as the lane after this one, and whether the
  simplification candidates in §10 become issues.

## 10. Design retrospective: what a better system would have done differently

Recorded on 2026-09-08 in answer to the owner's question "could we have designed a better
system?". The principles the causal path was built on hold up: evidence persisted before any
status changes, fail-closed everywhere, structure and statistics as separate channels,
provenance on every edge, planted-truth recovery gates in CI. The execution accreted. Four
debts, each exposed by a measurement in this session; none is in scope for this lane, and
none justifies a rewrite. They are simplification candidates to sequence after lane 1.

| # | Debt | Evidence | What a better design does |
|---|---|---|---|
| 1 | Refutation refits a *reconstruction* of the estimator rather than the fitted one, so the node carries a reconstruction tolerance guard, per-refit cost calibration, a stratified subsampler and a cooperative compute budget. | `nodes/refutation.py` is 1,943 lines with 15 distinct issue references; the reconstruction's own interval is unusable (SE 4.9 vs 0.034 reported); per-run latency 110 s for four fitted estimators. | One fitted object per run shared by estimation and refutation (same worker process), or an amortized estimator whose re-inference is a forward pass (§11), which makes placebo, random-common-cause, subset and bootstrap cheap by construction. |
| 2 | The gate is an uncalibrated aggregate: a weighted mean of PASSED / WARNING / FAILED labels with hard band edges. | REVIEW unreachable for the life of the system (0 of 96 runs); the bootstrap pass threshold inverted against its own comment, unnoticed because the test never scored. | An evidence-native gate: the effect's bootstrap or posterior distribution and the E-value as continuous quantities, with explicit decision rules calibrated against the DGP's planted truth, and a CI test that enumerates the reachable band values. |
| 3 | The review key is the structure hash. | Any covariate change mints a new review; the queue holds 39 BLOCK-band structures whose approval changes no outcome; `approval` had no downstream effect for three months without anyone noticing. | Key reviews on the estimand (brand, treatment, outcome, adjustment set), show DAG diffs between versions, and queue only structures whose approval would change an outcome. |
| 4 | Complexity that hides defects: two gates sharing a vocabulary (discovery accept / review / reject vs refutation proceed / review / block), three chat brains, a 6,368-line route module. | The threshold inversion in debt 2 and the discard-after-compute in §2 both lived in files nobody can hold. | Split `routes/causal.py` by concern (frames, discovery, agent run, jobs, history); one vocabulary per gate; a module-size guard in CI. |

Sequencing recommendation: lane 1 first (it makes the evidence real, which every later
comparison needs); then §11's trial (it decides whether debt 1 is removed or refactored);
then debt 2's evidence-native gate, which the impact-run data and the trial both feed; debts
3 and 4 as issues.

## 11. CausalPFN: assessment and recommended trial

Researched 2026-09-08 (paper, repository, PyPI, and the live container). CausalPFN
(Balazadeh, Kamkari, Thomas, Li, Ma, Cresswell, Krishnan; NeurIPS 2025; arXiv 2506.07918;
Apache-2.0; `pip install causalpfn`) is a single transformer trained once on simulated
data-generating processes that satisfy strong ignorability. Given an observational dataset it
returns CATE and ATE in-context with a quantized posterior per unit, from which credible
intervals are drawn, with no per-dataset fitting or tuning.

What it is and is not for this platform:

- It learns no structure and selects no adjustment set. It assumes exactly what the backdoor
  adjustment assumes. Discovery, the DAG, the adjustment guarantee, and expert review stay as
  they are; CausalPFN is at most a fifth candidate in the energy-score estimator selector.
- The strongest argument for it is not accuracy but cost of re-inference: refutation on an
  amortized estimator is a forward pass, which would let debt 1's reconstruction, calibration
  and budget machinery go away.
- The strongest arguments against: its training prior covers continuous outcomes only, and
  every outcome on this platform is binary 0/1, so our runs are out of the prior's
  distribution; the paper reports the model "becomes severely overconfident when evaluated on
  OOD DGPs", corrected by temperature scaling. It is also a black box next to a DML
  specification a pharma reviewer can read.

| Fact | Value | Fit |
|---|---|---|
| Treatment support | binary only; multi-arm in theory, continuous unexplored | fine, the platform binarizes |
| Outcome prior | continuous only | out of distribution for binary outcomes; the risk to test |
| Calibration | in-distribution calibrated; OOD overconfident until temperature-scaled | must be measured on our outcomes |
| Context limit | about 50k rows, degrades above | our runs are 1,500 |
| Benchmarks (paper) | best average rank on CATE across IHDP, ACIC, Lalonde; competitive on ATE | encouraging, all continuous-outcome |
| Runtime today | 110 s per run, four estimators fitted | inference would be seconds |
| Container readiness | torch 2.9.1 CPU present; huggingface_hub present; `faiss-cpu` missing; 2 CPUs; 3.7 GiB headroom under the 5 GiB limit | feasible; one dependency; memory must be measured |

Recommendation: a one-day scratch trial, after lane 1's engine work lands, with pass criteria
fixed before it runs, on frames whose truth we know. Not part of this lane.

Trial design:

1. Frames: the DGP recovery probe frames (`tests/integration/test_dgp_recovery_probe.py`:
   three brands, `n_records=3000`, seed 21, heterogeneous DGP, planted `true_ate_by_arm` and
   segment CATE map) plus the two 1,500-row live frames replicated in §2.
2. Estimator: `causalpfn` `ATEEstimator` / `CATEEstimator` on CPU, the adjustment set the
   platform would use for each frame, temperature scaling as the paper prescribes.
3. Pass criteria, all required:
   - |ATE − planted truth| < 0.15 on every frame (the gate LinearDML already passes);
   - segment CATE ordering high > medium > low preserved on every brand;
   - the 95% credible interval covers the planted truth in at least 18 of 20 seeds;
   - wall-clock under 10 s per frame and peak RSS under 1 GiB on 2 CPUs, measured in the
     `e2i_api` image.
4. On pass: integrate as a logged shadow candidate in the selector (never selected, always
   compared) for a few weeks of live runs; then allow selection; then decide whether debt 1's
   reconstruction path is retired.
5. On fail (most likely on credible-interval coverage for binary outcomes): stop, record the
   numbers here, and keep the DML / forest estimators.

The one fact that would reverse the recommendation to try it is a coverage collapse on binary
outcomes, which is exactly what criterion 3 measures. A later caution: a model trained on
synthetic priors may look best on this synthetic substrate and say less about Optum or CSU
data when they arrive; the recovery benchmark is still the right first test, and the
shadow period on live runs is the second.

**Trial result (2026-09-11) — FAIL, item 5 applies.** Run as designed (60 DGP frames = 3 brands ×
20 seeds at n = 3,000, plus the two §2 live pairs; `causalpfn 0.1.4` in a capped scratch container from
the deployed `e2i_api` image, 2 CPUs; full record in `docs/demos/results/2026-09-11_causalpfn_trial/`):

| criterion | uncalibrated (T = 1) | calibrated (paper's temperature scaling) |
|---|---|---|
| \|ATE − truth\| < 0.15, every frame | PASS 60/60 (max 0.067; LinearDML max 0.070) | PASS 3/3 |
| segment order high > medium > low, every brand | FAIL 46/60 (Kisqali 11/20) | 2/3 |
| 95 % interval covers truth in ≥ 18/20 seeds | FAIL 10 / 13 / 11 of 20 (LinearDML 18 / 16 / 19) | FAIL 0/3 |
| < 10 s per frame | FAIL median 60 s (23 s CATE passes + 36 s interval draws) | FAIL ≈ 470 s |
| peak RSS < 1 GiB | FAIL 2.4 GiB | FAIL 1.8–2.0 GiB |

The coverage collapse named above is what the sweep measured: intervals half the analytic width
(0.05 vs 0.10) with one-sided high misses. Calibration selects the floor temperature (0.001) on every
frame, DGP or live, and yields a 0.002-wide interval that excludes its own point estimate. Two package
findings: `CATEEstimator.estimate_ate_CI` raises `KeyError('ate')` in 0.1.4 (the helper omits the key),
and the interval samples use the calibrated `temperature` while the point estimate uses
`prediction_temperature`. On the live pairs the uncalibrated point estimates fall inside the reported
intervals (0.062 in [0.019, 0.152]; 0.156 in [0.117, 0.236]). Consequence for §10 debt 1: the
amortized-estimator route to retiring the reconstruction path is closed; debt 1 is a refactor question.
Not integrated, no shadow candidate, `causalpfn` not added to the image. Reopens only with a continuous
outcome on this platform or a release whose calibration gives non-degenerate intervals on a binary one.

Sources: https://github.com/vdblm/CausalPFN , https://arxiv.org/abs/2506.07918 ,
https://arxiv.org/html/2506.07918v2 , https://pypi.org/project/causalpfn/ .

## 12. Owner decisions after the pre-execution review (2026-09-09)

The plan was adversarially reviewed before Task 1 (four codex read-only rounds, 21 findings, five local
disproofs, live BEGIN/ROLLBACK experiments — recorded at the end of the plan). Six decisions were put to the
owner with recommendations; the owner's answers, now part of this design:

1. **Execution**: subagent-driven, one task at a time in the lane worktree, red-first TDD, codex fixed point per
   task (ralph-loop + codex-rescue), memory monitored, CI batched into one push/PR/deploy at the end.
2. **Degenerate resample distribution** (every re-fit returns the same effect) in a NON-critical test: an honest
   SKIPPED with reason `degenerate_resample_distribution`, never a placeholder p-value and never a fail-closed
   halt — the critical placebo gate catches an estimator that ignores its data. (§4.1 amended by this note;
   exceptions inside the loops still raise `RefutationError`, §5.)
3. **Promote concurrency**: `promote_causal_path_guarded` takes `LOCK TABLE public.expert_reviews IN SHARE MODE`
   before its UPDATE (measured live: blocks a racing resolve UPDATE and a racing renew INSERT, not reads). An
   advisory lock keyed by hash, shared with the resolve/renew paths, only if reviewer contention ever appears.
4. **Timestamp tie**: a pending row with the same `created_at` as the adjudication after it is NOT a reopen — in
   SQL (strict `>`) and in both Python readers (`_latest_adjudication` tie-only rule; the durable-rejection block
   in `check_approval` moved above the pending check). "Most recent adjudication wins" (#1971) is preserved.
5. **Live lock rehearsal** holds the table for 6 s, not 20 s.
6. **Evidence writer**: `causal_validations.details_json` / `test_config` are written as JSON objects (the writer
   json.dumps'ed into jsonb since `0742b81f6`; 480 live rows were strings), with backfill migration 135, so the
   lane's per-resample evidence is written, backfilled and tested in one shape. The same pattern on
   `expert_reviews.agent_assessment_json` / `checklist_json` is a filed follow-up, not this lane.
