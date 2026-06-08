# Twin H9 + R5 Fidelity-Outcome-Feed — Design Brief (2026-06-08)

Goal: fix **H9** (twin↔experiment resolution landmine) and complete the **R5
fidelity-boundary outcome feed** to the point where the twin-prediction-vs-actual
fidelity leg yields REAL rows from REAL data (no mocking). Worktree
`feat/twin-h9-r5-outcome-feed`. TDD red-first → codex fixed point → PR; CI batched at end.

## Intent (ascertained from code + schema, not assumed)

The fidelity loop: a digital twin predicts an intervention ATE (`twin_simulations.simulated_ate`);
later a real A/B experiment runs and produces an **actual** ATE; we compare predicted-vs-actual
to track twin calibration. Chain (all real columns, verified):

```
per-unit experiment outcomes  ──(MISSING: the "outcome feed")
  → ResultsAnalysisService.compute_itt_results(control_data, treatment_data)
  → ab_experiment_results.effect_estimate          (actual ATE; 0 rows today)
  → compare_experiment_to_twin → compare_with_twin_prediction
  → FidelityComparison → ab_fidelity_comparisons / twin_fidelity_tracking
```

The producer is `fidelity_tracking_update` (Celery), enqueued best-effort on a FINAL analysis
(`ab_testing_tasks.py` ~L693). It is the R4b/H9 wiring; it is currently DORMANT because the
outcome feed bails (`control_data=[]` placeholder → `insufficient_data`, the #422 fail-safe).

## Findings (data-driven, faithful prod DB)

- All A/B outcome tables EMPTY: `ab_experiment_assignments=0`, `ab_experiment_enrollments=0`,
  `ab_experiment_results=0`, `ab_fidelity_comparisons=0`. `ml_experiments=616` (MLflow rows).
- **No per-unit outcome storage exists**: assignments carry `(experiment_id, unit_id, unit_type, variant)`
  — NO outcome value; enrollments carry lifecycle (consent/withdrawal) — NO outcome. No dedicated
  `metric_observations` table ever landed (the #422 intent; #422 closed via the fail-safe ONLY).
- **Real per-unit outcome source = `business_metrics`** (`metric_type='per_hcp_rollup'`):
  per-`hcp_id` typed columns `trx_count, nrx_count, total_rx_count, market_share, conversion_rate,
  engagement_score, call_frequency` by `brand`/`metric_date` (the generic `value` col is NULL for
  these rows). 539 per-HCP rows, 163 distinct HCPs; Fabhalta 94 / Kisqali 82 / Remibrutinib 76 HCPs.
  FK `business_metrics.hcp_id → hcp_profiles.hcp_id`.
- **H9 link key = `twin_simulations.experiment_design_id`** (uuid, indexed, FK→`ml_experiments.id`).
  The H9 comment "twin_simulations has no experiment_id column" is imprecise — `experiment_design_id`
  IS the experiment link. The global `list_simulations(limit=1)` fallback is the cross-brand landmine.

## Design

### H9 — experiment-scoped, brand-safe sim resolution (no global fallback)
- Add `SimulationRepository.get_latest_for_experiment(experiment_design_id, *, brand=None)`:
  `SELECT … FROM twin_simulations WHERE experiment_design_id = :id [AND brand=:brand]
   ORDER BY created_at DESC LIMIT 1`. Returns None if none.
- `compare_experiment_to_twin`: when `twin_simulation_id is None`, resolve via the above using the
  experiment_id (== design id in the common 1:1 case). If none → raise ValueError (skip), **never**
  fall back to the newest sim globally. Keep explicit-id branch unchanged.
- Producer (`ab_testing_tasks.py` FINAL branch): keep best-effort enqueue (it self-skips); the
  scoping now lives in the resolver so a future explicit id is optional. Update the stale comment.

### R5 — real outcome feed
- New `ExperimentOutcomeRepository.load_arrays(experiment_id, primary_metric, *, window=None)`:
  1. assignments = `ABExperimentRepository.get_assignments(experiment_id)` → `(unit_id, variant)`.
  2. map `primary_metric` → a business_metrics typed column via an explicit allow-list
     (`trx|trx_count→trx_count`, `nrx→nrx_count`, `total_rx→total_rx_count`, `market_share`,
     `conversion_rate`, `engagement_score`, `call_frequency`). Unknown metric → raise (fail-closed,
     never silently pick a column).
  3. per assigned hcp_id: aggregate the column over `metric_type='per_hcp_rollup'` rows
     (brand-scoped via the experiment's brand; window-filtered on `metric_date` if a window given) →
     one real outcome value per unit (sum for counts, mean for rates — per-metric reducer).
  4. split by variant (control vs the treatment variant) → `(control_data, treatment_data)` np arrays.
  5. Return empty arrays when no assignments / no matching metrics → caller bails `insufficient_data`
     (the existing honest path). NEVER fabricate.
- Wire into BOTH placeholder sites in `ab_testing_tasks.py` (interim ~L223-227 and final/results
  ~L652-657): replace `control_data=[]; treatment_data=[]` with the loader; bail only on genuinely
  empty arrays (keep the NaN-safety guarantee from #422).
- Variant naming: assignments use `variant` (varchar). Treat the configured control label vs others;
  default control label "control" (confirm against ab_experiment design defaults).

### Faithful real-results proof (no mocking)
There are 0 executed experiments, so the proof builds ONE REAL experiment from REAL data:
- pick a brand with ≥ N HCPs in business_metrics (e.g. Fabhalta, 94 HCPs);
- create a real `ml_experiments` row + real `ab_experiment_assignments` over those real hcp_ids,
  randomized to control/treatment (real rows, a real experiment setup — not a mock);
- run the REAL loader → REAL per-HCP outcomes from business_metrics → REAL `compute_itt_results`
  → REAL `ab_experiment_results.effect_estimate`;
- create a REAL twin simulation row keyed by `experiment_design_id` (or use the existing model),
  run `compare_experiment_to_twin` → REAL `FidelityComparison` persisted.
- Assert: real finite ATE, a persisted fidelity row, H9 resolver picks the experiment-scoped sim
  (not a cross-brand one). Tear the fixture rows down after (user pattern: confirm then delete).
- The arithmetic (ATE, CI, fidelity score) is REAL over REAL metric values. Assignment rows are a
  real experiment configuration, not fabricated outcomes.

## Open questions for codex convergence (verify against source)
1. Is `business_metrics.per_hcp_rollup` the intended outcome source, or is a dedicated
   `ab_metric_observations` table the design intent? (An empty new table gives no real results —
   argue from intent + no-mocking.)
2. primary_metric→column mapping + per-metric reducer (sum vs mean) — correct for ITT?
3. Window semantics: post-assignment `metric_date` window vs all-time. What does the experiment
   design actually specify (experiment_designer / ab config)?
4. H9: is `experiment_id == experiment_design_id` safe to assume, or must we map design→run?
5. Anything that makes this a labeling fix rather than a functional one (flag HIGH).

## CONVERGED VERDICT (codex + independent source/DB verification, 2026-06-08)

Codex confirmed the design and surfaced 3 issues; ALL independently verified against source/DB:

- **Q1 source = `business_metrics.per_hcp_rollup`** — CONFIRMED. No `ab_metric_observations` table exists; an empty new table = no real results (labeling, not functional). Use the real per-HCP rows.
- **Q2 mapping/reducer** — CONFIRMED. `_compute_results` (results_analysis.py:262) does a pooled two-sample t-test on **per-unit scalar** arrays (`treatment_mean - control_mean`, `n=len(arr)`). Reducer = collapse multiple `metric_date` rows per HCP → one scalar: **SUM for counts** (trx/nrx/total_rx), **MEAN for rates** (market_share/conversion_rate/engagement_score/call_frequency). Filter out HCPs whose metric value is NULL (don't feed NaN). Unknown metric → raise (fail-closed).
- **Q3 window** — `ml_experiments.observation_window_days` exists (nullable) but is an ML-training window (MED-1), not an A/B measurement window; no A/B window columns exist. **Decision: default all-time** (`metric_date` unfiltered) since 0 real assignments; accept optional `window_days` param for future use. Documented.
- **Q4 H9** — `experiment_design_id` FK→`ml_experiments.id`, set explicitly by `link_experiment` (twin_repository.py:532), NOT auto. Same id space as `ab_experiment_assignments.experiment_id` ⇒ 1:1 safe **when linked**; if NULL → raise (skip), never global. Don't conflate with `twin_fidelity_tracking.actual_experiment_id`.
- **Q5 seams** — line numbers corrected: interim bail `ab_testing_tasks.py:241-252`, final bail `:663-713`, global fallback `results_analysis.py:636-638`, `_compute_results` `:262`.

### NEW issues found (verified):
- **HIGH-1 (on the R5 live path)**: `ab_results.py` `save_fidelity_comparison` insert dict uses keys `prediction_error_percent` + `ci_coverage` — **neither is a real column** (real: `relative_prediction_error`, `confidence_interval_coverage`). PostgREST errors on unknown cols ⇒ the FIRST real fidelity write fails. Persist IS wired (`compare_with_twin_prediction:593 → _persist_fidelity_comparison:740 → save_fidelity_comparison`). `_to_fidelity_record` reads the bad names too. FIX read+write mapping; keep the in-memory `FidelityComparison` dataclass names. Triggers compute prediction_error/abs/rel + fidelity_grade.
- **MED-2**: global-fallback docstring "twin_simulations has no experiment_id column" is INACCURATE — correct it (it's `experiment_design_id`).

### Implementation checklist (TDD red-first; both A/B sites wired; no feature omitted):
1. `SimulationRepository.get_latest_for_experiment(experiment_design_id, *, brand=None)` (twin_repository.py).
2. `compare_experiment_to_twin` None-branch → experiment-scoped resolver; raise if none; fix docstring (MED-2).
3. `ab_results.py` HIGH-1: fix insert + read column names (`relative_prediction_error`, `confidence_interval_coverage`).
4. New `src/repositories/experiment_outcome.py` `ExperimentOutcomeRepository.load_arrays(experiment_id, primary_metric, *, window_days=None)` → (control, treatment) np arrays from real business_metrics; empty → caller bails.
5. Wire BOTH `ab_testing_tasks.py` sites (final `compute_experiment_results` + interim `scheduled_interim_analysis`): fetch ml_experiments brand+prediction_target, load arrays, bail `insufficient_data` only on genuinely empty, else real compute → persist.
6. Faithful proof: real ml_experiments + real ab_experiment_assignments over real Fabhalta HCPs → real load_arrays → real compute_itt_results → real ab_experiment_results → real twin_simulations(experiment_design_id) → real compare_experiment_to_twin → real ab_fidelity_comparisons. Tear down after.

## IMPLEMENTATION OUTCOME (2026-06-08, TDD + faithful E2E)

Both A/B consumers of the feed wired (final-results + interim). The faithful real-DB E2E
(`tests/integration/test_twin_fidelity_outcome_feed_e2e.py`, gated `E2I_DB_INTEGRATION=1`) drove
the whole chain over REAL Fabhalta per-HCP `business_metrics` and CAUGHT 5 latent layers that had
never run end-to-end (the cheapest-disproof value — introspection could not see them):

1. **numpy bool not JSON serializable** — `_compute_results` set `is_significant = p_value < alpha`
   (np.bool_) + left `p_value`/`relative_lift`/`power` as numpy scalars → `save_results` insert blew
   up. Fixed: `bool()`/`float()`/`int()` coercion at construction (honours the dataclass contract).
2. **`save_results` wrong columns** — wrote `sample_size_control`/`sample_size_treatment`/
   `statistical_power`; real cols are `control_n`/`treatment_n`/`observed_power`. Fixed write+read.
3. **HIGH-1 fidelity columns** — `save_fidelity_comparison` wrote `prediction_error_percent`/
   `ci_coverage` (nonexistent) → PGRST204. Fixed to `relative_prediction_error`(×100 on read)/
   `confidence_interval_coverage`; added `comparison_timestamp`. Triggers compute the rest.
4. **interim `EnrollmentStats.target_sample_size`** — phantom attribute (AttributeError). Sourced
   from `interim_config` or the real `total_assigned` (assigned cohort = planned N).
5. **interim `MetricData(name=…)` required** + **double-persist** — `perform_interim_analysis` already
   persists internally (`_persist_analysis`); removed the redundant `record_interim_analysis`
   (was tripping `unique_analysis_number`).

Proven green: 167 unit tests + 2 faithful E2E (final fidelity chain → real `ab_fidelity_comparisons`
row; interim sequential test `p=0.111, decision=continue` → real `ab_interim_analyses` row). DB
returned to 0 rows after teardown. H9 decisively verified: the experiment-scoped resolver picks the
linked sim, NOT a newer cross-brand decoy (the old global fallback would have).
