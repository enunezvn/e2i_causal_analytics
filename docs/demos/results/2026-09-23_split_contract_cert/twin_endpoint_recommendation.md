# Twin endpoint recommendation (ADVISE only — nothing changed) — 2026-09-23

Question (owner): which REAL endpoint should the digital twin predict?
Evidence: read-only exploration `explore_twin_endpoint.md` (this scratchpad); every load-bearing count re-verified by me with psql on supabase-db.

## What is true today (measured)

- The twin's effect estimator predicts `business_metrics.cohort_conversion_outcome` — a constant in `src/data/per_hcp_cohort_columns.py:39`, written only by the plant script `scripts/backfill_segment_engagement.py`; values 0–3.9 with no declared unit; 13,797 rows, ALL is_synthetic=true.
- There is NO real per-HCP outcome anywhere: business_metrics, hcp_brand_adoption, patient_journeys, hcp_profiles, triggers, hcp_intent_surveys, ab_experiment_assignments are 100 % synthetic. The only is_synthetic=false outcome table is `optum_biologic_persistence_causal` (15,209 real patients, XOLAIR/DUPIXENT, persistent_at_180d 47.5 %) — it has no hcp_id/npi, joins 0 rows to patient_journeys, and its brands are not the platform's.
- The scoring loop has never closed: twin_fidelity_tracking = 0 rows, and `ab_experiment_assignments.unit_id` uses `hcp_00000..hcp_01334` while every HCP table uses `scvhcp_00000..scvhcp_04999` → 0 joinable keys. Changing the endpoint alone cannot close the loop.

## Candidates

| rank | endpoint | grain / rows | why | cost |
|---|---|---|---|---|
| 1 | `hcp_brand_adoption.adopted` | (hcp_id, brand), 15,000, ~40 % positive, exact 60/20/10/10 splits, own treatment_arm | the platform's canonical HCP endpoint: 3 real ml_experiments + 3 production goldstd models, the `hcp_adoption` causal dataset's declared outcome, cohort_spec's only HCP cohort; 10,136 (hcp,brand) pairs / all 4,813 twin-cohort HCPs join | loader becomes a two-read merge (hcp_brand_adoption + business_metrics channels), METRIC_COLUMN_MAP needs a table-keyed arm; temporal validity: only ~265 rows/brand have consideration_date after the planted channels start (2026-05) — below COHORT_MIN_ROWS=500 unless the plant is re-dated |
| 2 | `patient_journeys.persistent_180d` (or `treatment_initiated`) rolled to HCP | per patient, ~2.1 journeys/HCP/brand | strongest clinical meaning; 9 real experiments with locked leakage-safe covariates | grain change + noisy per-HCP rate; larger refactor (reuse resolve_cohort_outcome_frame) |
| 3 | `hcp_intent_surveys.intent_to_prescribe_change` | per HCP per survey, 50,000, continuous | cheapest swap; the only endpoint contemporaneous with the planted treatments (2026-03..09) | stated intent, not behaviour; no ML experiment registered against it |
| — | `optum_biologic_persistence_causal.persistent_at_180d` | per patient, REAL | the only real outcome | not wireable: no HCP key, no commercial exposure, wrong brands — a data-acquisition project, not a config change |

## Recommendation

Point the twin at `hcp_brand_adoption.adopted` at (hcp_id, brand) grain, and sequence the A/B key-namespace repair BEFORE the endpoint change.

Single strongest reason: it is the only choice where the twin stops inventing a private endpoint and predicts the same quantity the ML platform, the causal engine and the experiment feed already treat as the target, in an interpretable unit (adoption risk difference in percentage points) — so when real HCP data arrives, the twin is already pointed at the right name.

Confidence: moderate. It is a "platform-canonical" answer, not a "real data" answer — no real per-HCP endpoint exists to choose.

The one fact that would reverse it: if the planted treatment channels cannot be re-dated to precede `consideration_date`. Strict treatment-before-outcome leaves ~265 rows/brand (< 500 gate); if re-dating the plant is off the table, choose candidate 3 (`intent_to_prescribe_change`), the only endpoint whose window overlaps the treatments.

Not measured: whether `adopted` responds to any of the 8 planted channels at all (the two plants are independent DGPs — an estimation run would tell; cheapest disproof, not run under the read-only constraint); whether real HCP outcome data is scheduled to arrive; whether the `hcp_NNNNN` A/B namespace is a bug or a separate seed population.

Owner decision needed: which candidate. Nothing has been changed, retrained or re-pointed.

## Addendum (2026-09-23 03:20Z) — a REAL HCP-grain adoption asset exists on disk

Verified: `data/rwd/mart/hcp_adoption/e2i_ml_v3_patient_journeys.parquet` = 40,000 real Optum HCPs (`optum_mart.optum_hcp`, converter `scripts/convert_optum_hcp_adoption.py`), binary `adopted_target_brand` = 929 positives (2.32 %), real referral-network covariates (`influence_network_size`, `referral_in_degree`, `kol_score`, ...), stratified-random `data_split` 60/20/15/5 (`optum_hcp_adoption_v1`), ids `HCP_<npi-like>`. It is NOT loaded into any table.

This reinforces rank 1: adoption is the only endpoint with a real HCP-grain counterpart. Two consequences:
- Second reversing fact: the Optum HCP ids (`HCP_1560…`) must be reconcilable with the units an experiment can enrol (`scvhcp_…` / `hcp_…` today) — if they cannot, the real cohort can train an adoption model but never score a twin against an experiment.
- Its 2.32 % base rate is thin for detecting an absolute ATE, and it carries no manipulable commercial-channel exposure (the network position is a covariate, not a treatment) — so it is an outcome + covariate substrate, not a treatment substrate.
