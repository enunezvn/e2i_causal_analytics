# #1991 debts 3 and 4 with #2029 folded in — a four-lane wave

Date: 2026-09-12. Owner decisions taken in the brainstorm on 2026-09-12: sequenced wave; reviews mint
only where a decision can change an outcome, keyed on the estimand; full routes split by concern; a
1,500-line ratchet guard; Python-and-wire vocabulary renames without a column rename; widen and clamp
`delta_percent` and persist seeds.

Origin: `docs/superpowers/specs/2026-09-08-expert-review-loop-closure-design.md` §10 (debts 3 and 4),
issue #1991, issue #2029. Every claim below was verified in source and against the live database on
2026-09-12 (read-only); line numbers are as of main `2b43ee85e` and will drift, so the text names
symbols wherever a symbol exists.

## 1. Goal and non-goals

Goal: remove three sources of hidden defects that the retrospective measured — a review queue whose
approvals change nothing, gate vocabularies that collide by name, and a route module nobody can hold —
and make the refutation gate deterministic, without changing any served number, band, or route contract.

Non-goals, recorded so nobody re-derives them: the three chat brains (documented, not merged); renaming
the `discovered_dags.gate_decision` column (owner chose Python-and-wire renames only; a follow-up
issue may file the column rename); the seven status vocabularies beyond the three gates; the
CausalForestDML leaf-5 sites (PR #2047 scope note); any change to band thresholds or test weights.

## 2. What the code and the live database say (2026-09-12)

### Debt 3 — the review key is the structure hash
- Key: `compute_dag_hash` (`src/causal_engine/dag_hash.py`) = SHA-256 of sorted nodes, edges,
  treatment nodes and outcome nodes; the adjustment set is deliberately outside it. Pending uniqueness is
  the partial index `uq_er_pending_dag_brand` on `(dag_version_hash, COALESCE(brand,''))` (migration 062).
  The estimand columns `brand`, `treatment_variable`, `outcome_variable` exist on every row but key nothing.
- Minting: only the refutation node, on BLOCK and REVIEW bands (`_review_fields_for_band` →
  `ExpertReviewGate.check_approval` → `ExpertReviewRepository.create_review`). PROCEED never mints.
- Live: 37 pending / 2 rejected / 1 approved. All 37 pending carry `gate=block` in their context; 27
  distinct hashes over 27 distinct estimands; one estimand (Remibrutinib `treatment_arm → persistent_180d`)
  holds 4 pending reviews under 4 hashes. No review has been minted since 2026-09-08 because discovery now
  bands 11/11 PROCEED. All 37 carry `dag_structure_json`.
- Effect of a decision: rejection halts every band and blocks promotion (`dag_structure_rejected` inside
  `promote_causal_path_guarded`, migration 134). Approval only matters on a REVIEW band when
  `CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL` is on (default off) or to supersede an earlier rejection. A BLOCK
  row is already terminal (`status="failed"` is set before the review fields), so approving any of the 37
  changes nothing. The debt holds exactly as written.
- `get_dag_changes(old, new)` (`dag_hash.py`) already computes node/edge diffs and has zero production
  callers; nothing lists DAG versions; the frontend `DagPanel` renders one snapshot.

### Debt 4a — gate vocabularies
- Three gates, not two: discovery `accept / review / reject / augment` (`src/causal_engine/discovery/base.py`
  `GateDecision`; live `discovered_dags`: accept 154, augment 5), refutation `proceed / review / block`
  (`src/causal_engine/refutation_runner.py` `GateDecision`; live `causal_validations`: proceed 1,207, block
  336, review 18), expert review `proceed / pending_review / renewal_required / rejected / blocked /
  unavailable` (`src/causal_engine/expert_review_gate.py` `ReviewGateDecision`). Seven further status
  vocabularies ride the same flow (per-test status, `causal_paths.validation_status`, `approval_status`,
  run status, `quality_tier`, `dag_source`, negative-control `skip_reason`).
- Two Python classes named `GateDecision`, distinguished only by import path; two `public.*.gate_decision`
  columns of different enum types (migration 036 renamed the SQL type to `discovery_gate_decision` but
  not the column).
- A live defect: `_summarize_refutation_rows` in `src/api/routes/chatbot_tools.py` reads
  `causal_validations` rows (refutation vocabulary only) yet merges `{"block","reject"}` and
  `{"review","augment"}`, leaving dead branches, and its `else` maps a null or unknown gate to
  `"proceed"` — fail-open in a path that is fail-closed everywhere else. Its tests feed only refutation tokens.
- `config/domain_vocabulary.yaml` has no discovery-gate section; `scripts/validate_vocabulary_enum_sync.py`
  therefore does not check it, and `refutation_test_types` lacks `negative_control_outcome` (migration 138).
- Wire types: `gate_decision` and `expert_review_decision` are `str` in the API schemas and `string | null`
  in the frontend; the badge falls through to a dash on any unknown token.

### Debt 4b — `src/api/routes/causal.py`
- 6,601 lines, 23 routes, 111 top-level defs (88 private), 18 concern clusters, 46 commits in 60 days,
  18 distinct issue references.
- 50 symbols escape the module (47 private): `segments.py` (8 function-local import sites, 7 symbols),
  `scripts/calibration/reband_sensitivity_readings.py`, and 42 test files with 87 attribute-patch sites over
  23 symbols, 4 string patch paths, and one file-path allowlist
  (`tests/unit/test_utils/test_query_log_redaction_guard.py`).
- Coupling: the dataset registry (`_CAUSAL_DATASET_SPECS` and friends) is used by five clusters, `segments.py`
  and the script; the loaders (`_load_agent_estimation_frame`, `_coerce_estimation_row`, …) by four; the
  discover-effects job drives `_run_agent_analysis_task`; treatment-effects paging constants defined near
  line 6,250 are read by loaders near line 2,700; `_SurfaceCSequentialPipeline`, `_dowhy_interval` and
  `_te_pvalue_from_z` cross between the pipeline block and treatment-effects.
- Import-time side effect: `ClinicalContextService()` at module level builds four external HTTP clients;
  an `E402` import sits 1,700 lines in.
- Registration: one `include_router(causal_router, prefix="/api")` in `src/api/main.py`; router prefix
  `/causal`, tag `Causal Inference`, shared `responses`; three explicit `operation_id`s
  (`get_pipeline_status`, `get_causal_estimation_data`, `causal_health_check`).
- No module-size guard exists anywhere (ruff selects no `PL` rules; no meta test; pre-commit checks byte
  size of new files only). Measured 2026-09-12: **32 files** under `src/` exceed 1,500 lines, from
  `routes/causal.py` 6,601 and `data/causal_role_classifier.py` 6,525 down to `routes/predictions.py` 1,519;
  the guard's allowlist starts with those 32 minus `routes/causal.py` (31 pins), re-measured when the test
  is written.
- Three chat brains, confirmed: the orchestrator graph (`routes/chatbot_graph.py`, served through
  `/api/copilotkit/chat/stream`), the AG-UI runtime (`routes/copilotkit.py`, `/api/copilotkit/agent/default`,
  the one the browser sidebar uses), and the suggestion-pill call (`routes/chat.py`,
  `/api/chat/suggestions`); `routes/chat_bridge.py` re-runs a failed orchestrator turn through the runtime.

### #2029 — refutation determinism and `delta_percent`
- `RefutationRunner` has no seed config. `placebo_treatment_refuter` (`_run_placebo_test`) and
  `random_common_cause` (`_run_random_common_cause_test`) are called with no seed, so their draws come from
  global numpy state. `data_subset` and `bootstrap` are in-house resamples seeded from
  `_resample_seed_for(estimate_id)` (SHA-256 of the estimate id); the live path always passes the query id
  (11/11 post-deploy rows carry an `estimate_id`), but the seed is not persisted.
- DoWhy 0.14: the correct kwarg is `random_state`; the two unseeded refuters convert an int to a
  `RandomState` once and share it across simulations. `random_seed` only calls `np.random.seed` globally.
- `causal_validations.delta_percent` is `DECIMAL(8,4)` (`database/ml/010_causal_validation_tables.sql`).
  The clamp `_DELTA_PERCENT_COLUMN_MAX` is applied only in `_run_negative_control_test`; the four
  perturbation tests share an unclamped `abs(Δ) / max(|original|, 1e-10) * 100`. `save_suite` is one bulk
  insert, so a single overflow drops the whole suite silently (logged, returns `[]`). Migration 139 is free
  (highest existing: `database/migrations/138_…`).

## 3. Wave shape

Four lanes, serial, each in its own worktree with its own PR, deploy and live cert, under the standing
lane protocol (`feedback_lane_execution_protocol_20260909`: red-first, `pytest -n 0`, ralph-loop plus codex
read-only rounds to a fixed point per task, one push per lane, never squash). Order and reason:

| lane | scope | why this position |
|---|---|---|
| 1 | #2029 determinism + `delta_percent` | smallest, removes a live flake (placebo BLOCK flip at 10 unseeded sims) |
| 2 | one vocabulary per gate | contains a fail-open defect; small; lands before the split moves the summarizer's neighbours |
| 3 | routes split + size guard | largest blast radius (42 test files); goes after the two small lanes and before lane 4, which edits the agent-analyze module the split creates |
| 4 | reviews keyed on the estimand | needs migrations 140/141 and a live data fix; ships last with its own cert |

Each lane is independently shippable. No lane changes a served number, band, threshold, route path,
status code, auth dependency or operation id; every lane's cert includes a negative control that proves
its check would fail on the pre-lane image.

## 4. Lane 1 — refutation determinism (#2029)

### Seeding
- One integer seed per run, derived from the estimate id the same way `_resample_seed_for` already does,
  computed once in `run_all_tests` and passed to `_run_placebo_test` and `_run_random_common_cause_test`,
  which forward it as `random_state=` to `refute_estimate`. No new config key: identical inputs give
  identical verdicts; a run with no estimate id stays unseeded and records `random_state: null`.
- The calibration probe in `nodes/refutation.py` (`num_simulations=1`) also passes the seed; it measures
  wall time, so determinism there is free.
- The runner's `run_all_tests` signature gains nothing public; the seed rides the existing
  `estimate_id` argument.

### Persistence
- Every persisted test row's `details_json` carries the seed it used: `resample_seed` for subset and
  bootstrap (already computed, never written), `random_state` for placebo and random common cause,
  absent for the analytic tests. A flipped verdict is reproducible from the row.

### Overflow
- Migration `database/migrations/139_causal_validations_delta_percent_numeric_12_4.sql`:
  `ALTER TABLE public.causal_validations ALTER COLUMN delta_percent TYPE NUMERIC(12,4)`; idempotent
  (guard on the current type); the 119 synthetic seeding function's `round(…, 4)` is unchanged.
- One clamp at the write boundary: `CausalValidationRepository._test_to_row` (and `save_single_test`) clamp
  `delta_percent` to the new bound `99999999.9999` and keep the unclamped value in
  `details["delta_percent_exact"]` when clamping occurred. The runner's `_DELTA_PERCENT_COLUMN_MAX` and the
  negative-control-only clamp are retired; the exact ratio the negative-control test already stores in
  `details["control_to_claimed_ratio"]` is unchanged.

### Tests (red first)
- Two-run identity: `placebo_treatment` and `random_common_cause` at `num_simulations=2` return identical
  `refuted_effect`, `p_value` and status on two runs with the same estimate id, and differ across two
  estimate ids (the differing case is the positive control against a seed that ignores its input).
- Persisted seed: the row builder emits `random_state`/`resample_seed` in details for the four
  perturbation tests and omits them for `sensitivity_e_value` and `negative_control_outcome`.
- Near-zero claim: a suite with `original_effect = 1e-8` and a perturbation shift of 0.01 persists all rows
  through `save_suite` with `delta_percent` clamped and `delta_percent_exact` present; the existing
  negative-control clamp test moves to the repository boundary.
- Migration 139 rehearsed on the live database in `BEGIN … ROLLBACK` with a positive control (insert a
  `delta_percent` of 123456.7 succeeds after, fails before).
- `tests/unit/test_causal_engine/test_refutation_bands_enumeration.py` stays green unchanged (no band arithmetic moves).

### Cert (deployed image)
- Ledger shows migration 139; the column type reads NUMERIC(12,4).
- Two consecutive eleven-pair Remibrutinib discovery runs on the deployed image: placebo and
  random-common-cause verdicts, `refuted_effect` and `p_value` identical pair-for-pair between the runs;
  every perturbation row since the flip carries its seed in details (count = rows, nulls = 0).
- Negative control: the pre-lane image's rows carry no seed keys (measured 2026-09-12: 0 of 11).

## 5. Lane 2 — one vocabulary per gate

### Rename
- `src/causal_engine/discovery/base.py`: `GateDecision` → `DiscoveryGateDecision`; import sites updated
  (`discovery/gate.py`, `discovery/cache.py`, `discovery/observability.py`, tests). The field name
  `discovery_gate_decision` is unchanged everywhere; the refutation `GateDecision` keeps its name.

### The defect
- `_summarize_refutation_rows` (`routes/chatbot_tools.py`) reduces to the three refutation tokens. Gate
  priority stays block > review > proceed. A row set whose gates include a null or an unknown token yields
  `gate = "unknown"`, and the chat answer's provenance carries the caveat that the persisted gate could not
  be read; nothing maps to `proceed` by default. The docstring's cross-vocabulary sentence is removed.

### Typing
- `src/api/schemas/causal.py`: `gate_decision: Optional[Literal["proceed","review","block"]]`,
  `expert_review_decision: Optional[Literal[…six values…]]`, `discovery_gate_decision:
  Optional[Literal["accept","review","reject","augment"]]` where it is surfaced.
- `frontend/src/types/causal.ts` and `expert-review.ts`: matching unions; `frontend/src/types/generated/api.ts`
  regenerated (the verify-types gate is the arbiter; a Literal in the schema changes the generated enum, so
  this lane's `api.ts` diff is expected and reviewed line by line).
- `gateBadge` and `ReviewStatusPanel` render every union member; an out-of-union value is a type error,
  not a dash.

### Guard
- `config/domain_vocabulary.yaml`: new `discovery_gate_decisions` section with all four values, and
  `negative_control_outcome` added to `refutation_test_types`.
- `scripts/validate_vocabulary_enum_sync.py`: a fourth enum check binding the yaml section, the Python
  `DiscoveryGateDecision`, and the SQL type `discovery_gate_decision` (read from `database/ml/036_…`).

### Tests (red first)
- Summarizer: rows with `{"accept"}`, `{"augment"}`, `{None}` and `{"block"}` → the first three read
  `unknown`, the last `block`; a `proceed`-only set still reads `proceed`.
- Enum sync: the script fails when a value is removed from any of the three sources (mutation test in
  `tests/unit/test_scripts/`).
- Schema Literals: a response carrying `gate_decision="reject"` fails validation.

### Cert (deployed image)
- Enum-sync script green in CI on the merge commit; `api.ts` regenerated and committed; a live chat
  provenance answer on a path with block rows reads "block"; a path with no persisted rows reads the
  unknown caveat rather than "proceed" (negative control: the same question on the pre-lane image reads
  "proceed").

## 6. Lane 3 — the routes split and the size guard

### Package layout
`src/api/routes/causal.py` becomes the package `src/api/routes/causal/`:

| module | owns (today's clusters) |
|---|---|
| `datasets.py` | dataset registry `_CAUSAL_DATASET_SPECS`, default dataset, brand scoping, numeric/categorical columns, derivations, fill-zero outcomes, negative-control map, `_list_dataset_brands`, `_is_randomized_treatment`, the `column_labels` re-exports |
| `loaders.py` | `_coerce_estimation_row`, `_load_agent_estimation_frame`, the HCP/NBA/trigger/patient loaders, `_one_hot_categoricals`, `_resolve_requested_baselines`, paging (`_te_paged_select`, `_TE_PAGE_SIZE`, `_TE_MAX_PAGES`) |
| `_common.py` | `_GENERIC_500_DETAIL`, robustness/structural warning strings, `_NO_*_DETAIL`, `_opt_float`, `_as_float`, `_as_optional_float`, `_parse_occurred_at`, `_dowhy_interval`, `_te_pvalue_from_z`, `_CAUSAL_JOB_TTL_SECONDS`, `CAUSAL_COMPLETED_EVENT_TYPE` |
| `catalog.py` | `/brands`, `/variables`, `/propose-questions` (with `_adjusted_partial_corr`), `/clinical-context`, `/estimation-data`; the clinical-context service becomes a lazy accessor |
| `discovery.py` | `/discover-effects*` routes, the durable store, markers, heartbeat, `_prerank_*`, `_discover_candidate_questions`, ranking, `_run_discover_effects_task` |
| `agent.py` | `/agent-analyze*`, `_agent_analysis_store`, `_run_agent_analysis_task`, MLflow recording, `_agent_state_to_response`, `_refutation_tests_from_state`, `_estimator_comparison_from_estimation`, the refutation-config override |
| `pipelines.py` | `/pipeline/sequential`, `/pipeline/parallel`, `/pipeline/{id}`, `/validate`, the pipeline wiring block (`_SurfaceC*Pipeline`, `_build_pipeline_input_*`, output-to-response mapping, structural gate) |
| `hierarchical.py` | `/hierarchical/*`, `_resolve_hierarchical_dataframe`, `_execute_hierarchical_analysis` |
| `activity.py` | `/health`, `/history`, `/value-chains`, `/treatment-effects`, `/estimators`, `_ESTIMATOR_REGISTRY`, the activity cache, chain helpers, `_resolve_treatment_effect_frame`, `_run_treatment_effect_estimate` |
| `__init__.py` | builds one `router = APIRouter(prefix="/causal", tags=["Causal Inference"], responses=…)` and includes the sub-routers; exports `router` only |

Import direction is one-way: `_common` ← `datasets` ← `loaders` ← the route modules; route modules may
import `agent` from `discovery` (the job fans out over the agent task) and `pipelines` from `activity`
(`_SurfaceCSequentialPipeline` for treatment effects); nothing imports upward. The two seams that were
backward today (paging constants, interval/p-value helpers) move down to `loaders`/`_common`.

### Consumers
- `src/api/main.py` is unchanged (`from src.api.routes.causal import router as causal_router`).
- `segments.py` and `scripts/calibration/reband_sensitivity_readings.py` import from `datasets`/`loaders`.
- Tests patch the owning module; the four string patch paths and the 87 attribute patch sites are updated
  in the same PR; the redaction-guard allowlist lists `src/api/routes/causal/`.
- The patchable third-party seams (`get_recent_memories`, `count_memories_by_type`,
  `get_async_supabase_client`, `apply_provenance_filter`) are imported by name at module level in the
  module that uses them, so the existing patch style keeps working there.
- Behaviour is unchanged by construction: routes, paths, auth dependencies, status codes, response
  models, `operation_id`s and tags are moved, not edited. The OpenAPI document must be byte-identical to
  main's; the verify-types gate and a committed snapshot test enforce it.

### Size guard
- `tests/unit/test_tests_meta/test_module_size_ratchet.py`: walks `src/**/*.py`; any file over 1,500
  physical lines fails unless listed in an allowlist dict `{path: pinned_line_count}` kept in the test
  module; an allowlisted file fails if it exceeds its pin, and the test also fails if a pin is above the
  file's current count (so pins can only move down). The new `causal/` package has no entries. The
  allowlist is measured, not guessed, when the test is written; the split lands with `routes/causal.py`
  removed from it.
- The test is CI-visible: `tests/unit/test_tests_meta/` is listed explicitly in
  `.github/workflows/backend-tests.yml` (verified 2026-09-12), so no allowlist edit is needed.

### Tests (red first)
- The ratchet test lands red against `routes/causal.py` at 6,601 lines, then green after the split.
- OpenAPI snapshot: `tests/unit/test_api/test_causal_openapi_unchanged.py` compares the app's OpenAPI
  document for the `/api/causal/*` paths against a fixture captured from main before the split.
- Import side effect: importing `src.api.routes.causal` constructs no `ClinicalContextService` (assert
  on a spy), and the lazy accessor builds it once.
- Every existing test in the 42 files green after patch-site updates, `-n 0`.

### Cert (deployed image)
- OpenAPI document byte-identical to the pre-lane image's (fetched from both, diffed); `api.ts`
  byte-identical to main; ratchet green in CI; a live smoke of one route per module with an authenticated
  token (brands, discover-effects questions, an agent-analyze status read, pipeline status, health,
  history, value chains, treatment effects) returning the same shapes as before the flip.

## 7. Lane 4 — reviews keyed on the estimand

### Key
- A review's identity is `estimand_key = brand || ':' || treatment_variable || ':' || outcome_variable`
  (lower-cased, brand `''` when null). The adjustment set and the DAG hash are the review's version, not
  its identity: a covariate change must update the existing review, so it cannot be part of the key. This
  deviates on purpose from §10's wording "(brand, treatment, outcome, adjustment set)".
- Migration `140_expert_reviews_estimand_key.sql`: add `estimand_key TEXT`, backfill from the three
  columns (every row has them), `NOT NULL` after backfill, replace `uq_er_pending_dag_brand` with a partial
  unique index on `(estimand_key) WHERE approval_status = 'pending'`; keep `dag_version_hash` and its
  approval indexes (approval lookups by hash stay valid).
- Migration `141_expert_review_versions.sql`: table `expert_review_versions(version_id, review_id FK,
  dag_version_hash, adjustment_set_hash, dag_structure_json, query_id, created_at)`; backfill one version
  per existing review from its current hash and snapshot.

### Minting
- REVIEW band mints; BLOCK band does not; PROCEED never did. `_review_fields_for_band` on BLOCK runs
  only the read-only rejection probe.
- REVIEW band on an estimand with a pending review appends a version (hash, snapshot, adjustment-set hash,
  run id) when the hash differs from the latest version, and returns the existing review id; the same hash
  appends nothing.
- REVIEW band on an estimand whose latest review is `approved` for a different hash re-opens: a new pending
  review row keyed on the estimand with `supersedes_review_id` pointing at the approval, carrying the new
  version; the old approval keeps its validity for its own hash.
- Rejection semantics are unchanged: `dag_structure_rejected(hash, brand)` and the migration-134 RPC are
  untouched; a rejection is recorded against the version's hash.

### Diff
- `get_dag_changes` (existing, no callers) computes the diff between consecutive versions; the detail
  endpoint returns `versions[]` with `changes` (nodes/edges added/removed, adjustment-set delta) per version
  after the first.

### Surfaces
- `GET /expert-reviews/pending`: adds `version_count` and `last_changed_at`.
- `GET /expert-reviews/{id}`: adds `versions` with diffs, and its `history` becomes the estimand's reviews
  (not the hash's), each carrying the diff between its latest version and the previous review's — so the
  four backfilled `treatment_arm → persistent_180d` reviews show structural deltas across reviews too.
- `GET /expert-reviews/summary`: counts gain `superseded`.
- Frontend `DagPanel`: renders the latest structure and, when versions > 1, the added/removed nodes and
  edges against the previous version; `ExpertReviews.tsx` shows the version count column.

### Data fix
- A guarded one-shot in migration 140's tail: `UPDATE expert_reviews SET approval_status='superseded',
  resolved_at=now(), comments_json = comments_json || {"superseded_reason": "block-band review; approval
  could not change an outcome (#1991 debt 3)"} WHERE approval_status='pending' AND analysis_context LIKE
  '%gate=block%'`, wrapped in a `DO` block that asserts the affected count equals the count read at the
  top of the same transaction and raises otherwise. Nothing is deleted. Rehearsed in `BEGIN … ROLLBACK`
  on the live database with the count (37 on 2026-09-12) recorded in the lane's evidence.

### Tests (red first)
- BLOCK band mints nothing (node test with a repo spy); REVIEW band mints once and a second REVIEW run
  with a new hash appends a version, with the same hash appends nothing.
- Re-open after approval creates a pending row with `supersedes_review_id`.
- Diff: two versions differing by one confounder node and its edges yield exactly that delta.
- Migrations 140/141 rehearsed with the live copy; backfill counts asserted (rows = versions).
- API tests for the three routes' new fields; frontend `DagPanel` diff render test.

### Cert (deployed image)
- Ledger shows 140 and 141; `expert_reviews` reads 0 pending, 37 superseded, 2 rejected, 1 approved;
  `expert_review_versions` count equals the pre-migration review count.
- An eleven-pair discovery on the deployed image mints nothing (all PROCEED); the pending count stays 0.
- The detail endpoint for a superseded Remibrutinib `treatment_arm → persistent_180d` review shows its
  versions and an estimand history with diffs between the four historical hashes (backfilled as four
  single-version reviews), which exercises the diff path on real data.
- Negative control: on the pre-lane image the same discovery job's BLOCK-band path would mint; the lane's
  unit test proves the mint is gone and the live count proves nothing new appears.

## 8. Sequencing, gates and rollback

- Each lane: worktree → red-first tasks → codex fixed point → one push → CI green (actions runs API by
  full sha) → owner go → merge `--merge` → deploy watch (0 non-terminal runs, container tag == main HEAD,
  content marker) → live cert with a negative control → issue comments → memory → worktree removed.
- Migrations are idempotent and forward-only; lane 1's type widening and lane 4's new column/table need no
  rollback script beyond the standard `rollback_NNN.sql` convention if the owner wants one.
- Lane 3 is the only lane with a large diff; it ships in one PR (owner's choice A) with the OpenAPI
  snapshot as the contract.

## 9. Follow-ups filed, not done here
- Column rename `discovered_dags.gate_decision → discovery_gate_decision` (debt 4a, option B).
- The seven status vocabularies and the third gate's naming.
- Chat brains consolidation.
- `nuisance_config.py` docstring listing the out-of-scope CausalForestDML sites (from #2031's cert).
