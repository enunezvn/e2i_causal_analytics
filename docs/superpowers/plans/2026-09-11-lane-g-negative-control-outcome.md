# Lane G (#2007) — negative-control-outcome refutation test: task plan

**Written:** 2026-09-11, on branch `claude/2007-negative-control-outcome` (worktree `.worktrees/lane-g-2007`,
based on Lane E's tip `5c8731031` because G edits the same runner and node; the PR is opened after
#2017 merges so its diff is G's alone).
**Source:** `.claude/plans/2026-09-09-lane1-followups-task-list.md` §"Lane G" + issue #2007.
**Gate passed:** the disproof (`docs/demos/results/2026-09-11_negative_control_disproof/`, commit
`fa336bdc0`) — omitting an arm's declared confounders moves 3 of 9 structural-null outcomes out of their
CI (copay_support → treatment_initiated +0.017 → +0.052; psp_enrolled → treatment_initiated +0.005 →
+0.090; rep_detailing_high → persistent_180d +0.031 → +0.058); 0 adjusted false positives; 11/11
planted truths detected. `sample_dropped` and `trigger_accepted` have NO responding control at n = 1500.

## Design decisions (owner-visible; each traces to a measurement or an existing seam)

1. **Registry declares only measured responders.** `_CAUSAL_NEGATIVE_CONTROL_OUTCOMES = {"patient_journeys":
   {"copay_support": "treatment_initiated", "psp_enrolled": "treatment_initiated", "rep_detailing_high":
   "persistent_180d"}}`. A control that cannot move under confounding on the generator (sample_dropped,
   trigger_accepted) would PASS under confounding — false assurance — so it is NOT declared; the
   docstring says so with the numbers and requires re-verification per data source (Optum / CSU).
2. **Reading, not a gate, for the first live period.** `DEFAULT_CONFIG["negative_control_outcome"] =
   {"enabled": True, "critical": False}`; explicit weight `0.0` in `_calculate_confidence` (the dict
   defaults an unlisted test to 0.1 — that default must not apply). Promotion is an owner decision after
   live counts.
3. **Same fit as the estimate.** The NC effect comes from the node's DoWhy reconstruction with
   `outcome=nc_outcome` (same `common_causes`, same resolved method) — the model-build half of
   `_reconstruct_dowhy_artifacts` split out WITHOUT the reported-vs-reconstructed tolerance guard (there is
   no reported NC ATE to compare against). CI from DoWhy's `estimate.get_confidence_intervals()`; if the
   backend cannot give one, the test is an honest SKIPPED (`negative_control_ci_unavailable`), never a
   fabricated interval.
4. **The frame must carry the NC column.** `_load_agent_estimation_frame` selects only
   `[treatment, outcome, *covariates]` (routes/causal.py:3069) — the NC outcome is fetched as an extra
   PASSTHROUGH column that is never a covariate/adjustment member and whose NULLs never drop rows from the
   primary frame (the NC fit drops its own NaNs and records `nc_n`). The route sets the agent state key
   `negative_control_outcome` (pattern: `randomized_design` at routes/causal.py:3412 from
   `_is_randomized_treatment`), declared in `CausalImpactState` (LangGraph drops undeclared keys).
5. **Rule** (runner, non-critical): PASSED if the NC CI includes 0; WARNING if it excludes 0 and
   |nc_effect| < |original_effect|; FAILED if it excludes 0 and |nc_effect| ≥ |original_effect| ("the
   negative control moved at least as much as the claimed effect"). `refuted_effect = nc_effect`;
   `delta_percent = 100·|nc_effect| / |original_effect|` (0.0 when the original is 0 — it IS the FAILED
   ratio, so the DB column carries the verdict's number). `details`: `nc_outcome`, `nc_effect`, `nc_ci`,
   `nc_n`, `rule: "negative_control_ci_vs_zero"`, `weight: 0.0`, `reading` (one sentence for the
   narrative), `message` (the legacy `details` string the interpretation node prints).
6. **SKIPPED reasons** (spec §12 decision-2 vocabulary, persisted via `skipped_tests`):
   `no_negative_control_declared` (no registry entry / key absent — every non-agent path, treatment_arm,
   hcp_adoption, nba_triggers), `negative_control_column_missing` (declared but not in the frame),
   `negative_control_ci_unavailable`, `negative_control_too_few_rows` (< MIN rows after dropna; pin the
   constant), plus the budget skip the runner already applies.
7. **DB:** `database/migrations/138_refutation_test_type_negative_control.sql` — `ALTER TYPE
   refutation_test_type ADD VALUE IF NOT EXISTS 'negative_control_outcome';` on the 071 pattern (the
   runner detects ADD VALUE and applies it un-wrapped; no consumer statement in the file). The enum lives
   in `database/ml/010_causal_validation_tables.sql`; `save_suite` inserts ALL rows in one call
   (repositories/causal_validation.py:144), so a missing enum value would drop the whole suite's
   persistence — the migration rides the same deploy, applied before the app starts.
8. **Surfaces:** state `RefutationTest.test_name` Literal; frontend `RefutationTestId` union + a sixth
   `REFUTATION_TESTS` entry (content.test.ts pins "five … exactly two critical" → six / two; REFUTATION_INTRO,
   RefutationGate.tsx and Documentation.tsx say "five"); viz `RefutationMethod` union + label/description
   in `visualizations/causal/RefutationTests.tsx` (+ its test) and `REFUTATION_METHOD_MAP` in
   `CausalAnalysisDetail.tsx` (an unmapped name falls back to "Random Common Cause" — must not);
   `types/causal.ts:226` comment; lineage `docs/lineage/causal_dag_lineage.html` §4.2 row + "five refuters"
   wording. `config/agent_config.yaml`'s `refutation_tests` list is consumed only by the GEPA metric
   (`causal_impact_metric.py:125`, a pass-rate over LISTED tests) — leave the list alone, add a comment
   naming the reading (adding it would change GEPA scoring, an owner call).

## Standing rules for every task (memory `feedback_lane_execution_protocol_20260909`)

ONE implementer subagent at a time in `.worktrees/lane-g-2007`; every command starts with
`cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-g-2007 &&` and every python/pytest with
`python -c 'import src; assert "lane-g-2007" in src.__file__'`; venv `/home/enunez/Projects/e2i_causal_analytics/.venv`;
NEW COMMITS ONLY; red-first; `pytest -n 0 -p no:cacheprovider --timeout 600 <files>`; no mocks of production
behaviour; codex (read-only) to `VERDICT: ACCEPT` per task with the CLAUDE.md pushback paragraph; whole diff
again before the PR; `free -m` before heavy steps.

## Tasks (in order)

- [ ] **T1 runner.** `RefutationTestType.NEGATIVE_CONTROL_OUTCOME`, DEFAULT_CONFIG entry, weight 0.0,
      `_run_negative_control_test(original_effect, negative_control)` with decision 5/6, `run_all_tests(...,
      negative_control: Optional[Tuple[str, float, Tuple[float, float], int]] = None)` dispatching it
      (budget-skip like the others, SKIPPED with reason when None), `to_legacy_format` passthrough.
      Tests `tests/unit/test_causal_engine/test_refutation_runner_negative_control_2007.py`: three bands +
      each SKIPPED reason; details keys; confidence score and gate IDENTICAL with and without the NC row
      (weight 0); `test_refutation_bands_enumeration.py` reachable-band pins extended.
- [ ] **T2 migration.** File 138 + `tests/unit/test_database/test_migration_138_negative_control.py`
      (136/071 pattern: exact statement, no BEGIN/COMMIT, no UPDATE/consumer, caveat comment present).
- [ ] **T3 route + state.** Registry dict + `_negative_control_outcome(dataset, treatment_var, outcome_var)`
      (None when undeclared or equal to the primary outcome); loader `passthrough_columns`; initial_state
      key; `CausalImpactState.negative_control_outcome: NotRequired[Optional[str]]`. Tests under
      `tests/unit/test_api/test_causal_agent_analyze_negative_control_2007.py` (pattern
      `test_causal_randomized_flag.py`): mapped treatment → key + column fetched; sample_dropped → None;
      NC NULLs do not drop primary rows; the NC column is not in `confounders`/`modeled_confounders`.
- [ ] **T4 node.** Split the model-build out of `_reconstruct_dowhy_artifacts`; NC fit under the compute
      budget on `refutation_data.dropna(subset=[nc])`; measure `get_confidence_intervals()` cost and
      availability per resolved method on the seed-21 frame (record numbers in the commit body); pass the
      tuple to `run_all_tests`; key absent → the runner's SKIPPED. Tests
      `tests/unit/test_agents/test_causal_impact/test_refutation_negative_control_2007.py` (run ONLY this
      file, `--timeout 600`): reconstruction called with the NC outcome and the same common causes; tuple
      reaches the runner; persisted details carry the keys; absent key → SKIPPED reason.
- [ ] **T5 calibration pin** (heavy_ml, `tests/unit/test_causal_engine/test_negative_control_calibration_2007.py`,
      reuse `_fit`/frame/`NULL_PAIRS`; `pytest.mark.timeout` ≤ 600 — the heavy lane's stall window is 1200):
      the 11 planted truths with their registry NC outcome → PASSED ≥ 10/11 through the runner rule; the three
      disproof responders reproduce their recorded omitted-fit movement (CI excludes 0); the two undeclared
      arms are asserted undeclared for the measured reason.
- [ ] **T6 surfaces.** Decision 8 in full; `content.test.ts`, viz `RefutationTests.test.tsx` green;
      lineage row; YAML comment.
- [ ] **Close-out.** Whole-diff codex ACCEPT; ruff on changed files; targeted pytest `-n 0`; frontend tests;
      `docs/demos/results/2026-09-11_negative_control_disproof/` already committed; issue comment with the
      task table; PR after #2017 merges (base main); cert after deploy per the lane block.
