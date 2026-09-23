# split-contract lane — final report (2026-09-23)

PR **#2241** https://github.com/enunezvn/e2i_causal_analytics/pull/2241 — branch `claude/split-contract`, base `origin/main f1bb9e36c`, **final head `b1035af03`**. Not merged (by design). `closingIssuesReferences` = 0 (GraphQL-verified; body says "Part of #2207"). Follow-up issue filed: **#2242**.

## Commits (oldest first)

```
4d8d129d2 feat(data_preparer): table-route split contract honours data_split (Part of #2207)
e011e5b7d docs(retrain): describe the table cohort dict in the contract docstrings (Part of #2207)
9bd4742c3 feat(db): migration 151 seeds the goldstd cohort contracts - only what is provable (Part of #2207)
8ac67fd88 test(retrain): a table cohort dict passes through _cohort_input_from_training_config verbatim (Part of #2207)
1922df60d docs(data_preparer): the contract's columns list is the real leakage guard (Part of #2207)
317281e44 chore(types): regenerate OpenAPI types for the table cohort description (Part of #2207)
fd9f51dbf fix(data_preparer): the data_split probe tells a real 42703 from every other failure (Part of #2207)
e03a992fa fix(data_preparer): GE validation accepts a table cohort dict and filters the suite to its projection (Part of #2207)
aa1661316 fix(feature_store): the freshness probe resolves a table cohort dict to its table (Part of #2207)
2e1423e32 fix(api): keep monitoring.py at its module-size pin; regenerate the types line (Part of #2207)
f1c494021 feat(db): migration 151 also seeds cohort_feature_manifest_source='synthetic_csu' on the 9 patient rows (Part of #2207)
a86c1825b test(data_preparer): a table contract + its DGP manifest clears Layer 3; the control flags the driver (Part of #2207)
9b6a32a58 test(db): migration 151 teeth also pin resolve_manifest_source(dict, value) (Part of #2207)
99b12d7b3 fix(data_preparer): serialised dedupe key for contract suites; strict 42703 classifier (Part of #2207)
754854031 fix(retrain): derive brand from a table cohort contract's filters (Part of #2207)
b1035af03 fix(retrain): gate contract brand derivation on type=table and a dict filters (Part of #2207)
```

## CI — final head b1035af03 (runs API, all workflows)

```
35814319148 RPC vs DDL Column Guard completed success
35814319104 Lifecycle State Guard completed success
35814319110 Feast Apply & Idempotency completed success
35814319166 Tier 1-5 Agent Harness completed success
35814319212 Verify OpenAPI Types completed success
35814319192 Frontend Tests completed success
35814319128 Security Scanning completed success
35814319245 Backend Tests completed success
```
All workflows `success` (lead-verified independently: Backend Tests 35814319245, every job green including Type Check (MyPy)).

Earlier heads (evidence trail): 8ac67fd88 → Backend Tests FAILED on `test_module_size_ratchet` (monitoring.py 2163 > pin 2153) + Verify OpenAPI Types FAILED (api.ts drift); both fixed. 1922df60d → same two failures (superseded). 317281e44 / 2e1423e32 / a86c1825b / 99b12d7b3 → all workflows success (a86c1825b: Backend Tests 35811566740 success; 99b12d7b3: Backend Tests 35811957422 success).

## mypy artifact diff vs baseline (`mypy_main/mypy_report.txt`, 58 errors on 814eaa959)

Final head b1035af03 (Backend Tests 35814319245; lead-verified against main d5d9cc879, saved at `scratchpad/mypy_pr/`): `58 errors` = baseline `58`; **NEW: none; GONE: none.**

Preview on 99b12d7b3's run 35811957422 (differs from the final head only by the two-line brand gate): `Found 58 errors in 42 files (checked 1095 source files)`; NEW in head: none; GONE: none.

## Codex verdicts

| round | verdict | findings → closing commit |
|---|---|---|
| r1 | FINDINGS (1 HIGH, 1 MED, 0 LOW) | HIGH `ge_validator` rejected table dicts → e03a992fa; MED probe swallowed errors → fd9f51dbf (`MLDataLoader.has_column`) |
| r2 | FINDINGS (0 HIGH, 1 MED, 2 LOW); r1 HIGH closed, freshness correct, no other false greens | MED unhashable dedupe key → 99b12d7b3; LOW bare 42703 substring → 99b12d7b3; LOW zero-expectations test false green → 99b12d7b3 |
| r3 | FINDINGS (1 HIGH, 0 MED, 0 LOW); r2 all CLOSED; migration-151 manifest audit clean | HIGH sweep contract carries no brand / model identity → brand half 754854031 (derive from contract filters); registry-linkage half pre-existing → **#2242** |
| r4 | FINDINGS (0 HIGH, 1 MED, 0 LOW); confirms the migration-151 path correct end to end | MED derivation ran for any dict / assumed `.get()` on filters → **b1035af03** (gated on `type=="table"` + dict filters; 2 tests) |

No r5 (lead decision; the brief's cap was 3 rounds). Briefs/outputs: `split_contract_codex_r{1..4}.{md,out}`.

## Probes (all READ-ONLY, real PostgREST with the repo .env service key, worktree code)

**E — Kisqali initiation contract → `load_data` → split_loader math → real `enforce_splits` → schema/quality/leakage nodes** (`split_contract_probe.out`):
```
counts = {'train': 5294, 'validation': 1791, 'test': 869, 'holdout': 884} total = 8838
ratios = {'train': 0.599, 'validation': 0.2026, 'test': 0.0983, 'holdout': 0.1}
split_ratios_valid = True
split_ratio_checks = ['Train split: 59.90% ✓', 'Validation split: 20.26% ✓', 'Test split: 9.83% ✓', 'Holdout split: 10.00% ✓']
qc_status = passed overall_score = 1.0 blocking_issues = []
leakage_detected = False severity = none
CRITICAL/HIGH findings = 0
PROBE VERDICT: PASS
```
Persistence / discontinuation (Kisqali): identical counts 5294/1791/869/884, `split_ratios_valid=True`, zero leakage findings, PASS (`split_contract_probe_other_cohorts.out`). All nine contracts through GE: 39/39 (initiation) or 36/36 on the contract suite, gate inputs `blocking_issues=[] qc_status=passed score=1.0 leakage_severity=none`; Fabhalta 5367/1834/906/918, Remibrutinib 5418/1727/901/874 (`split_contract_probe_ge_after.out`).

**Full data_preparer graph walk WITHOUT a manifest** (`split_contract_probe_graph.out`): every node through `detect_leakage` clean; then Layer 3 escalates and routes to LLM remediation on all three cohorts:
```
[adaptive_validity_check 14.6s] -> {"leakage_detected": true, "leakage_severity": "high", "leaked_features": ["age_at_diagnosis", "disease_severity"], "blocking_issues": []}
[route_after_leakage_detection] -> remediate
leakage_remediation would run (LLM) — NOT executed by this probe; FAIL
[adaptive_validity_check 5.8s] -> {"leakage_detected": true, "leakage_severity": "high", "leaked_features": ["disease_severity"], "blocking_issues": []}
[route_after_leakage_detection] -> remediate
leakage_remediation would run (LLM) — NOT executed by this probe; FAIL
[adaptive_validity_check 6.0s] -> {"leakage_detected": true, "leakage_severity": "high", "leaked_features": ["disease_severity"], "blocking_issues": []}
[route_after_leakage_detection] -> remediate
leakage_remediation would run (LLM) — NOT executed by this probe; FAIL
SUMMARY: {'initiation': False, 'persistence': False, 'discontinuation': False} | SOME FAIL
```
(initiation: `disease_severity` z=40.14σ AUC 0.713, `age_at_diagnosis` z=41.10σ AUC 0.703 → high/drop; persistence/discontinuation: `disease_severity` z=25.96σ AUC 0.625 → high/drop; all other covariates info/keep — `split_contract_probe_adaptive.out`.)

**Disproof WITH `feature_manifest_source="synthetic_csu"`** (`split_contract_probe_manifest.out`; resolved via the real `resolve_manifest_source(dict, "synthetic_csu")`):
```
=== initiation / Kisqali === resolve_manifest_source(dict, 'synthetic_csu') -> 'synthetic_csu'
WARNING src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check: Declared-safe immunity: exempted 5 manifest pre-index feature(s) from leakage (contract is authoritative); severity none -> info. Features: ['
  feature=rep_detailing_high     severity=info      z=8.83     remediation=keep       decided_by=adversarial layer=3 declared_safe=None
  feature=sample_dropped         severity=info      z=5.32     remediation=keep       decided_by=adversarial layer=3 declared_safe=None
  feature=trigger_accepted       severity=info      z=17.94    remediation=keep       decided_by=adversarial layer=3 declared_safe=None
[route_after_leakage_detection] -> continue
[finalize_output 0.0s] {"gate_passed": true, "qc_passed": true, "is_ready": true, "qc_score": 1.0} blocking=[]
GATE initiation/Kisqali: gate_passed=True qc_status=passed qc_passed=True is_ready=True qc_score=1.0 missing_required_features=[] blocking_issues=[]
=== persistence / Kisqali === resolve_manifest_source(dict, 'synthetic_csu') -> 'synthetic_csu'
WARNING src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check: Declared-safe immunity: exempted 5 manifest pre-index feature(s) from leakage (contract is authoritative); severity none -> info. Features: ['
  feature=copay_support          severity=info      z=2.03     remediation=keep       decided_by=adversarial layer=3 declared_safe=None
  feature=psp_enrolled           severity=info      z=4.76     remediation=keep       decided_by=adversarial layer=3 declared_safe=None
[route_after_leakage_detection] -> continue
[finalize_output 0.0s] {"gate_passed": true, "qc_passed": true, "is_ready": true, "qc_score": 1.0} blocking=[]
GATE persistence/Kisqali: gate_passed=True qc_status=passed qc_passed=True is_ready=True qc_score=1.0 missing_required_features=[] blocking_issues=[]
=== discontinuation / Kisqali === resolve_manifest_source(dict, 'synthetic_csu') -> 'synthetic_csu'
WARNING src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check: Declared-safe immunity: exempted 5 manifest pre-index feature(s) from leakage (contract is authoritative); severity none -> info. Features: ['
  feature=copay_support          severity=info      z=2.03     remediation=keep       decided_by=adversarial layer=3 declared_safe=None
  feature=psp_enrolled           severity=info      z=4.76     remediation=keep       decided_by=adversarial layer=3 declared_safe=None
[route_after_leakage_detection] -> continue
[finalize_output 0.0s] {"gate_passed": true, "qc_passed": true, "is_ready": true, "qc_score": 1.0} blocking=[]
GATE discontinuation/Kisqali: gate_passed=True qc_status=passed qc_passed=True is_ready=True qc_score=1.0 missing_required_features=[] blocking_issues=[]
SUMMARY: {'initiation': True, 'persistence': True, 'discontinuation': True} | ALL PASS
```

## Migration 151 rehearsal (one psql call: BEGIN → 151 → SELECT → 151 again → ROLLBACK; `split_contract_rehearsal2.out`)

`UPDATE 1` × 12, then `UPDATE 0` × 12 (idempotent), `ROLLBACK`. Rows inside the transaction (data_source truncated to 60 chars) and the live re-read afterwards (`non_null_live|total`):
```
csu_treatment_initiation_lr_balanced_v1|||
csu_treatment_initiation_lr_full_v1|||
discontinuation_fabhalta_goldstd_lr_v1|{"columns": ["disease_severity", "academic_hcp", "geographic|discontinued_180d|synthetic_csu
discontinuation_kisqali_goldstd_lr_v1|{"columns": ["disease_severity", "academic_hcp", "geographic|discontinued_180d|synthetic_csu
discontinuation_remibrutinib_goldstd_lr_v1|{"columns": ["disease_severity", "academic_hcp", "geographic|discontinued_180d|synthetic_csu
hcp_adoption_fabhalta_goldstd_lr_v1||adopted|
hcp_adoption_kisqali_goldstd_lr_v1||adopted|
hcp_adoption_remibrutinib_goldstd_lr_v1||adopted|
initiation_fabhalta_goldstd_lr_v1|{"columns": ["disease_severity", "academic_hcp", "geographic|treatment_initiated|synthetic_csu
initiation_kisqali_goldstd_lr_v1|{"columns": ["disease_severity", "academic_hcp", "geographic|treatment_initiated|synthetic_csu
initiation_remibrutinib_goldstd_lr_v1|{"columns": ["disease_severity", "academic_hcp", "geographic|treatment_initiated|synthetic_csu
persistence_fabhalta_goldstd_lr_v1|{"columns": ["disease_severity", "academic_hcp", "geographic|persistent_180d|synthetic_csu
persistence_kisqali_goldstd_lr_v1|{"columns": ["disease_severity", "academic_hcp", "geographic|persistent_180d|synthetic_csu
persistence_remibrutinib_goldstd_lr_v1|{"columns": ["disease_severity", "academic_hcp", "geographic|persistent_180d|synthetic_csu
0|14
```

## Tests

New files (all `-n 0`, hermetic unless noted): `test_data_loader_table_2207_split_contract.py` (13), `test_ge_validator_table_2207_split_contract.py` (10), `test_ml_data_loader_has_column_2207_split_contract.py` (9), `test_mig151_goldstd_cohort_contracts_2207_split_contract.py` (14), `test_feast_source_freshness_table_2207_split_contract.py` (2), `tests/integration/test_adaptive_manifest_immunity_table_contract_2207_split_contract.py` (3; runs the real Layer-3 check on a 2000-row fixture, ~10 s), + 10 tests added to `test_execute_model_retraining_real.py`. Red-first shown for every group (loader 10/12 red, migration 14/14 red, GE 4/6 red, has_column 6/6 red, brand 3 then 2 red). Fixture changes: `test_data_loader_concat_f16.py` gains `loader.has_column = AsyncMock(return_value=False)` (lead-approved).

Local runs: broad pin set 1160 passed / 5 skipped (`split_contract_pins_final.out`); remaining 7.4 unit files 288 passed; `tests/api/test_monitoring_endpoints.py -k retrain`: 2 pre-existing failures (`TestEvaluateRetraining::*`, HTTP 500) identical on the untouched base f1bb9e36c and not in CI's list. Ruff whole-tree `--no-cache` check + format: clean at every commit.

## Files changed

```
.../151_registry_cohort_contracts_goldstd.sql      | 185 +++++++++++
 frontend/src/types/generated/api.ts                |   2 +-
 .../data_preparer/nodes/data_loader.py             | 182 ++++++++++-
 .../data_preparer/nodes/ge_validator.py            | 117 ++++++-
 src/agents/tier_0/split_handoff.py                 |  14 +-
 src/api/routes/monitoring.py                       |  12 +-
 src/feature_store/feast_source_freshness.py        |  15 +-
 src/repositories/ml_data_loader.py                 |  46 +++
 src/services/cohort_contract.py                    |   7 +-
 src/tasks/drift_monitoring_tasks.py                |  19 +-
 ..._immunity_table_contract_2207_split_contract.py | 146 +++++++++
 .../test_data_loader_concat_f16.py                 |   4 +
 .../test_data_loader_table_2207_split_contract.py  | 341 +++++++++++++++++++++
 .../test_ge_validator_table_2207_split_contract.py | 264 ++++++++++++++++
 ...goldstd_cohort_contracts_2207_split_contract.py | 177 +++++++++++
 ...t_source_freshness_table_2207_split_contract.py |  44 +++
 ...l_data_loader_has_column_2207_split_contract.py | 125 ++++++++
 .../test_execute_model_retraining_real.py          | 114 +++++++
 18 files changed, 1788 insertions(+), 26 deletions(-)
```

## Known limits (unchanged, documented in the PR body)

- `run_schema_validation` validates the column-scoped projection against the full `patient_journeys` Pandera schema → 6 `column_in_dataframe` errors, but `run_quality_checks` overwrites `blocking_issues` before the gate (pre-existing, documented in `graph.py`); telemetry-only.
- `will_adopt`: the scope_definer rewrites any target containing "adopt" → HCP rows keep `cohort_data_source` NULL (the new target guard would refuse them).
- **#2242**: retrain candidates are registered from a generated experiment id, not attached to the model being retrained (pre-existing; this PR fixes only the brand half of the scope name).
- A table WITHOUT `data_split` still takes the temporal split and still fails the enforcer.

## Could not do / did not do

- Did not run the LLM `leakage_remediation` node, `register_features_in_feast`, `kg_role_enrichment`, `qc_remediation` or the memory hooks (writers / paid LLM) — the graph walk stops at the routing decision or skips them, as agreed.
- No mypy on the box (rule); CI's artifact is the arbiter (diff above).
- Did not merge, deploy, or run the manual retrain trigger (owner action per the brief).
