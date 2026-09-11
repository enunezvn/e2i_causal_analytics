# Sensitivity gate calibration: the E-value as a benchmarked reading, not a cutoff

**Date:** 2026-09-10. **Lane:** Wave 2, lane D′ (#1991 debt 2, first slice). **Status:** design approved in
brainstorm by the owner on 2026-09-10; awaiting owner review of this written spec before the plan.
**Closes:** #1988 (option 3), #1994 (option 1), #1989 (docstring pin). **Updates:** #1991.
**Evidence:** memory `sensitivity_evalue_gate_is_effect_size_gate_20260909`, the disproof scripts in this
session's scratchpad (`evalue_benchmark.py`, `evalue_dml_calib.py`, `evalue_reading_dgp.py`,
`live_reband_preview.py`), the issue comments on #1988 and #1991 dated 2026-09-09.

## 0. The principle this lane serves

Leaders make commercial decisions on what this platform reports. Every verdict the refutation gate
emits must therefore mean something a leader can act on, in words a leader can read, with the number
behind it visible. A gate that blocks correct answers teaches leaders to distrust the tool; a gate that
labels most answers "warning" teaches them to ignore it. Both have been true of the sensitivity test
since it was written. This lane replaces it with a reading that is right on planted truth, rare in its
caveats, and explicit about what each caveat means.

## 1. Problem

`RefutationRunner._run_sensitivity_test` (`src/causal_engine/refutation_runner.py`) converts the reported
effect and its CI bound to an approximate risk ratio, computes the VanderWeele–Ding E-value on each, and
scores the CI-bound E-value against fixed cutoffs: PASSED at 2.0, WARNING at 1.5, FAILED below. The test is
`critical`, so FAILED means BLOCK: the estimate is withheld and an expert review is queued. Both cutoffs
date from the first RefutationRunner commit (`0742b81f6`, 2025-12-19) and were never calibrated.

Measured 2026-09-09 on the live table (124 agent runs, `estimate_source = causal_impact_query`): 65 BLOCK,
every one of them on this test alone (placebo passed 231/231; random common cause passed 225/231); of the
59 PROCEED runs, 42 carry a sensitivity WARNING. On a binary outcome (SD ≈ 0.48) the cutoffs are effect-size
bands: a CI-lower-bound effect under ~6 pp FAILS, 6–15 pp WARNS, above 15 pp PASSES. The DGP's planted true
effects fall in those bands by construction, and the live BLOCKs are exactly the pairs whose planted truth
is small.

## 2. Measured findings that fix the design

All fits below use LinearDML with the production RandomForest nuisances (the estimator family the agents
and the recovery probe use), seed 42, on the seed-21 Remibrutinib patient frame, at the live row cap
n = 1,500 and at n = 5,000. Peak RSS 468 MiB, 66 s for all 33 fits at n = 5,000.

### 2.1 The estimator is right; the gate is wrong

At n = 1,500, correctly adjusted on the arm's declared confounders (`ARM_REGISTRY`), every one of the 11
planted true effects was recovered within 0.03 of truth and every 95 % CI covered it. Today's rule then
scores them:

| rule applied to 11 correctly recovered truths at n = 1,500 | PASSED | WARNING | FAILED (= BLOCK) |
|---|---|---|---|
| today: E-value(CI bound) vs 2.0 / 1.5, SMD conversion | 2 | 2 | 7 |
| today's cutoffs with the binary risk-ratio conversion | 2 | 5 | 4 |

The smallest truth (`sample_dropped → treatment_initiated`, 4 pp) has E-value(CI) 1.13. No cutoff above
1.10 passes all truths; a cutoff at or below 1.10 is equivalent to "the CI excludes zero", which the
null-crossing guard already enforces.

### 2.2 The E-value cannot detect confounding, so it cannot be a confounding gate

Refitting each true pair with its strongest declared confounder omitted produces estimates biased in
either direction; their E-values are indistinguishable from the correctly specified fits and are often
higher (`trigger_accepted → treatment_initiated` at n = 5,000: correct 1.46, confounder omitted 1.71). The
E-value is monotone in effect size divided by uncertainty; bias that inflates the estimate inflates its
E-value. It answers "how strong would a confounder have to be", never "is there one". The tests that do
detect a mis-specified estimator are the adjustment-set contract (`ARM_CONFOUNDERS`,
`tests/unit/test_synthetic/test_arm_confounder_contract.py`), placebo, random common cause, and the FCI
latent-confounder diagnostic. They are unchanged by this lane and stay critical where they are critical.

### 2.3 Spurious associations are caught by the interval, not by the cutoff

Each arm fitted against an outcome it does not move, correctly adjusted: 10 of 12 fits (both sample sizes)
have a CI that includes zero. The remaining two are a chance positive at n = 1,500
(`sample_dropped → adherent_180d`, gone at n = 5,000) and `treatment_arm → persistent_180d`, which is not
spurious: persistence is built from the same adherence machinery the arm moves.

### 2.4 Measured confounding is small and measurable per run

The confounding the adjustment removed, expressed on the risk-ratio scale as the ratio of the naive
(unadjusted) risk ratio to the adjusted one, ranges 1.00–1.33 on the DGP and 1.01–1.31 on the live runs.
The estimation node already computes the naive contrast (`naive_ate`, `confounding_bias_removed`). This
is the natural per-run benchmark: an unmeasured confounder would have to carry more confounding than the
whole measured set did to explain the effect away.

### 2.5 Benchmark the point estimate; state precision separately

Benchmarking the CI bound against measured confounding re-creates the defect for small, imprecise effects:
4 of 11 truths read WARNING at n = 1,500. Benchmarking the point estimate reads all 11 as robust at both
sample sizes, while the interval carries the precision statement on its own:

| case at n = 1,500 (LinearDML) | beyond measured confounding | within | null finding |
|---|---|---|---|
| 11 planted truths, correctly adjusted | 11 | 0 | 0 |
| 11 planted truths, a confounder omitted | 10 | 0 | 1 |
| 6 null pairs, correctly adjusted | 1 (the chance positive) | 0 | 5 |

Identical at n = 5,000 except the chance positive disappears and `treatment_arm → persistent_180d` reads
beyond (a real indirect effect).

### 2.6 Re-band preview of the 124 live runs

Stored per-test statuses and stored E-value inputs, with baseline risk and the naive contrast computed from
frames pulled the way the route pulls them (same table, brand filter, row cap; `E2I_INCLUDE_SYNTHETIC=true`
on the live container, so no provenance filter):

| reading | runs |
|---|---|
| beyond measured confounding (PASSED) | 91 |
| within measured confounding (WARNING) | 6 |
| null finding (WARNING) | 5 |
| randomized design, not applicable (SKIPPED) | 6 |
| continuous treatment (benchmarked in §4.3, measured in §7) | 13 |
| unmapped legacy pair `accepted → converted` (Kisqali, 3 rows) | 3 |

| gate move | runs |
|---|---|
| BLOCK → PROCEED | 56 |
| PROCEED → PROCEED | 46 |
| BLOCK → BLOCK | 6 (all on a random-common-cause FAILED; 4 are also null findings) |

Nothing stays blocked on the E-value. The six WARNINGs sit on one question family
(`acceptance_status → conversion_flag`, three brands), where the adjustment removed more than the
remaining effect — a caveat a decision maker should see. Five percent of runs carry a WARNING, against
the 34 percent that would carry one under #1988 option 2.

### 2.7 Conversion defect

The runner converts a risk difference on a 0/1 outcome with the continuous-outcome approximation
`RR ≈ exp(0.91·d)` (Chinn 2000 / VanderWeele 2017, meant for a standardized mean difference). VanderWeele
and Ding give a separate path for a risk difference on a binary outcome that uses the control-arm
baseline risk. Every outcome the platform serves is binary. The agent's sensitivity node uses a third
variant (`exp(d)`, no 0.91). Three engines, two formulas, one number reported to leaders.

## 3. Owner decisions (2026-09-10, brainstorm)

1. **Sensitivity stops being a gate.** It is non-critical and has no FAILED outcome. Placebo and random
   common cause remain the critical tests.
2. **A CI that includes zero is served as a null finding**, with an explicit caveat in the record and the
   drill-down. It is never blocked and never queued for expert review.
3. **All four E-value surfaces move together**: the runner, the agent's sensitivity node, the
   interpretation node, and the chat tool `sensitivity_analyzer`, plus the documentation page.
4. **The calibration must not produce a warning-heavy population.** A correctly recovered planted truth
   reads as robust; WARNING is reserved for a concrete, actionable caveat and is expected to be rare.
5. **Leader confidence is paramount.** Verdict words lead; numbers follow; every caveat says what it means.

## 4. Design

### 4.1 One E-value module

New `src/causal_engine/evalue.py` owns the math the three engines currently copy. Pure functions, no I/O,
numpy only. Every function is total on its documented domain and raises `ValueError` outside it.

| function | contract |
|---|---|
| `e_value_from_rr(rr)` | `E = RR + sqrt(RR·(RR−1))` for RR ≥ 1; a protective RR < 1 is inverted first; RR = 1 → 1.0. |
| `rr_from_smd(d)` | `exp(0.91·|d|)`. Continuous outcomes, or any case without a baseline risk. |
| `rr_from_risk_difference(rd, baseline_risk)` | `p1 = p0 + rd`; RR = `p1/p0` oriented ≥ 1 (a negative RD reverses the exposure coding, per the EValue package). Requires `0 < p0 < 1` and `0 < p1 < 1`; returns `None` otherwise. |
| `bias_factor(rr_eu, rr_ud)` | Ding & VanderWeele 2016 joint bounding factor `B = RR_EU·RR_UD / (RR_EU + RR_UD − 1)`, inputs oriented ≥ 1. |
| `joint_confounding_benchmark(naive, adjusted, baseline_risk, outcome_std)` | `B_obs = RR(naive) / RR(adjusted)` oriented ≥ 1, each RR via the risk-difference path when a baseline risk exists, else the SMD path. `None` when `naive` is `None`. |
| `covariate_bias_factors(frame, treatment, outcome, covariates)` | Per covariate: binary covariates as-is, continuous ones split at the median; `RR_EU` = share of high-covariate units among treated ÷ among controls (treated = `T == 1`, or `T > median(T)` for a continuous treatment); `RR_UD` = outcome rate among high-covariate ÷ low-covariate **controls** (SMD path on the mean difference for a continuous outcome). Each turned into a bias factor. A categorical covariate is scored level by level (each level a 0/1 indicator) and keeps the max under its own name. A categorical level with fewer than 5 rows in either arm (`MIN_CELL_SIZE`) is skipped as a sparse cell. Empty dict for an empty covariate list. |
| `measured_confounding_benchmark(joint, covariate_factors)` | Returns `(value, basis)`: the joint benchmark when available (`basis = joint_naive_vs_adjusted`), else the largest covariate factor (`strongest_covariate`), else `None` (`none_measured`). |
| `classify(effect, ci, *, randomized, baseline_risk, outcome_std, naive_effect, covariate_factors, n_rows)` | Returns a `SensitivityReading` (§4.4). |

The joint benchmark is preferred over the covariate factors because it is estimand-consistent: it is the
confounding the estimator actually removed for this treatment and outcome. The covariate factors are
always computed and reported, so a reader can see which measured covariate carried the most confounding.
On the DGP the strongest covariate factor sometimes exceeds the joint benchmark (`trigger_accepted`,
disease severity 1.23 vs joint 1.15) and would flip a planted truth to WARNING; that is why it is the
fallback, not the maximum.

### 4.2 Conversion by outcome and treatment type

| treatment | outcome | risk ratio for the E-value | benchmark |
|---|---|---|---|
| binary 0/1 | binary 0/1 | risk-difference path with the control-arm baseline risk on the full estimation frame | joint (naive vs adjusted) |
| binary 0/1 | continuous | SMD path with the outcome SD | joint, both RRs via the SMD path |
| continuous | any | SMD path | strongest covariate factor (no naive contrast exists; the estimation node's `naive_ate` is `None` by design for non-binary treatment) |
| randomized design (dataset spec) | any | computed for information | none; SKIPPED as today |

Binary is `set(unique(values)) == {0, 1}` after coercion, mirroring `_compute_naive_contrast`.

### 4.3 The benchmark inputs and where they are computed

The refutation node (`src/agents/causal_impact/nodes/refutation.py`) computes, on the **full**
`estimation_data` frame before any subsampling: the control-arm baseline risk (binary treatment only),
the naive effect (read from `estimation_result["naive_ate"]`; recomputed from the frame only when that is
missing), and the covariate bias factors for `estimation_result["covariates_adjusted"]` (the backdoor set
the estimator used; `baseline_covariates_adjusted` are efficiency controls and are excluded). It passes
them to `RefutationRunner.run_all_tests` as new optional keyword arguments `baseline_risk`,
`naive_effect`, `covariate_bias_factors`, exactly as `outcome_std` is threaded today (#1419: the runner's
`data` may be the refutation subsample, and a benchmark must describe the frame the reported effect came
from). When a caller passes none of them and `data`/`treatment`/`outcome` are present, the runner computes
them from `data` itself using the DoWhy model's common causes; caller-supplied values win. The agent's
sensitivity node computes the same three inputs from the same state with the same helper, so the two
engines cannot disagree on a run.

### 4.4 Readings, statuses, and the words a leader sees

`classify` returns a `SensitivityReading` dataclass: `reading`, `status`, `headline`, `message`,
`e_value_point`, `e_value_ci`, `rr_point`, `rr_ci`, `ci_includes_null`, `conversion`
(`risk_ratio` | `standardized_difference`), `baseline_risk`, `naive_effect`, `benchmark`,
`benchmark_basis`, `covariate_bias_factors`, `n_rows`, `covariates_measured`. Evaluated in this order:

| # | condition | reading | status | headline |
|---|---|---|---|---|
| 1 | `randomized` | `not_applicable_randomized` | SKIPPED | Not applicable: randomized design |
| 2 | `ci_lo ≤ 0 ≤ ci_hi` | `null_finding` | WARNING | No detectable effect at this sample size |
| 3a | no benchmark, basis `none_measured` (no covariate measured on the frame, no naive contrast) | `unbenchmarked` | WARNING | Robustness not benchmarked: no measured confounders |
| 3b | no benchmark, basis `measured_unscoreable` (≥ 1 covariate measured on the frame, every one skipped for want of a usable contrast, no naive contrast) | `unbenchmarked` | WARNING | Robustness not benchmarked: measured confounders could not be scored |
| 4 | `rr_point > benchmark` | `beyond_measured_confounding` | PASSED | Robust to confounding at measured strength |
| 5 | otherwise (ties included) | `within_measured_confounding` | WARNING | Sensitive to confounding |

Message templates (one sentence each, numbers to two decimals, the verdict first):

- **beyond**: "Explaining this effect away would need an unmeasured confounder with a risk ratio of at
  least {e_value_point} with both treatment and outcome ({e_value_ci} at the CI bound), stronger than all
  measured confounding combined ({benchmark}, {basis in words})."
- **within**: "A confounder no stronger than the measured set ({benchmark}, {basis in words}) could account
  for the whole effect (risk ratio at the estimate {rr_point}; E-value {e_value_point}). Treat the direction
  as more reliable than the size."
- **null_finding**: "The 95 % CI [{lo}, {hi}] includes zero at n = {n_rows}. The estimate is reported as a
  null finding; no unmeasured confounder is needed to explain it."
- **unbenchmarked / `none_measured`**: "The interval excludes zero (E-value {e_value_point}), but no measured
  confounders exist for this design, so robustness to confounding cannot be benchmarked."
- **unbenchmarked / `measured_unscoreable`**: "The interval excludes zero (E-value {e_value_point}), but none of
  the {covariates_measured} measured confounder(s) could be scored on this frame (no usable contrast — for
  example a covariate collinear with the treatment), so robustness to confounding cannot be benchmarked."
- **not_applicable_randomized**: today's text, unchanged.

`unbenchmarked` is one reading with two sub-cases, told apart by `benchmark_basis`, the headline and the
message only (reading name, status and weight are identical):

- `none_measured`: a non-randomized run with no covariate measured on the frame and no naive contrast.
- `measured_unscoreable`: at least one covariate WAS measured on the frame (`BenchmarkInputs.covariates_measured
  > 0`, the requested covariates present in the frame) but every one was skipped by `covariate_bias_factors`
  (no usable contrast), and there is no naive contrast. The confounder was measured and adjusted for; it simply
  cannot benchmark the effect. Saying "no measured confounders exist for this design" there is false, and every
  caveat must say what it means.

Measured on the live population (re-band, 2026-09-10, `docs/demos/results/2026-09-10_sensitivity_calibration/reband.md`):
6 live runs read `unbenchmarked`, all of them `peer_influence_score → adopted` (dataset `hcp_adoption`,
continuous treatment, the single declared covariate `centrality_z`). On the live frame
corr(peer_influence_score, centrality_z) = 0.9995 for both brands, so the covariate's median split coincides
with the treatment's, no control sits in its high stratum, and the factor is skipped. All 6 are the
`measured_unscoreable` sub-case; the population has zero `none_measured` runs. `covariates_measured` is
threaded exactly like `n_rows` (agent nodes pass the full-frame count; the runner falls back to the count its
own benchmark-inputs branch measured, else 0) and is persisted in `details` with the other reading fields.
The chat tool `sensitivity_analyzer` has no frame and stays on `none_measured` with its own text. The CI test
in §6 pins both sub-cases so neither can silently absorb a benchmarked case.

Status scores and weights are unchanged (PASSED 1.0, WARNING 0.6, SKIPPED excluded; sensitivity weight
0.25). Only criticality and the labels change. The confidence score and band arithmetic are debt 2's
second slice and are out of scope here beyond the enumeration test.

### 4.5 Runner changes (`src/causal_engine/refutation_runner.py`)

- `DEFAULT_CONFIG["sensitivity_e_value"]` becomes `{"enabled": True, "critical": False}`. The
  `e_value_threshold` key and `PASS_THRESHOLDS["e_value_min"]` are deleted, not kept at a new value.
- `_determine_gate_decision` derives the critical set from the config's `critical` flags instead of the
  hardcoded three-member set, so config and behaviour cannot drift apart again.
- `_run_sensitivity_test` delegates to `evalue.classify` and writes the full reading into
  `RefutationResult.details` (all `SensitivityReading` fields, plus the existing `standardized`,
  `outcome_std`, `gate_applicable`). `message` is the leader-facing sentence; `headline` is new.
- The `original_ci` parameter's docstring gains the #1989 pin: the reference interval is the **reported**
  interval from the estimation node; a reference interval is never derived from the reconstruction
  (its SE measured 4.9 against 0.034 reported).
- `PASS_THRESHOLDS["placebo_p_value"]` comment and the lineage threshold table say what the code does
  (PASSED at p ≥ 0.05, FAILED below, no WARNING band); the code is unchanged (#1994 option 1).
- `config/agent_config.yaml` and `tests/unit/test_causal_engine/test_lineage_residue_1975_1979.py` move
  with the runner: the YAML entry describes the benchmark rule and the residue test pins the absence of
  `e_value_threshold` from both files, replacing today's pin of the hardcoded 2.0.

### 4.6 Agent nodes

**Refutation node.** Computes the benchmark inputs (§4.3), passes them to the runner, and when the
sensitivity reading is `null_finding` appends its message once to the run's `warnings` accumulator, so the
API record and the drill-down show it through the existing warnings path.

**Sensitivity node** (`nodes/sensitivity.py`). Drops its private `_calculate_e_value` and the 2.0 / 1.5 /
3.0 verdicts; calls `evalue.classify` with the same inputs. Writes `sensitivity_analysis` with the existing
keys plus the reading: `e_value`, `e_value_ci`, `interpretation` (= message), `robust_to_confounding`
(= reading is `beyond_measured_confounding`), `unmeasured_confounder_strength` (kept for consumers; its
value becomes the reading name), and new `reading`, `headline`, `rr_point`, `rr_ci`, `benchmark`,
`benchmark_basis`, `conversion`. The `SensitivityAnalysis` TypedDict in `state.py` declares the new keys
(LangGraph drops undeclared keys). The point E-value changes numerically for the same run: the node adopts
the 0.91 factor and the risk-ratio path, and the two engines now report the same number.

**Interpretation node** (`nodes/interpretation.py`). The robustness sentence is built from the reading:
the headline, then the message. The 2.0 and 3.0 narrative bands are removed. `confidence` keeps its
derivation (high needs `robust_to_confounding`), which now means "beyond measured confounding". The FCI
latent-confounder corroboration policy is unchanged: it suppresses the warning only when
`robust_to_confounding` is true, surfaces it on `within`, `null_finding`, `unbenchmarked`, or a missing
sensitivity result (fail-open, as measured in `test_latent_warning_policy.py`).

### 4.7 Chat tool `sensitivity_analyzer` (`src/agents/tool_composer/tool_registrations.py`)

Signature becomes `sensitivity_analyzer(ate, ci_lower, ci_upper=None, baseline_risk=None,
naive_ate=None)`. The math comes from `evalue`; the risk-ratio path is used when `baseline_risk` is given.
When `naive_ate` is given the tool returns the same reading and headline the runner would; otherwise it
returns `reading = "unbenchmarked"` with an interpretation stating that no universal E-value threshold
exists and that a benchmark needs the measured confounders. The `robustness` weak/moderate/strong field is
removed; `e_value_point` and `e_value_ci` stay. The registry entry's description in `tool_registry.py`
is updated to match. The tool still refuses non-finite inputs.

**Finding, not fixed here.** The registry's `ToolSchema` for `sensitivity_analyzer` declares inputs
(`causal_result`, `gamma_range`) and outputs (`e_value`, `robustness_value`, `sensitivity_plot_data`)
that do not match the registered function. Filed as a separate issue at the lane's close-out.

### 4.8 Documentation surfaces

**Lineage page** `docs/lineage/causal_dag_lineage.html`: the refutation scoring table row for
`sensitivity_e_value` (critical no, pass rule = point risk ratio above the measured-confounding benchmark,
warning = within / null finding / unbenchmarked), the placebo row (no warning band; matches #1994 option
1), the "Measured" callout and the REVIEW-band list item (the 2026-09-10 counts and the new reachable
bands from §6), the E-value details block (conversion paths, benchmark, readings), the refutation gate
calculator (sensitivity non-critical, statuses PASSED/WARNING/SKIPPED), the randomized-skip rows, and the
`review_caveat` row reserved for this lane. Anchors need a rendered `getBBox` check per memory.

**Documentation page** (`frontend/src/components/documentation/content.ts`, `RefutationGate.tsx`): the
sensitivity entry (`critical: false`, the pass rule and warning sign in the words of §4.4) and the E-value
illustration (the fixed "threshold 2.0" line becomes a per-run benchmark marker; the fail state renders as
the WARNING reading). `type Outcome = 'pass' | 'fail'` stays; the copy changes.

### 4.9 What does not change

No database migration: the `status` enum already holds PASSED/WARNING/SKIPPED and `details_json` is
free-form. No OpenAPI change: `RefutationSummary.sensitivity_e_value` keeps its type, the per-test
`details` string carries the message, so `api.ts` is not regenerated. The drill-down's three-state verdict
(#1867) renders the new statuses as-is. The `causal_paths` promotion rules (PROCEED → validated, BLOCK →
refuted) are untouched; fewer BLOCKs mean fewer paths marked refuted. The 109 seeded `causal_paths`
sensitivity rows and the existing pending expert-review rows are untouched (debt 3).

## 5. Error handling

- `evalue` functions raise `ValueError` on non-finite or out-of-domain inputs; the runner and the node
  catch nothing new — a failure in the sensitivity computation surfaces as today (`sensitivity_error` on the
  node; `RefutationError` on the runner), never as a fabricated reading.
- A missing baseline risk or naive contrast is not an error: the classifier falls back per §4.2 and records
  `benchmark_basis`, so the reader always knows what the number was benchmarked against.
- A frame whose treatment is binary but whose control arm has no outcome events (`p0 = 0`) has no
  risk-ratio path; the SMD path is used and `conversion` says so.

## 6. Testing (red-first, one task at a time)

**Unit, `tests/unit/test_causal_engine/test_evalue.py`** (new): every function in §4.1 against hand
values (the EValue package's worked examples where available), orientation of negative effects, domain
errors, and the classifier's five readings including tie handling and `unbenchmarked`.

**Runner** (`test_refutation_runner*.py`, `test_evalue_standardization_p4.py`): sensitivity is never
FAILED and never BLOCKs; `critical` is read from config; details carry the reading; the reported-interval
docstring pin; the residue test's new assertions.

**Band enumeration** (new): every combination of statuses over the five tests with sensitivity in
{PASSED, WARNING, SKIPPED} and the others in {PASSED, WARNING, FAILED, SKIPPED}; asserts the exact
reachable confidence values, that REVIEW is reachable without a critical failure, and that when the four
other tests pass the band is PROCEED for every sensitivity status. The enumerated table is written into
the lineage page callout.

**Calibration, heavy lane** (`tests/unit/test_causal_engine/test_sensitivity_calibration.py`,
`heavy_ml`): LinearDML on the seed-21 Remibrutinib frame at n = 1,500 (21 s, ~400 MiB measured): all 11
planted truths read `beyond_measured_confounding`; the five null pairs of §2.3 read `null_finding`; and, as
a pinned known limit, the omitted-confounder fits read the same as the correct fits on at least 9 of 11
pairs, so nobody re-adds a cutoff gate believing the E-value detects confounding.

**Nodes** (`test_sensitivity*.py`, `test_interpretation*.py`, `test_latent_warning_policy.py`,
`test_refutation*.py` under `test_agents/test_causal_impact/`): the reading and headline in state, the
narrative sentence, the null caveat appended once to `warnings`, `robust_to_confounding` semantics, and
the corroboration policy unchanged.

**Chat tool** (`test_tools_fail_closed.py`): shared math, `unbenchmarked` without a naive contrast, a
reading with one, non-finite refusal.

**Frontend** (`frontend/src/components/documentation/*.test.tsx` as they exist): the sensitivity entry
renders `critical: false` and the new pass rule; run only the documentation tests on the box.

All CI-visible: new unit files under `tests/unit/test_causal_engine/` and `tests/unit/test_agents/` are in
the backend allowlist; the calibration test carries the `heavy_ml` marker.

## 7. Impact measurement before merge

A committed script, `scripts/calibration/reband_sensitivity_readings.py`, re-bands every
`causal_impact_query` estimate in `causal_validations` under the new rule using the stored per-test
statuses and stored E-value inputs, with baseline risk, naive contrast and covariate factors computed from
frames pulled through the route's own loaders (`_load_agent_estimation_frame`,
`_load_hcp_adoption_join_frame`) at the stored `refutation_n_rows_total`, and applies the runner's own
`_calculate_confidence_score` / `_determine_gate_decision`. Output: a per-pair table of today's band vs
the new band and the reading distribution, written to
`docs/demos/results/<run-date>_sensitivity_calibration/reband.md` (dated the day it runs) and posted on #1988 and #1991. The
owner sees this table before the merge. Caveats stated in the output: row order of a `limit N` pull is
not guaranteed identical to the original run's, so population quantities may differ slightly; the three
`accepted → converted` rows have no current dataset mapping and are listed as unmapped.

## 8. Live verification after deploy

1. Container content: the new module is present in the `e2i_api` image and the runner config reads
   `critical: False` for `sensitivity_e_value` (read from the container, never from the job conclusion).
2. Re-run the 11-question Remibrutinib patient-journeys discovery job on the deployed image (the
   lane-1 `run_discovery.py` method). Expected: the six pairs BLOCKed on 2026-09-09 PROCEED with
   `beyond_measured_confounding` readings; no BLOCK unless a critical test fails; readings and headlines
   present in `causal_validations.details_json`.
3. Null finding: pose the brand-less `treatment_arm → persistent_180d` probe pair (null-crossing on 2 of 6
   live runs). Expected: the API record's `warnings` carries the null-finding sentence, the drill-down
   shows the sensitivity row as WARNING with that message, `gate_decision = proceed`, no expert-review row
   created.
4. Narrative: the interpretation text for a `beyond` run contains the headline and the benchmark number;
   for a null run it does not call the effect robust.
5. Expert review queue: no new pending rows from PROCEED runs; the count before and after the job is
   recorded.

## 9. Rollout

One worktree (`.worktrees/lane-d-1991`, branch `claude/1991-sensitivity-calibration`), one implementer
subagent at a time, red-first per task, codex fixed point per task, whole-diff codex before the single PR,
`--merge` never squash, one deploy. No migration rides the deploy. Memory guard: `free -m` before the
calibration test and the re-band script (measured peaks ~470 MiB); stop below 1.5 GiB available.

## 10. Known limits, stated so nobody rediscovers them

- The E-value does not detect omitted confounders (§2.2). The calibration test pins this.
- "Beyond measured confounding" is relative to what this frame measured. A design with weak measured
  confounding sets a low bar; the message always prints the benchmark and its basis so a reader can judge.
- The joint benchmark exists only for a binary treatment; continuous treatments use the strongest covariate
  factor, a cruder measure. The re-band table shows both populations separately.
- Debt 2's second slice — continuous scores replacing PASSED/WARNING labels in the confidence mean — waits
  until the bootstrap distribution is also continuous; a single continuous term in a mean of labels has no
  calibration anchor.
- Sparse high-cardinality categoricals inflate the fallback benchmark on a column that carries no
  confounding: measured (30 levels, n = 200, 30 % treated, 10 % events, seeds 0..49, no relation between the
  column and T or Y) the covariate factor had median 5.71 and max 12.75 before any cell rule (Task 4's
  frames measured median 4.70 / max 27.0 at 30 levels and 8.70 / 91.0 at 80). Tiny cells give ratios of
  the order of the arm ratio and the covariate keeps the max. The direction is safe (a larger benchmark
  cannot manufacture false robustness) but it can push a run to `within_measured_confounding` on nothing.
  `MIN_CELL_SIZE = 5` (the chi-square rule of thumb) skips a level with fewer than 5 rows in either arm:
  the same frames measure median 1.81, max 3.00, with a factor on 28 of 50 seeds. The residual is a
  selection artefact of the 30 % arm at that sparsity (a surviving level needs 5 treated rows out of ~6.7,
  so survivors are treated-heavy; the same rule at a 50 % arm measures median 1.20). Live cardinality is
  unaffected (4 levels: median 1.016; 12 levels: 1.091; n = 1500). A larger cell size (8 or 10) silences
  the column on every n = 200 seed. The numeric route is not subject to the rule.

## 11. Issues

Closes #1988 (option 3 delivered: bands redesigned on planted truth, enumerated by a CI test), #1994
(comment and threshold table fixed, behaviour unchanged), #1989 (docstring pin at the `original_ci`
consumer). Updates #1991 (debt 2 first slice shipped; second slice and its precondition recorded). Files
one new issue: the `sensitivity_analyzer` registry schema mismatch (§4.7).
