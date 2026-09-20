# `dml_learner` adversarial review and remediation (2026-09-20)

## Verdict

The original registration was incomplete and was not an optimal production
integration. It made the estimator forceable, but CATE output, heterogeneity
claims, refutation reconstruction, public discovery, and Auto eligibility were
separate contracts that could drift.

The remediated implementation is suitable as an **expert opt-in estimator**.
It is intentionally not in Auto: on the production-shaped benchmark it added
about 18 seconds, did not win, and did not materially change the conclusion.
Promotion to Auto needs repeated-dataset evidence, not registration alone.

## Applied fixes

1. The quadratic final-stage basis is rank-safe. Constant and linearly
   dependent terms (including binary `x² == x`) are pruned with scale-normalized
   pivoted QR. Mixed boolean/numeric frames are normalized to a float design.
   An underdetermined or biased-inference warning makes the wrapper fail closed.
2. `cate_available` is separate from `heterogeneity_detected`. A row-level CATE
   array is a model capability, not evidence that treatment effects vary.
3. Heterogeneity is inference-backed: linear final stages test effect-modifier
   coefficients with Bonferroni control; non-parametric models use
   multiplicity-controlled pointwise effect-deviation inference. Score strata
   are emitted only after detection and only when both strata have model-based
   intervals. They are explicitly labeled exploratory.
4. A canonical estimator registry now drives selector construction, the
   default chain, aliases, forceability, public metadata, sampling-interval
   eligibility, exact DoWhy method mapping, and refutation reconstruction.
   `backdoor.econml.dml.DML` is matched exactly and cannot capture
   `LinearDML` or `CausalForestDML` by substring.
5. Contract tests cover rank safety, invalid-inference rejection, CATE versus
   detection semantics, interval-bearing segments, registry parity, the API
   catalog, exact refutation mapping, and a real heterogeneous DML fit.
6. Auto eligibility is now explicit registry metadata. `dml_learner` remains
   `default_enabled = false`; the API/UI expose it as an expert override.

## Production-shaped forced run

Source: checked-in `data/rwd/synthetic_CSU/patient_journeys.parquet`.
The current `EstimationNode` was forced to `dml_learner` on a deterministic,
treatment/outcome-stratified 2,500-row sample with 10 mixed continuous,
binary, and low-cardinality covariates.

| Measure | Result |
|---|---:|
| Selected estimator | `dml_learner` |
| ATE | 0.094969 |
| 95% CI | [0.038416, 0.151522] |
| Standard error | 0.028854 |
| CATE available | true |
| Heterogeneity detected | false |
| Heterogeneity test | effect-modifier coefficients, Bonferroni |
| Adjusted p-value | 0.224747 |
| Segments published | 0 |
| Invalid-inference warnings | 0 |
| Wall time | 14.606 s |

This is the intended negative case: the estimator returns finite row-level
CATE predictions, but the agent does not claim heterogeneity or publish
segments when the inference-backed test does not reject constant effects.

## Auto-chain benchmark

The same checked-in patient-journey substrate was benchmarked on a deterministic
2,500-row sample. Results are one-run engineering evidence, not a statistical
performance study.

| Estimator | Energy score | Fit time | ATE | 95% CI |
|---|---:|---:|---:|---:|
| Causal forest | 0.316606 | 21.255 s | -0.005264 | [-0.155965, 0.145437] |
| Linear DML | 0.317398 | 19.009 s | 0.000378 | [-0.040152, 0.040907] |
| DML learner | 0.316862 | 18.330 s | 0.001749 | [-0.038620, 0.042119] |
| DRLearner | 0.317420 | 16.052 s | -0.003233 | [-0.143391, 0.136924] |
| OLS | 0.317865 | 14.405 s | 0.004262 | [-0.042287, 0.050811] |

All scores fell inside the selector's 0.05 tie band; the rank-aware tie-break
selected `linear_dml`. Adding `dml_learner` therefore increased tournament cost
without changing the selected estimator or substantive conclusion. The full
tournament warning capture also exposed two underdetermined-inference warnings
from existing peer estimators; the isolated forced `dml_learner` run above had
none. Peer warning cleanup is separate from this estimator's Auto decision.

## Remaining limits

- The high/low output is a model-score stratum, not a human-interpretable
  segment rule and not an independently validated policy subgroup.
- One dataset and one run cannot establish general Auto superiority. Promotion
  should require repeated seeds and multiple representative datasets, with
  incremental latency, selection frequency, interval validity, and planted
  CATE recovery reported together.
- The local HTTP service was not restarted for this review; validation ran
  in-process against the edited code, avoiding a stale-worker false result.
