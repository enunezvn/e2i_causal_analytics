# ADR-017: Discovery corroboration is bootstrap edge stability; diagnostics and warnings annotate, never gate; a prior-asserted DAG is never reported as discovered

**Date**: 2026-09-02 | **Status**: Accepted | **Implemented by**: PRs #1869, #1871 (refutation three-state + warning-aware phrasing), #1879 (`dag_source`), #1883 (bootstrap corroboration), #1886 (FCI diagnostic), #1887 (two-channel confounder wiring), #1888 (corroboration-gated latent warning)

## Context

Guided causal discovery on `/api/causal/*` is single-algorithm PC with background knowledge: tiers anchor the treatment as cause and the outcome as effect, and required edges seed the estimand. Three separate ways of overstating evidence surfaced across the 08-31 → 09-04 arc.

- **A single algorithm agrees with itself by construction.** The gate's corroboration axis was cross-algorithm vote agreement; on a one-algorithm run that is a vacuous signal, so an uncorroborated run could still score well on its own say-so.
- **Priors were being read back as findings.** The dataset spec's covariate list was declared as a *structural* prior, forcing `conf→treatment` and `conf→outcome` as required edges for every covariate. The shipped DAG was then identical for real data and pure noise (measured F1 0.78, SHD 4 on the recovery benchmark), and the API still labelled it `discovered`.
- **Soft signals rendered as hard ones.** The refutation engine is three-state per test (PASSED / WARNING / FAILED) but the contract carried only `passed: bool`, so a soft WARNING (E-value CI bound 1.51, inside the [1.5, 2.0) warning band) displayed as a red Failed X next to a PROCEED gate chip — and the narrative still said "survived all robustness checks".

## Decision

Three clauses, each verified against the code independently.

### 1. Corroboration for a single-algorithm run is bootstrap edge stability, measured only beyond the priors

`DiscoveryGate._calculate_corroboration` returns a score **and a basis label**:

- ≥2 converged algorithms → `algorithm_agreement` (existing vote math, unchanged).
- One algorithm, some edge beyond the required set → `bootstrap_stability`, averaged over **only** the edges the priors did not force. A required edge appears in every resample, so its stability is assertion, not evidence.
- One algorithm, **every** edge prior-required → `prior_determined`: the axis is *not applicable* (not "failed"). It is dropped from the weighted average and the remaining components are renormalized — and this is checked **before** whether stability data exists, so a fully prior-forced graph is never mistaken for a run that had no evidence to offer.
- One algorithm, edges beyond the priors, no stability data → `uncorroborated_single_run`, score 0.0, and the confidence **is** that 0.0: a single algorithm's self-reported edge confidence is the vacuous signal being removed and may not rescue the score.

Guided runs therefore default to 20 bootstrap resamples (`DISCOVERY_BOOTSTRAP_RESAMPLES`); 0 would make every guided run auto-REJECT. Multi-algorithm ensembles default to 0 — they are already corroborated by cross-algorithm agreement, and switching them on would be a 20× runtime surprise for existing consumers. An edge is augment-eligible only if it is corroborated *and* beyond the priors.

### 2. The FCI latent-confounding diagnostic and refutation warnings annotate; they never gate

- The `latent_diagnostic` payload rides through `DiscoveryGate.evaluate` as **metadata pass-through on every path, early REJECTs included**, and no decision or confidence math reads it — a new gate input would invalidate the accept/reject calibration.
- `GraphBuilderNode` only annotates and logs (the log line is the durable base-rate record; the job store's TTL is 8 h). The human-readable warning is raised by `InterpretationNode` and only when **two independent signals corroborate**: the flag is up AND the E-value says the estimate is fragile — or the sensitivity analysis produced no result at all, in which case it surfaces as a precaution and says so (fail-open). Measured basis: the estimand mark alone cannot separate a real latent confounder from orientation noise on an effectful frame, and no discovery-side knob changes that; the E-value can. The payload stays on `causal_graph` unconditionally, so suppressing the *warning* loses no data.
- Refutation stays **WARNING-tolerant PROCEED** — the gate logic is deliberately unchanged. What changed is honesty of display: each test carries an additive `status: passed|warning|failed` (absent → the legacy two-state `passed`), warnings are counted and coloured separately from failures, and `robustness_phrase` makes the narrative follow the per-test outcomes — "survived all N robustness checks" only when every executed test passed.

### 3. A prior-asserted DAG is never reported as discovered

- **Two confounder channels, split** (#1887). `modeled_confounders` is the *adjustment-guarantee* channel: every covariate listed is unioned into the final adjustment set regardless of the DAG, so the conditioning set stays exactly what was declared. `anchored_confounders` is the *structural-prior* channel and is deliberately **empty** at the dataset-spec call site: a covariate list is a role allowlist, not a per-question assertion that each column is a genuine confounder. With only tiers and the estimand edge as priors, the data selects the confounder edges (measured F1 mean 0.93, SHD ≤ 1 at n=2000) and the gate scores real evidence.
- **`dag_source` is computed from what the DAG carries beyond the prior-implied edge set**, not from the gate decision alone: `discovered` (accept **and** edges beyond the priors), `prior_asserted` (accept or augment, but every shipped edge is prior-implied — agreement by the data is indistinguishable from assertion, so no data contribution is claimed), `augmented`, `domain_knowledge`. Treatment and outcome are read from the **DAG's own node lists**, because a divergence from the request would silently empty the prior set and label every run `discovered` — failing toward the overstating label.
- `discovered_confounders` is the backdoor adjustment set **minus** everything declared up front, and is empty unless `dag_source` is `discovered`/`augmented`. Echoing the caller's own covariate list back as "discovered" is the same overstatement.
- Per-edge `edge_provenance` (`required_prior` | `discovered` | `curated`) ships with the DAG, so the claim is auditable edge by edge rather than taken on the summary label.

## Consequences

- (+) The recovery benchmark separates signal from noise: under the old single-channel wiring the shipped DAG was the same for real data and pure noise; under the split it is not.
- (+) A reader can tell what the data contributed at three grains — the `dag_source` label, `discovered_confounders`, and per-edge provenance — and the three cannot disagree, because all are derived from the same beyond-priors set.
- (−) Guided discovery costs ~20× more compute than a single PC fit. That is the price of having any corroboration signal at all on a single-algorithm run; the per-run `discovery_bootstrap_resamples` state key is the escape hatch.
- (−) Runs whose graph is entirely prior-implied now report `prior_asserted` and read as a *weaker* result than before. This is an honest regression, not a capability loss — do not "fix" it by trusting the old `discovered` label.
- (−) The latent warning is deliberately suppressed when the E-value says the estimate is robust, so a real latent confounder on a robust-looking estimate is not surfaced to the analyst. The flag remains in the payload and in the logs; this is a surfacing policy chosen against a measured false-positive rate, not a claim that the diagnostic was wrong.
- (−) `status` is absent on legacy cached refutation payloads, so consumers must keep the two-state fallback.

## References

- `src/causal_engine/discovery/gate.py` — `_calculate_corroboration`, the basis labels and the renormalization; `_is_corroborated`
- `src/agents/causal_impact/nodes/graph_builder.py` — `DISCOVERY_BOOTSTRAP_RESAMPLES`, guided vs ensemble defaults, diagnostic annotation
- `src/agents/causal_impact/nodes/interpretation.py` — `_latent_warning_entries` (the corroborated-surfacing policy)
- `src/api/routes/causal.py` — the two confounder channels and the `dag_source` / `discovered_confounders` computation; `src/api/schemas/causal.py` — `dag_source`, `EdgeProvenanceModel`
- `src/insights/robustness_phrase.py` — warning-aware verdict phrasing
