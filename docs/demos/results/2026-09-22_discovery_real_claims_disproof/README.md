# Can structure discovery build the causal DAG on real claims data? (2026-09-22)

**Question (owner):** is a causal DAG built for real claims data, and why can the
discovery route not take over when the domain expert (authored structure) is not
available?

**Cheapest disproof:** run the PRODUCTION guided-discovery path
(`GraphBuilderNode._run_discovery` — `src/agents/causal_impact/nodes/graph_builder.py`:
PC + tiers `[covariates < T < Y]` + required estimand edge + bootstrap resamples —
the scripts pass `discovery_bootstrap_resamples: 20` (`run_disproof.py:28`), the same
value as production's `DISCOVERY_BOOTSTRAP_RESAMPLES = 20` at `graph_builder.py:26` — then `DiscoveryGate`)
on real and planted-truth frames. The disproof script never calls `execute()`
(it persists to `discovered_dags`); tracing is off (no Opik, no Redis).

## Frames (`run_disproof.py`, `run_timing.py`)

| Label | Frame | T | Y | k | Rows |
|---|---|---|---|---|---|
| A0 (`disproof_runs.txt:5`: `n=4000 … k=57`) | REAL Optum persistence cohort `data/rwd/mart/persistence/e2i_ml_v3_patient_journeys.parquet`, train split, production shape: every numeric non-id column tiered as a covariate, as `graph_builder.py` does (`covariate_cols`) | `lis_dual_flag` — a pre-index exposure; the mart frames carry NO treatment column | `persistent_at_180d` | 57 | 4,000 (`sample(n=4000, random_state=0)`) |
| A1 (`disproof_runs.txt:16`: `n=4000 … k=14`) | same, pruned: `keep = [age_at_index, enrollment_duration_days, comorbidity_diag_distinct_count, charlson_score]` + every `cci_*` flag with prevalence `>= 0.02` (`run_disproof.py:75-76`) | same | same | 14 | 4,000 |
| A2 (`disproof_runs.txt:27`: `n=4000 … k=13`) | A1 without `charlson_score` | same | same | 13 | 4,000 |
| B (`disproof_runs.txt:38`: `n=4000 … k=10`) | synthetic `data/rwd/synthetic_CSU/patient_journeys.parquet` (Remibrutinib rows), planted confounders `disease_severity`, `academic_hcp` (`data/synthetic/ground_truth_20260611T150429.json`) | `treatment_arm` | `treatment_initiated` | 10 | 4,000 |
| timing (`timing_run.txt:2`: `kept k= 43`; `run_timing.py:8`: `n=4000`) | A0 rank-pruned greedily (a column is kept iff it raises the correlation-matrix rank; `run_timing.py`), ONE PC run, `discovery_bootstrap_resamples=0`, latent diagnostic off | same as A | same as A | 43 | 4,000 |

## Captured results

Every value below is quoted from the named file.

| Run | File:line | Gate | Conf | Wall | Captured facts |
|---|---|---|---|---|---|
| A0, pre-fix (`src` = main `0d1730255`) | `disproof_runs.txt:3,5-6,13` | reject | 0.000 | 0.5 s | `A0 corr-matrix rank 45 of 59`; reason `Too few edges discovered: 0 < 1`; latent diagnostic error `Data correlation matrix is singular. Cannot run fisherz test.` |
| A0, post-fix (`src` = this branch `cbe7b14d8`) | `a0_postfix_run.txt:2-4,7-12` | reject | — | 0.24 s / 0.19 s (two runs: `_run_discovery`, then `execute`) | `success=False edges=0`; reasons `['Discovery failed', 'Discovery could not run: pc: Data correlation matrix is singular. …']`; `execute -> discovery_skip_reason: auto-discovery could not run, falling back to manual DAG: pc: …`; the same string in `warnings`; `manual DAG edges: 115`, `gate decision: reject` |
| A1 | `disproof_runs.txt:16-21` | augment | 0.739 | 180.6 s | corroboration (bootstrap stability) 67.35 %; `T->Y in ensemble: False` although `(T, Y)` was a REQUIRED prior edge; `Edge recall: 0.00%` |
| A2 | `disproof_runs.txt:27-32` | augment | 0.734 | 289.4 s | corroboration 66.81 %; `T->Y in ensemble: False` |
| B | `disproof_runs.txt:38-47` | accept | 0.818 | 1.7 s | `T->Y in ensemble: True`; `PLANTED confounders: {'disease_severity': 'confounder', 'academic_hcp': 'instrument'}` |
| timing | `timing_run.txt:2-4` | reject (`uncorroborated_single_run`) | 0.00 | 230.3 s | `kept k= 43`; the 14 columns named on that line were dropped as linearly dependent (57 − 43 = 14); `edges=130`; `T->Y: False` |

Derived (not captured, arithmetic on the lines above): the production path
runs `1 + DISCOVERY_BOOTSTRAP_RESAMPLES` PC fits, so the timing frame under
production settings is about `21 × 230.3 s ≈ 81 min` for one causal question.

## What this establishes

1. **No causal DAG is built from real claims data today.** The DAG-building
   pipeline (`graph_builder.py`) is reachable only for the datasets registered in
   `src/api/routes/causal/datasets.py::_CAUSAL_DATASET_SPECS` (`datasets.py:60-170`:
   `patient_journeys`, `hcp_adoption`, `nba_triggers`, all synthetic); the real
   Optum mart cohort frames are prediction frames (pre-index covariates + one
   label, no treatment column — see the A0 frame row). The Layer-4 "structural
   attestations" in `src/data/manifests/optum_feature_manifest.py` (edges over
   `{feature, T, Y}`, `_optum_attestation`) feed a per-feature role decider that
   is off by default (`src/agents/tier_0/pipeline.py::adaptive_structural_decider_enabled = False`),
   are never assembled into a cohort DAG, and their research record
   (`docs/layer4/optum_initiation_attestation_research.md`, "Still deferred")
   states activation is a no-op for keep/drop on the initiation cohort.
2. **Discovery cannot "take over" on a real claims frame as shipped**, for two
   measured reasons: (a) the Charlson/Elixhauser flag families and composite
   scores are linearly dependent (rank 45 of 59 — A0 pre-fix row), fisherz refuses the singular
   correlation matrix, and — before this branch — the failure was reported as an
   empty structure; (b) even rank-pruned, one PC fit on 43 covariates takes
   230.3 s (timing row), so the production bootstrap makes one question an ~81 min job (derived, see above)
   against the agent's hard timeout.
3. **Expert approval is already advisory** in the causal agent
   (`CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL`, default off — `refutation.py:90`,
   `config/agent_config.yaml:1165`): a REVIEW-band DAG queues a review and the
   estimate proceeds with a caveat; only a human REJECTION halts. On the
   synthetic path discovery therefore *does* take over today.

## Fixed on this branch — scope: correctness of the reported outcome, not discovery takeover

`DiscoveryRunner` returned `success=True` when no algorithm converged. That
boolean is consumed: `DiscoveryGate.evaluate` branches on it, the persisted
`discovered_dags.metadata.success` records it, and `graph_builder` decides
whether to report a skip. A run that could not execute was therefore recorded
and gated as a run that found nothing. This branch derives `success` from
whether any algorithm converged and carries the cause into the gate reason,
`discovery_skip_reason` and the API warnings (post-fix A0 row above). It does
not change what ships (the manual DAG, `dag_source='domain_knowledge'`) and it
does not make discovery succeed on claims frames — that is the design decision
below.

## Deliberately NOT done here (owner decision)

- **Collinearity pre-pruning before PC** (turns A0 into A1/A2) **only together
  with a covariate cap or screening for the DAG-learning frame** — pruning alone
  turns a 0.24 s honest failure (A0 post-fix row) into an ~81 min run (derived, see above). Dropped/capped covariates
  would stay in the adjustment guarantee (`graph_builder._apply_adjustment_guarantee`),
  so the estimate would still condition on them.
- **A real-data entry point.** The mart frames have no treatment column; the
  raw enriched drop `data/rwd/Optum_Parquet/Optum_enriched.parquet` carries
  `index_biologic_brand`, `index_biologic_molecule`, `biologic_switch_180d_flag`,
  which could define a comparative-effectiveness question. That is a
  converter + registry lane, and `datasets.py:290-293` requires the
  negative-control registry to be re-verified per data source before it scores.
- **Discovery as a fallback structural source in the Layer-4 role ladder** for
  un-attested features. Intent investigation (2026-09-22): `gh issue list
  --search "structural decider discovery fallback"` returns nothing relevant;
  no `.claude/plans/*layer4*` or `docs/layer4/*` file mentions discovery; the
  attestation commits (`19a76f5a1`, `33527da0f`, PR #543) describe an authored,
  literature-grounded source by design ("role is DERIVED from edges, never
  declared"). Nothing recorded asks for a discovered fallback. Recommendation:
  do not add one unless the owner states that requirement — it would inherit
  (2a)/(2b) and would let an uncorroborated structure decide feature drops
  above the LLM rung. The fact that would reverse this: an owner requirement
  for role decisions on real cohorts without authored diagrams.

## Reproduce

```bash
# from the tree under test, cwd-first import; E2I_DATA_ROOT = a checkout that has data/ (gitignored)
cd <tree-under-test>
E2I_DATA_ROOT=/path/to/checkout-with-data python - < docs/demos/results/2026-09-22_discovery_real_claims_disproof/run_disproof.py
```
The scripts print `src resolves to: …` — check it names the tree you mean. A
script run BY PATH inside a worktree imports `src` from the main checkout
(editable `.pth`), which is how `disproof_runs.txt` / `timing_run.txt` came to
be pre-fix captures (their headers say so).
