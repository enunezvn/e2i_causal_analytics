# Can structure discovery build the causal DAG on real claims data? (2026-09-22)

**Question (owner):** is a causal DAG built for real claims data, and why can the
discovery route not take over when the domain expert (authored structure) is not
available?

**Cheapest disproof:** run the PRODUCTION guided-discovery path
(`GraphBuilderNode._run_discovery`: PC + tiers `[covariates < T < Y]` + required
estimand edge + 20 bootstrap resamples, then `DiscoveryGate`) on real and
planted-truth frames. No `execute()` in the disproof script (it persists to
`discovered_dags`); no Redis / Opik (tracing off).

## Frames

| Label | Frame | T | Y | k | Rows |
|---|---|---|---|---|---|
| A0 | REAL Optum persistence cohort `data/rwd/mart/persistence`, train split, production shape (every numeric non-id column tiered as a covariate, as `graph_builder.py` does) | `lis_dual_flag` (pre-index exposure; the mart frames carry NO treatment column) | `persistent_at_180d` | 57 | 4,000 |
| A1 | same, pruned to one comorbidity family (`cci_*` ≥ 2 % prevalence) + 4 scalars | same | same | 14 | 4,000 |
| A2 | A1 without the composite `charlson_score` | same | same | 13 | 4,000 |
| B | synthetic `patient_journeys` (Remibrutinib), planted confounders `disease_severity`, `academic_hcp` | `treatment_arm` | `treatment_initiated` | 10 | 4,000 |
| timing | A0 rank-pruned greedily (drop a column iff it does not raise the correlation-matrix rank) | same as A | same as A | 43 | 4,000 |

## Results (`disproof_runs.txt`, `timing_run.txt`, `a0_postfix_run.txt`)

| Run | Gate | Confidence | Wall | Notes |
|---|---|---|---|---|
| A0 (pre-fix, main `0d1730255`) | REJECT | 0.00 | 0.5 s | corr-matrix rank **45 of 59**; fisherz: *"Data correlation matrix is singular"*; runner returned `success=True`, 0 edges; reason shown: **"Too few edges discovered: 0 < 1"** |
| A0 (post-fix, this branch) | REJECT | 0.00 | ~0.5 s | `success=False`; reason: **"Discovery could not run: pc: Data correlation matrix is singular…"**; `execute()` surfaces `discovery_skip_reason` + warning; manual DAG ships |
| A1 | AUGMENT | 0.739 | 181 s | corroboration 67 %; **no T→Y edge** in the ensemble although it was a REQUIRED prior edge (causal-learn honours required edges in orientation only); FCI flags bidirected edges into T |
| A2 | AUGMENT | 0.734 | 289 s | same shape |
| B | ACCEPT | 0.818 | 1.7 s | T→Y recovered; `disease_severity` → confounder (correct); `academic_hcp` → instrument (planted confounder, its → Y edge missed) |
| timing (43 covs, **no** bootstrap) | REJECT (uncorroborated single run) | 0.00 | **230 s** | 130 edges; production's 20 resamples ⇒ ~75–80 min for one question |

## What this establishes

1. **No causal DAG is built from real claims data today.** The causal-agent
   pipeline that builds DAGs (`src/agents/causal_impact/nodes/graph_builder.py`)
   is reachable only for the three synthetic tables in
   `src/api/routes/causal/datasets.py::_CAUSAL_DATASET_SPECS`; the real Optum
   mart cohort frames are prediction frames (pre-index covariates + one label,
   no treatment column). The Layer-4 "structural attestations" in
   `src/data/manifests/optum_feature_manifest.py` are per-feature 3-node
   fragments read by a feature-validity decider that is off by default
   (`tier_0/pipeline.py::adaptive_structural_decider_enabled=False`); they are
   never assembled into a cohort DAG, and their own research record says
   activation is a no-op for keep/drop on the initiation cohort.
2. **Discovery cannot "take over" on a real claims frame as shipped**, for two
   measured reasons: (a) the Charlson/Elixhauser flag families and composite
   scores are linearly dependent, fisherz refuses the singular correlation
   matrix, and — before this branch — the failure was reported as an empty
   structure; (b) even collinearity-pruned, PC on 43 covariates takes ~4 min
   per un-bootstrapped run, ~80 min under the production bootstrap.
3. **Expert approval is already advisory** in the causal agent
   (`CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL` default off): a REVIEW-band DAG queues
   a review and the estimate proceeds with a caveat; only a human REJECTION
   halts. So on the synthetic path discovery *does* take over today.

## Fixed on this branch

The silent part of (2a): `DiscoveryRunner` now derives `success` from whether
any algorithm converged; the gate's REJECT reason and `graph_builder`'s
`discovery_skip_reason` carry the cause into the API warnings.

## Deliberately NOT fixed here (design decision for the owner)

- Collinearity pre-pruning before PC (turns A0 into A1/A2) **only together with**
  a covariate cap / screening for the DAG-learning frame — pruning alone turns
  a 0.5 s honest failure into an ~80 min run against the agent's hard timeout.
  Dropped/capped covariates would stay in the adjustment guarantee channel
  (`_apply_adjustment_guarantee`), so the estimate still conditions on them.
- A real-data entry point: the mart frames have no treatment; the raw enriched
  drop (`Optum_enriched.parquet`) carries `index_biologic_brand` /
  `index_biologic_molecule` / `biologic_switch_180d_flag`, which could define a
  comparative-effectiveness question — a converter + registry + per-source
  negative-control re-verification lane (`datasets.py` mandates the latter).
- Wiring discovery into the Layer-4 role ladder as a fallback for un-attested
  features: not recommended — it would inherit (2a)/(2b) and let an
  uncorroborated structure decide feature drops above the LLM.

## Reproduce

```bash
# from a checkout that has data/ (gitignored); E2I_DATA_ROOT points at it
cd <tree-under-test> && E2I_DATA_ROOT=/path/to/checkout-with-data python - < docs/demos/results/2026-09-22_discovery_real_claims_disproof/run_disproof.py
```
The scripts print `src resolves to: …` — check it names the tree you mean
(a script run BY PATH inside a worktree imports `src` from the main checkout).
