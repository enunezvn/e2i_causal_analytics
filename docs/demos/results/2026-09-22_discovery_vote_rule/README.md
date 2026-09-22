# Discovery ensemble: does the "vote" mean agreement, and who should vote? (2026-09-22)

**Owner's question (verbatim):** "Instead of trusting one algorithm to guess the causal
arrows from data, DiscoveryRunner runs an ensemble (GES, PC, FCI, DirectLiNGAM,
ICA-LiNGAM via causal-learn), votes edges into a single DAG" — versus the observed
"Two votes. FCI, DirectLiNGAM and ICA-LiNGAM are wired but not on by default."

Every number below is quoted from a captured file at the cited `file:line`. Captures
were produced by `run_vote_rule.py` (this directory) with a cwd-first import; each
capture prints `src resolves to:` naming the tree it ran against (pre-fix = the main
checkout at `efaad6eeb`, post-fix = this branch).

## 1. What the code did (intent, shape, harm)

- **Intent.** `DiscoveryConfig.ensemble_threshold` is documented as "Minimum fraction
  of algorithms that must agree on an edge" (`src/causal_engine/discovery/base.py`,
  `DiscoveryConfig` docstring); the gate's corroboration axis is "agreement between
  >=2 converged algorithms" (`gate.py::_calculate_corroboration`). The default
  ensemble `[GES, PC]` dates from the module's first commit `5148f0d10` (2025-12-30);
  FCI (`57bb08d88`) and the LiNGAMs (`745f95440`) were added on 2025-12-31 as opt-ins
  and never promoted (`git log --diff-filter=A` on the wrappers). FCI was later
  repurposed as the latent-confounding DIAGNOSTIC (`runner.py::_run_latent_diagnostic`),
  which annotates and never gates.
- **Shape.** `runner.py::_build_ensemble` computed
  `min_votes = max(1, int(n_converged * threshold))` (unchanged since `5148f0d10`;
  `9d02055ec` only switched the denominator to the converged count). With two voters
  and threshold 0.5 that is `int(1.0) = 1`: an edge found by EITHER algorithm was
  included, at confidence `1/2 = 0.5`. The vote was a union.
- **Harm, measured.** On the rank-pruned real claims frame GES and PC produced 55
  candidate directed edges of which they agreed on 10
  (`real_ensemble_postfix.txt:109`: `"n_candidate_edges": 55, "n_agreed_edges": 10`).
  The union rule shipped a 32-edge DAG (`real_ensemble_prefix.txt:182`:
  `n_edges=55 dag_edges=32`), i.e. 22 of the 32 shipped orientations were a single
  algorithm's, and 18 candidate pairs were reciprocal (`real_ensemble_postfix.txt:114`:
  `"reciprocal_pairs": 18`) — both directions at confidence 0.5, resolved by
  `_remove_cycles` dropping whichever equal-confidence edge it met first. The gate
  called that DAG `augment confidence=0.673` (`real_ensemble_prefix.txt:182`).
  Single-voter edges at 0.5 are not below the gate's `review_threshold` (0.5), so the
  gate never flagged them.

## 2. Frames

| Frame | Rows | T | Y | Covariates | Source |
|---|---|---|---|---|---|
| planted synthetic | 4,000 | `treatment_arm` | `treatment_initiated` | 10 (`synthetic_ges.json` `columns`) | `data/rwd/synthetic_CSU/patient_journeys.parquet`, Remibrutinib rows, the discovery-disproof lane's frame B byte-for-byte (`run_vote_rule.py::synthetic_frame`) |
| real claims | 15,209 (`real_algo_runs.txt:4`) | `treatment_dupixent` | `persistent_at_180d_g28` | 10 (`real_algo_runs.txt:8`: `capped_covs(10)`) | `data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet`; 57 numeric `MART_SAFE_FEATURES`, 0 constant, 12 dropped as exactly collinear by the greedy rank rule (`real_algo_runs.txt:4-5`), 45 kept, capped to top-7 by \|corr T\| ∪ top-7 by \|corr Y\| (`real_algo_runs.txt:6-7`) — the two sets overlap on 4 columns, so the cap yields 10, of which 4 are binary (`real_algo_runs.txt:9`) |

Planted edges of the synthetic frame (8), read from the generator and listed in
`run_vote_rule.py::SYNTH_PLANTED` with the source lines: `disease_severity -> T`,
`academic_hcp -> T` (`src/ml/synthetic/dgp/treatment_arm.py:295-301`), `T -> Y`,
`disease_severity -> Y`, `academic_hcp -> Y` (`patient_generator.py:320-335`),
`age_at_diagnosis -> Y` (`treatment_arm.py:455-478`), `disease_severity ->
engagement_score`, `academic_hcp -> engagement_score` (`patient_generator.py:1080-1087`).
"Skeleton recall" counts a planted pair found in either direction; "oriented recall"
counts it found in the planted direction only; "false pairs" are adjacent pairs that
are not planted; "reciprocal" pairs are emitted in both directions (a CPDAG/PAG
undirected mark — `ges_wrapper.py:179-183`, `pc_wrapper.py:244-246`,
`fci_wrapper.py:252-278` all emit those as both directions).

All runs are UNGUIDED (no priors, `bootstrap_resamples=0`, latent diagnostic off),
through the production `DiscoveryRunner` with its 300 s per-algorithm timeout, one
algorithm per process under the box lock. The production agent path is guided
single-PC + bootstrap (`graph_builder.py`), where the vote rule does not apply
(`n_converged = 1`); the vote rule governs every multi-algorithm consumer
(`graph_builder`'s unguided branch, the tool-registry `discover_dag` without an
estimand, the feature analyzer's causal ranker).

## 3. Single algorithms (deliverable b, part 1)

| Frame | Algorithm | Converged | Edges | Runtime | Recovery / shape | File:line |
|---|---|---|---|---|---|---|
| synthetic | GES | yes | 8 | 2.12 s | skeleton 7/8, oriented 6/8, 0 false pairs, 1 reciprocal (T–Y undirected), missed `age_at_diagnosis -> Y` | `synthetic_algo_runs.txt:3-4` |
| synthetic | PC | yes | 7 | 0.07 s | skeleton 6/8, oriented 4/8, 0 false pairs, `T -> Y` REVERSED, `disease_severity – Y` undirected | `synthetic_algo_runs.txt:8-9` |
| synthetic | FCI | yes | 7 | 0.06 s | skeleton 6/8, oriented **1/8**, 4 planted edges reversed (`disease_severity -> T`, `academic_hcp -> T`, both `-> engagement_score`); edge types 5 undirected, 1 directed | `synthetic_algo_runs.txt:13-15` |
| synthetic | DirectLiNGAM | **no** | 0 | 0.00 s | `refused: LiNGAM assumes non-Gaussian continuous variables, but 4 column(s) are binary` (T, Y, `ecog_performance_status`, `academic_hcp`) | `synthetic_algo_runs.txt:20` |
| synthetic | ICA-LiNGAM | **no** | 0 | 0.00 s | same refusal | `synthetic_algo_runs.txt:25` |
| real | PC | yes | 29 | 1.08 s | 29 pairs, 0 reciprocal, no `T -> Y`, into T: `cci_dementia`, `comorbidity_diag_distinct_count`, `enrollment_duration_days` | `real_algo_runs.txt:10-11` |
| real | GES | yes | 36 | 10.89 s | 35 pairs, 1 reciprocal, no `T -> Y` | `real_algo_runs.txt:21-22` |
| real | FCI | yes | 43 | 2.69 s | 26 pairs, **17 reciprocal**; edge types 7 directed, **17 bidirected**, 2 undirected | `real_algo_runs.txt:56-58` |
| real | DirectLiNGAM | **no** | 0 | 0.00 s | `refused … 6 column(s) are binary` (T, Y, `lis_dual_flag`, `high_comorbidity_burden_flag`, `cci_dementia`, `cci_chronic_pulmonary`) | `real_algo_runs.txt:69` |
| real | ICA-LiNGAM | **no** | 0 | 0.00 s | same refusal (this one capture ran outside the box lock, `real_algo_runs.txt:71`: the wrapper refuses before any fit) | `real_algo_runs.txt:81` |

Two facts decide the LiNGAM question before any fit: the wrappers refuse any frame
with a binary column (`lingam_wrapper.py:52-85`, #2009) and every causal frame here has
a binary treatment and outcome; and the `lingam` package is not a declared dependency
(`pyproject.toml:42` lists only `causal-learn`), is not importable from the venv
(`ModuleNotFoundError: No module named 'lingam'`) and is absent from the prod image
(`docker exec e2i_api python -c "import importlib.util as u; print(u.find_spec('lingam') is not None)"`
-> `False`, run 2026-09-22). Even with the refusal lifted, `from lingam import
DirectLiNGAM` (`lingam_wrapper.py:147`) would fail.

## 4. Ensembles: union (pre-fix) vs agreement (post-fix) (deliverables a and b)

`tree_build_ensemble` is the tree's own `_build_ensemble`; `ref_union` /
`ref_agreement` are reference rules computed in the script from the same captured
edge lists. Gate = `DiscoveryGate().evaluate` with default thresholds (ACCEPT >= 0.8,
REVIEW >= 0.5, AUGMENT edge >= 0.9).

### 4.1 Planted synthetic frame

| Voters | Rule (tree) | Voted edges | Final DAG | Skeleton / oriented recall (final DAG) | False pairs | Gate | File:line |
|---|---|---|---|---|---|---|---|
| GES+PC | union (main) | 9 (2 reciprocal) | 7 | 7/8 / 6/8, `T -> Y` reversed | 0 | augment 0.699 (corroboration 0.833, edge conf 0.833, structure 0.163) | `synthetic_ensemble_prefix.txt:40-42` |
| GES+PC | **agreement (this branch)** | 6 (census: 9 candidates, 6 agreed, rate 0.667) | 6 | 6/8 / 5/8, `T -> Y` reversed | 0 | augment 0.699 (corroboration 0.667, edge conf 1.0, structure 0.163) | `synthetic_ensemble_postfix.txt:27-30` |
| GES+PC+FCI | union (main) | 13 | 7 | 7/8 / 6/8 | 0 | augment 0.484 | `synthetic_ensemble_prefix.txt:53-55` |
| GES+PC+FCI | agreement | 8 (13 candidates, rate 0.615; 2 reciprocal) | 6 | 6/8 / 5/8 | 0 | augment 0.562 | `synthetic_ensemble_postfix.txt:40-43` |
| GES+FCI | agreement | 2 (13 candidates, rate 0.154) | 2 | 2/8 / 2/8 | 0 | augment 0.480 | `synthetic_ensemble_postfix.txt:53-56` |
| PC+FCI | agreement | 2 (12 candidates, rate 0.167) | 1 | 1/8 / 1/8 | 0 | augment 0.480 | `synthetic_ensemble_postfix.txt:77-80` |
| any single algorithm, no bootstrap | — | — | — | — | — | reject 0.000 (`uncorroborated_single_run`) | `synthetic_ensemble_postfix.txt:90-93` |

**The cost of the agreement rule on this frame, stated plainly.** Neither GES nor PC
emitted a false pair at n=4,000, so the union had no false edge to lose. The three
single-voter edges the agreement rule drops (`synthetic_ensemble_prefix.txt:40` 9 voted
vs `synthetic_ensemble_postfix.txt:28` 6 voted) are: `academic_hcp -> Y` (planted, GES
only — the one edge by which the final DAGs differ: skeleton 7/8 vs 6/8), `T -> Y`
(planted direction, GES only; the union's `_remove_cycles` dropped it anyway at 0.50
against the 2-vote `Y -> T`, `synthetic_ensemble_prefix.txt` "Removing cycle edge
treatment_arm -> treatment_initiated (confidence: 0.50)"), and `Y -> disease_severity`
(wrong direction, PC's undirected half; also cycle-removed). Of the three single-voter
edges, one was a correct orientation that survived; of the six agreed edges, five are
correct orientations (`synthetic_ensemble_postfix.txt:29`: oriented 5/8 of 6 edges). The
gate confidence is the same number under both rules (0.699), by construction — see §5.

### 4.2 Real claims frame (no planted truth; shape only)

| Voters | Rule (tree) | Voted edges | Final DAG | Reciprocal (voted) | Gate | File:line |
|---|---|---|---|---|---|---|
| GES+PC | union (main) | 55 | 32 | 18 | augment 0.673 (corroboration 0.591, structure 1.0) | `real_ensemble_prefix.txt:182` |
| GES+PC | **agreement (this branch)** | 10 (55 candidates, rate 0.182) | 10 | 0 | augment 0.608 (corroboration 0.182, edge conf 1.0, structure 0.675) | `real_ensemble_postfix.txt:109-112` |
| GES+PC+FCI | union (main) | 64 | 31 | 27 | augment 0.650 | `real_ensemble_prefix.txt:195` |
| GES+PC+FCI | agreement | 36 (64 candidates, rate 0.563) | 26 | 10 | augment 0.721 | `real_ensemble_postfix.txt:122-125` |
| GES+FCI | agreement | 22 (57 candidates, rate 0.386) | 22 | 0 | augment 0.754 | `real_ensemble_postfix.txt:135-138` |
| PC+FCI | agreement | 20 (52 candidates, rate 0.385) | 20 | 0 | augment 0.723 | `real_ensemble_postfix.txt:194-197` |

Neither rule and no voter set puts `T -> Y` in the unguided ensemble (`"T->Y": false`
on every `VOTES` line) — the same finding the discovery-disproof lane recorded for the
guided path; Lane D owns the required-edge question.

## 5. The fix (deliverable a) and its gate interaction

`_build_ensemble` now uses `min_votes = max(2, ceil(n_converged * threshold - 1e-9))`
when at least two algorithms converged, and 1 when exactly one did (its corroboration
stays the bootstrap path). **`ceil`, not `int`:** "at least a fraction t of the
voters" is `votes >= n * t`, whose smallest integer solution is `ceil(n * t)`; `int`
turned 3 voters at 0.5 into a 1-vote quorum. **Floor 2:** one voter cannot agree with
anyone; a threshold that resolves below two votes is a union, which the config's
documented meaning does not describe. The `- 1e-9` keeps `10 * 0.3 =
3.0000000000000004` from rounding the quorum up to 4.

**Gate interaction.** Under an agreement filter every surviving edge is agreed on by
construction — with two voters every survivor is `2/2 = 1.0` — so the gate's former
corroboration math (mean `votes / n` over the survivors) would return 1.0 for ANY
two-voter run and ACCEPT it regardless of how much the voters disagreed. The reference
`ref_agreement` rows (built in the script without a census, so the gate falls back to
main's survivors math) show exactly that degeneracy: `accept 0.833` with corroboration 1.0 on the synthetic frame
(`synthetic_ensemble_postfix.txt:38`) and `accept 0.935` on the real frame where the
voters agreed on 10 of 55 candidates (`real_ensemble_postfix.txt:120`). The fix records a
**vote census** on the DAG (`dag.graph["vote_census"]`: converged voters, quorum,
candidate edges, agreed edges, agreement rate) and `DiscoveryGate._calculate_corroboration`
reads the census's `agreement_rate` for >=2 converged voters. For two voters at rate r
the gate then computes `0.4 * r + 0.4 * 1.0 + 0.2 * structure`, and the union rule
computed `0.8 * (0.5 + 0.5 * r) + 0.2 * structure` (its mean edge confidence over the
union is `(2a + u) / (2(a + u)) = 0.5 + 0.5 * r`) — the same number, so the
ACCEPT/REVIEW/REJECT bands keep their calibration and only the DAG changes. Measured:
synthetic GES+PC 0.699 under both rules (`synthetic_ensemble_prefix.txt:42`,
`synthetic_ensemble_postfix.txt:30`, identical structure score 0.1634). On the real
frame the union's 0.673 and the agreement rule's 0.608 differ only through the
structure score (1.0 vs 0.6747, `real_ensemble_prefix.txt:182`,
`real_ensemble_postfix.txt:112`): the 10-edge DAG leaves isolated nodes, and
`0.4 * 0.1818 + 0.4 + 0.2 * 1.0 = 0.673` reproduces the union's number exactly at the
union's structure score. `DiscoveryResult.algorithm_agreement`
(persisted by `repositories/discovered_dag.py`) reads the same census, so the stored
value is the agreement rate, not a vacuous 1.0. A result built without a census (tests,
hand-built results) keeps the old votes-per-edge math.

**Tests** (`tests/unit/test_causal_engine/test_discovery/test_ensemble_vote_rule.py`,
15 tests, all through the real `_build_ensemble` and the real gate): agreement at the
default threshold, `ceil` vs `int` at 3 voters, unanimity at 1.0, floor of 2 at 0.1,
single-converged keeps every edge, failed algorithms are not voters, float rounding,
census contents, gate reads the census (disagreement blocks ACCEPT; full agreement
scores 1.0; the calibration identity; the no-census fallback), the `algorithm_agreement`
property, and the single source of truth for the default ensemble (config, tool schema
and input model, graph_builder's unguided branch).

**Teeth.** With the old quorum planted back (`teeth_plant_a.txt`): `7 failed, 8 passed`.
With the census not recorded (`teeth_plant_b.txt`): `7 failed, 8 passed`. Green on the
branch: `131 passed` across the new file plus `test_runner.py`,
`test_ensemble_confidence_p12.py`, `test_gate.py`, `test_base.py`,
`test_graph_builder_discovery_bootstrap.py`, `test_causal_discovery_tool.py`.

## 6. Recommendation for the default voter set (deliverable b) — OWNER DECISION

**Keep `[GES, PC]`.** The task's bar for a change was a strict improvement on BOTH
frames; no candidate meets it:

- **FCI must not vote.** Its PAG marks reach the DAG merge as votes for both directions
  (`fci_wrapper.py:252-278`): 17 bidirected pairs on the real frame
  (`real_algo_runs.txt:58`), 5 of 6 typed edges undirected on the synthetic frame
  (`synthetic_algo_runs.txt:15`). A both-direction vote agrees with whichever direction
  GES or PC emitted, so adding FCI raised the real frame's agreement rate from 0.182 to
  0.563 and the gate from 0.608 to 0.721 (`real_ensemble_postfix.txt:109,112,122,125`)
  while adding 10 reciprocal pairs the cycle remover then resolves arbitrarily
  (`real_ensemble_postfix.txt:123`) — corroboration without orientation evidence. On
  the planted frame FCI's orientations were reversed on 4 of 8 planted edges
  (`synthetic_algo_runs.txt:14`); it agreed with GES on 2 and with PC on 2 candidate
  edges (`synthetic_ensemble_postfix.txt:53,77`), added no planted edge to GES+PC
  (skeleton 6/8 either way, `synthetic_ensemble_postfix.txt:29,42`) and lowered the gate
  from 0.699 to 0.562 (`synthetic_ensemble_postfix.txt:30,43`). A bidirected FCI edge
  MEANS "latent confounder", which is exactly what the merge cannot represent
  (`EdgeType.BIDIRECTED` exists, but `_build_ensemble` types every ensemble edge
  `DIRECTED`); its right place is the latent diagnostic it already serves.
- **DirectLiNGAM and ICA-LiNGAM cannot vote** on any frame in this product: the
  wrappers refuse binary columns by design (#2009) and both frames carry a binary
  treatment and outcome (`synthetic_algo_runs.txt:20,25`, `real_algo_runs.txt:69`); the
  `lingam` package is not installed in the venv or the prod image (§3). Promoting them
  would add a guaranteed `converged=False` result to every run.
- **Runtime is not the constraint at this width** (10 covariates): PC 1.08 s, GES 10.89 s,
  FCI 2.69 s on 15,209 rows (`real_algo_runs.txt:10,21,56`); the wall-clock cap of 300 s
  was never approached.

The single fact that would reverse this: a DAG merge that represents FCI's
bidirected/circle marks as their own edge types (so a `<->` never votes for an
orientation) — then FCI's skeleton could corroborate GES/PC adjacencies without
polluting orientation. That is a merge-representation change, not a default flip.

## 7. Single source of truth (deliverable c)

`DEFAULT_DISCOVERY_ALGORITHMS = (GES, PC)` and `DEFAULT_DISCOVERY_ALGORITHM_NAMES`
(`base.py`) are read by `DiscoveryConfig.algorithms`, `graph_builder`'s unguided
branch (state default and the empty-list fallback), the tool registry's
`DiscoverDagInput.algorithms`, its invalid-name fallback, the module-level
`discover_dag(algorithms=None)` and the registered `ToolParameter` default, and the
observability docstring. `state.py` comments point at the constant. Remaining literal
`["ges", "pc"]` occurrences are in tests and docstring examples only.

## 8. Not done here (owner decisions / other lanes)

- The FCI merge representation (bidirected/circle marks as typed ensemble edges).
- The PC/GES CPDAG undirected marks are still emitted as both directions and, when
  two voters both leave a pair undirected, both directions reach the quorum and
  `_remove_cycles` picks one at equal confidence (`synthetic_ensemble_postfix.txt:41`:
  2 reciprocal pairs among the 8 agreed GES+PC+FCI edges). A typed undirected
  ensemble edge would be the honest output; out of this lane's `_build_ensemble`-only
  scope for runner.py.
- `_discover_dag_with_tracing` still reports `agreement = mean(e.confidence)` to Opik
  (vacuous 1.0 for a two-voter agreement ensemble); Opik is intentionally stopped on
  the droplet and the persisted value (`algorithm_agreement`) is fixed; left for a
  tracing lane because it sits outside `_build_ensemble`.
- The unguided ensemble never contains `T -> Y` on the real frame under any rule; the
  guided path's required-edge honesty is Lane D's item 3.

## Reproduce

```bash
cd <tree-under-test>   # cwd-first import; the script prints "src resolves to:"
E=docs/demos/results/2026-09-22_discovery_vote_rule
export VOTE_RULE_OUT=$PWD/$E
for a in ges pc fci direct_lingam ica_lingam; do
  flock /tmp/claude-1000/e2i_box_heavy.lock timeout 400 \
    .venv/bin/python - --step algo --frame synthetic --algo $a --cap 300 < $E/run_vote_rule.py
done
.venv/bin/python - --step ensemble --frame synthetic --combos "ges,pc;ges,pc,fci;ges,fci;pc,fci" < $E/run_vote_rule.py
# same with --frame real (reads the parquet by absolute path; data/ is gitignored)
```
