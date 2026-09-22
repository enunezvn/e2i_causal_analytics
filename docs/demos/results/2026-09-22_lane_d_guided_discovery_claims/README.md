# Lane D — guided discovery on claims frames: evidence (2026-09-22)

Spec: `docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md`,
section "Lane D — guided discovery on claims frames" (on the program branch, not
on main). Prior evidence: `docs/demos/results/2026-09-22_discovery_real_claims_disproof/`
(the measured starting point the spec quotes).

Every number below is quoted from a captured file in this directory, cited as
`file:line`. The scripts print `src resolves to: …/.worktrees/lane-d-guided-discovery-claims/src/__init__.py`
(`d0_frame_shape.txt:1`, `d4_fci_depth_cost.txt:1`) — a script run BY PATH in a
worktree imports `src` from the main checkout, so every probe was run as a
heredoc / `python - <` from the worktree root with a cwd-first import and an
assert on `src.__file__`. Supabase / Redis environment variables were blanked
before importing `src` in every run that touches the agent node; nothing was
persisted.

## The frame (`d0_frame_shape.txt`, `frame_resolver.py`)

The real causal cohort parquet
`data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet`
(Lane A's export; the registry entry lives only on Lane A's unmerged branch), resolved
as the Lane A loader would — numeric coercion of the non-categorical
`MART_SAFE_FEATURES`, drop-first `<col>=<level>` one-hot of the 7 text categoricals
with a `<col>=__missing__` dummy where NULLs exist (`frame_resolver.py`, read from
`_resolve_agent_estimation_frame` on branch `claude/real-data-causal-estimation`).
The loader's own exact-collinearity prune is deliberately NOT replicated, so the
pre-flight's rank prune is exercised on the un-pruned frame.

| Fact | Value | Source |
|---|---|---|
| Rows × columns, covariates | n = 15,209, 79 columns, k = 77 | `d0_frame_shape.txt:2` |
| NaN cells | 0 | `d0_frame_shape.txt:3` |
| Binary covariates | 71 of 77; the 6 non-binary are `age_at_index`, `enrollment_duration_days`, `charlson_score`, `elixhauser_van_walraven_score`, `comorbidity_diag_distinct_count`, `comorbidity_diag_claim_count` | `d0_frame_shape.txt:5` |
| Correlation-matrix rank (T, Y + 77) | 63 of 79 | `d0_frame_shape.txt:6` |
| Marginal fisherz p-value of the estimand pair (T = `treatment_dupixent`, Y = `persistent_at_180d_g28`) | 0.2100 — independent at alpha 0.05 | `d0_frame_shape.txt:7` |
| Raw Y rate by arm | 0.734 (Xolair) vs 0.744 (Dupixent) | `d0_frame_shape.txt:8` |

## Item 3 first — why the required (T, Y) edge was missing (`d1_required_edge_mechanism.txt`)

Cheapest disproof: a 3-column frame with `c -> t`, `c -> y` and NO `t -> y`
(`t ⊥ y | c` exactly), a guided prior with tiers `[[c], [t], [y]]` and
`required_edges = [(t, y)]`, through the production `PCAlgorithm.discover`.

| Fact | Value | Source |
|---|---|---|
| PC output edges | `[('c','t'), ('c','y')]`; required `(t, y)` present: **False** | `d1_required_edge_mechanism.txt:1-2` |
| causal-learn `SkeletonDiscovery.skeleton_discovery` consults `is_required` | **False**; consults `is_forbidden`: True | `d1_required_edge_mechanism.txt:3` |
| `orient_by_background_knowledge` consults `is_required` | True (orientation only) | `d1_required_edge_mechanism.txt:4` |
| Control with a real `t -> y` effect | `(t, y)` present: True | `d1_required_edge_mechanism.txt:6` |
| causal-learn version | 0.1.4.3 (the D1 probe's own version line printed `?`; captured separately from the worktree venv) | `d6_causal_learn_version.txt:3` |

So the mechanism is causal-learn's: a required edge constrains ORIENTATION of an
adjacency that survived the skeleton phase; it never protects the adjacency. On the
real frame the estimand pair is marginally independent (`d0_frame_shape.txt:7`), so
PC removes it at depth 0. Nothing of ours drops it: the single-algorithm ensemble
keeps every edge the converged run draws (`runner._build_ensemble`, `min_votes = 1`),
and no cycle can involve Y under the tiers. The bootstrap ensemble threshold named
in the spec is not the cause (the base run never had the edge).

What shipped for item 3 (honesty, not a rewrite of the data's testimony):
`DiscoveryResult.metadata["required_edges_missing"]` + `..._cause` (graph_builder
`_annotate_required_edges`), a warnings line naming the estimand edge, and the
existing ACCEPT-path assertion of `T -> Y` with provenance `required_prior` pinned
by `tests/unit/test_agents/test_causal_impact/test_graph_builder_preflight.py::TestRequiredEdgeHonesty`.
Restoring the edge inside the PC wrapper was considered and rejected: it would
turn a pure-noise frame's REJECT ("too few edges") into a prior-determined REVIEW
and would inflate the gate's mean edge confidence with an edge the data refused —
the provenance label already keeps the shipped DAG honest. On the REVIEW / REJECT
paths the shipped DAG is the manual construction and the estimand edge is labelled
`curated`, per the documented `_compute_edge_provenance` design (codex iter-1 HIGH
in that PR: a manual DAG's edges are `curated` even where a prior agrees).

## Item 1 + the lane's single premise (`d3_budget_premise.txt`)

Premise: capping the DAG-learning frame at 20 covariates makes one guided PC fit
fast enough for at least 10 resamples inside the budget. Prototype pre-flight
(the same three steps the module implements) on the real frame, then ONE guided PC
fit through `GraphBuilderNode._run_discovery` (bootstrap 0, latent diagnostic off),
then ONE unguided FCI fit on the capped frame.

| Fact | Value | Source |
|---|---|---|
| Rank prune | kept 61, dropped 16: `cci_hiv`, 11 `elx_*` flags duplicating Charlson flags, `payer_category=commercial_exchange`, `payer_category=medicare_lis_dual`, `payer_bus=MCR`, `charlson_risk_band=5 plus` | `d3_budget_premise.txt:9` |
| Correlation rank after the prune | 63 of 63 (full rank) | `d3_budget_premise.txt:10` |
| Cap 20 by the screening rule | 20 selected (the line prints the selection size as `k=`); names and their \|corr\| with T and with Y on the two lines below | `d3_budget_premise.txt:11-13` |
| ONE guided PC fit, n = 15,209, k = 20 | wall 9.5 s, converged, 61 edges, fisherz | `d3_budget_premise.txt:14` |
| Gate at B = 0 | reject (uncorroborated single run), as designed | `d3_budget_premise.txt:15` |
| `T -> Y` in the ensemble | False; parents(T) = 5 covariates, parents(Y) = 3 | `d3_budget_premise.txt:16` |
| Projection (hypothesis, measured in item 5) | 1 + 20 fits ≈ 199 s; ≈ 17 resamples fit in 180 s | `d3_budget_premise.txt:17` |
| ONE unguided FCI fit on the capped frame | **382.5 s**, converged, 95 edges | `d3_budget_premise.txt:31` |

The 16 dropped columns are exactly the 16 the Lane A loader's own prune reports
on this frame (`d5_prune_cross_check_lane_a.txt:1-4`: identical sets, identical
order; Lane A source cited on line 4), which cross-validates the two
implementations. The production
module's decisions on the same frame: kept 20, constant 0, collinear 16,
capped 41, screening k = 14 (`d4_fci_depth_cost.txt:2`).

## Item 2 — the budget, and the latent diagnostic under it (`d4_fci_depth_cost.txt`)

Default `DISCOVERY_TIME_BUDGET_S = 180` = `_AGENT_HARD_TIMEOUT_S` (900) −
`_REFUTATION_COMPUTE_BUDGET_S` (720), both in `src/api/routes/causal/_common.py`
(`d8_budget_constants.txt:2-3`; the defaults on lines 4-6, the pinning test on
line 7 — `TestDefaultsAreDerivedNotInvented`). Discovery runs before estimation
and refutation, so it may use only the headroom the graph has beyond the
refutation node's cooperative deadline.

The FCI latent diagnostic is ON by default for guided runs and, at 382.5 s
unlimited (`d3_budget_premise.txt:31`), costs more than twice the budget. Could a
conditioning-depth limit rescue it?

| Fact | Value | Source |
|---|---|---|
| FCI at `max_cond_vars = 1` on the capped frame | wall 356.7 s, converged, 100 edges, 44 bidirected pairs, estimand pair bidirected: **False** | `d4_fci_depth_cost.txt:12` |
| depth 2 / 3 | not completed: the probe was stopped by hand at 396 s total (other lanes queued on the box lock) | `d4_fci_depth_cost.txt:14` |

Finding: FCI's cost on this frame is not driven by the conditioning depth
(356.7 s at depth 1 vs 382.5 s unlimited), so no depth limit brings the
diagnostic under the budget. What shipped: the diagnostic falls under the same
budget in the runner (`_maybe_latent_diagnostic`): not started when the budget
is already spent (`ran=False`, reason recorded), otherwise bounded by the
remaining budget with a timeout reported as `ran=True, converged=False` — never
silently absent. On the real frame under production defaults the diagnostic is
therefore skipped and the response says so (item 5 below). Whether the real
dataset should pay ~6 min for it (a larger budget, or a smaller diagnostic frame)
is an owner decision recorded in the PR.

## Item 4 — independence test, measured (`indep_test_runs.txt`, `indep_test_runs_r2.txt`, `indep_test_runs_r3_synthetic.txt`, `d7_gsq_arm_stopped.txt`)

Arms run through the production node path (`run_indep_test.py`: pre-flight →
guided PC → bootstrap under the 180 s budget → gate; latent diagnostic OFF so the
arms compare the skeleton test alone; `gsq`/`chisq` need discrete data, so for
those arms the non-binary columns are quantile-binned to ≤ 10 levels; fisherz
runs on the unbinned frame). The real arms were first launched as one chain; the
`gsq` arm never returned from its first bootstrap resample fit and was stopped
by hand after the chain had held the box lock 63 min (`d7_gsq_arm_stopped.txt`); the
remaining arms were re-run ONE AT A TIME under a hard 300 s `timeout`.

| Arm | Result | Source |
|---|---|---|
| real / fisherz | wall 199.8 s, primary fit 8.4 s, **AUGMENT 0.797**, 16/20 resamples attempted and succeeded inside the budget (`budget_exhausted`), mean stability beyond the prior 0.746, 44.4 % of edges at stability ≥ 0.9, 20 covariates kept, T→Y absent from the ensemble | `indep_test_runs.txt:12` |
| real / gsq | primary fit converged (cycle-removal lines written 12:17:27), then the FIRST resample fit had not returned 3,224 s later; the python process had run 3,484 s (55 CPU-min) when stopped by hand — **did not complete** | `d7_gsq_arm_stopped.txt:3-4,7-8`, `indep_test_runs.txt:20` |
| real / chisq | **did not complete in 300 s** (hard timeout, rc = 124) | `indep_test_runs_r2.txt:1,6` |
| synthetic / fisherz (planted patient_journeys, n = 4,000, k = 10) | wall 1.9 s, **ACCEPT 0.818**, 20/20 resamples, T→Y drawn, `disease_severity` = confounder, `academic_hcp` = instrument | `indep_test_runs_r3_synthetic.txt:4` |
| synthetic / gsq | wall 152.4 s: primary fit 0.1 s but the first resample consumed the budget — 1/20 resamples, **REJECT** (uncorroborated single run); `disease_severity` = confounder, `academic_hcp` = unrelated | `indep_test_runs_r3_synthetic.txt:9,11` |
| synthetic / chisq | wall 110.0 s, 1/20 resamples, **REJECT** (uncorroborated); `disease_severity` = ancestor (NOT recovered as confounder), `academic_hcp` = unrelated | `indep_test_runs_r3_synthetic.txt:16,18` |

**Verdict: fisherz, for both frame types.** On the capped real frame neither
discrete test completes a single bootstrap resample inside any budget the agent
can afford (gsq: > 53 min on one resample fit — 3,224 s, in a process that had run 3,484 s = 58 min; chisq: > 300 s), and on the synthetic
control they cost 50–80× fisherz's wall while losing corroboration (1 resample)
and, for chisq, the planted confounder role. Production keeps the wrapper's
auto-selection (fisherz on these frames); the `discovery_indep_test` override
stays as the measurement instrument. Two mechanisms behind the numbers, both
recorded so the pick is not mistaken for a tuning preference: (a) a bootstrap
resample duplicates rows, which for the contingency-table tests changes the
conditioning-set search enough that a resample fit costs orders of magnitude
more than the primary fit on the same frame (0.1 s → > 100 s on the synthetic
frame; 8.4 s primary vs > 53 min on the real frame); (b) a resample fit could
not be interrupted from inside the process — the budget was predictive
(estimate-before-start) and could not see a fit slower than its predecessors.
Mechanism (b) is fixed in this PR after codex round 1 (finding 5): each
resample is now waited for only as long as the budget has left and an overrun
is abandoned and reported (`n_abandoned`), pinned by
`test_a_resample_that_outlives_the_budget_is_abandoned_not_awaited`.

## Item 5 — acceptance measurement (`acceptance_runs.txt`, `acceptance_runs_synthetic.txt`, `acceptance_runs_final.txt`)

Through the production node `GraphBuilderNode.execute` with defaults exactly as
wired (cap 20, budget 180 s, B = 20, min_resamples 10, latent diagnostic ON;
`run_acceptance.py`), Supabase/Redis blanked (the "NOT persisted" warning is the
script environment, not a defect). `acceptance_runs.txt` is the real frame on the
lane's code BEFORE the codex-round-1 fixes; `acceptance_runs_final.txt` is the
same run on the code that ships (strict top-k union, rank basis seeded with T
and Y, bounded resample wait) and is the table the acceptance is judged on.

| Run | Result | Source |
|---|---|---|
| real / persistence_g28, pre-fix code | node wall 504.6 s, discovery 202.7 s, **AUGMENT 0.793**, 13/20 resamples in 176.4 s (budget exhausted), latent diagnostic timed out with 3.6 s left (reported), pre-flight kept 20 / constant 0 / collinear 16 / capped 41 (k = 14), shipped DAG 177 edges over 79 nodes, adjustment set 77 = every declared covariate, estimand edge shipped with provenance `required_prior` | `acceptance_runs.txt:16-19` |
| real / persistence_g28, final code (6ee05be5b: strict union, seeded basis, bounded resample) | node wall 502.2 s, discovery 185.9 s, **AUGMENT 0.793**, 14/20 resamples in 174.0 s, 0 abandoned, pre-flight kept 20 / collinear 16 / capped 41 (k = 14, `protected_capped` 0), shipped 176 edges / 79 nodes, adjustment set 77 of 77, provenance `required_prior`; **phase timers: `_find_adjustment_sets` 316.2 s** of the 502.2 s | `acceptance_runs_final.txt:16` |
| real / persistence_g28, final code + isolated-node fix (5a08bb89d) | node wall **424.0 s** (RUN line, measured by the script), discovery 182.0 s, **AUGMENT 0.797**, 16/20 resamples in 171.2 s, 0 abandoned, latent diagnostic timed out with 8.8 s left (reported), shipped 176 edges / 79 nodes, adjustment set 77 of 77, provenance `required_prior`; phase timers: `_find_adjustment_sets` **241.9 s**. The script itself exited rc = 124 at the 600 s hard timeout AFTER printing the RUN and WARNINGS lines: the timed-out FCI diagnostic's worker thread (a ~380 s fit, `d3_budget_premise.txt:31`) kept the interpreter alive at exit — see the note below | `acceptance_runs_final2.txt:16,21` |
| synthetic / patient_journeys, pre-fix code | wall 1.9 s, **ACCEPT 0.818**, 20/20 resamples, latent diagnostic ran and converged (no flag), 6 edges / 12 nodes, adjustment set 10 of 10, `disease_severity` = confounder, `academic_hcp` = instrument on the shipped DAG | `acceptance_runs_synthetic.txt:6` |
| synthetic / patient_journeys, final code | wall 2.3 s, **ACCEPT 0.818**, 20/20 resamples, 0 abandoned, latent diagnostic converged (no flag), 6 edges / 12 nodes, adjustment set 10 of 10, `disease_severity` = confounder, `academic_hcp` = instrument — unchanged | `acceptance_runs_final.txt:39` |

The codex-round-1 fixes did not move the real-frame verdict (AUGMENT 0.793 →
0.797; the strict union still selects 20 covariates at k = 14 on this frame, and
one to three more resample fits land in the budget).

**Where the node wall beyond discovery goes, and whose it is.** The phase timers
put 316 s (then 242 s) of the node wall in `_find_adjustment_sets`. My first
reading — that the 57 covariates the pre-flight keeps from the learner return as
isolated nodes and inflate the search — was only half right: it is true on the
ACCEPT path (the shipped DAG is the ensemble plus the isolated add-backs), and
the search now skips degree-0 nodes (commit 5a08bb89d,
`TestIsolatedNodesAreNotBackdoorCandidates`, a correct and cheap exclusion). But
the real frame's path is AUGMENT, whose shipped DAG is the MANUAL construction
(every declared covariate drawn as a common cause, 155 edges) plus the
corroborated extras — no isolated nodes — so on that path the exclusion is a
no-op and the 316 s → 242 s difference between the two runs is run-to-run
variance on a shared box, not the fix's effect; what the timers measure there
is the pre-existing minimal-set search: on a
manual DAG with 77 independent confounders no set of size ≤ 3 can d-separate T
and Y, so the search exhausts C(77,0) + C(77,1) + C(77,2) + C(77,3) = 76,154
criterion checks (the empty set included) before its documented fallback to
the full candidate set (one more check). That cost
exists on every manual-DAG path (REJECT / REVIEW / AUGMENT) for any wide claims
frame — Lane A's post-prune 61-covariate frame would pay C(61, ≤ 3) = 37,882 checks —
and it is not this lane's code, so it is reported, not changed: **owner /
follow-up item**, with the number, in the PR body.

**The timed-out latent diagnostic keeps computing.** When the FCI diagnostic
times out under the budget it is reported honestly (`ran = True, converged =
False`, seconds left), but its worker thread cannot be cancelled and runs the
fit to completion — ~380 s on this frame — in the background. In the
acceptance script that thread kept the interpreter alive past the 600 s hard
timeout (rc = 124 after the results were printed); in the API worker it costs
one CPU for ~6 min per query on this dataset. That is the same contract as the
per-algorithm timeout, but on the real frame it is paid on every guided query
under production defaults, which is why the latent-diagnostic default is an
owner decision (PR body, decision 2: accept the skip and its background cost /
turn the diagnostic off for this dataset / raise the budget).

Spec acceptance, real frame: a gate decision inside the agent timeout with at
least 10 resamples — met; the shipped adjustment set contains every declared
covariate — met (77 of 77 declared, none missing); the response names what was
pruned and capped — met (the pre-flight warning line names all 16 collinear and
every capped covariate). Synthetic planted frame: unchanged — ACCEPT,
`disease_severity` recovered as confounder (and `academic_hcp` as instrument)
on the shipped DAG.

Two facts for the owner, not defects of the lane: the estimand edge is absent
from every guided ensemble on the real frame (item 3; the shipped DAG carries it
with provenance `required_prior` and the warning says so), and the FCI latent
diagnostic cannot run under the 180 s budget on this frame (item 2; reported as
a timeout with the seconds left, never silently absent).

## Reproduce

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane-d-guided-discovery-claims
PY=/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python
L=/tmp/claude-1000/e2i_box_heavy.lock
D=docs/demos/results/2026-09-22_lane_d_guided_discovery_claims
# one arm per invocation, each under a HARD timeout (a gsq/chisq resample fit on
# the real frame does not return; see item 4)
for t in fisherz gsq chisq; do flock $L timeout 300 $PY - real $t < $D/run_indep_test.py; done
for t in fisherz gsq chisq; do flock $L timeout 300 $PY - synthetic $t < $D/run_indep_test.py; done
flock $L timeout 600 $PY - real < $D/run_acceptance.py
flock $L timeout 600 $PY - synthetic < $D/run_acceptance.py
```

The D0/D1/D3/D4 probes were heredocs; their full text is the captured file's
header comment where one exists (`d3_budget_premise.txt`, `d4_fci_depth_cost.txt`)
and otherwise the numbers are reproducible from `frame_resolver.py` +
`src/causal_engine/discovery/preflight.py` on the same parquet.
