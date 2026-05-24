# RCA: D6 Per-Pipeline Overhead

**Issue:** #482
**Date:** 2026-05-24
**Author:** Claude (Opus 4.7)
**Branch:** `perf/482-d6-overhead-rca`

## TL;DR

"D6" is **not a pipeline stage** — it is the design-decision label
**"Pre-flight always runs — no skip flag"** from the data-sufficiency rollout
plan. It refers to the `run_sufficiency_check` node in `data_preparer`.

The per-invocation cost is **~2.5 ms** on a representative n=10,000 binary
classification cohort. In the current CI integration lane, end-to-end
data_preparer graph invocations happen in **zero non-skipped tests** —
not because every such test is on the `--ignore` list (it isn't), but
because the tests that survive the `--ignore` list and the
default-runs-slow path skip at runtime when their data fixtures are missing
(`data/rwd/` is gitignored).

**Recommendation:** Close #482 as accept-as-cost. A trim of ~68 % of D6's
per-call cost via `lru_cache` on `_z_scores` is shown below as
empirically verified, but the savings are at micro-second scale per
production cohort and do not move the CI integration lane (which already
doesn't exercise the graph end-to-end).

## What is D6?

D6 is the sixth row of the **"Locked decisions"** table in the
data-sufficiency rollout plan:

> `docs/superpowers/plans/2026-05-22-data-sufficiency-diagnostics-rollout.md:35`
>
> | D6 | Pre-flight always runs — no skip flag (cheap formula evaluation; exercises invariants). … | DONE in PR #462 + PR #462 hotfix |

The decision is specifically about the **invariant** (no skip flag; audit
semantics for the SKIPPED / INCONCLUSIVE verdicts). Implementation
behaviors inside the sufficiency node are not "D6" per se; they implement
the D6 contract.

In code, D6 is implemented by `run_sufficiency_check`:

- **Definition:** `src/agents/ml_foundation/data_preparer/nodes/sufficiency_check.py:88`
- **Wiring:** `src/agents/ml_foundation/data_preparer/graph.py:269` adds the
  node; `graph.py:333–334` wires it between `compute_baseline_metrics`
  and `kg_role_enrichment`.
- **References to "D6":** lines 102, 106, 110, 792, 904 in
  `sufficiency_check.py` — all are comments documenting how the hotfix
  preserved the D6 invariant.

The original concern (issue #473 §"Proposed fixes" item 3, restated in
PR #476's follow-up note):

> *"The data-sufficiency pre-flight (run_sufficiency_check) now runs on
> every data_preparer pipeline pass (D6, PR #462/#472). Confirm whether
> it added measurable per-pipeline overhead in the integration tests
> that exercise the data_preparer graph; gate/optimize if material."*

## Where it shows up

### CI lane (where the perf concern was raised)

`.github/workflows/backend-tests.yml:353–378` defines the
`integration-tests` step. The relevant facts:

1. It runs `pytest tests/integration/ --ignore=...` with a fixed `--ignore`
   list (lines 356–372).
2. **There is no `-m "not slow"` filter.** `slow` is registered as a
   marker (`pyproject.toml:206`) but not excluded by default
   (`pyproject.toml:226` `addopts = "-v --tb=short -n 4 --dist=loadscope"`).
3. The lane is sharded 2-way via `pytest-split` (`--splits 2`).

Graph-invoking integration tests, audited against the actual CI behavior:

| Test file | Invokes `data_preparer` graph end-to-end? | In `--ignore` list? | Slow-marked? | Runs in CI? | Notes |
|---|---|---|---|---|---|
| `tests/integration/test_agents/test_data_preparer/test_data_preparer_pipeline.py` | Yes (3 `.ainvoke()` sites) | **Yes** (`--ignore=tests/integration/test_agents/test_data_preparer/`) | No | **No** | excluded by directory ignore |
| `tests/integration/test_tier0_e2e.py` | Indirectly via tier0 runner subprocess | **Yes** (`--ignore=tests/integration/test_tier0_e2e.py`) | 2 slow marks | **No** | excluded by file ignore |
| `tests/integration/test_csu_full_data_preparer_e2e.py` | Yes (`DataPreparerAgent().run(...)` via `_run_pipeline` at L209–235) | No | **5 slow marks** (but slow runs in CI) | **Skipped at runtime** | fixture `csu_data_source` requires `data/rwd/csu/e2i_ml_v3_patient_journeys.json`; `data/rwd/` is gitignored, so the file is absent on GH runners → `pytest.skip(f"CSU journeys file not present at {CSU_JOURNEYS_PATH}")` at L61–62 of the fixture |
| `tests/integration/test_csu_val_auc_measurement.py` | Yes, transitively — launches `scripts/run_tier0_test.py` whose step-2 instantiates `DataPreparerAgent` and calls `agent.run(...)` (`scripts/run_tier0_test.py:2082–2205`) | No | 8 slow marks | **Skipped in CI** (fixture absence at L109–110); would invoke D6 if real CSU data were present | |
| `tests/integration/test_tier0_runner_csu_raw_json.py` | **No** — directly calls `_load_from_files` and `transform_data` (subset of data_preparer node functions); does NOT instantiate `DataPreparerAgent` or invoke the compiled graph, so `run_sufficiency_check` is NOT executed | No | 2 slow marks | Yes, but does not exercise D6 — corrected from earlier draft of this RCA, which mis-described this test as hitting the SKIPPED branch | |
| `tests/integration/test_falkordb_role_persistence.py` | No — constructs graph for structure assertions (walks edges at L314,346); does NOT `.invoke()` it | No | No | Yes; graph construction only | |
| `tests/integration/test_agents/test_state_checkpoint_replay.py` | No — its `graph.ainvoke()` calls (L583, L650) build **custom test graphs** (`StateGraph(ModelTrainerState)` with `node_a`/`node_b`/etc.), NOT the data_preparer graph | No (NOT in `--ignore`) | No | Yes, but doesn't touch D6 | corrected from earlier draft of this RCA, which mis-stated the ignore status |

**Net for the CI integration lane:**
- Tests that would invoke D6 via the data_preparer graph end-to-end are
  either explicitly excluded (`test_agents/test_data_preparer/`,
  `test_tier0_e2e.py`) or runtime-skipped because their data fixtures
  are missing from CI checkouts (`test_csu_full_data_preparer_e2e.py`,
  `test_csu_val_auc_measurement.py`).
- **No surviving CI integration test invokes `run_sufficiency_check` on
  the data_preparer graph end-to-end.** D6's cumulative wall-clock
  contribution to the CI integration lane is effectively zero.

### Slow lane / nightly / local dev

The fully-exercised paths (`test_csu_full_data_preparer_e2e.py`, the
slow-marked CSU tests, the ignored `test_agents/test_data_preparer/`
suite when run locally) do call `run_sufficiency_check` with real data.
There D6 contributes ~2.5 ms per pipeline pass against a documented
~30 s pipeline budget (per `test_csu_full_data_preparer_e2e.py:21`
docstring) — ~0.008 %. Undetectable.

### Production

D6 runs on every real-cohort data_preparer pass in production: ~2.5 ms
per cohort. Not a perf concern at human-scale cohort throughput.

## Profile results

### Reproducible setup

The numbers below come from this script run on this branch in this
worktree's venv:

```python
# /tmp/profile_d6.py
import asyncio, time, cProfile, pstats, io
import pandas as pd, numpy as np
from src.agents.ml_foundation.data_preparer.nodes.sufficiency_check import run_sufficiency_check

rng = np.random.RandomState(42)
df = pd.DataFrame(rng.randn(10000, 20), columns=[f'feat_{i}' for i in range(20)])
df['outcome'] = (rng.rand(10000) < 0.1).astype(int)
state = {
    'experiment_id': 'profile_test',
    'scope_spec': {'problem_type': 'binary_classification', 'prediction_target': 'outcome'},
    'train_df': df, 'target_rate': 0.1,
    'blocking_issues': [], 'power_warnings': [],
}
asyncio.run(run_sufficiency_check(state))  # warmup
N = 50
t0 = time.perf_counter()
for _ in range(N): asyncio.run(run_sufficiency_check(state))
print(f'avg: {(time.perf_counter()-t0)/N*1000:.2f} ms')

pr = cProfile.Profile(); pr.enable()
for _ in range(N): asyncio.run(run_sufficiency_check(state))
pr.disable()
pstats.Stats(pr).sort_stats('cumulative').print_stats(15)
```

Env: Python 3.12.3, scipy installed (per `.venv` import-time check), repo
at branch `perf/482-d6-overhead-rca` HEAD.

### Measured numbers (N=50 invocations, consistent counting)

| Metric | Value |
|---|---|
| Avg wall-clock per call | **2.48 ms** |
| `_z_scores` calls per invocation | **5** (250 over 50 iters) |
| Underlying `scipy.stats.norm.ppf` calls per invocation | **10** (500 over 50 iters; each `_z_scores` does 2 ppf) |
| `binary_outcome_power` calls per invocation | **3** (150 over 50 iters; `sensitivity_grid` evaluates 3 effect-size candidates) |
| `mde_for_sample_size` calls per invocation | **2** (100 over 50 iters; once in `_classify_classification`, once in `sensitivity_grid` directly) |
| `_z_scores` + downstream `ppf` cumulative share | **76 %** of D6 wall-clock |
| `_classify_classification` cumulative share | **82 %** |

Top hotspots (file:line):

1. `src/utils/power_analysis_lib.py:54` `_z_scores` — wraps
   `scipy.stats.norm.ppf` calls for `alpha` and `power`. Pure function
   of `(alpha, power)`.
2. `src/utils/power_analysis_lib.py:302` `sensitivity_grid` — calls
   `binary_outcome_power` for each of 3 candidates.
3. `src/utils/power_analysis_lib.py:238` `mde_for_sample_size` — direct
   closed-form formula (NOT bisection; an earlier draft of this RCA was
   incorrect on this point).
4. `src/utils/power_analysis_lib.py:90` `binary_outcome_power` — itself
   triggers a `_z_scores` call.

### Empirically measured `lru_cache` benefit

Setup: monkey-patch `power_analysis_lib._z_scores` with
`functools.lru_cache(maxsize=32)`, re-warmup, re-time (N=100):

| Configuration | avg per D6 call |
|---|---|
| baseline (no cache) | **1.71 ms** |
| `_z_scores` lru_cached | **0.54 ms** |
| **saved** | **1.17 ms (68 %)** |

The cache is effective because `(alpha, power)` is the same pair on
every call across a typical cohort run (default `(0.05, 0.80)`).

(Note: 1.71 ms baseline here vs 2.48 ms above reflects within-bench
noise on this dev box; the cache savings ratio is the reproducible
number.)

## Root cause(s)

Ranked, with evidence:

1. **The "per-pipeline overhead" concern in the parent issue was
   hypothetical, not measured.** PR #473 §3 inherited the assumption
   that adding a node to the data_preparer graph would slow down
   integration tests that exercise it. In practice, the tests that
   would exercise it end-to-end either don't run in CI (ignored
   directories / files) or skip at runtime (data fixtures absent
   from gitignored `data/rwd/`). No surviving CI integration test
   invokes the data_preparer graph end-to-end on the current main.

2. **D6's intrinsic per-call cost is 2.5 ms** — fundamentally bounded
   by 10 `scipy.stats.norm.ppf` lookups (via 5 `_z_scores` calls). The
   function IS pure formula evaluation as the rollout plan claims;
   there are no DB hits, network calls, fixture-setup overhead, or
   non-trivial recomputation inside D6 itself.

3. **The sharded integration lane's residual wall-clock imbalance**
   (shard 1 ≈ 6:41 vs shard 2 ≈ 14:00 on the latest main run) is the
   problem #480 tracks; D6 is not on its critical path.

## Recommendations

1. **Close #482 as accept-as-cost.** D6 contributes no measurable
   wall-clock to the CI integration lane (no surviving CI test invokes
   the data_preparer graph end-to-end). No fix moves the
   integration-lane needle. **Effort: S. Lane savings: 0 s.
   Risk: low** (the residual risk is that a future CI change un-ignores
   `test_agents/test_data_preparer/` or commits CSU/PNH/BC fixtures into
   `data/rwd/`, which would add ~2.5 ms × N invocations — still
   negligible against the 6:41–14:00 shard wall-clocks).

2. **(Optional, separate tracker) `lru_cache(maxsize=32)` on
   `power_analysis_lib._z_scores`.** Empirically measured 68 %
   reduction in D6 wall-clock (1.17 ms saved per call). Beneficial
   to any caller of `power_analysis_lib` in production cohort sweeps,
   not just D6. **Effort: S (one-line decorator + 1 unit test).
   Savings: 1.17 ms × N production calls. Risk: low** —
   `lru_cache` on a deterministic pure function of two floats is the
   canonical pattern; the only failure mode is if someone later makes
   `_z_scores` non-pure (closure over mutable state), which a unit
   test pinning cache behavior would catch. **This is NOT a #482 fix**
   because #482 was scoped to integration-test wall-clock; defer to a
   focused production-perf tracker if/when demand surfaces.

3. **(Optional) Make `sensitivity_grid` candidates configurable / lazy.**
   The hardcoded 3 candidates `[0.05, 0.10, 0.20]` at
   `sufficiency_check.py:444` drive ~50 % of D6's cost. If a future
   consumer never reads `sensitivity_grid`, it could be opt-in.
   **Effort: M (config surface + audit consumers). Risk: medium**
   (might surprise downstream report consumers). Defer unless
   production perf becomes the binding constraint.

## Decision

- [x] **Accept-as-cost. Close #482.**
- [ ] Surgical fix (PR `<link>`)
- [ ] Defer (file follow-up `#`)

Rationale: REASON-BEFORE-RULES applies — the issue asks "is there
overhead?" and the honest answer is "yes, ~2.5 ms per call, in zero
non-skipped CI invocations." Implementing recommendation #2
(`lru_cache`) would *legitimately* save ~1.2 ms per production cohort
call but would *NOT* address the wall-clock concern that motivated
#482 (integration-test runtime). Shipping it under the #482 banner
would conflate two distinct concerns. The right call is to close #482
with the empirical evidence above; if production-side power-analysis
perf becomes a real concern, file a focused issue and ship the cache
under that PR.

## What I couldn't determine

- **Could not run the actual sharded CI lane locally** — would require
  the full service-container stack (Supabase, Redis, MLflow, FalkorDB)
  not provisioned in this dev env, plus the gitignored CSU/PNH/BC data
  fixtures. Conclusions above come from
  (a) workflow YAML inspection (`grep --ignore`),
  (b) per-file inspection of `.ainvoke()` / `.invoke()` callsites,
  (c) per-file inspection of skip conditions and slow marks,
  (d) micro-profile of `run_sufficiency_check` in isolation.
  This is sufficient to conclude "D6 is not on the CI lane's hot path"
  but does not give a full integration-lane profile flame-graph. If
  the team wants higher confidence, run a one-shot `pytest --profile`
  on the actual CI matrix and grep for `run_sufficiency_check` in the
  output.
- **Did not profile the slow / nightly lane.** Out of scope per the
  issue ("integration-test wall-clock NOT solved by sharding").
- **Did not audit whether `_z_scores`'s float-key cache safety has any
  edge cases.** Standard `lru_cache` on `(float, float)` is robust for
  the actual call sites (`(0.05, 0.80)` is constant) but if a future
  call site passes `numpy.float64` instances the cache key may not hit;
  rec #2's implementation should include a unit test pinning the cache
  hit rate.

## References

- Issue: https://github.com/enunezvn/e2i_causal_analytics/issues/482
- Parent issue: https://github.com/enunezvn/e2i_causal_analytics/issues/473
- PR that introduced D6: #462 (data-sufficiency rollout Phase 1)
- PR that scoped this follow-up: #476 (CI infra-flake fixes + sharding)
- Sister residuals: #480 (shard rebalance), #481 (slow-tests partition)
- Rollout plan: `docs/superpowers/plans/2026-05-22-data-sufficiency-diagnostics-rollout.md`
- Implementation: `src/agents/ml_foundation/data_preparer/nodes/sufficiency_check.py:88`
- Wiring: `src/agents/ml_foundation/data_preparer/graph.py:269,333–334`
- CI workflow: `.github/workflows/backend-tests.yml:353–378`
- Pytest config (no slow exclusion): `pyproject.toml:201–226`
