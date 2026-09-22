# Live probe — `auto_discover: true` on the real `optum_biologic_persistence` frame (2026-09-22)

**Verdict: FAILED TO COMPLETE.** Guided discovery on the real frame (n = 15,209) blocked the API worker in the backdoor adjustment-set search after PC; gunicorn aborted the worker (code 134, its `--timeout 120` heartbeat) about 67 s after the discovery gate returned AUGMENT and ~4 min after PC itself finished; the job is orphaned in `running` with no failure surfaced to the caller. Tracked as issue #2233. Run once, not re-run (per the brief; the container did not flip, so there was no environment cause to retry against). The four default-path outcomes in `../cert.md` (discovery off) are unaffected — the discovery-off default is what let the cert complete.

## The run

| item | value |
|---|---|
| request | `{"treatment_var": "treatment_dupixent", "outcome_var": "discontinued_180d", "dataset": "optum_biologic_persistence", "limit": 20000, "auto_discover": true}` |
| analysis_id / request id | `b0cf6946-ecd8-41ab-8aee-e123fd731dc8` / `req:e4f530b9` |
| submitted | 2026-09-22T22:13:40Z; submit warnings = `['Analysis submitted; poll GET /causal/agent-analyze/{id} for the result.']` (no discovery-off warning: the explicit `true` was honored) |
| poll | 65 polls, cap hit at 1005.7 s (22:30:25Z), last status `running`; re-polled at 22:33:25Z: still `running`, `latency_ms = 0`, `dag.nodes = []` |
| container | `e2i-api:c0860bbf4`, `StartedAt 2026-09-22T21:22:09.275118978Z` — identical in the response's `_cert.container_started_at` and in `docker inspect` at 22:32:53Z. **No flip during the probe.** e2i_api at 0.58 % CPU / 867 MiB afterwards: nothing is still computing for this job |
| script | `../run_cert.py --probe-discovery` (log `run_probe.log`, exit 2 = poll cap) |

## Timeline from `docker logs e2i_api` (excerpt in `docker_logs_excerpt.txt`; full abort dump in `worker_abort_faulthandler_dump.txt`)

```
22:13:50 [WARNING] src.api.routes.causal.loaders - causal loader: dropping numerically collinear covariate(s) [16 names] for dataset 'optum_biologic_persistence' brand=None (collinear at machine precision with the intercept and earlier resolved columns; design rank 61 of 77) [req:e4f530b9]
22:13:50 [WARNING] src.api.middleware.timing - Slow request: POST /api/causal/agent-analyze took 10196.64ms (threshold: 1000.0ms) [req:e4f530b9]
22:13:50 [INFO] ..._impact.nodes.graph_builder - Auto-discovery enabled, attempting structure learning [req:e4f530b9]
22:13:55 [INFO] ...sal_engine.discovery.runner - Starting causal discovery with 1 algorithms: ['pc'] [req:e4f530b9]
         (18 × "Removing cycle edge" warnings from the PC result, 22:13:55–22:14:09)
22:16:42 [WARNING] ...sal_engine.discovery.runner - Bootstrap stopped by the time budget (180s): 11/20 resamples attempted, 11 succeeded
22:16:55 [WARNING] ...sal_engine.discovery.runner - Latent-confounding diagnostic (FCI) timed out after 12.1s (discovery time budget 180s) [req:e4f530b9]
22:16:55 [INFO] ...ine.discovery.observability - Ensemble complete: 74 edges, 71.74% agreement in 180.01s [req:e4f530b9]
22:16:55 [INFO] ...sal_engine.discovery.runner - Causal discovery complete: 74 edges found in 180.01s [req:e4f530b9]
22:16:55 [INFO] ...ausal_engine.discovery.gate - Gate decision: augment (confidence: 77.40%, edges: 74, high-conf: 29) [req:e4f530b9]
22:16:55 [INFO] ..._impact.nodes.graph_builder - Discovery complete: 74 edges, gate decision: augment [req:e4f530b9]
22:16:55 [INFO] ..._impact.nodes.graph_builder - Augmenting manual DAG with discovered edges (AUGMENT) [req:e4f530b9]
[2026-09-22 22:18:03 +0000] [1] [CRITICAL] WORKER TIMEOUT (pid:104)
Current thread 0x000075a11f250b80 (most recent call first):
  File ".../networkx/algorithms/d_separation.py", line 293 in is_d_separator
  File "/app/src/agents/causal_impact/nodes/graph_builder.py", line 747 in _satisfies_backdoor_criterion
  File "/app/src/agents/causal_impact/nodes/graph_builder.py", line 689 in _find_adjustment_sets
  File "/app/src/agents/causal_impact/nodes/graph_builder.py", line 304 in execute
  File "/app/src/agents/causal_impact/nodes/graph_builder.py", line 1566 in build_causal_graph
  File "/app/src/agents/causal_impact/graph.py", line 170 in wrapper
  File ".../langgraph/_internal/_runnable.py", line 501 in ainvoke
  File "/usr/local/lib/python3.12/asyncio/runners.py", line 118 in run
  File ".../uvicorn/workers.py", line 107 in run
[2026-09-22 22:18:04 +0000] [1] [ERROR] Worker (pid:104) was sent code 134!
[2026-09-22 22:18:04 +0000] [924] [INFO] Booting worker with pid: 924
```

The same dump shows a second thread still inside `causallearn/.../FCI.py:980 removeByPossibleDsep ← fci_wrapper.py:104 discover ← runner.py:976 _run_latent_diagnostic` at 22:18:03 — 68 s after the runner logged that the FCI diagnostic "timed out after 12.1s". The timeout abandoned the future; the thread kept computing.

## What the evidence says (reasoned, each step tied to a line above)

1. **Discovery itself ran to a decision.** Guided PC (single algorithm, by design — see below) fit, 11/20 bootstrap resamples inside the 180 s budget (≥ `DISCOVERY_MIN_RESAMPLES = 10`, so corroborated), FCI diagnostic timed out, ensemble 74 edges at 71.74 % agreement, gate **AUGMENT** at 77.40 % confidence. This is NOT the PR #2203 "could not run: singular…" failure; the fixed runner did not need to fire.
2. **The kill happened in the graph builder's adjustment-set search on the augmented graph.** The current thread at the abort is `_find_adjustment_sets → _satisfies_backdoor_criterion → networkx.is_d_separator`, running on the worker's event loop (`asyncio.run → langgraph ainvoke → build_causal_graph`), not in the bounded compute executor. The curated DAG has 123 edges; AUGMENT added 74 discovered edges. Task 9 (`../preflight.md`) measured this exact search's cost profile — the d-separation pass "burns its 100,000-iteration cap … k=17: 16.8 s, k=77: 467 s" — which is why the default path stays on the curated DAG and completed in 279–362 s.
3. **Why 120 s and not the 900 s cap:** the search blocks the loop, so the uvicorn worker stops answering gunicorn's heartbeat; gunicorn (`--timeout 120`, from `docker inspect`'s Cmd) sends SIGABRT (code 134) at 22:18:03, 68 s after AUGMENT. The agent's `_AGENT_HARD_TIMEOUT_S = 900` and the refutation deadline are cooperative asyncio deadlines and cannot fire while the loop is blocked.
4. **Orphaned job.** The killed worker owned the running task; the replacement worker (pid 924, forked warm) knows nothing about it, and the cross-worker job store keeps the last-written `running` row. The caller sees `running` forever; no `failed`, no warning, `latency_ms 0`. A cooperative timeout would have written `failed`; a SIGABRT cannot.

## Fields grepped in the raw response (`raw_discontinued_180d.json`) — Lane D / Lane E / vote rule

| field / token | present? |
|---|---|
| `feature_role_panel` (Lane E) | absent |
| `anchored_confounders` (Lane E channel) | absent |
| `estimand` / estimand-edge warning (Lane D) | absent |
| `uncorroborated` (Lane D) | absent |
| `pre_flight` / `preflight` (Lane D) | absent |
| `vote` / `voter` (vote rule) | absent |
| `GES` | absent (`ges` matches only inside the word `edges`) |
| `pc` | absent |
| `dag` | `nodes: [], edges: []`, `dag_source: "domain_knowledge"` (the pending placeholder), `discovered_confounders: []`, `refutation: null` |

**None of the Lane D / Lane E / vote-rule fields can be read from this path**: the response is the pending handle frozen by the worker death. The only live evidence of those lanes on the real frame is in the container log (guided PC ran, bootstrap corroborated, AUGMENT decided) — it never reached a response. Their acceptance evidence stays in their own evidence dirs (`2026-09-22_lane_d_guided_discovery_claims/`, `2026-09-22_lane_e_feature_role_voters/`, `2026-09-22_discovery_vote_rule/`).

**Observation on "1 algorithms: ['pc']" vs the `[GES, PC]` default.** `DEFAULT_DISCOVERY_ALGORITHMS = (GES, PC)` (`src/causal_engine/discovery/base.py`) is the ensemble default; the API submits `discovery_guided: True`, and guided discovery is single-algorithm PC with 20 bootstrap resamples, a 20-covariate cap and a 180 s budget **by design** (`graph_builder.py` constants, Lane D). So the vote rule's distinct-voter agreement is not what corroborated this run; resample stability (11/20) was. Not a defect; recorded so nobody reads "1 algorithm" as the ensemble default having been lost.

## Consequences and follow-ups (not fixed here) — tracked as issue #2233 (part of, not closed by this evidence)

- The default-path cert stands: `auto_discover` defaults to `false` for this dataset (`_default_auto_discover → False`, asserted in-process on c0860bbf4) and every submit warned so, verbatim (`../cert.md`).
- Follow-up A (blocking for any discovery-on run on this frame): the post-AUGMENT `_find_adjustment_sets` must run off the event loop under a bound (it already has a cap concept; the cap did not save a 197-edge graph), or the augmented graph must be pruned before the search.
- Follow-up B: a worker death must surface as `failed` on the job (arbiter-side or a heartbeat-stale sweep); today it is a permanent `running`.
- Follow-up C: the FCI diagnostic's thread outlives its timeout and keeps the CPU (the dump shows it 68 s later).
- Not re-run: one run was the brief's cap, the container never flipped, and a second identical run would only orphan a second job.
