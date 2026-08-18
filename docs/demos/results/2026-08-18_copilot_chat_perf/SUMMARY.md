# Copilot Chat Performance Test — 2026-08-18

Full 51-turn V2 eval (`docs/demos/COPILOT_CHAT_DEMO_SCENARIOS_V2.md`) against prod, compared
against the 2026-08-15 baseline (`docs/demos/results/2026-08-15_copilot_chat_perf/`).

Purpose: measure what waves 9–14 bought — the two 08-15-FAIL fixes (#1634 router agent budgets,
#1635 experiment_designer retry), the five grading-derived fixes (#1636 classifier-JSON leak,
#1637 KPI-name fallback, #1638 agent-roster SSOT, #1639 experiment-designer bounds, #1640 TRx
substrate fence), the AG-UI stream-health/TTFB chain (#1658–#1673), and the memory/RAG adapter
fixes (#1681/#1683/#1685).

Surface: `copilot_agui_runner.py` → `POST /api/copilotkit/agent/default` (AG-UI), same brain and
protocol as the 08-11 and 08-15 baselines. Container running the #1686 deploy (merge `1a17b73e7`),
up ~10 h, smoke-exercised; nothing src-touching merged since (`7a8f9f896` is workflow-only).

**Environment confounds, measured at run start (13:14 UTC):**
- **ChEMBL/EBI fully down** (molecule endpoint timeout, mechanism endpoint HTTP 500 — the #1688
  outage). Contained: only the clinical-context *mechanism* fragment depends on ChEMBL; graders
  verified it degraded to a labelled `static_fallback` on 3.1 / 6.4 / A.9-followup and never
  masqueraded as live. No grade was driven by it.
- **hcp_adoption champion models demoted** — discovered *by* this run; see the platform
  regression below. This one is NOT contained: it drives the run's only FAIL and several PARTIALs.

## Headline

**51 ok / 0 failed / 0 empty / 0 non-200.** Total wall 983.3 s → **1,113.4 s** (+130.1 s; mildly
slower across the board, no single-turn blowup — see Latency).

**Grades: 37 PASS / 13 PARTIAL / 1 FAIL** (baseline: 38 / 11 / 2). Totals are near-flat;
composition moved substantially in both directions:

- **Both 08-15 FAILs are fixed and verified at the payload layer** (5.1 and 3.4, below).
- **The one new FAIL (3.3) and most new PARTIALs share a single root cause outside the chat
  code**: the champion-model registry lost its production champions on Monday 03:05 — a design
  conflict between the reseed pipeline and the serving path, fully root-caused below.

**Zero hallucinations again.** Graders traced every numeric claim to an `on_tool_end`/event
payload; several values corroborated outside the run (1.1/A.8's TRx 10,273 vs payload
`data_through 2026-08-17`; 5.2's 22-agent roster verified against `src/agents/factory.py`).

| subset | PASS | PARTIAL | FAIL | baseline (08-15) |
|---|---|---|---|---|
| Narratives 1–2 (14) | 9 | 5 | 0 | 10 / 4 / 0 |
| Narratives 3–4 (13) | 8 | 4 | 1 (3.3) | 11 / 1 / 1 (3.4) |
| Narratives 5–6 (12) | 10 | 2 | 0 | 8 / 3 / 1 (5.1) |
| Appendix A (12) | 10 | 2 | 0 | 9 / 3 / 0 |
| **Total (51)** | **37** | **13** | **1** | **38 / 11 / 2** |

## Verified fixes (each checked on the exact numbers that were broken)

| fix | evidence |
|---|---|
| **5.1 (#1634)** — 08-15 FAIL | routes to `e2i_data_query_tool`, returns the real 10-row per-brand health_score log (every cell payload-verified, confidence 0.730–0.989) plus the correct "no single unified score" framing. 298 → 1,809 chars. |
| **3.4 (#1635)** — 08-15 FAIL | genuinely succeeded: `status: completed`, `fallback: false`, a complete **instrumental-variable** design (randomize invitation, instrument attendance) that correctly resolves the speaker-program self-selection confound. Not a longer apology. |
| **3.6 (#1639 bounds)** | the absurd numbers are gone at the payload layer: `duration_estimate_days` 94,115 → **784**; `required_sample_size` 672,206 → **5,582**; the MDE-vs-expected-effect contradiction is gone. Residuals flagged (below). |
| **2.1 (#1636 classifier leak)** | the raw classifier blob is gone; opens with prose, zero code fences; severity tiers and lines-of-therapy each sum exactly to the payload's 2,341. |
| **5.2 (#1638 roster SSOT)** | answers from its own system prompt: **22/22 agent names, correct count, correct tiers**, verified against `AGENT_REGISTRY_CONFIG`. Baseline named 2 agents, substituted tool names, and stated the wrong count. |
| **A.9-followup (template bloat)** | all seven 08-15 template markers absent by scan; answer is Kisqali/Northeast/Q1-specific from the title line; quantified causal paths are back (competitor_activity → trx_market_share −0.073, ~−$24,218, 70-day lag, conf 0.793). |
| **1.7 / A.2b / A.8 (#1640 substrate fence)** | the `measure_basis` fence **fires and is surfaced**: answers now label KPI-ledger vs episodic-corpus substrates and decline to mix them (A.2b explains the 73× gap and attributes it to the tool). Note: the fence *labels* the incoherence; the underlying scale gap itself still exists (below). |
| **4.6 (WS2-TR-005)** | False Alert Rate now **resolves from chat** — both asked-for metrics answered; settles the 08-15 finding that the alias map lacked it. (New window-attribution flags, below.) |
| **4.5 (window fix persists)** | `window_requested == window_applied == 2026-08-01..2026-09-01`, `window_status: applied`, stated in prose. Third consecutive run green. |
| **6.4 ("moved off clinical_context") — REFUTED** | grader disproved the CSV's premise: 6.4 called the **same four tools as baseline** (kpi, causal, experiments, clinical_context) — the CSV's `agents_dispatched` column under-recorded. %PNH-tested is still correctly refused; the coverage warning is quoted verbatim from the payload. |

## THE new platform regression — hcp_adoption champions demoted (root-caused, verified live)

**Symptom in the run**: `predict_hcp_segment_likelihood_tool` fails closed on **all three brands**
("no production champion registered"); the failure string appears in **13 turns'** event payloads
and materially degrades ≥6 turns: **3.3 FAIL** (Fabhalta ranking gone), **1.6 PARTIAL** (Kisqali),
**2.5 PARTIAL** (Remibrutinib, both cuts), and thinner substitute content in 2.6, 6.1, 6.2. On
08-15 all of these served full propensity rankings (e.g. 6.1: oncology 56.9 % n=1,662, AUC 0.791).

**Root cause (every link measured, not inferred):**

1. `resolve_hcp_adoption_champion` (`src/services/hcp_segment_likelihood.py:205`) requires a
   `ml_model_registry` row with `stage='production' AND is_champion AND artifact_path AND NOT
   is_synthetic` — "the honesty gate: never serve a staging model."
2. `cohort_deployer.register_cohort_model` (`src/mlops/gold_standard_eval/cohort_deployer.py`)
   **by documented design registers gold-standard models at `stage='staging'`, `is_champion=False`
   and hard-refuses `stage='production'`** — because a second production row would make the
   orchestrator resolver's serving ensemble return both models.
3. Live table today: all three `hcp_adoption_*_goldstd_lr_v1` rows sit at
   `staging / is_champion=f / artifact-backed / non-synthetic` — **re-registered
   2026-08-17 03:04–03:05 UTC**, while `promoted_at` still reads **2026-08-12 01:23:37** (all
   three the same second — a scripted/ad-hoc promotion; no promotion script exists in the repo
   and no commit references one).
4. The 03:05 Monday timestamps match the host cron `0 3 * * 1 reseed_synthetic.sh` (the known
   Mon-3AM reseed guardrail), whose `stage_goldstd_retrain` → `scripts/retrain_goldstd.sh`
   retrains the 12 gold-standard models on the fresh substrate and re-registers them — at
   staging, per the deployer's contract. Reseed log: all stages OK, done 03:07:41.

**So**: someone promoted the three rows out-of-band on 08-12 (which is what made the 08-11/08-15
evals serve propensity tables), and the next weekly reseed correctly — *per the deployer's own
contract* — wiped that promotion. **This is a design conflict, not a wave-9–14 regression**: the
serving path demands the exact registry state the retrain pipeline is designed to refuse.
It will recur every Monday until the contract conflict is resolved (serve staging goldstd rows
behind the same artifact/synthetic checks? auto-re-promote post-retrain after gates pass? keep
fail-closed and drop the capability?) — that decision needs the dispatcher's
`_probe_prediction_champions` collision constraint in the room.

## New findings from grading (not present or not visible on 08-15)

1. **False-superlative prose defect, three instances in one subset (5.1, 5.6, 5.7)**: a min/max
   claim asserted in the concluding prose contradicts the (exact, payload-faithful) table the same
   answer just printed. Worst: 5.7 crowns 0.198 "largest effect size" directly under a table
   listing 0.267 (−35 % off). Retrieval and tables are right; the summary layer doesn't check
   itself against them. This is what turned 5.7 PARTIAL/regressed.
2. **1.5 provenance misstatement (PARTIAL/regressed)**: with no tool call, the answer recalled
   the prior turn's causal payload correctly but described it as "causal modeling on **real**
   patient-journey data" while the payload says `data_source: "synthetic"`, and invented
   "(DoWhy/EconML-style)" specificity (payload: `backdoor.linear_regression`). Numbers right,
   provenance wrong — the exact inversion of the platform's honesty labelling.
3. **4.4 axis conflation (PARTIAL/regressed)**: silently swaps patient-severity segments in for
   the asked-about **HCP** segments; baseline had refused the premise explicitly.
4. **4.7 narrowed query lost the on-point evidence**: queried `market_share` instead of the
   trigger-lift KPI, so the one validated trigger_accepted → treatment_initiated chain (+0.059,
   $21,479) the baseline cited is absent, and the headline overclaims trigger attribution.
5. **4.1 scope label dropped (PARTIAL/regressed)**: did *not* assume a brand (all-brands profile,
   correctly labelled in the payload) but dropped the payload's own "all brands" label from the
   prose, so 3,428 HCPs / 37,006 TRx read as Northeast-oncologist figures. Also flagged: the
   "Northeast" cohort is 69 % of the whole 5,000-HCP universe — region filter possibly not
   applied; needs a cross-run check.
6. **4.6 window attribution**: false-alert payload returns `data_through: null` /
   `reporting_window: null`, yet the answer asserts a 30-day window for both metrics, and claims
   "same trailing-30-day window as 4.5" when 4.5 ran on the applied **calendar-August** window.
7. **Scale gap behind the #1640 fence is real and unreconciled**: regional monthly TRx
   (South 238,449) vs national 30-day TRx (10,273) — the fence now *labels* the substrates
   honestly, but ~20× regional>national remains unexplained by any measure_basis note (1.7,
   A.2b: table range "207,270–238,449" also excludes its own midwest row 182,115).
8. **5.5 over-abstention, third consecutive run** — and the scenario doc's line 458 ("4.3/5.5/5.7
   do not clarify… no over-abstention", the #1407 measured-clean state) is stale; correct or
   re-measure it.
9. **3.6 residual implausibilities** (bounds fix holds, plausibility doesn't): required n=5,582
   HCPs exceeds the platform's entire 5,000-HCP universe; `duration_estimate_days: 784` sits
   unreconciled next to the design's own 3+6-month field period.
10. **A.10 unearned region scoping (PARTIAL but improved)**: table headed "(Northeast)" over
    paths that are provably region-insensitive (A.9-seed got identical paths with `region: null`),
    and the payload's own $520,809 gap analysis went unused. The 08-15 256× contradiction is gone.

## Latency

+130 s total, spread thin (median turn +2–3 s; no new outlier). TTFBs are similar-to-slightly-up
vs 08-15 — the wave-11 gzip/keepalive work shows no measurable TTFB gain on this surface at n=1.
3.4/3.6 still burn ~165 s each (the validity-audit sub-step still times out inside an otherwise
successful design — handled honestly in the answer). Not worth chasing until n>1 characterises
run-to-run variance.

## Caveats

- **n=1 per turn**, same as every prior run; ±6–8 s same-tool deltas are noise.
- **Champion-registry outage contaminates the vs-baseline comparison on ≥6 turns** — those
  regressions measure the registry state, not waves 9–14. Re-run narratives 1–3 + 6 after the
  registry question is settled before reading those turns as chat regressions.
- **ChEMBL outage (#1688)**: measured, labelled, contained to mechanism fragments; no grade driven
  by it. Re-probe 3.1/6.4/A.9-followup after recovery if exactness matters.

## Recommended follow-ups

1. **File + decide: champion-registry design conflict** (the Monday demotion loop) — the only
   P1 here; recurs weekly.
2. **File: false-superlative summary-layer defect** (5.1/5.6/5.7) — cheap to fix (have the
   synthesis check min/max claims against its own table), high demo visibility.
3. 1.5 provenance misstatement + unlabelled in-session recall (no-tool turns should restate
   source or re-query).
4. Fix stale `COPILOT_CHAT_DEMO_SCENARIOS_V2.md:458` (5.5 has clarified three runs straight).
5. 4.1 region-filter suspicion: verify cohort_profiler actually applies region.
6. On ChEMBL recovery: rerun nightly job A (#1688 plan unchanged).

## Artifacts

| file | what |
|---|---|
| `raw_agui.jsonl` | 51 turns, full SSE event streams + `response_text` |
| `measurements_agui.csv` | per-turn ttfb / first-progress / total / tools (NB: `agents_dispatched` under-records multi-tool turns — see 6.4) |
| `grades_n1n2.json` … `grades_appendix.json` | four independent graders, per-question PASS/PARTIAL/FAIL + payload-traced evidence |
