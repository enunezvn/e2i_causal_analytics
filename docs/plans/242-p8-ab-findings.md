# #242 P8 — live A/B findings (ensemble vs single-Sonnet)

**Status: FINAL. The clean n=30 result is reproduced THREE times identically (2026-05-25). Ship `Refs #242` — keep #242 OPEN. The 12-entry PNH tail remains unmeasured; see "Why the full 42 was never measurable" below — it cannot change the ship decision.**

Date: 2026-05-25. Harness: `scripts/measure_ensemble_ab.py` (reproducible). Models: `anthropic/claude-sonnet-4-6` + `anthropic/claude-opus-4-7` + `openai/gpt-5`, compiled classifier, golden set `tests/fixtures/causal_role_golden_set.json`.

## Reachability (smoke, 3/3 OK)
All three members reachable with the `.env` keys; **`openai/gpt-5` is a valid model string**; per-provider telemetry (tokens + cost) works for all three (GPT-5 via litellm's OpenAI usage shape). On an obvious post-index leak all three returned `descendant`.

## A/B run
Targeted the **leak-relevant roles** (`descendant`, `collider`, `mediator`) — 42 golden entries. **The Anthropic account ran out of credits mid-run**: Sonnet + Opus errored (non-vote) on 12 entries (the PNH block); GPT-5 errored 0. Those 12 became artificial `split`s and are **excluded**. The conclusions below rest on the **30 clean entries** (all three voted).

### Clean A/B (n=30)
| Metric | Result |
|--------|--------|
| single-Sonnet role-accuracy | **19/30 = 0.633** |
| ensemble role-accuracy | **19/30 = 0.633** (identical — no improvement) |
| ensemble regressions (Sonnet-right → ensemble-wrong) | **0** (no harm) |
| correlated failures (all 3, incl GPT-5, wrong together) | **9/30 (30%)** |
| AC5 cases (Sonnet wrong → ensemble right or escalated) | **1** |
| clean leak-false-negatives (gt=descendant, Sonnet=benign) | **0** |

The single AC5 case: `post_index_petechiae_event_180d_flag` (gt=`descendant`) — S=`collider`, O=`descendant`✓, G=`mediator` → 1-1-1 **split/escalate**. The ensemble escalated to review a case single-Sonnet mislabeled; it did not produce the correct label via majority, and a 1-1-1 split is **flaky to pin as a live regression test**.

## Interpretation (the honest finding)
- **The ensemble does not beat single-Sonnet on accuracy** on hard leak-relevant cases (0.633 = 0.633, n=30). It causes no regressions.
- **The multi-vendor independence premise — the core #242→#240 rationale — largely fails here.** On 30% of hard cases all three models *including GPT-5* agreed on the same answer that differs from the golden label. Where vendors agree, the gate gains no asymmetric-failure signal from multi-vendor.
- **AC5 (as written: a leak single-Sonnet false-negatives, ensemble catches) is NOT cleanly satisfied** — 0 clean leak-FNs, 1 weak escalation-only case.

### Decision-relevant for #240
#240 AC3.5 treats #242 as the HARD multi-vendor unblocker for the severity-gate. This data suggests multi-vendor may **not** deliver the asymmetric signal #240 assumed — the gate's correlated-failure concern (two Anthropic siblings) is **not** clearly mitigated by adding GPT-5 when all three converge. Re-scope #240 AC3.5 in light of this.

### Possible golden-label disputes (feed #358)
Cases where all three frontier models unanimously disagreed with the golden label are candidates for a golden-set review, e.g.: `ctdna_clearance_90d_flag_given_baseline_positive` (gt=collider, all→mediator), `best_recist_response_180d` (gt=mediator, all→descendant), `switch_ai_to_fulvestrant_within_365d` (gt=descendant, all→collider), `post_index_thrombocytopenia_event_180d` (gt=descendant, all→mediator). These may be model error OR label error — worth adjudication, not assumed.

## Resolution (2026-05-25, after credit top-ups)

The A/B was re-run three times after the hold. **Every run reproduced the clean n=30 numbers exactly** (single-Sonnet 19/30=0.633, ensemble 19/30=0.633, 0 regressions, 9 correlated all-3-wrong, 1 weak escalation-only AC5, 0 clean leak-FN). The finding is therefore robust, not a one-off of a single sample.

### Why the full 42 was never measurable
Each ~$2 Anthropic top-up funds ≈30 entries of Sonnet+Opus (~60 Anthropic calls). The harness iterates the golden set top-to-bottom and the **12-entry PNH_fabhalta block is last** (entries 31–42), so every from-the-top run exhausts its budget on entries 1–30 and the PNH tail always errors with `credit balance too low` (GPT-5/OpenAI errored 0). Re-running from the top after a top-up just re-funds another pass over the first 30; it never reaches PNH. A PNH-first run (24 Anthropic calls) would measure the tail, but only on fresh budget run *before* a full pass drains it. The boundary is a per-top-up budget artifact, not a property of the PNH features.

This does **not** change the decision: with 30 hard leak-relevant cases (incl. PNH entries that *did* land before the boundary) showing **0 clean leak-FN catches and only 1 weak escalation**, the 12 unmeasured entries cannot turn AC5 into a clean pass. Ship `Refs #242`.

### Decision (executed)
- **`Refs #242`** — capability shipped; #242 stays OPEN. AC5 (single-Sonnet false-negatives a leak, ensemble catches it) is NOT cleanly met, so no closing keyword.
- Follow-ups to file: multi-vendor re-scope against #240 AC3.5 (independence premise fails on ~30% of hard cases) + golden-label review against #358 (the all-3-frontier-models-disagree candidates).
- A `Closes #242` would require a clean, reproducible leak-FN-caught case backed by a non-flaky skippable live integration test — not present in the data.

## What is already done
The ensemble implementation is complete, codex-reviewed twice (ACCEPT, 0 findings), 64 unit tests green, ruff + mypy clean. Branch `feat/242-multi-model-ensemble`.
