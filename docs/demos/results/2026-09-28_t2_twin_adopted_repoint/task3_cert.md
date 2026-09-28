# Lane T2 — Task 3 live acceptance (2026-09-28)

**Verdict: PASS.** The deployed twin estimates on `hcp_brand_adoption.adopted` and reports the causal
forest's doubly-robust interval. Every planted focus channel is significant in every brand; the planted
null is calibrated at the family level; the chat route, the API and the fidelity feed agree.

## What is deployed

- `image_sha: b150ae6013387d7f6198b89060d011742d452c50` on `e2i_api` + `e2i_frontend` (StartedAt 20:13:36Z / 20:14:43Z, healthy, `/health` 200).
- It carries #2305 (T2), #2307 (peer, bento tag parse) and #2309 (#2292 fix). Deploy run 36469691787, `Deploy to Droplet` 19:59:48Z → 20:15:44Z.
- Ledger: `record_deploy_attempt.sh pr2305 b150ae601…` bound run 36469691787 (snapshot `pr2305.pre`, dispatched 18:11:33Z). `prove_state.sh new` → **76 OK / 0 BAD**, which clears the stale `e2i_feast` BAD carried since 2026-09-24.

## 1. API probe on the deployed image — `task3_api_probe.py` → 9 ok / 0 bad

| brand | digital_engagement ATE | DR 95% CI | half-width | is_significant | recommendation |
|---|---|---|---|---|---|
| Kisqali | +0.1069 | (+0.0607, +0.1531) | 0.046 | true | deploy |
| Fabhalta | +0.1560 | (+0.1154, +0.1967) | 0.041 | true | deploy |
| Remibrutinib | +0.1643 | (+0.1213, +0.2072) | 0.043 | true | deploy |

The plan's gate was "engagement on Fabhalta, `is_significant=true`, interval ≈ ±0.04". Measured: ±0.041.

| brand | rep_training_quality (planted null) ATE | DR 95% CI | significant |
|---|---|---|---|
| Kisqali | −0.0109 | (−0.0500, +0.0282) | no |
| Fabhalta | +0.0058 | (−0.0348, +0.0465) | no |
| Remibrutinib | +0.0472 | (+0.0083, +0.0861) | yes — the realised seed-427 draw, see §4 |

- `/intervention-types`: 8/8 `available_for_effect` on the `cohort_causal` basis, in every brand. The availability gate now runs the same rule `/simulate` does.
- `/proposed-experiments`: the envelope `outcome_column` is `adopted`. The stored rows resolve by the date rule: 24 `conversion_rate`, 3 `cohort_conversion_outcome`, 3 unknown (`synthetic_uplift_v1`; drafting them is refused), plus 4 new `adopted` rows written by this probe's own `/simulate` calls.
- These numbers equal the lane's pre-deploy read-only probe to 4 decimals.

## 2. Chat route — `copilot_chat_perf_runner.py`, `raw_task3_chat.jsonl`

The question was: *"Use the digital twin to simulate a digital engagement intervention for Fabhalta HCPs…"*. The answer reported **+0.156, 95% interval 0.115–0.197, DEPLOY**, with the synthetic-gold cohort provenance and the region effects. Those are the same numbers as the API.

Observation, not a gate: the answer justifies "Significant? Yes" by the policy's minimum-effect threshold (0.050). Significance is the interval excluding 0, so the conclusion is right but the stated reason is wrong. This is chat-prose wording, not the estimator.

## 3. Fidelity feed — `task3_live_feed_probe.py` → PASS

This is a re-run of the 2026-09-23 probe on the post-deploy code and today's cron-refreshed A/B data. It reads 360/360 experiments from `unit_outcomes`; effects equal the stored results within 1e-9 (360/360); array sizes equal the stored n (360/360).

## 4. Recovery gate + interval reversal condition

The certifying run is in PR #2313, the Option A gate, owner decision 2026-09-28: `t1_gate_family_null.log`. It ran `--live --via-twin-loader` in-process on source `b150ae601`, which is the deployed sha. Result: **PASS**.

- Point gate: 8/8 per brand. Focus channels exclude 0 in every brand.
- Null calibration over 100 fixed seeds × 3 brands: false-positive rate **11/300 = 0.037** (Wilson 95% 0.021–0.064; gate ≤ 0.10). SE/empirical SD **1.15** (gate ≥ 0.9).
- The plan's reversal condition ("DR SE / empirical SD < 0.9 or null FP > 0.10 → LinearDML") is therefore measured and **not triggered**.

The Remibrutinib null is a realised draw, not a defect: `../2026-09-28_remi_null_caveat/summary.txt`. The +0.045 of the +0.047 is the final Bernoulli coin flips of seed 427. Placebo false-positive rates are 6/100, 2/40 and 3/40. Omitted-confounding and channel-leakage refits do not remove it.
