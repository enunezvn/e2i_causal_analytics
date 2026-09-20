# Lane B (#2115) — `forecast_kpi_tool` live certification

**Date:** 2026-09-20 · **Branch:** `claude/2115-forecast-kpi-tool` · **Base:** `5d0d3c19a`

Demo 6.5 — *"Forecast Kisqali TRx volume for the next two quarters and tell me the
biggest risk to that forecast"* — refused on every run from 2026-07-29 because the
platform had no forecaster. This certifies the forecast half against the real
production substrate, in the environment that actually runs it.

## What was certified, and where

The **faithful environment for TimesFM is the prod api image, not a CI runner** — the
local `.venv` carries transformers 4.57.3, which does not ship
`TimesFm2_5ModelForPrediction`. Everything below ran inside
the prod api image tagged `5d0d3c19a` (`ghcr.io/enunezvn/e2i-api`), against the live
`supabase-db` and the live `e2i_redis`.

(The image tag is written short on purpose: a full 40-character commit SHA in a code
span trips Gitleaks' `generic-api-key` rule, which cannot tell one from an API key. The
honest fix is the document, not an allowlist entry that would weaken the scanner.)

Provenance was asserted inside each run, not assumed:
`src.__file__ == /work/src/__init__.py` (the lane's code, never the image's baked
copy), `transformers 5.10.1`, `statsmodels 0.14.6`, `torch 2.9.1+cpu`.

| Run | What it proves | Result |
|---|---|---|
| `timesfm_image_suite.log` | the model + backtest suites under real transformers 5.10.1 | **33 passed, 1 failed** (the failure needed a live broker; re-run below) |
| `live_cert.log` / `live_cert.json` | broker → worker → weights → real PROD series, end to end | **PASS**, all three brands |

A real `worker_forecast` container was started with the compose service's own settings
(`--concurrency=1 --queues=forecast`, `HF_HOME` on the named `e2i_hf_cache` volume,
2.5 G limit, 2 CPU) and registered `src.tasks.forecast_timesfm_batch` on the `forecast`
queue. `worker_available()` → `True | worker_forecast@e99a1b181ce8 is consuming forecast`.
Dispatch round trip: **3.74 s**. Peak worker RSS: **878 MiB of 2.5 G**.

## The measured result (live PROD series, 164 complete months, 24 rolling origins)

Every model backtested on the *same* origins; the lowest monthly MAPE is served.

| Brand | Champion | seasonal-add | seasonal-mul | trend | TimesFM 2.5 |
|---|---|---|---|---|---|
| Kisqali | `holt_winters_seasonal_add` | **6.85%** | 6.89% | 7.38% | 7.58% |
| Fabhalta | `holt_winters_seasonal_add` | **7.06%** | 7.50% | 7.09% | 8.00% |
| Remibrutinib | `holt_winters_seasonal_mul` | 7.22% | **7.06%** | 8.09% | 7.40% |

(monthly MAPE, h=6. Naive baseline on Kisqali: 10.42%; seasonal-naive: 9.80%.)

**The champion is not the same model for every brand.** That is why the owner's
"swappable model interface, backtested per request" is load-bearing rather than
ceremony: any hard-coded single model would be wrong on at least one brand, and at 12
origins instead of 24 the ranking flips again.

**TimesFM ran on all three brands and lost narrowly every time.** It is close enough
(0.34–0.94 pp) to be worth keeping in the contest and not close enough to justify
making the answer depend on it — which is exactly how it is wired.

Kisqali, two quarters from `data_through` 2026-08-31 (80% band):

| Month | Forecast | Band |
|---|---|---|
| 2026-09 | 848,332 | 779,230 – 947,209 |
| 2026-10 | 858,332 | 788,180 – 955,470 |
| 2026-11 | 875,064 | 801,955 – 978,742 |
| 2026-12 | 907,097 | 826,036 – 1,008,798 |
| 2027-01 | 809,803 | 735,904 – 909,021 |
| 2027-02 | 853,614 | 783,734 – 956,157 |
| **total** | **5,152,242** | |

The December peak and January trough are Lane A's planted ±8% calendar seasonality
(#2114) surviving into the forecast — measured on the live series, the detrended
month-of-year means run 0.919 (Jan) to 1.071 (Dec).

## Latency, measured rather than assumed

Per-model backtest cost on the live Kisqali series, 24 origins, in-image, 2 CPU:

| Component | Cost |
|---|---|
| `holt_winters_seasonal_add` | 12.32 s |
| `holt_winters_seasonal_mul` | 17.80 s |
| `holt_winters_trend` | 3.60 s |
| `timesfm_2_5` (batched, via worker) | **5.83 s** |
| champion's final fit | 0.28 s |

Two findings changed the design:

1. **TimesFM is not the bottleneck; Holt-Winters is** (33.7 s of ~40 s). The batching
   works — 24 origins cost one round trip and one weight load, not 24 of each.
2. **Threading the fits was measured and is SLOWER** — 29.67 s sequential vs 39.30 s
   across a 4-thread pool (0.75×), because the scipy optimiser holds the GIL. So the
   answer is not parallelism but **not recomputing a forecast that cannot have
   changed**: results are cached under a key containing `data_through`, so an entry
   built on August is structurally unable to answer a September request.

As shipped (`worker_forecast` at `replicas: 0`) the contest is Holt-Winters only,
~33 s cold and cached thereafter.

## Honesty properties certified in the payload

* `measure_basis.comparison_key == ["business_metrics"]` — a forecast of canonical TRx
  is a canonical-TRx figure, so the #1640 scale guard still fences it off the
  patient-panel KPIs.
* `limitations` names the blind spots (competitor launch, payer/formulary change, label
  change, regional step) and `risk_analysis_requires` routes the risk half to the gap
  analyzer and `causal_analysis_tool`.
* The band is the champion's **own measured miss distribution** at each horizon step —
  not statsmodels' analytic interval and not TimesFM's quantile head. That is the only
  mechanism under which the two model families are comparable, and it is what lets the
  payload describe the band as measured error rather than as a confidence interval.

The planted Kisqali midwest −15% step from 2026-10 lies **inside** this forecast window
and is invisible to every model here. That is not a defect being disclosed after the
fact — it is why the payload carries `limitations` and why the prompt block forbids
narrating a risk as though the forecast had priced it in.

## Reproducing

`live_cert.py` is the script that produced `live_cert.log` / `live_cert.json`. It needs
a `worker_forecast` container on `e2i_network` and the api service's Supabase env.

---

# POST-DEPLOY LIVE VERIFICATION — 2026-09-20, merge `096ceecc3`

Run **inside the deployed `e2i_api` container**, against the live PROD Supabase, after
PR #2198 merged and the rollout flipped the app tier. **VERDICT: PASS — 18/18 checks,
0 failures.** Evidence: `postdeploy_verify.py` (the script), `postdeploy_verify.log`
(its output), `postdeploy_verify.json` (the structured result).

## The flip is proven, not assumed

A one-sided "the module is there" check cannot tell a successful deploy from a module
that was always there, so the control was taken **before** the rollout:

| | image tag | `src.kpi.forecast` present |
|---|---|---|
| before | `ghcr.io/enunezvn/e2i-api:97f83be64` | **False** |
| after | `ghcr.io/enunezvn/e2i-api:096ceecc3` | **True** |

`e2i_api` `StartedAt` moved `2026-09-20T17:22:24Z` → `18:53:26Z`, health `healthy`,
`/health` 200. Every app-tier container (`api`, `frontend`, `scheduler`,
`worker_light` x2, `worker_medium`) is on `096ceecc3`.

## What was verified, and why these things

**Registration (3).** `forecast_kpi_tool` is in `E2I_CHATBOT_TOOLS` and `E2I_TOOL_MAP`
(11 tools), and `kpi_forecaster` imports from the registry.

**The real deployed prompts (6).** Asserted against `E2I_COPILOT_SYSTEM_PROMPT` and
`E2I_CHATBOT_SYSTEM_PROMPT` themselves — not by calling `render_blocks` and checking its
return, which would prove only that the helper works. Each prompt carries the forecast
guidance, names `forecast_kpi_tool`, and has **no unsubstituted `{capability_guidance}`
or `{breakdown_guidance}` slot** — an unsubstituted brace would ship to the model as a
literal.

**The degraded path PROD actually runs (5).** `worker_forecast` ships `replicas: 0`, so
there is no forecast worker and `worker_available()` is `False`:
`no worker is consuming the 'forecast' queue`. TimesFM is therefore **refused with that
reason in `models_not_run`, not silently dropped**, and the forecast still serves on a
Holt-Winters champion. Certifying only the worker-up path would have certified a
configuration that is not deployed.

```
champion=holt_winters_seasonal_add   n=164   data_through=2026-08-31
  holt_winters_seasonal_add   MAPE  6.85%  origins=24
  holt_winters_seasonal_mul   MAPE  6.89%  origins=24
  holt_winters_trend          MAPE  7.38%  origins=24
  NOT RUN timesfm_2_5: the TimesFM forecast worker is not available:
          no worker is consuming the 'forecast' queue

  2026-09    848,332   [ 779,230 ..   947,209]
  2026-10    858,332   [ 788,180 ..   955,470]
  2026-11    875,064   [ 801,955 ..   978,742]
  2026-12    907,097   [ 826,036 .. 1,008,798]
  2027-01    809,803   [ 735,904 ..   909,021]
  2027-02    853,614   [ 783,734 ..   956,157]
```

Every month's band contains its point, and the payload survives
`json.dumps(allow_nan=False)` — the strict encoder FastAPI uses, which rejects the
`Infinity` a degenerate band produced during review.

**Routing, at the capability level (4).** The first version of this check asserted that
the compound demo 6.5 ask satisfies both `is_forecast_question` and
`is_forecast_risk_question`. **It does not, by design** — `planner.py:1259` returns
`False` from `is_forecast_question` when the risk predicate fires, because each
predicate grades a *decomposed sub-question*, not the compound ask. The assertion was
mine and it was wrong; asserting predicate truthiness was a proxy for the thing that
matters. Replaced with the capability:

| decomposed ask | tool reached |
|---|---|
| "What's the Kisqali TRx forecast for the next two quarters?" | `kpi_forecaster` |
| "What are the risks to that forecast?" | `causal_effect_estimator` |
| CONTROL — "Which HCP segments are highest risk of churn?" | `risk_scorer` |
| DEGRADED — forecast ask, forecaster unregistered | `risk_scorer` |

The control matters: a fix that sent *every* PREDICTIVE ask to the forecaster would
pass the first two rows and break entity scoring, which this map has served all along.

## Known and intentional

`worker_forecast` stays at `replicas: 0` until the box has headroom (same rule as
`worker_heavy`). Until it is scaled up, PROD forecasts are Holt-Winters only, and the
payload says so in `models_not_run` rather than pretending a four-model contest ran.
Champion selection across unequal origin sets on short series is tracked in **#2199**.
