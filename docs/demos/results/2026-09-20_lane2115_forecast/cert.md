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
`ghcr.io/enunezvn/e2i-api:5d0d3c19ada8db08854f0cb1a9e4668a47c6d9b6` against the live
`supabase-db` and the live `e2i_redis`.

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
