## Fabhalta Launch-Readiness Snapshot

### 1. % PNH Patients Tested
Not available. There's no tool or data source in this platform that tracks PNH diagnostic/testing rates — that's a clinical epidemiology metric outside the commercial data model (`treatment_events`, `business_metrics`). I won't fabricate a figure here.

### 2. NRx Trend
Two different substrates are in play — reporting each separately, not combined:

**From `treatment_events` (computed on demand), last 6 months (2026-02-18 to 2026-08-18):** NRx = **3,630**. This figure carries a coverage warning I have to flag rather than use for trend: *"61% of this 181-day total falls in its most recent 30 days — the data is not evenly distributed across the window. Do NOT treat the full-window figure as a baseline for the recent period."* So I can't draw a trend conclusion from this number alone.

**From `business_metrics` (modeled market-scale, monthly grain), national total across all 4 regions:**

| Month | Regions reporting | National NRx total |
|---|---|---|
| Aug 2026 | 4/4 (west, midwest, south, northeast) | **37,767.6** |
| Jul 2026 | 4/4 (west, midwest, south, northeast) | **33,368.0** |
| Jun 2026 | 2/4 (west, midwest only — south/northeast not returned) | not computable (partial) |

Comparing the two complete, non-overlapping monthly buckets: Jul → Aug NRx rose **+13.2%** (33,368.0 → 37,767.6). Jun 2026 can't be totaled — only west (9,045.0) and midwest (7,152.0) rows were returned, so it's a partial read, not a real decline or discontinuity.

*Note: the `treatment_events` figure (3,630) and the `business_metrics` totals (~33K–38K) are not comparable — they measure different things (raw event count vs. modeled market-scale level), not a market-share split.*

### 3. Top Adoption Barriers (validated causal drivers of NRx, min confidence 0.7)

| Rank | Driver → NRx | Effect Size | Confidence | Est. Impact | Lag |
|---|---|---|---|---|---|
| 1 | Intent-to-prescribe → new patient starts | +0.336 | 0.766 | +$69,399 | 43 days |
| 2 | Treatment initiated → patient onboarding | +0.240 | 0.917 | +$41,363 | 30 days |
| 3 | Sample dropped → trial experience | +0.074 | 0.808 | +$14,383 | 18 days |

**Read as barriers to close:** the biggest lever is converting HCP *intent* into actual new-patient starts (+0.336, the largest of the three) — suggesting a gap between HCPs willing to prescribe and HCPs who actually pull the trigger. Onboarding friction (treatment-initiation → onboarding, +0.240) is the second-largest and fastest-materializing (30-day lag). "Sample dropped" affecting trial experience is the smallest lever (+0.074) but shortest lag (18 days) — a quick-turnaround fix.

**Clinically grounded context (FDA label, real, via OpenFDA):** Fabhalta carries a **boxed warning** for serious encapsulated-bacteria infections and is dispensed only through a **REMS program** requiring vaccination ≥2 weeks pre-dose. This REMS/vaccination burden is a plausible real-world contributor to the "intent-to-prescribe → new patient starts" gap (rank 1 above) — HCPs may express intent but delay starts pending vaccination workflow completion.

### 4. Recommended Next Experiment
Design a **REMS/onboarding-acceleration test**: run a controlled pilot (via `experiment_designer`) that pre-schedules vaccination and REMS enrollment logistics support at the point of intent-to-prescribe, measuring time-to-first-fill against a control cohort. This directly targets the largest driver (+0.336, intent → starts) and the second-largest (+0.240, onboarding), and has a short enough lag (30–43 days) to read out within a quarter.

Want me to pull the regional NRx trend split (rather than national) or check the current Fabhalta experiment roster for anything already running in this space?