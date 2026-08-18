## Remibrutinib NRx — Northeast Region, This Quarter

**Headline number:** NRx = **448** for Remibrutinib in the northeast region, covering **Q3 2026 to date** (window applied: 2026-07-01 through 2026-10-01, but only **48 days of data** are actually in — this is a partial, in-progress quarter, not a completed one). *Source: `kpi_calculate_tool`, computed from `treatment_events`.*

**Signal of deceleration within the quarter itself:**
- Trailing-30-day NRx = **172** of that 448 total → a **38.4% share** of the quarter's volume.
- If volume were spread evenly across the 48 days elapsed, the trailing 30 days should hold **62.5%** of the total.
- The actual 38.4% share vs. the 62.5% uniform-expected share indicates prescriptions have **slowed in the most recent 30 days relative to the earlier part of the quarter** — this is a within-quarter deceleration signal, not a comparison to a prior quarter.

**Important caveat on "drop":** I don't have a valid prior-quarter (or any non-overlapping, equal-length prior period) NRx figure for the northeast region to compare against, so I can't quantify a period-over-period decline — only the within-quarter deceleration pattern above.

**Causal drivers found — none of them explain a decline:**
The `causal_analysis_tool` returned 3 validated causal paths into `nrx_volume` for Remibrutinib/northeast, and **all three carry positive effect sizes** (i.e., they describe what *increases* NRx when present — their absence or weakening would be consistent with softer volume, but the tool itself reports positive relationships, not negative/declining ones):

| Cause | Via | Effect size | Confidence | Est. business impact |
|---|---|---|---|---|
| Treatment initiated | patient_onboarding | +0.41 | 0.816 | $72,247 |
| Intent to prescribe | new_patient_starts | +0.27 | 0.816 | $111,256 |
| Sample dropped | trial_experience | +0.092 | 0.841 | $30,580 |

Ranked by effect size: **treatment_initiated (+0.41)** is the largest positive driver, **intent_to_prescribe (+0.27)** second, **sample_dropped (+0.092)** smallest. None of these is a negative driver, so the causal engine isn't attributing the slowdown to an identified negative cause — it's more consistent with these positive drivers weakening in magnitude (e.g., fewer new patient starts or onboarding events) than with a new negative force appearing.

**Triggers query — not usable for this region/brand cut:** The `e2i_data_query_tool` trigger pull was filtered for Remibrutinib/northeast, but the 10 records returned span **Remibrutinib, Fabhalta, and Kisqali** brand IDs and carry no region field — the filter doesn't appear to have been applied to the returned rows. I can't reliably extract northeast-specific Remibrutinib trigger activity from this result, so I'm not using it to explain the drop.

**Bottom line:** The data supports a real within-quarter deceleration (recent 30 days underperforming an even pace), but there's no valid baseline to size a "drop" against a prior period, and the causal analysis surfaced only positive-effect drivers rather than an identified negative cause. Want me to pull a same-length prior-quarter NRx figure for northeast so we have a real baseline to compare against?