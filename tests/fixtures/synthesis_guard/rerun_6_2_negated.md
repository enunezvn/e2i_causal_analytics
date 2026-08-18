## Remibrutinib Conversion Rate by Region — Most Recent 30 Days (through 2026-08-17)

*Note: no specific window was requested for this metric, and the tool's default reporting window is the trailing 30 days of trigger data — that's the period covered below.*

| Region | Conversion Rate | vs. 4-region avg (63.40%) |
|---|---|---|
| West | 64.12% | +0.72 pts |
| Northeast | 63.59% | +0.19 pts |
| Midwest | 63.45% | +0.05 pts |
| **South** | **62.43%** | **-0.97 pts** |

**Underperforming regions:** **South** (62.43%) is the lowest of the four and the only region meaningfully below average; **Midwest** (63.45%) sits just barely above average and is not a real laggard — so on this data, **South** is the one region genuinely underperforming on conversion rate. Midwest, Northeast, and West are clustered tightly together (63.45%–64.12%).

## ROI Context by Region (business_metrics substrate — NOT comparable to the conversion-rate figures above, different substrate)

| Region | ROI (point estimate) | 12-mo monthly ROI range (temporal variability, n=12) |
|---|---|---|
| Northeast | 2.142 | 0.96–2.04 (mean 1.50) |
| Midwest | 1.836 | 0.99–1.91 (mean 1.44) |
| South | 1.742 | 1.06–1.91 (mean 1.44) |
| West | 1.626 | 0.56–1.91 (mean 1.47) |

These ROI bands describe each region's recent month-to-month variability, not a confidence interval on the current value.

## On "ROI of shifting 20% more rep capacity to South"

I don't have a tool that models the marginal/incremental ROI of a hypothetical capacity reallocation — none of the calls above compute that scenario, and I won't fabricate one. What the data does support:

- South's current headline ROI is **1.742**, the second-lowest of the four regions (only West, at 1.626, is lower) — so a capacity shift into South isn't obviously the highest-ROI redeployment on ROI alone; it's the lowest-converting region but not the highest-ROI region.
- The dominant validated conversion driver for Remibrutinib overall (not region-specific — the causal tool returned no region split) is **acceptance_status → conversion_flag** (effect size +0.2, confidence 0.849, 34-day lag, ~$83,965 estimated impact) — i.e., payer/patient acceptance status is the biggest lever on conversion, more than rep activity per se. A second driver of the same pair has effect size +0.147, confidence 0.833, 22-day lag, ~$29,455 impact.

**Bottom line:** South is the clearest conversion-rate laggard, but before shifting rep capacity there, note that its ROI (1.742) isn't the region's strongest lever either — West's ROI is actually lower. If you want, I can run the segment-likelihood tool to see which HCP segments in South are most likely to respond to increased rep engagement, which would be a more direct way to size the opportunity than an unsupported capacity-shift ROI estimate.