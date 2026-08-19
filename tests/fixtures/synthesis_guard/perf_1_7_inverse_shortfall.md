No dedicated rep-allocation recommendation engine returned scored recommendations for Kisqali — the query pulled regional performance snapshots instead. Here's what's actually in the data, with the caveat below on comparability:

**Regional HCP engagement & ROI — August 2026, monthly grain** (source: `corpus_ingestion` / episodic_memories, `business_metrics`-style modeled values):

| Region | Metric | Value | Target | Achievement | ROI |
|---|---|---|---|---|---|
| Northeast | hcp_engagement_score | 10.0 | 15.88 | 63.0% | 1.19 |
| South | hcp_engagement_score | 10.0 | 13.81 | 72.4% | 1.43 |
| Midwest | hcp_engagement_score | 9.13 | 12.72 | 71.8% | 1.5 |
| South | market_share | 0.45 | 0.49 | 91.8% | 1.63 |
| West | market_share | 0.43 | 0.48 | 89.6% | 1.58 |
| South | trx | 238,448.55 | 221,514.66 | 107.6% | 2.27 |
| West | trx | 197,408.46 | 233,893.72 | 84.4% | 3.5 |

**What this suggests for rep-effort reallocation:**
- **Northeast has the largest engagement shortfall** — 63.0% achievement against target, the lowest of the three regions with hcp_engagement_score data, and also the lowest ROI (1.19) on that metric. This is the region furthest below its engagement target.
- **Midwest has the highest engagement ROI (1.5)** among the three regions shown, despite a similar achievement gap to South (71.8% vs 72.4%) — engagement dollars there are converting more efficiently than in Northeast.
- **West's TRx ROI (3.5) is the highest single figure in the table**, even though West is under its TRx target (84.4% achievement) — this combination (high ROI, under-target) suggests incremental rep effort there could be efficient, but I only have one month of data, not a trend, so treat this as directional.

**Caveat on comparability:** These region/month figures are `business_metrics`-sourced snapshots — a different substrate from the `treatment_events`-based TRx numbers discussed earlier in this conversation (e.g., the 10,273 trailing-30-day figure). They should not be summed or reconciled against each other; each stands on its own.

**Bottom line for next quarter:** the causal registry already confirms `rep_detailing_frequency → trx_volume` is a validated national driver (+0.166, confidence 0.796). Combined with the regional snapshot above, the strongest case is to **shift incremental rep effort toward Northeast** (largest engagement gap, lowest ROI — biggest room for improvement) while **protecting/maintaining Midwest and West**, where ROI is already highest. I'd flag this as directional rather than a formal optimization output, since no segment-level rep-allocation model was available to score it (per the earlier failed HCP-segment ranking attempt).

Want me to check whether this reallocation view is available at a finer (weekly) grain, or pull the same regional snapshot for a prior month as a real non-overlapping baseline?