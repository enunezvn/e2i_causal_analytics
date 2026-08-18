Rep-visit ("rep detailing") effects on Kisqali outcomes are well-supported and **passed all refutation tests** — confidence in these paths is high.

**Direct rep_visits causal paths (Kisqali, all_time):**

| Cause → Effect | Effect Size | Confidence | Refutation Tests | Gate Decision |
|---|---|---|---|---|
| rep_detailing_high → treatment_initiated | +0.043 | 0.930 | 5/5 passed | proceed |
| rep_detailing_frequency → roi (via trx_volume) | +0.198 | 0.851 | 5/5 passed | proceed |
| rep_detailing_frequency → trx_volume (via hcp_engagement) | +0.166 | 0.796 | 5/5 passed | proceed |
| rep_detailing_frequency → intent_to_prescribe (via message_recall) | +0.267 | 0.772 | 5/5 passed | proceed |

All four paths cleared **5 of 5 refutation tests with 0 failures and 0 warnings**, each yielding a "proceed" gate decision. Confidence ranges from **0.772** (frequency → intent_to_prescribe) to **0.930** (rep_detailing_high → treatment_initiated), so the strongest-confidence rep effect is on treatment initiation directly, while the effect on ROI (0.198, via trx_volume) carries the largest estimated effect size among these four.

Note: this data is flagged `evidence_is_synthetic: true` in the refutation evidence — it's a validated synthetic-data test, not a claim about production-grade real-world statistical robustness.

Source: `e2i_data_query_tool` (causal_chain, Kisqali, all_time). A broader query also surfaced treatment_arm-driven paths (0.943–0.944 confidence, also 5/5 passed) but those are not rep-visit specific — they concern treatment_arm as the cause, so I excluded them from this rep-visit answer.