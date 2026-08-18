There's no single "system health score" — the health_score agent logs per-run analyses per brand rather than one aggregate number. Here are its 10 most recent runs (last 30 days window requested, data actually spans **2026-07-24 to 2026-08-16**):

| Timestamp | Brand | Confidence | Insights Found | Anomalies | ROI Estimate | Impact Estimate |
|---|---|---|---|---|---|---|
| 2026-08-16 | Remibrutinib | 0.808 | 7 | 2 | 6.21 | $128,439 |
| 2026-08-15 | Fabhalta | 0.837 | 12 | 0 | 4.80 | $135,022 |
| 2026-08-14 | Kisqali | 0.844 | 6 | 0 | 6.23 | $153,790 |
| 2026-08-08 | Remibrutinib | 0.877 | 11 | 0 | 8.69 | $185,851 |
| 2026-08-08 | Kisqali | 0.896 | 19 | 3 | 5.39 | $348,523 |
| 2026-08-03 | Fabhalta | 0.896 | 8 | 1 | 8.23 | $320,002 |
| 2026-08-01 | Remibrutinib | 0.989 | 19 | 3 | 1.62 | $35,827 |
| 2026-07-29 | Fabhalta | 0.974 | 1 | 1 | 7.07 | $100,126 |
| 2026-07-28 | Kisqali | 0.730 | 3 | 0 | 6.40 | $169,637 |
| 2026-07-24 | Kisqali | 0.858 | 18 | 3 | 2.28 | $265,200 |

Key observations from this table:
- **Confidence levels** range from **0.730** (Kisqali, 2026-07-28, lowest) to **0.989** (Remibrutinib, 2026-08-01, highest).
- **Anomalies detected** are highest for Remibrutinib (2026-08-16) and Kisqali (2026-08-08, 2026-07-24) at 3 each; several runs (Fabhalta 2026-08-15, Remibrutinib 2026-08-08, Kisqali 2026-07-28) show **0 anomalies**.
- **ROI estimate** is highest for Remibrutinib (2026-08-08) at **8.69** and lowest for Remibrutinib (2026-08-01) at **1.62**.

This is a per-run monitoring log, not a composite score — there's no single blended "health score" value in these results. Source: `e2i_data_query_tool` (agent_analysis, health_score agent, synthetic data).

Would you like this narrowed to one brand, or would a trend chart of confidence/anomalies over time be useful?