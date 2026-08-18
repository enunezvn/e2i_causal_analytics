**Fabhalta HCP segment likelihood-to-prescribe, by specialty** (current adoption propensity — the requested "next quarter" horizon is context only; this ranking reflects current propensity, not a horizon-specific forecast)

| Rank | Specialty | Mean Propensity | n | SE |
|---|---|---|---|---|
| 1 | Hematology | 58.4% | 1,016 | 0.007 |
| 2 | Internal Medicine | 51.1% | 478 | 0.011 |
| 3 | Neurology | 46.2% | 156 | 0.020 |
| 4 | Dermatology | 36.5% | 854 | 0.008 |
| 5 | Oncology | 34.9% | 1,662 | 0.006 |
| 6 | Rheumatology | 34.5% | 257 | 0.015 |
| 7 | Allergy/Immunology | 34.3% | 577 | 0.010 |

- **Model:** `hcp_adoption_fabhalta_goldstd_lr_v1` (holdout AUC 0.811), scored over **5,000 HCPs**, overall mean propensity **41.8%**
- No `low_confidence` flags — all segments have adequate sample sizes.
- **Hematology** is the top segment (highest propensity, n=1,016), followed by Internal Medicine and Neurology; Oncology has the largest n (1,662) but ranks near the bottom on propensity.

Since the score is a current adoption-propensity ranking rather than a next-quarter-specific increase forecast, hematology and internal medicine HCPs are the strongest current targets for expanded Fabhalta engagement. Happy to re-run this by geographic region if useful.