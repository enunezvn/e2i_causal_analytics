## Kisqali TRx Share — Last 6 Months (2026-02-18 to 2026-08-18)

**Important framing correction:** The available TRx Share metric is Kisqali's share of the **tracked Novartis portfolio** (Fabhalta + Kisqali + Remibrutinib, cross-indication) — it does **not** measure share against external oncology competitors (e.g., Verzenios, Ibrance). Those competitor brands aren't in the data model, so I can't report a true Kisqali-vs-competitor market share comparison. What I can give you:

| Metric | Value | Period | Source |
|---|---|---|---|
| Kisqali TRx Share (of tracked portfolio) | **30.9%** | Last 6 months (2026-02-18 to 2026-08-18) | `treatment_events`, computed on demand |

## What's Driving TRx Share/Volume (causal analysis, Kisqali)

Ranked by effect size (all validated, 5/5 refutation tests passed):

| Rank | Driver | Effect on | Effect Size | Confidence | Est. Business Impact | Lag |
|---|---|---|---|---|---|---|
| 1 | Persistent 180-day refill continuity | TRx volume | +0.285 | 0.892 | +$127,512 | 24 days |
| 2 | Formulary status → patient access | TRx volume | +0.223 | 0.805 | +$100,658 | 78 days |
| 3 | Rep detailing frequency → HCP engagement | TRx volume | +0.166 | 0.796 | +$38,244 | 77 days |
| 4 | HCP coverage → prescriber base | TRx market share | +0.09 | 0.899 | +$29,541 | 55 days |
| 5 | Copay support program → adherence | TRx volume | +0.088 | 0.794 | +$17,025 | 64 days |
| 6 | Competitor activity → share of voice | TRx market share | **-0.073** | 0.793 | **-$24,218** | 70 days |

**Read:** The single largest positive driver is refill continuity/adherence (+0.285, +$127,512) — patients staying on therapy past 180 days is doing more for volume than any access or promotional lever. The only negative driver is competitor share-of-voice activity (-0.073, -$24,218), a modest headwind. Rep detailing frequency has a meaningful positive effect (+0.166, +$38,244) with a 77-day lag, making it the most actionable near-term rep lever.

## Where to Focus Reps Next Quarter

Predicted adoption propensity by specialty (champion model `hcp_adoption_kisqali_goldstd_lr_v1`, AUC 0.791, n=5,000 HCPs):

| Rank | Specialty | Mean Propensity | n | SE |
|---|---|---|---|---|
| 1 | Oncology | **56.9%** | 1,662 | 0.006 |
| 2 | Hematology | 38.9% | 1,016 | 0.007 |
| 3 | Rheumatology | 34.3% | 257 | 0.014 |
| 4 | Dermatology | 30.3% | 854 | 0.007 |
| 5 | Allergy/Immunology | 28.8% | 577 | 0.008 |

No segments flagged low-confidence. **Oncology** is the clear top target (56.9% propensity, largest cohort n=1,662) — reinforced by the causal finding that rep detailing frequency and HCP coverage are validated, actionable levers on volume/share. Hematology is a secondary tier worth sustaining coverage in.

**Recommendation:** Concentrate incremental rep detailing capacity on oncology HCPs (highest propensity + largest base), while maintaining formulary-access and adherence/copay-support efforts (persistent refill continuity is your top driver) to defend against the competitor share-of-voice headwind.

*Note: propensity scores measure current adoption likelihood, not a "next quarter" horizon specifically — treat this ranking as horizon-agnostic prioritization.*