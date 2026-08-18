## Heterogeneous Treatment Effects (HTE) — What This Means in Our Analyses

**Heterogeneous treatment effects** (HTE) means a treatment/intervention doesn't have the same effect on everyone — the effect size varies systematically by subgroup. Rather than reporting one average treatment effect (ATE) for a full population, the `heterogeneous_optimizer` agent estimates **CATE** (Conditional Average Treatment Effect) — the effect *conditional on* a subgroup (in our data, HCP/patient severity segments) — so you can see which segments respond most/least to a given commercial action.

**Recent CATE analyses from `heterogeneous_optimizer`** (source: `e2i_data_query_tool`, agent_analysis, synthetic data):

| Timestamp | Brand | Treatment Var | Overall ATE | Low Severity CATE | Medium Severity CATE | High Severity CATE | Heterogeneity Score |
|---|---|---|---|---|---|---|---|
| 2026-08-17 21:22 | Kisqali | rep_detailing_high | 0.2567 | 0.070 | 0.196 | 0.504 | 0.434 |
| 2026-08-17 18:16 | Fabhalta | psp_enrolled | 0.1307 | 0.035 | 0.091 | 0.266 | 0.231 |
| 2026-08-17 17:58 | Kisqali | treatment_arm | 0.5133 | 0.140 | 0.420 | 0.980 | 0.840 |
| 2026-08-16 19:45 | Kisqali | trigger_accepted | 0.2800 | 0.070 | 0.210 | 0.560 | 0.490 |
| 2026-08-16 10:57 | Kisqali | sample_dropped | 0.1587 | 0.056 | 0.140 | 0.280 | 0.224 |

**How to read this:**
- **Overall ATE** is the population-average effect of the treatment variable on the outcome (`treatment_initiated` for Kisqali runs; `adherent_180d` for the Fabhalta run).
- **CATE by segment** breaks that average effect out by severity tier. In every run above, **high_severity** is the top-responding segment and **low_severity** the weakest — e.g., in the Kisqali `treatment_arm` run, high-severity CATE (0.980) is roughly **7x** the low-severity CATE (0.140).
- **Heterogeneity score** quantifies how much the effect varies across segments — higher means the treatment's impact is more concentrated in specific subgroups. The Kisqali `treatment_arm` run has the highest heterogeneity score in this table (0.840), meaning that treatment's effect is the most unevenly distributed across severity tiers of the five shown; the Fabhalta `psp_enrolled` run has the lowest (0.231), meaning its effect is comparatively more uniform.

**Why it matters commercially:** each run's `recommendations` field consistently prioritizes high-severity segments for that treatment (e.g., "prioritize_high_severity_for_rep_detailing_high", ROI estimate 5.42 for that Kisqali run). This is how the platform turns HTE/CATE findings directly into targeting guidance — instead of applying a uniform tactic everywhere, it identifies which segment yields the largest incremental lift for that specific lever, and estimates the associated impact/ROI (e.g., $156,606 impact / 5.04 ROI for the Kisqali `treatment_arm` run).

Confidence levels on these runs range **0.807–0.937**, all "completed" status.

Would you like me to pull a specific brand's most recent CATE run in more depth, or route this through `orchestrator_tool` for a fuller heterogeneous treatment effect + gap analysis (as seen in the retrieved orchestration logs)?