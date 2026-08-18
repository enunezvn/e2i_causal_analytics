**Rep-detailing → TRx volume (Kisqali): confidence 0.796, passed all refutation tests.**

| Cause | Effect (via) | Effect size | Confidence | Refutation tests | Gate decision |
|---|---|---|---|---|---|
| rep_detailing_frequency | trx_volume (via hcp_engagement) | +0.166 | 0.796 | 5/5 passed, 0 failed, 0 warning | proceed |

- **Effect size +0.166**: a positive, validated causal link from rep detailing frequency to TRx volume, mediated through HCP engagement.
- **Estimated business impact**: **+$38,244.12**.
- **Method**: backdoor.linear_regression, with a **77-day** time lag between the tactic and the observed effect.
- **Refutation robustness**: all 5 refutation tests passed (0 failed, 0 warning), giving a `gate_decision` of "proceed" — this is the causal-validity gate this system uses before trusting a path.
- **Caveat**: this record is flagged `evidence_is_synthetic: true` and `data_source: synthetic`, and no specific date window was requested/returned — this is the engine's standing causal-paths registry for Kisqali/TRx, last tested 2026-07-31, not a figure tied to a particular calendar period you named.

For context, rep detailing is the 4th-strongest of 6 validated Kisqali→TRx causal paths currently in the registry, ranked by confidence: hcp_coverage (0.899), persistent_180d (0.892), formulary_status (0.805), rep_detailing_frequency (0.796), copay_support_program (0.794), competitor_activity (0.793, negative effect −0.073).

Want me to pull the same refutation breakdown for another driver (e.g., formulary_status) or for a different brand?