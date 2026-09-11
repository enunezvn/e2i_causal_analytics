# Live re-band of `random_common_cause` under the shift-vs-reported-SE rule as implemented (2026-09-11)

Agent runs: 136 (`estimate_source = causal_impact_query`, one rcc row each). Stored (the retired |Δ|/|ATE| rule): {'failed': 7, 'passed': 129}. Rule as implemented, scored through the runner's own `_score_common_cause_shift`: `shift_se = |refuted − original| / (se × scale)`, PASSED ≤ 1, WARNING ≤ 2, else FAILED (`PASS_THRESHOLDS["common_cause_shift_se"]`); `se = (ci_hi − ci_lo) / (2 × 1.959964)` from the reported interval; `scale = sqrt(n / refit n)` when the refutation ran on a #1419 subsample (refit n < n), else 1.0. Rows scaled: 3 — `ca4e8529` n=37371 refit n=5000 scale 2.73 (unscaled 1.07 → 0.39 SE, passed), `39b03eeb` n=37515 refit n=5000 scale 2.74 (unscaled 1.48 → 0.54 SE, passed), `c3676bb0` n=37515 refit n=5000 scale 2.74 (unscaled 1.51 → 0.55 SE, passed).

## Status moves (today → new)

| today → new | runs |
|---|---|
| failed → failed | 1 |
| failed → passed | 3 |
| failed → warning | 3 |
| passed → passed | 129 |

## Gate moves (today, recomputed from stored statuses → new, rcc swapped)

| today → new | runs |
|---|---|
| block → block | 1 |
| block → proceed | 1 |
| block → review | 5 |
| proceed → proceed | 81 |
| review → review | 48 |

## Headline counts

- FAILED today → new: {'failed': 1, 'warning': 3, 'passed': 3} (7 rows).
- PASSED today → FAILED new: **0**.
- shift_se on today's PASSED rows (n=129): max 0.694, p95 0.549, p50 0.157 (unscaled, as in the disproof re-band: max 1.514, p95 0.563).
- SE source used, by fidelity: {'naive': 4, 'evalue_inv': 103, 'reported': 29}. E-value inversion branches: {'ci_includes_zero': 6, 'old_standardized': 119, 'laneD_risk_ratio': 11}. Naive proxy kinds: {'naive_ols': 7, 'naive_2prop': 126, 'unmapped_pair': 3}.
- Rows with NO SE source: 0.
- Sanity: stored rcc status ≠ status recomputed from stored delta_percent on 0 rows. Stored gate ≠ gate recomputed by the CURRENT runner from the stored statuses on 59 rows, of which 59 are stored BLOCK with sensitivity FAILED (gated before Lane D′ made sensitivity non-critical — the criticality change, not this rule); none unexplained. The gate columns below hold every stored status fixed except rcc (the old-format sensitivity statuses included, which the Lane D′ live re-band has since re-read), so the gate delta is attributable to rcc alone.
- Rows whose primary SE is the continuous-treatment OLS proxy: 1 — `5f7a5c0c` Kisqali peer_influence_score→adopted (shift 2.44, failed). Rows whose refutation refits ran on a SUBSAMPLE of the estimation frame (`refutation_subsampled`): 3 — `ca4e8529` refit n=5000 of n=37371 (shift 0.39, passed), `39b03eeb` refit n=5000 of n=37515 (shift 0.54, passed), `c3676bb0` refit n=5000 of n=37515 (shift 0.55, passed).

## Proxy calibration (ratio of SE sources on rows carrying both)

| ratio | stats |
|---|---|
| evalue_inv / reported | n=27, median 1.000, min 1.000, max 1.000 |
| boot_sd / reported | n=29, median 1.103, min 0.824, max 1.495 |
| boot_ci / reported | n=29, median 0.937, min 0.726, max 1.299 |
| naive / reported | n=29, median 0.996, min 0.858, max 1.052 |
| naive / evalue_inv | n=127, median 0.998, min 0.471, max 1.131 |
| naive_2prop / evalue_inv (binary treatment & outcome) | n=121, median 0.999, min 0.854, max 1.131 |
| naive_ols / evalue_inv (continuous treatment) | n=6, median 0.471, min 0.471, max 0.482 |
| naive / boot_sd | n=29, median 0.889, min 0.679, max 1.156 |
| std_refits / se | n=133, median 0.351, min 0.007, max 27.123 |

`reported` and `evalue_inv` are two independent exact inversions of the SAME reported interval (`bootstrap_ci / ci_ratio` and the stored `e_value_ci` bound); their ratio is the cross-check. `boot_sd` is a 20-resample SE of the estimator, `naive` a covariate-free SE on the re-pulled frame, `std_refits` the spread of the 20 random-common-cause refits (from the stored p-value; rows with p stored as 0 excluded).

## Per run

| estimate | brand | pair | n (refit n if subsampled) | original | refuted | Δ | today % / status | se source | se | scale | shift_se (unscaled) | new status | gate today → new | std_refits (rcc noise) | p | reported / inv / boot_sd / naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `5f7a5c0c` | Kisqali | peer_influence_score→adopted | 1500 | -0.0018 | -0.0281 | 0.0263 | 1451.9 / failed | naive | 0.0108 | 1.00 | 2.44 (2.44) | failed | block → block | 0.0478 | 0.29126 | - / - / - / 0.0108 |
| `e0da01eb` | Kisqali | sample_dropped→treatment_initiated | 5000 | 0.0129 | 0.0290 | 0.0161 | 124.2 / failed | naive | 0.0140 | 1.00 | 1.15 (1.15) | warning | block → review | 0.0070 | 0.01063 | - / - / - / 0.0140 |
| `d0c49855` | Remibrutinib | treatment_arm→persistent_180d | 5000 | 0.0472 | 0.0642 | 0.0170 | 36.1 / failed | evalue_inv | 0.0204 | 1.00 | 0.84 (0.84) | passed | block → review | 0.0085 | 0.02282 | - / 0.0204 / - / 0.0188 |
| `3b938dd6` | Remibrutinib | treatment_arm→persistent_180d | 5000 | 0.0472 | 0.0663 | 0.0191 | 40.5 / failed | evalue_inv | 0.0204 | 1.00 | 0.94 (0.94) | passed | block → review | ≤ 0.0043 | 0.00000 | - / 0.0204 / - / 0.0188 |
| `bae7fd9a` | Remibrutinib | treatment_arm→persistent_180d | 5000 | 0.0357 | 0.0608 | 0.0251 | 70.2 / failed | naive | 0.0188 | 1.00 | 1.34 (1.34) | warning | block → review | ≤ 0.0057 | 0.00000 | - / - / - / 0.0188 |
| `7f18139e` | Remibrutinib | treatment_arm→persistent_180d | 5000 | 0.0357 | 0.0587 | 0.0229 | 64.2 / failed | reported | 0.0197 | 1.00 | 1.16 (1.16) | warning | block → review | 0.0071 | 0.00065 | 0.0197 / - / 0.0163 / 0.0188 |
| `ad3fbcc1` | Remibrutinib | treatment_arm→persistent_180d | 5000 | 0.0357 | 0.0542 | 0.0185 | 51.7 / failed | reported | 0.0197 | 1.00 | 0.94 (0.94) | passed | block → proceed | 0.0073 | 0.00551 | 0.0197 / - / 0.0239 / 0.0188 |
| `af27c964` | <all> | treatment_arm→persistent_180d | 1500 | 0.1094 | 0.1240 | 0.0146 | 13.4 / passed | evalue_inv | 0.0361 | 1.00 | 0.40 (0.40) | passed | review → review | 0.0143 | 0.15415 | - / 0.0361 / - / 0.0339 |
| `ef04ef1d` | <all> | treatment_arm→persistent_180d | 1500 | 0.1094 | 0.1251 | 0.0157 | 14.3 / passed | evalue_inv | 0.0361 | 1.00 | 0.43 (0.43) | passed | review → review | 0.0248 | 0.26365 | - / 0.0361 / - / 0.0339 |
| `cef308f1` | <all> | treatment_arm→persistent_180d | 1500 | 0.0856 | 0.0887 | 0.0032 | 3.7 / passed | evalue_inv | 0.0339 | 1.00 | 0.09 (0.09) | passed | review → review | 0.0074 | 0.33546 | - / 0.0339 / - / 0.0339 |
| `6cf41d45` | <all> | treatment_arm→persistent_180d | 1500 | 0.0856 | 0.0876 | 0.0020 | 2.4 / passed | reported | 0.0339 | 1.00 | 0.06 (0.06) | passed | proceed → proceed | 0.0057 | 0.36028 | 0.0339 / 0.0339 / 0.0356 / 0.0339 |
| `ef86bef4` | <all> | treatment_arm→persistent_180d | 1500 | 0.0856 | 0.0865 | 0.0009 | 1.0 / passed | reported | 0.0339 | 1.00 | 0.03 (0.03) | passed | proceed → proceed | 0.0030 | 0.38233 | 0.0339 / 0.0339 / 0.0382 / 0.0339 |
| `52245f41` | <all> | treatment_arm→persistent_180d | 1500 | 0.0856 | 0.0912 | 0.0056 | 6.6 / passed | reported | 0.0339 | 1.00 | 0.17 (0.17) | passed | proceed → proceed | 0.0099 | 0.28460 | 0.0339 / 0.0339 / 0.0355 / 0.0339 |
| `e0bbfd51` | <all> | treatment_arm→persistent_180d | 1500 | 0.0856 | 0.0880 | 0.0025 | 2.9 / passed | reported | 0.0339 | 1.00 | 0.07 (0.07) | passed | proceed → proceed | 0.0481 | 0.47962 | 0.0339 / 0.0339 / 0.0367 / 0.0339 |
| `965744d4` | <all> | treatment_initiated→persistent_180d | 1500 | 0.0953 | 0.0868 | 0.0085 | 8.9 / passed | evalue_inv | 0.0317 | 1.00 | 0.27 (0.27) | passed | review → review | 0.0048 | 0.03961 | - / 0.0317 / - / 0.0271 |
| `e2f8d0aa` | <all> | treatment_initiated→persistent_180d | 1500 | 0.0953 | 0.0924 | 0.0029 | 3.0 / passed | evalue_inv | 0.0317 | 1.00 | 0.09 (0.09) | passed | review → review | 0.0032 | 0.18412 | - / 0.0317 / - / 0.0271 |
| `5c5fa7b6` | <all> | treatment_initiated→persistent_180d | 1500 | 0.0953 | 0.0861 | 0.0092 | 9.7 / passed | evalue_inv | 0.0317 | 1.00 | 0.29 (0.29) | passed | review → review | 0.0050 | 0.03282 | - / 0.0317 / - / 0.0271 |
| `e01727df` | <all> | treatment_initiated→persistent_180d | 1500 | 0.0722 | 0.0729 | 0.0007 | 1.0 / passed | reported | 0.0316 | 1.00 | 0.02 (0.02) | passed | proceed → proceed | 0.0007 | 0.15833 | 0.0316 / 0.0316 / 0.0265 / 0.0271 |
| `d18b3d84` | <all> | treatment_initiated→persistent_180d | 1500 | 0.0722 | 0.0754 | 0.0033 | 4.5 / passed | reported | 0.0316 | 1.00 | 0.10 (0.10) | passed | proceed → proceed | 0.0027 | 0.11273 | 0.0316 / 0.0316 / 0.0291 / 0.0271 |
| `9260ca38` | <all> | treatment_initiated→persistent_180d | 1500 | 0.0722 | 0.0711 | 0.0011 | 1.5 / passed | reported | 0.0316 | 1.00 | 0.03 (0.03) | passed | proceed → proceed | 0.0058 | 0.42852 | 0.0316 / 0.0316 / 0.0357 / 0.0271 |
| `02546c65` | Fabhalta | acceptance_status→conversion_flag | 5000 | 0.0611 | 0.0612 | 0.0001 | 0.1 / passed | evalue_inv | 0.0110 | 1.00 | 0.01 (0.01) | passed | review → review | 0.0001 | 0.25402 | - / 0.0110 / - / 0.0109 |
| `45b27347` | Fabhalta | control_group_flag→action_taken | 5000 | -0.0786 | -0.0785 | 0.0001 | 0.1 / passed | evalue_inv | 0.0147 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0002 | 0.36164 | - / 0.0147 / - / 0.0149 |
| `1849dea6` | Fabhalta | peer_influence_score→adopted | 5000 | 0.4212 | 0.4212 | 0.0000 | 0.0 / passed | evalue_inv | 0.0126 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0001 | 0.46582 | - / 0.0126 / - / 0.0060 |
| `0fd3afe0` | Fabhalta | treatment_arm→adopted | 5000 | 0.1826 | 0.1867 | 0.0041 | 2.2 / passed | evalue_inv | 0.0133 | 1.00 | 0.31 (0.31) | passed | proceed → proceed | 0.0077 | 0.29716 | - / 0.0133 / - / 0.0132 |
| `29d74728` | Kisqali | acceptance_status→conversion_flag | 5000 | 0.0743 | 0.0743 | 0.0000 | 0.0 / passed | evalue_inv | 0.0112 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0002 | 0.46187 | - / 0.0112 / - / 0.0112 |
| `4e79867b` | Kisqali | acceptance_status→conversion_flag | 5000 | 0.0754 | 0.0755 | 0.0000 | 0.0 / passed | evalue_inv | 0.0113 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0002 | 0.43132 | - / 0.0113 / - / 0.0112 |
| `db089904` | Kisqali | acceptance_status→conversion_flag | 1500 | 0.0350 | 0.0350 | 0.0001 | 0.1 / passed | naive | 0.0210 | 1.00 | 0.00 (0.00) | passed | review → review | 0.0004 | 0.44497 | - / - / - / 0.0210 |
| `745e2359` | Kisqali | acceptance_status→conversion_flag | 5000 | 0.0587 | 0.0586 | 0.0001 | 0.1 / passed | evalue_inv | 0.0111 | 1.00 | 0.01 (0.01) | passed | review → review | 0.0001 | 0.33830 | - / 0.0111 / - / 0.0112 |
| `ca4e8529` | Kisqali | accepted→converted | 37371 (5000) | 0.0352 | 0.0395 | 0.0043 | 12.3 / passed | evalue_inv | 0.0041 | 2.73 | 0.39 (1.07) | passed | review → review | 0.0059 | 0.23149 | - / 0.0041 / - / - |
| `39b03eeb` | Kisqali | accepted→converted | 37515 (5000) | 0.0392 | 0.0452 | 0.0060 | 15.4 / passed | evalue_inv | 0.0041 | 2.74 | 0.54 (1.48) | passed | review → review | 0.1103 | 0.47820 | - / 0.0041 / - / - |
| `c3676bb0` | Kisqali | accepted→converted | 37515 (5000) | 0.0392 | 0.0454 | 0.0062 | 15.7 / passed | evalue_inv | 0.0041 | 2.74 | 0.55 (1.51) | passed | review → review | 0.0232 | 0.39526 | - / 0.0041 / - / - |
| `61fa9429` | Kisqali | control_group_flag→action_taken | 5000 | -0.0710 | -0.0709 | 0.0001 | 0.1 / passed | evalue_inv | 0.0148 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0001 | 0.29199 | - / 0.0148 / - / 0.0149 |
| `fa2c0fde` | Kisqali | control_group_flag→action_taken | 5000 | -0.0797 | -0.0796 | 0.0000 | 0.0 / passed | evalue_inv | 0.0147 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0002 | 0.43789 | - / 0.0147 / - / 0.0149 |
| `2e1a77f5` | Kisqali | control_group_flag→action_taken | 5000 | -0.0635 | -0.0636 | 0.0000 | 0.0 / passed | evalue_inv | 0.0149 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0002 | 0.45096 | - / 0.0149 / - / 0.0149 |
| `930af2f5` | Kisqali | copay_support→adherent_180d | 5000 | 0.0998 | 0.0975 | 0.0023 | 2.3 / passed | evalue_inv | 0.0147 | 1.00 | 0.16 (0.16) | passed | proceed → proceed | 0.0031 | 0.23226 | - / 0.0147 / - / 0.0144 |
| `fd0a46a2` | Kisqali | copay_support→adherent_180d | 5000 | 0.1023 | 0.1035 | 0.0012 | 1.2 / passed | evalue_inv | 0.0147 | 1.00 | 0.08 (0.08) | passed | proceed → proceed | 0.0063 | 0.42339 | - / 0.0147 / - / 0.0144 |
| `a9798c98` | Kisqali | copay_support→low_gap_180d | 5000 | 0.0856 | 0.0856 | 0.0000 | 0.0 / passed | evalue_inv | 0.0141 | 1.00 | 0.00 (0.00) | passed | review → review | 0.0074 | 0.49957 | - / 0.0141 / - / 0.0139 |
| `5d88a6b8` | Kisqali | copay_support→low_gap_180d | 5000 | 0.0802 | 0.0876 | 0.0074 | 9.2 / passed | evalue_inv | 0.0142 | 1.00 | 0.52 (0.52) | passed | review → review | 0.0078 | 0.17382 | - / 0.0142 / - / 0.0139 |
| `2aa18acd` | Kisqali | copay_support→persistent_180d | 5000 | 0.0738 | 0.0690 | 0.0048 | 6.6 / passed | evalue_inv | 0.0143 | 1.00 | 0.34 (0.34) | passed | review → review | 0.0059 | 0.20507 | - / 0.0143 / - / 0.0146 |
| `5eafede3` | Kisqali | copay_support→persistent_180d | 5000 | 0.0810 | 0.0732 | 0.0078 | 9.6 / passed | evalue_inv | 0.0147 | 1.00 | 0.53 (0.53) | passed | review → review | 0.0053 | 0.06991 | - / 0.0147 / - / 0.0146 |
| `0863b867` | Kisqali | disease_stage→persistent_180d | 5000 | -0.1153 | -0.1149 | 0.0004 | 0.4 / passed | evalue_inv | 0.0134 | 1.00 | 0.03 (0.03) | passed | proceed → proceed | 0.0077 | 0.47807 | - / 0.0134 / - / 0.0139 |
| `7d93bf6c` | Kisqali | disease_stage→persistent_180d | 5000 | -0.1283 | -0.1233 | 0.0050 | 3.9 / passed | evalue_inv | 0.0132 | 1.00 | 0.38 (0.38) | passed | proceed → proceed | 0.0076 | 0.25397 | - / 0.0132 / - / 0.0139 |
| `0f6dc974` | Kisqali | peer_influence_score→adopted | 5000 | 0.3920 | 0.3920 | 0.0000 | 0.0 / passed | evalue_inv | 0.0127 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0001 | 0.41074 | - / 0.0127 / - / 0.0061 |
| `bb2e6b6f` | Kisqali | peer_influence_score→adopted | 5000 | 0.3468 | 0.3468 | 0.0000 | 0.0 / passed | evalue_inv | 0.0130 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0001 | 0.38387 | - / 0.0130 / - / 0.0061 |
| `1fd63390` | Kisqali | psp_enrolled→adherent_180d | 5000 | 0.0538 | 0.0567 | 0.0029 | 5.4 / passed | evalue_inv | 0.0141 | 1.00 | 0.21 (0.21) | passed | review → review | 0.0095 | 0.37897 | - / 0.0141 / - / 0.0140 |
| `1897f19c` | Kisqali | psp_enrolled→adherent_180d | 5000 | 0.0723 | 0.0661 | 0.0062 | 8.5 / passed | evalue_inv | 0.0141 | 1.00 | 0.44 (0.44) | passed | review → review | 0.0037 | 0.04798 | - / 0.0141 / - / 0.0140 |
| `c5c0965a` | Kisqali | psp_enrolled→persistent_180d | 5000 | 0.0697 | 0.0672 | 0.0025 | 3.5 / passed | evalue_inv | 0.0143 | 1.00 | 0.17 (0.17) | passed | review → review | 0.0064 | 0.35030 | - / 0.0143 / - / 0.0143 |
| `b81ff097` | Kisqali | psp_enrolled→persistent_180d | 5000 | 0.0629 | 0.0650 | 0.0021 | 3.3 / passed | evalue_inv | 0.0143 | 1.00 | 0.15 (0.15) | passed | review → review | 0.0043 | 0.31354 | - / 0.0143 / - / 0.0143 |
| `5559f1dc` | Kisqali | rep_detailing_high→treatment_initiated | 5000 | 0.0641 | 0.0631 | 0.0010 | 1.6 / passed | evalue_inv | 0.0133 | 1.00 | 0.08 (0.08) | passed | review → review | 0.0041 | 0.39875 | - / 0.0133 / - / 0.0134 |
| `d5165fbf` | Kisqali | rep_detailing_high→treatment_initiated | 5000 | 0.0435 | 0.0507 | 0.0072 | 16.5 / passed | evalue_inv | 0.0133 | 1.00 | 0.54 (0.54) | passed | review → review | 0.0033 | 0.01360 | - / 0.0133 / - / 0.0134 |
| `d17611ba` | Kisqali | sample_dropped→treatment_initiated | 5000 | 0.0426 | 0.0389 | 0.0037 | 8.6 / passed | evalue_inv | 0.0139 | 1.00 | 0.26 (0.26) | passed | review → review | 0.0051 | 0.23869 | - / 0.0139 / - / 0.0140 |
| `16b528d9` | Kisqali | treatment_arm→adopted | 5000 | 0.1410 | 0.1396 | 0.0014 | 1.0 / passed | evalue_inv | 0.0129 | 1.00 | 0.11 (0.11) | passed | proceed → proceed | 0.0072 | 0.42066 | - / 0.0129 / - / 0.0135 |
| `8eedd3e2` | Kisqali | treatment_arm→adopted | 5000 | 0.1094 | 0.1134 | 0.0041 | 3.7 / passed | evalue_inv | 0.0134 | 1.00 | 0.30 (0.30) | passed | proceed → proceed | 0.0050 | 0.21090 | - / 0.0134 / - / 0.0135 |
| `a78ef2b6` | Kisqali | treatment_arm→adopted | 1500 | 0.2634 | 0.2635 | 0.0000 | 0.0 / passed | evalue_inv | 0.0244 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0003 | 0.44647 | - / 0.0244 / - / 0.0244 |
| `4a2303f4` | Kisqali | treatment_arm→persistent_180d | 5000 | 0.1581 | 0.1670 | 0.0089 | 5.6 / passed | evalue_inv | 0.0182 | 1.00 | 0.49 (0.49) | passed | proceed → proceed | 0.0065 | 0.08607 | - / 0.0182 / - / 0.0177 |
| `9d98ad31` | Kisqali | treatment_arm→persistent_180d | 5000 | 0.1533 | 0.1507 | 0.0025 | 1.7 / passed | evalue_inv | 0.0184 | 1.00 | 0.14 (0.14) | passed | proceed → proceed | 0.0050 | 0.30623 | - / 0.0184 / - / 0.0177 |
| `bd8ce86f` | Kisqali | treatment_arm→treatment_initiated | 5000 | 0.1569 | 0.1594 | 0.0025 | 1.6 / passed | evalue_inv | 0.0169 | 1.00 | 0.15 (0.15) | passed | proceed → proceed | 0.0068 | 0.35683 | - / 0.0169 / - / 0.0187 |
| `b0a9dceb` | Kisqali | treatment_arm→treatment_initiated | 5000 | 0.1624 | 0.1602 | 0.0022 | 1.3 / passed | evalue_inv | 0.0165 | 1.00 | 0.13 (0.13) | passed | proceed → proceed | 0.0030 | 0.23499 | - / 0.0165 / - / 0.0187 |
| `13ad2b2a` | Kisqali | trigger_accepted→treatment_initiated | 5000 | 0.0721 | 0.0741 | 0.0020 | 2.7 / passed | evalue_inv | 0.0132 | 1.00 | 0.15 (0.15) | passed | review → review | 0.0043 | 0.32453 | - / 0.0132 / - / 0.0132 |
| `bfd26231` | Kisqali | trigger_accepted→treatment_initiated | 5000 | 0.0669 | 0.0706 | 0.0037 | 5.6 / passed | evalue_inv | 0.0129 | 1.00 | 0.29 (0.29) | passed | review → review | 0.0021 | 0.04050 | - / 0.0129 / - / 0.0132 |
| `9a7dece1` | Remibrutinib | acceptance_status→conversion_flag | 5000 | 0.0764 | 0.0764 | 0.0000 | 0.0 / passed | evalue_inv | 0.0111 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0001 | 0.35395 | - / 0.0111 / - / 0.0113 |
| `6321aa5a` | Remibrutinib | acceptance_status→conversion_flag | 5000 | 0.0726 | 0.0726 | 0.0000 | 0.0 / passed | evalue_inv | 0.0111 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0000 | 0.49925 | - / 0.0111 / - / 0.0113 |
| `674b8506` | Remibrutinib | control_group_flag→action_taken | 5000 | -0.0909 | -0.0908 | 0.0000 | 0.0 / passed | evalue_inv | 0.0148 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0005 | 0.46603 | - / 0.0148 / - / 0.0149 |
| `b9d646da` | Remibrutinib | control_group_flag→action_taken | 5000 | -0.0967 | -0.0968 | 0.0000 | 0.0 / passed | evalue_inv | 0.0147 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0001 | 0.42152 | - / 0.0147 / - / 0.0149 |
| `daedc7ab` | Remibrutinib | copay_support→adherent_180d | 5000 | 0.0885 | 0.0964 | 0.0080 | 9.0 / passed | evalue_inv | 0.0148 | 1.00 | 0.54 (0.54) | passed | review → review | 0.0076 | 0.14811 | - / 0.0148 / - / 0.0143 |
| `eac854c6` | Remibrutinib | copay_support→adherent_180d | 5000 | 0.1042 | 0.1013 | 0.0029 | 2.7 / passed | evalue_inv | 0.0148 | 1.00 | 0.19 (0.19) | passed | proceed → proceed | 0.0059 | 0.31241 | - / 0.0148 / - / 0.0143 |
| `23a6fcb4` | Remibrutinib | copay_support→adherent_180d | 5000 | 0.1042 | 0.1040 | 0.0002 | 0.2 / passed | evalue_inv | 0.0148 | 1.00 | 0.01 (0.01) | passed | proceed → proceed | 0.0047 | 0.48552 | - / 0.0148 / - / 0.0143 |
| `3e7a6340` | Remibrutinib | copay_support→adherent_180d | 5000 | 0.0988 | 0.0953 | 0.0035 | 3.6 / passed | evalue_inv | 0.0144 | 1.00 | 0.24 (0.24) | passed | proceed → proceed | 0.0038 | 0.17495 | - / 0.0144 / - / 0.0143 |
| `9d9ad690` | Remibrutinib | copay_support→adherent_180d | 5000 | 0.0988 | 0.0955 | 0.0034 | 3.4 / passed | reported | 0.0144 | 1.00 | 0.23 (0.23) | passed | proceed → proceed | 0.0069 | 0.31249 | 0.0144 / 0.0144 / 0.0162 / 0.0143 |
| `c28f062b` | Remibrutinib | copay_support→adherent_180d | 5000 | 0.0988 | 0.1012 | 0.0023 | 2.4 / passed | reported | 0.0144 | 1.00 | 0.16 (0.16) | passed | proceed → proceed | 0.0061 | 0.35151 | 0.0144 / 0.0144 / 0.0206 / 0.0143 |
| `1efb753c` | Remibrutinib | copay_support→low_gap_180d | 5000 | 0.0928 | 0.0951 | 0.0023 | 2.5 / passed | evalue_inv | 0.0141 | 1.00 | 0.16 (0.16) | passed | proceed → proceed | 0.0059 | 0.34654 | - / 0.0141 / - / 0.0139 |
| `1da3a77f` | Remibrutinib | copay_support→low_gap_180d | 5000 | 0.1053 | 0.1036 | 0.0016 | 1.6 / passed | evalue_inv | 0.0143 | 1.00 | 0.12 (0.12) | passed | proceed → proceed | 0.0057 | 0.38594 | - / 0.0143 / - / 0.0139 |
| `03779dc5` | Remibrutinib | copay_support→low_gap_180d | 5000 | 0.1053 | 0.1017 | 0.0036 | 3.4 / passed | evalue_inv | 0.0143 | 1.00 | 0.25 (0.25) | passed | proceed → proceed | 0.0076 | 0.31920 | - / 0.0143 / - / 0.0139 |
| `466751b8` | Remibrutinib | copay_support→low_gap_180d | 5000 | 0.1032 | 0.0992 | 0.0040 | 3.9 / passed | evalue_inv | 0.0140 | 1.00 | 0.29 (0.29) | passed | proceed → proceed | 0.0042 | 0.16807 | - / 0.0140 / - / 0.0139 |
| `020cdeef` | Remibrutinib | copay_support→low_gap_180d | 5000 | 0.1032 | 0.0992 | 0.0040 | 3.9 / passed | reported | 0.0140 | 1.00 | 0.28 (0.28) | passed | proceed → proceed | 0.0049 | 0.20885 | 0.0140 / 0.0140 / 0.0174 / 0.0139 |
| `1bab6aaf` | Remibrutinib | copay_support→low_gap_180d | 5000 | 0.1032 | 0.0974 | 0.0058 | 5.6 / passed | reported | 0.0140 | 1.00 | 0.42 (0.42) | passed | proceed → proceed | 0.0059 | 0.16031 | 0.0140 / 0.0140 / 0.0174 / 0.0139 |
| `d40cf304` | Remibrutinib | copay_support→persistent_180d | 5000 | 0.0856 | 0.0755 | 0.0101 | 11.8 / passed | evalue_inv | 0.0146 | 1.00 | 0.69 (0.69) | passed | review → review | 0.0038 | 0.00365 | - / 0.0146 / - / 0.0147 |
| `8f66f069` | Remibrutinib | copay_support→persistent_180d | 5000 | 0.0789 | 0.0776 | 0.0013 | 1.6 / passed | evalue_inv | 0.0148 | 1.00 | 0.09 (0.09) | passed | review → review | 0.0058 | 0.41160 | - / 0.0148 / - / 0.0147 |
| `310ebb76` | Remibrutinib | copay_support→persistent_180d | 5000 | 0.0789 | 0.0766 | 0.0023 | 3.0 / passed | evalue_inv | 0.0148 | 1.00 | 0.16 (0.16) | passed | review → review | 0.0061 | 0.35163 | - / 0.0148 / - / 0.0147 |
| `c74451d7` | Remibrutinib | copay_support→persistent_180d | 5000 | 0.0663 | 0.0652 | 0.0011 | 1.6 / passed | evalue_inv | 0.0146 | 1.00 | 0.07 (0.07) | passed | review → review | 0.0062 | 0.43152 | - / 0.0146 / - / 0.0147 |
| `c98828c4` | Remibrutinib | copay_support→persistent_180d | 5000 | 0.0663 | 0.0691 | 0.0028 | 4.3 / passed | reported | 0.0146 | 1.00 | 0.19 (0.19) | passed | proceed → proceed | 0.0064 | 0.32993 | 0.0146 / 0.0146 / 0.0150 / 0.0147 |
| `08cfeabd` | Remibrutinib | copay_support→persistent_180d | 5000 | 0.0663 | 0.0678 | 0.0015 | 2.3 / passed | reported | 0.0146 | 1.00 | 0.11 (0.11) | passed | proceed → proceed | 0.0038 | 0.34312 | 0.0146 / 0.0146 / 0.0186 / 0.0147 |
| `643d175a` | Remibrutinib | peer_influence_score→adopted | 5000 | 0.3676 | 0.3675 | 0.0001 | 0.0 / passed | evalue_inv | 0.0128 | 1.00 | 0.01 (0.01) | passed | proceed → proceed | 0.0002 | 0.29570 | - / 0.0128 / - / 0.0060 |
| `a61f4911` | Remibrutinib | peer_influence_score→adopted | 5000 | 0.3676 | 0.3677 | 0.0001 | 0.0 / passed | evalue_inv | 0.0128 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0001 | 0.35404 | - / 0.0128 / - / 0.0060 |
| `9b7dc08b` | Remibrutinib | peer_influence_score→adopted | 5000 | 0.3676 | 0.3676 | 0.0000 | 0.0 / passed | evalue_inv | 0.0128 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0001 | 0.29804 | - / 0.0128 / - / 0.0060 |
| `af68739c` | Remibrutinib | psp_enrolled→adherent_180d | 5000 | 0.0891 | 0.0874 | 0.0018 | 2.0 / passed | evalue_inv | 0.0144 | 1.00 | 0.12 (0.12) | passed | review → review | 0.0051 | 0.36365 | - / 0.0144 / - / 0.0141 |
| `eae696c9` | Remibrutinib | psp_enrolled→adherent_180d | 5000 | 0.0885 | 0.0849 | 0.0036 | 4.1 / passed | evalue_inv | 0.0143 | 1.00 | 0.25 (0.25) | passed | review → review | 0.0050 | 0.23379 | - / 0.0143 / - / 0.0141 |
| `04337c7f` | Remibrutinib | psp_enrolled→adherent_180d | 5000 | 0.0885 | 0.0844 | 0.0041 | 4.7 / passed | evalue_inv | 0.0143 | 1.00 | 0.29 (0.29) | passed | review → review | 0.0058 | 0.23875 | - / 0.0143 / - / 0.0141 |
| `303bf57c` | Remibrutinib | psp_enrolled→adherent_180d | 5000 | 0.0840 | 0.0838 | 0.0001 | 0.2 / passed | evalue_inv | 0.0144 | 1.00 | 0.01 (0.01) | passed | review → review | 0.0005 | 0.37849 | - / 0.0144 / - / 0.0141 |
| `090fc07e` | Remibrutinib | psp_enrolled→adherent_180d | 5000 | 0.0840 | 0.0861 | 0.0022 | 2.6 / passed | reported | 0.0144 | 1.00 | 0.15 (0.15) | passed | proceed → proceed | 0.0070 | 0.37869 | 0.0144 / 0.0144 / 0.0153 / 0.0141 |
| `de10133e` | Remibrutinib | psp_enrolled→adherent_180d | 5000 | 0.0840 | 0.0831 | 0.0009 | 1.0 / passed | reported | 0.0144 | 1.00 | 0.06 (0.06) | passed | proceed → proceed | 0.0029 | 0.38319 | 0.0144 / 0.0144 / 0.0173 / 0.0141 |
| `324e584d` | Remibrutinib | psp_enrolled→persistent_180d | 5000 | 0.0598 | 0.0612 | 0.0014 | 2.3 / passed | evalue_inv | 0.0146 | 1.00 | 0.09 (0.09) | passed | review → review | 0.0017 | 0.21069 | - / 0.0146 / - / 0.0145 |
| `e5497265` | Remibrutinib | psp_enrolled→persistent_180d | 5000 | 0.0690 | 0.0667 | 0.0023 | 3.3 / passed | evalue_inv | 0.0145 | 1.00 | 0.16 (0.16) | passed | review → review | 0.0057 | 0.34326 | - / 0.0145 / - / 0.0145 |
| `af5baba6` | Remibrutinib | psp_enrolled→persistent_180d | 5000 | 0.0690 | 0.0664 | 0.0026 | 3.8 / passed | evalue_inv | 0.0145 | 1.00 | 0.18 (0.18) | passed | review → review | 0.0038 | 0.24407 | - / 0.0145 / - / 0.0145 |
| `3105db99` | Remibrutinib | psp_enrolled→persistent_180d | 5000 | 0.0830 | 0.0777 | 0.0053 | 6.4 / passed | evalue_inv | 0.0148 | 1.00 | 0.36 (0.36) | passed | review → review | 0.0045 | 0.12253 | - / 0.0148 / - / 0.0145 |
| `742320ff` | Remibrutinib | psp_enrolled→persistent_180d | 5000 | 0.0830 | 0.0763 | 0.0067 | 8.1 / passed | reported | 0.0148 | 1.00 | 0.46 (0.46) | passed | proceed → proceed | 0.0050 | 0.08855 | 0.0148 / 0.0148 / 0.0159 / 0.0145 |
| `c92426dc` | Remibrutinib | psp_enrolled→persistent_180d | 5000 | 0.0830 | 0.0747 | 0.0083 | 10.0 / passed | reported | 0.0148 | 1.00 | 0.56 (0.56) | passed | proceed → proceed | 0.0049 | 0.04290 | 0.0148 / 0.0148 / 0.0163 / 0.0145 |
| `f4396ad9` | Remibrutinib | rep_detailing_high→treatment_initiated | 5000 | 0.0766 | 0.0692 | 0.0074 | 9.7 / passed | evalue_inv | 0.0135 | 1.00 | 0.55 (0.55) | passed | review → review | 0.0062 | 0.11547 | - / 0.0135 / - / 0.0134 |
| `c7e00c92` | Remibrutinib | rep_detailing_high→treatment_initiated | 5000 | 0.0671 | 0.0657 | 0.0014 | 2.1 / passed | evalue_inv | 0.0135 | 1.00 | 0.11 (0.11) | passed | review → review | 0.0035 | 0.34053 | - / 0.0135 / - / 0.0134 |
| `2673da7f` | Remibrutinib | rep_detailing_high→treatment_initiated | 5000 | 0.0671 | 0.0706 | 0.0034 | 5.1 / passed | evalue_inv | 0.0135 | 1.00 | 0.25 (0.25) | passed | review → review | 0.0064 | 0.29832 | - / 0.0135 / - / 0.0134 |
| `7d7aa027` | Remibrutinib | rep_detailing_high→treatment_initiated | 5000 | 0.0723 | 0.0674 | 0.0049 | 6.7 / passed | evalue_inv | 0.0135 | 1.00 | 0.36 (0.36) | passed | review → review | 0.0073 | 0.25140 | - / 0.0135 / - / 0.0134 |
| `ed5ea6e9` | Remibrutinib | rep_detailing_high→treatment_initiated | 5000 | 0.0723 | 0.0646 | 0.0077 | 10.7 / passed | reported | 0.0135 | 1.00 | 0.57 (0.57) | passed | proceed → proceed | 0.0057 | 0.08861 | 0.0135 / 0.0135 / 0.0176 / 0.0134 |
| `f8e62229` | Remibrutinib | rep_detailing_high→treatment_initiated | 5000 | 0.0723 | 0.0650 | 0.0073 | 10.1 / passed | reported | 0.0135 | 1.00 | 0.54 (0.54) | passed | proceed → proceed | 0.0069 | 0.14391 | 0.0135 / 0.0135 / 0.0198 / 0.0134 |
| `1e176cd7` | Remibrutinib | sample_dropped→treatment_initiated | 5000 | 0.0366 | 0.0406 | 0.0041 | 11.2 / passed | evalue_inv | 0.0137 | 1.00 | 0.30 (0.30) | passed | review → review | 0.0068 | 0.27309 | - / 0.0137 / - / 0.0140 |
| `5aac098f` | Remibrutinib | sample_dropped→treatment_initiated | 5000 | 0.0417 | 0.0399 | 0.0018 | 4.3 / passed | evalue_inv | 0.0135 | 1.00 | 0.13 (0.13) | passed | review → review | 0.0060 | 0.38087 | - / 0.0135 / - / 0.0140 |
| `94e0f4b2` | Remibrutinib | sample_dropped→treatment_initiated | 5000 | 0.0417 | 0.0429 | 0.0013 | 3.0 / passed | evalue_inv | 0.0135 | 1.00 | 0.09 (0.09) | passed | review → review | 0.0075 | 0.43342 | - / 0.0135 / - / 0.0140 |
| `5143e819` | Remibrutinib | sample_dropped→treatment_initiated | 5000 | 0.0425 | 0.0344 | 0.0081 | 19.0 / passed | evalue_inv | 0.0138 | 1.00 | 0.59 (0.59) | passed | review → review | 0.0055 | 0.06932 | - / 0.0138 / - / 0.0140 |
| `b6b5b1c4` | Remibrutinib | sample_dropped→treatment_initiated | 5000 | 0.0425 | 0.0389 | 0.0036 | 8.4 / passed | reported | 0.0138 | 1.00 | 0.26 (0.26) | passed | proceed → proceed | 0.0066 | 0.29440 | 0.0138 / 0.0138 / 0.0151 / 0.0140 |
| `24c1c101` | Remibrutinib | sample_dropped→treatment_initiated | 5000 | 0.0425 | 0.0365 | 0.0060 | 14.1 / passed | reported | 0.0138 | 1.00 | 0.44 (0.44) | passed | proceed → proceed | 0.0055 | 0.13827 | 0.0138 / 0.0138 / 0.0176 / 0.0140 |
| `fd34c6d6` | Remibrutinib | treatment_arm→adopted | 5000 | 0.1469 | 0.1457 | 0.0013 | 0.9 / passed | evalue_inv | 0.0134 | 1.00 | 0.10 (0.10) | passed | proceed → proceed | 0.0046 | 0.39097 | - / 0.0134 / - / 0.0133 |
| `d28af0f3` | Remibrutinib | treatment_arm→adopted | 5000 | 0.1469 | 0.1432 | 0.0037 | 2.5 / passed | evalue_inv | 0.0134 | 1.00 | 0.28 (0.28) | passed | proceed → proceed | 0.0050 | 0.22852 | - / 0.0134 / - / 0.0133 |
| `90b892c1` | Remibrutinib | treatment_arm→adopted | 5000 | 0.1469 | 0.1470 | 0.0001 | 0.0 / passed | evalue_inv | 0.0134 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0089 | 0.49758 | - / 0.0134 / - / 0.0133 |
| `46faebfb` | Remibrutinib | treatment_arm→persistent_180d | 5000 | 0.0867 | 0.0893 | 0.0026 | 3.0 / passed | evalue_inv | 0.0199 | 1.00 | 0.13 (0.13) | passed | review → review | 0.0101 | 0.39660 | - / 0.0199 / - / 0.0188 |
| `05507f0d` | Remibrutinib | treatment_arm→persistent_180d | 1500 | 0.1256 | 0.1195 | 0.0061 | 4.9 / passed | evalue_inv | 0.0324 | 1.00 | 0.19 (0.19) | passed | review → review | 0.0076 | 0.21062 | - / 0.0324 / - / 0.0331 |
| `02c40d84` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.2068 | 0.2026 | 0.0042 | 2.0 / passed | evalue_inv | 0.0179 | 1.00 | 0.24 (0.24) | passed | proceed → proceed | 0.0035 | 0.11259 | - / 0.0179 / - / 0.0186 |
| `e9942378` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.1821 | 0.1879 | 0.0058 | 3.2 / passed | evalue_inv | 0.0177 | 1.00 | 0.33 (0.33) | passed | proceed → proceed | 0.0068 | 0.19630 | - / 0.0177 / - / 0.0186 |
| `dd0419ad` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.1821 | 0.1872 | 0.0051 | 2.8 / passed | evalue_inv | 0.0177 | 1.00 | 0.29 (0.29) | passed | proceed → proceed | 0.0109 | 0.32008 | - / 0.0177 / - / 0.0186 |
| `3f5c6ba6` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.1821 | 0.1873 | 0.0052 | 2.8 / passed | evalue_inv | 0.0177 | 1.00 | 0.29 (0.29) | passed | proceed → proceed | 0.0053 | 0.16378 | - / 0.0177 / - / 0.0186 |
| `4bc1effd` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.1821 | 0.1868 | 0.0046 | 2.6 / passed | evalue_inv | 0.0177 | 1.00 | 0.26 (0.26) | passed | proceed → proceed | 0.0039 | 0.11376 | - / 0.0177 / - / 0.0186 |
| `202ec0e7` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.1821 | 0.1868 | 0.0047 | 2.6 / passed | evalue_inv | 0.0177 | 1.00 | 0.26 (0.26) | passed | proceed → proceed | 0.0054 | 0.19541 | - / 0.0177 / - / 0.0186 |
| `4a6e6302` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.1821 | 0.1896 | 0.0075 | 4.1 / passed | evalue_inv | 0.0177 | 1.00 | 0.42 (0.42) | passed | proceed → proceed | 0.0079 | 0.17147 | - / 0.0177 / - / 0.0186 |
| `d9e40f34` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.1915 | 0.1908 | 0.0007 | 0.4 / passed | evalue_inv | 0.0177 | 1.00 | 0.04 (0.04) | passed | proceed → proceed | 0.0049 | 0.43990 | - / 0.0177 / - / 0.0186 |
| `41e8a64a` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.1915 | 0.1897 | 0.0018 | 0.9 / passed | reported | 0.0177 | 1.00 | 0.10 (0.10) | passed | proceed → proceed | 0.0023 | 0.21565 | 0.0177 / 0.0177 / 0.0170 / 0.0186 |
| `1bb7f7ce` | Remibrutinib | treatment_arm→treatment_initiated | 5000 | 0.1915 | 0.1890 | 0.0025 | 1.3 / passed | reported | 0.0177 | 1.00 | 0.14 (0.14) | passed | proceed → proceed | 0.0034 | 0.22971 | 0.0177 / 0.0177 / 0.0185 / 0.0186 |
| `8d54bf77` | Remibrutinib | trigger_accepted→treatment_initiated | 5000 | 0.0946 | 0.0931 | 0.0015 | 1.6 / passed | evalue_inv | 0.0131 | 1.00 | 0.12 (0.12) | passed | proceed → proceed | 0.0012 | 0.09656 | - / 0.0131 / - / 0.0132 |
| `1f9b82c2` | Remibrutinib | trigger_accepted→treatment_initiated | 5000 | 0.0923 | 0.0919 | 0.0004 | 0.4 / passed | evalue_inv | 0.0131 | 1.00 | 0.03 (0.03) | passed | proceed → proceed | 0.0042 | 0.46458 | - / 0.0131 / - / 0.0132 |
| `9a67513b` | Remibrutinib | trigger_accepted→treatment_initiated | 5000 | 0.0923 | 0.0887 | 0.0036 | 3.9 / passed | evalue_inv | 0.0131 | 1.00 | 0.28 (0.28) | passed | proceed → proceed | 0.0056 | 0.26111 | - / 0.0131 / - / 0.0132 |
| `de237134` | Remibrutinib | trigger_accepted→treatment_initiated | 5000 | 0.0960 | 0.0979 | 0.0019 | 2.0 / passed | evalue_inv | 0.0133 | 1.00 | 0.15 (0.15) | passed | proceed → proceed | 0.0040 | 0.31606 | - / 0.0133 / - / 0.0132 |
| `7c36f63f` | Remibrutinib | trigger_accepted→treatment_initiated | 5000 | 0.0960 | 0.0959 | 0.0001 | 0.1 / passed | reported | 0.0133 | 1.00 | 0.00 (0.00) | passed | proceed → proceed | 0.0041 | 0.49422 | 0.0133 / 0.0133 / 0.0155 / 0.0132 |
| `7a8f0753` | Remibrutinib | trigger_accepted→treatment_initiated | 5000 | 0.0960 | 0.0964 | 0.0004 | 0.4 / passed | reported | 0.0133 | 1.00 | 0.03 (0.03) | passed | proceed → proceed | 0.0013 | 0.37911 | 0.0133 / 0.0133 / 0.0122 / 0.0132 |
| `e92eaf3e` | Remibrutinib | urticaria_severity_uas7→persistent_180d | 5000 | 0.1370 | 0.1403 | 0.0033 | 2.4 / passed | evalue_inv | 0.0135 | 1.00 | 0.24 (0.24) | passed | proceed → proceed | 0.0060 | 0.29085 | - / 0.0135 / - / 0.0140 |
| `3783d448` | Remibrutinib | urticaria_severity_uas7→persistent_180d | 5000 | 0.1507 | 0.1494 | 0.0014 | 0.9 / passed | evalue_inv | 0.0133 | 1.00 | 0.10 (0.10) | passed | proceed → proceed | 0.0047 | 0.38718 | - / 0.0133 / - / 0.0140 |
| `ed376940` | Remibrutinib | urticaria_severity_uas7→persistent_180d | 5000 | 0.1507 | 0.1480 | 0.0027 | 1.8 / passed | evalue_inv | 0.0133 | 1.00 | 0.20 (0.20) | passed | proceed → proceed | 0.0056 | 0.31526 | - / 0.0133 / - / 0.0140 |
| `c9895f97` | Remibrutinib | urticaria_severity_uas7→persistent_180d | 5000 | 0.1505 | 0.1436 | 0.0070 | 4.6 / passed | evalue_inv | 0.0135 | 1.00 | 0.52 (0.52) | passed | proceed → proceed | 0.0075 | 0.17835 | - / 0.0135 / - / 0.0140 |
| `3de14338` | Remibrutinib | urticaria_severity_uas7→persistent_180d | 5000 | 0.1505 | 0.1430 | 0.0076 | 5.0 / passed | reported | 0.0135 | 1.00 | 0.56 (0.56) | passed | proceed → proceed | 0.0057 | 0.09341 | 0.0135 / 0.0135 / 0.0124 / 0.0140 |
| `98d63b21` | Remibrutinib | urticaria_severity_uas7→persistent_180d | 5000 | 0.1505 | 0.1443 | 0.0062 | 4.1 / passed | reported | 0.0135 | 1.00 | 0.46 (0.46) | passed | proceed → proceed | 0.0066 | 0.17243 | 0.0135 / 0.0135 / 0.0202 / 0.0140 |

## Per pair

| brand | pair | runs | today status | new status | gate today | gate new | median |ATE| | median shift_se | max shift_se | se sources |
|---|---|---|---|---|---|---|---|---|---|---|
| <all> | treatment_arm→persistent_180d | 7 | {'passed': 7} | {'passed': 7} | {'review': 3, 'proceed': 4} | {'review': 3, 'proceed': 4} | 0.0856 | 0.09 | 0.43 | {'evalue_inv': 3, 'reported': 4} |
| <all> | treatment_initiated→persistent_180d | 6 | {'passed': 6} | {'passed': 6} | {'review': 3, 'proceed': 3} | {'review': 3, 'proceed': 3} | 0.0837 | 0.10 | 0.29 | {'evalue_inv': 3, 'reported': 3} |
| Fabhalta | acceptance_status→conversion_flag | 1 | {'passed': 1} | {'passed': 1} | {'review': 1} | {'review': 1} | 0.0611 | 0.01 | 0.01 | {'evalue_inv': 1} |
| Fabhalta | control_group_flag→action_taken | 1 | {'passed': 1} | {'passed': 1} | {'proceed': 1} | {'proceed': 1} | 0.0786 | 0.00 | 0.00 | {'evalue_inv': 1} |
| Fabhalta | peer_influence_score→adopted | 1 | {'passed': 1} | {'passed': 1} | {'proceed': 1} | {'proceed': 1} | 0.4212 | 0.00 | 0.00 | {'evalue_inv': 1} |
| Fabhalta | treatment_arm→adopted | 1 | {'passed': 1} | {'passed': 1} | {'proceed': 1} | {'proceed': 1} | 0.1826 | 0.31 | 0.31 | {'evalue_inv': 1} |
| Kisqali | acceptance_status→conversion_flag | 4 | {'passed': 4} | {'passed': 4} | {'proceed': 2, 'review': 2} | {'proceed': 2, 'review': 2} | 0.0665 | 0.00 | 0.01 | {'evalue_inv': 3, 'naive': 1} |
| Kisqali | accepted→converted | 3 | {'passed': 3} | {'passed': 3} | {'review': 3} | {'review': 3} | 0.0392 | 0.54 | 0.55 | {'evalue_inv': 3} |
| Kisqali | control_group_flag→action_taken | 3 | {'passed': 3} | {'passed': 3} | {'proceed': 3} | {'proceed': 3} | 0.0710 | 0.00 | 0.00 | {'evalue_inv': 3} |
| Kisqali | copay_support→adherent_180d | 2 | {'passed': 2} | {'passed': 2} | {'proceed': 2} | {'proceed': 2} | 0.1010 | 0.12 | 0.16 | {'evalue_inv': 2} |
| Kisqali | copay_support→low_gap_180d | 2 | {'passed': 2} | {'passed': 2} | {'review': 2} | {'review': 2} | 0.0829 | 0.26 | 0.52 | {'evalue_inv': 2} |
| Kisqali | copay_support→persistent_180d | 2 | {'passed': 2} | {'passed': 2} | {'review': 2} | {'review': 2} | 0.0774 | 0.43 | 0.53 | {'evalue_inv': 2} |
| Kisqali | disease_stage→persistent_180d | 2 | {'passed': 2} | {'passed': 2} | {'proceed': 2} | {'proceed': 2} | 0.1218 | 0.21 | 0.38 | {'evalue_inv': 2} |
| Kisqali | peer_influence_score→adopted | 3 | {'failed': 1, 'passed': 2} | {'failed': 1, 'passed': 2} | {'block': 1, 'proceed': 2} | {'block': 1, 'proceed': 2} | 0.3468 | 0.00 | 2.44 | {'naive': 1, 'evalue_inv': 2} |
| Kisqali | psp_enrolled→adherent_180d | 2 | {'passed': 2} | {'passed': 2} | {'review': 2} | {'review': 2} | 0.0630 | 0.32 | 0.44 | {'evalue_inv': 2} |
| Kisqali | psp_enrolled→persistent_180d | 2 | {'passed': 2} | {'passed': 2} | {'review': 2} | {'review': 2} | 0.0663 | 0.16 | 0.17 | {'evalue_inv': 2} |
| Kisqali | rep_detailing_high→treatment_initiated | 2 | {'passed': 2} | {'passed': 2} | {'review': 2} | {'review': 2} | 0.0538 | 0.31 | 0.54 | {'evalue_inv': 2} |
| Kisqali | sample_dropped→treatment_initiated | 2 | {'failed': 1, 'passed': 1} | {'warning': 1, 'passed': 1} | {'block': 1, 'review': 1} | {'review': 2} | 0.0278 | 0.71 | 1.15 | {'naive': 1, 'evalue_inv': 1} |
| Kisqali | treatment_arm→adopted | 3 | {'passed': 3} | {'passed': 3} | {'proceed': 3} | {'proceed': 3} | 0.1410 | 0.11 | 0.30 | {'evalue_inv': 3} |
| Kisqali | treatment_arm→persistent_180d | 2 | {'passed': 2} | {'passed': 2} | {'proceed': 2} | {'proceed': 2} | 0.1557 | 0.31 | 0.49 | {'evalue_inv': 2} |
| Kisqali | treatment_arm→treatment_initiated | 2 | {'passed': 2} | {'passed': 2} | {'proceed': 2} | {'proceed': 2} | 0.1597 | 0.14 | 0.15 | {'evalue_inv': 2} |
| Kisqali | trigger_accepted→treatment_initiated | 2 | {'passed': 2} | {'passed': 2} | {'review': 2} | {'review': 2} | 0.0695 | 0.22 | 0.29 | {'evalue_inv': 2} |
| Remibrutinib | acceptance_status→conversion_flag | 2 | {'passed': 2} | {'passed': 2} | {'proceed': 2} | {'proceed': 2} | 0.0745 | 0.00 | 0.00 | {'evalue_inv': 2} |
| Remibrutinib | control_group_flag→action_taken | 2 | {'passed': 2} | {'passed': 2} | {'proceed': 2} | {'proceed': 2} | 0.0938 | 0.00 | 0.00 | {'evalue_inv': 2} |
| Remibrutinib | copay_support→adherent_180d | 6 | {'passed': 6} | {'passed': 6} | {'review': 1, 'proceed': 5} | {'review': 1, 'proceed': 5} | 0.0988 | 0.21 | 0.54 | {'evalue_inv': 4, 'reported': 2} |
| Remibrutinib | copay_support→low_gap_180d | 6 | {'passed': 6} | {'passed': 6} | {'proceed': 6} | {'proceed': 6} | 0.1032 | 0.27 | 0.42 | {'evalue_inv': 4, 'reported': 2} |
| Remibrutinib | copay_support→persistent_180d | 6 | {'passed': 6} | {'passed': 6} | {'review': 4, 'proceed': 2} | {'review': 4, 'proceed': 2} | 0.0726 | 0.13 | 0.69 | {'evalue_inv': 4, 'reported': 2} |
| Remibrutinib | peer_influence_score→adopted | 3 | {'passed': 3} | {'passed': 3} | {'proceed': 3} | {'proceed': 3} | 0.3676 | 0.00 | 0.01 | {'evalue_inv': 3} |
| Remibrutinib | psp_enrolled→adherent_180d | 6 | {'passed': 6} | {'passed': 6} | {'review': 4, 'proceed': 2} | {'review': 4, 'proceed': 2} | 0.0863 | 0.14 | 0.29 | {'evalue_inv': 4, 'reported': 2} |
| Remibrutinib | psp_enrolled→persistent_180d | 6 | {'passed': 6} | {'passed': 6} | {'review': 4, 'proceed': 2} | {'review': 4, 'proceed': 2} | 0.0760 | 0.27 | 0.56 | {'evalue_inv': 4, 'reported': 2} |
| Remibrutinib | rep_detailing_high→treatment_initiated | 6 | {'passed': 6} | {'passed': 6} | {'review': 4, 'proceed': 2} | {'review': 4, 'proceed': 2} | 0.0723 | 0.45 | 0.57 | {'evalue_inv': 4, 'reported': 2} |
| Remibrutinib | sample_dropped→treatment_initiated | 6 | {'passed': 6} | {'passed': 6} | {'review': 4, 'proceed': 2} | {'review': 4, 'proceed': 2} | 0.0421 | 0.28 | 0.59 | {'evalue_inv': 4, 'reported': 2} |
| Remibrutinib | treatment_arm→adopted | 3 | {'passed': 3} | {'passed': 3} | {'proceed': 3} | {'proceed': 3} | 0.1469 | 0.10 | 0.28 | {'evalue_inv': 3} |
| Remibrutinib | treatment_arm→persistent_180d | 7 | {'failed': 5, 'passed': 2} | {'passed': 5, 'warning': 2} | {'block': 5, 'review': 2} | {'review': 6, 'proceed': 1} | 0.0472 | 0.94 | 1.34 | {'evalue_inv': 4, 'naive': 1, 'reported': 2} |
| Remibrutinib | treatment_arm→treatment_initiated | 10 | {'passed': 10} | {'passed': 10} | {'proceed': 10} | {'proceed': 10} | 0.1821 | 0.26 | 0.42 | {'evalue_inv': 8, 'reported': 2} |
| Remibrutinib | trigger_accepted→treatment_initiated | 6 | {'passed': 6} | {'passed': 6} | {'proceed': 6} | {'proceed': 6} | 0.0953 | 0.07 | 0.28 | {'evalue_inv': 4, 'reported': 2} |
| Remibrutinib | urticaria_severity_uas7→persistent_180d | 6 | {'passed': 6} | {'passed': 6} | {'proceed': 6} | {'proceed': 6} | 0.1505 | 0.35 | 0.56 | {'evalue_inv': 4, 'reported': 2} |

## Footnotes

- Seeded rows excluded: 109 rcc rows with `estimate_source <> causal_impact_query` ({'passed': 109}); they are not agent runs.
- Pairs without a dataset mapping (no re-pulled frame, stored sources only): Kisqali accepted→converted.
- The `<all>` brand rows carry a NULL brand on every test row; their frame is the all-brands pull (brand=None), as in the sensitivity re-band.
- SE constants: the rule's 1.959964. The reported interval on the agent path is the estimator's own `ate_ci_lower / ate_ci_upper`; on the pipeline path it is `effect ± 1.96·se`, a 0.002 % difference from the rule's constant.
- `std_refits` is recovered from DoWhy's one-tailed normal p (`z = |Δ| / std_refits`, `np.std` ddof=0 over 20 refits). A stored `p = 0.00000` means p < 5e-6 (z > 4.42) and the column shows an upper bound.
- `std_refits` is a ratio of two small numbers when p is near 0.5 (z near 0); read it as an order of magnitude there.
- Nothing is restated: the cutoffs 1 / 2 are read from the runner's `PASS_THRESHOLDS` and the verdict comes from its `_score_common_cause_shift`; the refit-frame scale is the runner's, applied from the stored `refutation_n_rows_total` / `refutation_n_rows`.

## Reading

The premise does NOT fully survive as measured: not every FAILED row moves to PASSED/WARNING: `5f7a5c0c` Kisqali peer_influence_score→adopted → failed (se source naive/naive_ols, shift 2.44). 6 of the 7 FAILED rows move to {'warning': 3, 'passed': 3}; no PASSED row becomes FAILED (max shift on a PASSED row 0.69 SE, p95 0.55, p50 0.16).
`5f7a5c0c` stays FAILED on the least faithful SE source — the covariate-free OLS proxy for a continuous treatment — which the calibration measures at n=6, median 0.471, min 0.471, max 0.482 of the reported SE on the same pair (naive_ols / evalue_inv). Its verdict is therefore UNRESOLVED by this measurement, not a counter-example. Hypothesis, not tuned in: the same pair's 2 exact inversions at n=[5000] scale by √(n_i/1500) to se ≈ 0.0234, i.e. shift ≈ 1.12 SE (WARNING). Its own DoWhy p-value (0.291) puts the shift inside the refit spread (std_refits 0.0478 is 4.4× the proxy SE): a null effect (ATE -0.0018) perturbed by noise.
What would reverse the reading: a PASSED row whose shift exceeds 2 refit-scaled reference SE (none), or the two exact inversions of the reported interval disagreeing (max |ratio − 1| = 1.69e-05 over 27 rows carrying both).
