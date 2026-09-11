# random_common_cause calibration on planted truth at n = 1500 (2026-09-11, #2005)

`tests/unit/test_causal_engine/test_rcc_calibration_2005.py`, the green run of the file as committed (5 passed, 7 min 39 s). Frame: `test_sensitivity_calibration.frame` (seed 21, Remibrutinib, heterogeneous DGP, 1500 rows). Estimate: LinearDML with RandomForest nuisance (`_fit`). Refutation: production's `_reconstruct_dowhy_artifacts` + the runner's `_run_random_common_cause_test`, 20 simulations, effect strength 0.1. Rule: shift in units of the reported interval's SE, PASSED ≤ 1.0, WARNING ≤ 2.0. Scale 1.0 (refit frame = estimation frame). `old Δ%` is the retired `|Δ| / |ATE|` statistic (FAILED above 30 % under the old rule), still persisted as `delta_percent`.

| kind | pair | ATE | 95 % CI | refit mean | shift_se | status | DoWhy p | old Δ% |
|---|---|---|---|---|---|---|---|---|
| truth | treatment_arm->treatment_initiated | 0.1531 | (0.0809, 0.2252) | 0.1644 | 0.31 | passed | 0.155 | 7.4 |
| truth | treatment_arm->adherent_180d | 0.2591 | (0.1877, 0.3304) | 0.2697 | 0.29 | passed | 0.259 | 4.1 |
| truth | treatment_arm->low_gap_180d | 0.2426 | (0.1725, 0.3126) | 0.2438 | 0.03 | passed | 0.458 | 0.5 |
| truth | copay_support->adherent_180d | 0.1209 | (0.0684, 0.1733) | 0.1117 | 0.34 | passed | 0.226 | 7.6 |
| truth | copay_support->low_gap_180d | 0.0994 | (0.0504, 0.1485) | 0.0909 | 0.34 | passed | 0.198 | 8.6 |
| truth | copay_support->persistent_180d | 0.1150 | (0.0633, 0.1667) | 0.1083 | 0.25 | passed | 0.301 | 5.8 |
| truth | psp_enrolled->adherent_180d | 0.0990 | (0.0476, 0.1503) | 0.0990 | 0.00 | passed | 0.283 | 0.1 |
| truth | psp_enrolled->persistent_180d | 0.0879 | (0.0354, 0.1404) | 0.0891 | 0.05 | passed | 0.336 | 1.4 |
| truth | rep_detailing_high->treatment_initiated | 0.0522 | (0.0041, 0.1003) | 0.0511 | 0.04 | passed | 0.492 | 2.1 |
| truth | sample_dropped->treatment_initiated | 0.0548 | (0.0071, 0.1024) | 0.0508 | 0.16 | passed | 0.324 | 7.2 |
| truth | trigger_accepted->treatment_initiated | 0.0570 | (0.0104, 0.1035) | 0.0527 | 0.18 | passed | 0.038 | 7.5 |
| null | copay_support->treatment_initiated | 0.0172 | (-0.0307, 0.0652) | 0.0014 | 0.65 | passed | 0.045 | 92.1 |
| null | psp_enrolled->treatment_initiated | 0.0047 | (-0.0429, 0.0523) | 0.0130 | 0.34 | passed | 0.093 | 174.0 |
| null | rep_detailing_high->adherent_180d | -0.0106 | (-0.0602, 0.0390) | -0.0095 | 0.04 | passed | 0.412 | 10.6 |
| null | trigger_accepted->adherent_180d | -0.0017 | (-0.0520, 0.0485) | -0.0024 | 0.03 | passed | 0.448 | 38.0 |
| null | rep_detailing_high->low_gap_180d | -0.0053 | (-0.0521, 0.0415) | -0.0016 | 0.15 | passed | 0.265 | 69.0 |
| null | trigger_accepted->low_gap_180d | 0.0031 | (-0.0456, 0.0518) | -0.0004 | 0.14 | passed | 0.354 | 112.0 |
| null | rep_detailing_high->persistent_180d | 0.0307 | (-0.0196, 0.0809) | 0.0224 | 0.32 | passed | 0.184 | 27.0 |
| null | sample_dropped->persistent_180d | -0.0284 | (-0.0776, 0.0208) | -0.0313 | 0.12 | passed | 0.410 | 10.2 |
| null | trigger_accepted->persistent_180d | -0.0086 | (-0.0595, 0.0424) | -0.0150 | 0.25 | passed | 0.242 | 75.5 |

Max shift_se: truths 0.34 (copay_support->adherent_180d, copay_support->low_gap_180d), nulls 0.65 (copay_support->treatment_initiated). No row WARNING or FAILED. 6 of the 9 nulls exceed the retired 30 % cutoff (38–174 %) while reading 0.03–0.65 SE: the same refits the old rule FAILED by its denominator alone. DoWhy's random common cause is drawn unseeded, so a re-run moves the values (an earlier green run of the same file read 0.46 / 0.43; the pre-flight probe read 0.52 on the copay null pair).
