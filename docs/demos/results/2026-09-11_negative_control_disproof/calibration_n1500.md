# Lane G (#2007) calibration pin: the negative-control reading on planted truth (seed 21, n = 1500)

Printed by `tests/unit/test_causal_engine/test_negative_control_calibration_2007.py` (`pytest -n 0 -s`, 6 passed in 31 s, droplet, 2026-09-11): each planted truth whose arm has a registry control (`_CAUSAL_NEGATIVE_CONTROL_OUTCOMES`) is fitted with the production LinearDML (`_fit`, RF nuisances 50 / leaf 5 / seed 42), its control is fitted adjusted (same confounders) and omitted (`run_disproof.py::fit_omitted`, seeded noise column), and both are scored by the real `RefutationRunner._run_negative_control_test` (PASSED = CI includes 0; WARNING = excludes 0, |nc| < |original|; FAILED = excludes 0, |nc| >= |original|).

## Planted truths with a declared control (6 rows: copay_support x3, psp_enrolled x2, rep_detailing_high x1)

| truth arm → outcome | original ATE | control | adjusted nc [95 % CI] → status | omitted nc [95 % CI] → status |
|---|---|---|---|---|
| copay_support → adherent_180d | +0.1209 | treatment_initiated | +0.0172 [-0.0307, +0.0652] → **passed** | +0.0517 [+0.0005, +0.1029] → warning |
| copay_support → low_gap_180d | +0.0994 | treatment_initiated | +0.0172 [-0.0307, +0.0652] → **passed** | +0.0517 [+0.0005, +0.1029] → warning |
| copay_support → persistent_180d | +0.1150 | treatment_initiated | +0.0172 [-0.0307, +0.0652] → **passed** | +0.0517 [+0.0005, +0.1029] → warning |
| psp_enrolled → adherent_180d | +0.0990 | treatment_initiated | +0.0047 [-0.0429, +0.0523] → **passed** | +0.0895 [+0.0402, +0.1388] → warning |
| psp_enrolled → persistent_180d | +0.0879 | treatment_initiated | +0.0047 [-0.0429, +0.0523] → **passed** | +0.0895 [+0.0402, +0.1388] → failed |
| rep_detailing_high → treatment_initiated | +0.0522 | persistent_180d | +0.0307 [-0.0196, +0.0809] → **passed** | +0.0577 [+0.0056, +0.1099] → failed |

Adjusted: 6/6 PASSED, 0 WARNING, 0 FAILED. Omitted: 0 PASSED, 4 WARNING, 2 FAILED — the reading detects the leak this DGP plants on every declared control; it reads FAILED where the leaked control moves at least as much as the claimed effect. The three omitted points reproduce `disproof.md` to four decimals (+0.0517 / +0.0895 / +0.0577; pin tolerance 0.01).

## Undeclared arms, would-be controls (from `NULL_PAIRS`; none leaves its CI under omitted confounding → declaring them would be false assurance)

| arm → would-be control | omitted nc [95 % CI] | excludes 0 |
|---|---|---|
| sample_dropped → persistent_180d | -0.0267 [-0.0788, +0.0254] | no |
| trigger_accepted → adherent_180d | -0.0106 [-0.0583, +0.0370] | no |
| trigger_accepted → low_gap_180d | -0.0105 [-0.0558, +0.0349] | no |
| trigger_accepted → persistent_180d | -0.0165 [-0.0668, +0.0338] | no |

`treatment_arm` has no registry entry (it moves every outcome in the generator; no structural null to declare).
