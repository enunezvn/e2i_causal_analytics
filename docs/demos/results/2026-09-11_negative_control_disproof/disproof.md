# Lane G (#2007) disproof: omitted-confounder fits on the 9 structural nulls (seed 21, n = 1500)

Generated 2026-09-11T17:23:04Z by `run_disproof.py` in 35 s. Estimator = production LinearDML (RF nuisances 50 trees / leaf 5 / seed 42). `omitted` conditions on a seeded noise column only.

**Nulls that RESPOND (adjusted CI ∋ 0, omitted CI ∌ 0): 3/9.** Adjusted false positives on nulls: 0/9. Positive control: planted truths with adjusted CI ∌ 0: 11/11.

| kind | arm → outcome | confounders | adjusted ATE [95 % CI] | omitted ATE [95 % CI] | naive Δ [95 % CI] | shift | responds |
|---|---|---|---|---|---|---|---|
| null | copay_support → treatment_initiated | insurance_access_score, disease_severity | +0.0172 [-0.0307, +0.0652] | +0.0517 [+0.0005, +0.1029] | +0.0609 [+0.0094, +0.1124] | +0.0345 | **yes** |
| null | psp_enrolled → treatment_initiated | disease_severity, engagement_score, academic_hcp | +0.0047 [-0.0429, +0.0523] | +0.0895 [+0.0402, +0.1388] | +0.0875 [+0.0374, +0.1375] | +0.0847 | **yes** |
| null | rep_detailing_high → adherent_180d | academic_hcp, engagement_score | -0.0106 [-0.0602, +0.0390] | +0.0198 [-0.0298, +0.0694] | +0.0159 [-0.0324, +0.0642] | +0.0304 | no |
| null | trigger_accepted → adherent_180d | disease_severity, engagement_score | -0.0017 [-0.0520, +0.0485] | -0.0106 [-0.0583, +0.0370] | -0.0239 [-0.0729, +0.0251] | -0.0089 | no |
| null | rep_detailing_high → low_gap_180d | academic_hcp, engagement_score | -0.0053 [-0.0521, +0.0415] | +0.0313 [-0.0155, +0.0781] | +0.0269 [-0.0194, +0.0733] | +0.0366 | no |
| null | trigger_accepted → low_gap_180d | disease_severity, engagement_score | +0.0031 [-0.0456, +0.0518] | -0.0105 [-0.0558, +0.0349] | -0.0232 [-0.0703, +0.0239] | -0.0136 | no |
| null | rep_detailing_high → persistent_180d | academic_hcp, engagement_score | +0.0307 [-0.0196, +0.0809] | +0.0577 [+0.0056, +0.1099] | +0.0397 [-0.0107, +0.0901] | +0.0271 | **yes** |
| null | sample_dropped → persistent_180d | academic_hcp, engagement_score | -0.0284 [-0.0776, +0.0208] | -0.0267 [-0.0788, +0.0254] | -0.0246 [-0.0765, +0.0274] | +0.0017 | no |
| null | trigger_accepted → persistent_180d | disease_severity, engagement_score | -0.0086 [-0.0595, +0.0424] | -0.0165 [-0.0668, +0.0338] | -0.0379 [-0.0888, +0.0131] | -0.0080 | no |
| truth | treatment_arm → treatment_initiated (truth +0.161) | disease_severity, academic_hcp | +0.1531 [+0.0809, +0.2252] | +0.2981 [+0.2331, +0.3631] | +0.2815 [+0.2148, +0.3483] | +0.1450 | no |
| truth | treatment_arm → adherent_180d (truth +0.243) | disease_severity, academic_hcp | +0.2591 [+0.1877, +0.3304] | +0.3244 [+0.2610, +0.3878] | +0.2961 [+0.2297, +0.3626] | +0.0653 | no |
| truth | treatment_arm → low_gap_180d (truth +0.234) | disease_severity, academic_hcp | +0.2426 [+0.1725, +0.3126] | +0.3143 [+0.2491, +0.3795] | +0.2781 [+0.2112, +0.3451] | +0.0717 | no |
| truth | copay_support → adherent_180d (truth +0.116) | insurance_access_score, disease_severity | +0.1209 [+0.0684, +0.1733] | +0.1298 [+0.0779, +0.1818] | +0.1173 [+0.0656, +0.1691] | +0.0090 | no |
| truth | copay_support → low_gap_180d (truth +0.109) | insurance_access_score, disease_severity | +0.0994 [+0.0504, +0.1485] | +0.0874 [+0.0363, +0.1385] | +0.0980 [+0.0480, +0.1481] | -0.0120 | no |
| truth | copay_support → persistent_180d (truth +0.089) | insurance_access_score, disease_severity | +0.1150 [+0.0633, +0.1667] | +0.0276 [-0.0255, +0.0807] | +0.0481 [-0.0049, +0.1011] | -0.0874 | no |
| truth | psp_enrolled → adherent_180d (truth +0.094) | disease_severity, engagement_score, academic_hcp | +0.0990 [+0.0476, +0.1503] | +0.1278 [+0.0792, +0.1765] | +0.1015 [+0.0514, +0.1516] | +0.0289 | no |
| truth | psp_enrolled → persistent_180d (truth +0.076) | disease_severity, engagement_score, academic_hcp | +0.0879 [+0.0354, +0.1404] | +0.0631 [+0.0109, +0.1152] | +0.0861 [+0.0347, +0.1375] | -0.0248 | no |
| truth | rep_detailing_high → treatment_initiated (truth +0.061) | academic_hcp, engagement_score | +0.0522 [+0.0041, +0.1003] | +0.0665 [+0.0167, +0.1163] | +0.0852 [+0.0372, +0.1333] | +0.0143 | no |
| truth | sample_dropped → treatment_initiated (truth +0.040) | academic_hcp, engagement_score | +0.0548 [+0.0071, +0.1024] | +0.0766 [+0.0273, +0.1259] | +0.0701 [+0.0200, +0.1202] | +0.0218 | no |
| truth | trigger_accepted → treatment_initiated (truth +0.066) | disease_severity, engagement_score | +0.0570 [+0.0104, +0.1035] | +0.0933 [+0.0464, +0.1402] | +0.1099 [+0.0619, +0.1579] | +0.0363 | no |
