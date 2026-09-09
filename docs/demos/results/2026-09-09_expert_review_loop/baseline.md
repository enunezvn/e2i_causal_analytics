# Lane 1 — baseline discovery run on the CURRENT (pre-lane) image

- Image: `ghcr.io/enunezvn/e2i-api:f30e9e9df4fdd3b6350e3ecc47f891a72902c596`
- Job: `b827adfe-7c8f-4077-b4be-cde1cec3d09b` (POST /api/causal/discover-effects, dataset `patient_journeys`, brand `Remibrutinib`), started 2026-09-09T13:24:40Z, 11/11 rows after 1887 s
- Live DB before: pending|39; rejected|1; 1025 (expert_reviews by status; causal_validations count) — after: pending 39 / rejected 1 (all six BLOCK rows re-used existing pending rows by hash), causal_validations 1080 (+55), discovered_dags 12
- The live question registry returned **11** questions for this brand (spec §7 and the plan say 12); the job's own `total` is the source of truth.
- `data_subset` / `bootstrap` columns read straight from `public.causal_validations` (the API omits SKIPPED tests); `n` = number of recorded per-resample effects.

| question | row | run | band | sensitivity | data_subset | bootstrap | review decision | review id | discovered_dag_id | ate |
|---|---|---|---|---|---|---|---|---|---|---|
| treatment_arm → treatment_initiated | completed | completed | proceed | passed | skipped (n=0) | skipped (n=0) | — | — | 9da1aaac-d781-44f3-9a18-646409b0810a | 0.1915 |
| urticaria_severity_uas7 → persistent_180d | completed | completed | proceed | warning | skipped (n=0) | skipped (n=0) | — | — | 26a23584-94f7-4218-ad18-5a8809c9e7be | 0.1505 |
| copay_support → low_gap_180d | completed | completed | proceed | warning | skipped (n=0) | skipped (n=0) | — | — | c626710a-9f5d-4b9e-a6a9-183b7ce5da20 | 0.1032 |
| copay_support → adherent_180d | completed | completed | proceed | warning | skipped (n=0) | skipped (n=0) | — | — | 3e8cc4d0-00f9-49bd-874d-8cd8808a580b | 0.0988 |
| trigger_accepted → treatment_initiated | completed | completed | proceed | warning | skipped (n=0) | skipped (n=0) | — | — | 216c7fdc-fb35-4ee8-85c5-4a8d548370f0 | 0.0960 |
| psp_enrolled → adherent_180d | blocked | failed | block | failed | skipped (n=0) | skipped (n=0) | pending_review | 8317fde2-742a-42a2-8412-27df439c7c91 | 5655762f-9835-4f73-884a-61e972edb68f | 0.0840 |
| psp_enrolled → persistent_180d | blocked | failed | block | failed | skipped (n=0) | skipped (n=0) | pending_review | 4be7e6b3-a0df-4b6a-b1c6-c88bd842767c | 722863ff-b95b-4fdd-9759-71bc24896411 | 0.0830 |
| rep_detailing_high → treatment_initiated | blocked | failed | block | failed | skipped (n=0) | skipped (n=0) | pending_review | df88e399-4364-44c2-a0e8-d6eea1864da8 | ab22a6d0-89b2-4ed3-a933-3e1deb1fc921 | 0.0723 |
| copay_support → persistent_180d | blocked | failed | block | failed | skipped (n=0) | skipped (n=0) | pending_review | c296ce6a-8e1a-40ae-b7f5-8002f6ea0c7a | 38c2d56a-6166-477a-ad4e-37e9897f8063 | 0.0663 |
| sample_dropped → treatment_initiated | blocked | failed | block | failed | skipped (n=0) | skipped (n=0) | pending_review | 703d03d0-6ae7-4d66-b5c2-6c83f8f0ead4 | 00cedd66-b809-4de9-b878-bf64f810d33d | 0.0425 |
| treatment_arm → persistent_180d | blocked | failed | block | failed | skipped (n=0) | skipped (n=0) | pending_review | 678e28f3-21c5-4b91-82eb-1a1098403f79 | b0fb4994-de54-41d7-a359-f8a9dfc7249a | 0.0357 |

Bands: {'proceed': 5, 'block': 6} — **0 REVIEW**, consistent with the spec §2 arithmetic (only the three critical tests score on this image; both non-critical tests are SKIPPED with 0 recorded effects on 11/11 rows).
