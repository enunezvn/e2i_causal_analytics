# Structural author measurement

- lm: `fake` (fake source: replay)
- resolver: `offline`; cohort: `CSU_remibrutinib`; briefs: 31
- measured_at: 2026-09-22T18:21:22.262348+00:00
- tree: commit edb96fbd24ddb63ca6b2e3bb93fa619c86df9af1 (dirty src/scripts/tests: False)

## Score

    PASS: gate missed_leaks == 0 — missed leaks 0 (rate 0.000 over 14 scored golden-leak features; 31 scored, 0 routed to review, n=31)
    exact role agreement 28/31 (0.903); leak-decision agreement 31/31 (1.000); conservative errors 0
      ancestor    support=5   predicted=5   tp=5   precision=1.00 recall=1.00
      collider    support=5   predicted=8   tp=5   precision=0.62 recall=1.00
      confounder  support=6   predicted=6   tp=6   precision=1.00 recall=1.00
      descendant  support=4   predicted=4   tp=4   precision=1.00 recall=1.00
      instrument  support=6   predicted=6   tp=6   precision=1.00 recall=1.00
      mediator    support=5   predicted=2   tp=2   precision=1.00 recall=0.40
      cohort CSU_remibrutinib: n=31 scored=31 exact=28 missed_leaks=0 review=0

## Missed leaks

- none

## Routed to review

- none

## Cost estimate for the real run

- prompt tokens 180831, output tokens 27900 → USD 0.58 at ASSUMED 2.0/8.0 per Mtok

NOTE: a fake-LM run measures the PIPELINE (parse → extract_role → grade → score), not the author. The replayed CSU edges reproduce the committed validation record; the other cohorts' stand-in fragments are not authored claims.
