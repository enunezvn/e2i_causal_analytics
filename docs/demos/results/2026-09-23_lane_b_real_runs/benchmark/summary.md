# Structural author measurement

- lm: `openai/gpt-5.6-terra`
- resolver: `live`; cohort: `all`; briefs: 91
- measured_at: 2026-09-23T01:35:53.001063+00:00
- tree: commit 8615c0cadce9cd157b8bfd84e2dff625313187de (dirty src/scripts/tests: False)

## Score

    PASS: gate missed_leaks == 0 — missed leaks 0 (rate 0.000 over 42 scored golden-leak features; 91 scored, 0 routed to review, n=91)
    exact role agreement 60/91 (0.659); leak-decision agreement 91/91 (1.000); conservative errors 0
      ancestor    support=13  predicted=10  tp=8   precision=0.80 recall=0.62
      collider    support=14  predicted=21  tp=11  precision=0.52 recall=0.79
      confounder  support=17  predicted=25  tp=16  precision=0.64 recall=0.94
      descendant  support=13  predicted=13  tp=5   precision=0.38 recall=0.38
      instrument  support=19  predicted=14  tp=14  precision=1.00 recall=0.74
      mediator    support=15  predicted=8   tp=6   precision=0.75 recall=0.40
      cohort BC_kisqali: n=30 scored=30 exact=17 missed_leaks=0 review=0
      cohort CSU_remibrutinib: n=31 scored=31 exact=23 missed_leaks=0 review=0
      cohort PNH_fabhalta: n=30 scored=30 exact=20 missed_leaks=0 review=0

## Missed leaks

- none

## Routed to review

- none

## Cost estimate for the real run

- prompt tokens 531375, output tokens 81900 → USD 2.56 at ASSUMED 2.5/15.0 per Mtok
