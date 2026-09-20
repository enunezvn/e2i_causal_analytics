# #2199 — the champion is chosen on the origins every model shares

2026-09-20. Real PROD `business_metrics` TRx, real `forecast_series`, no mocks.

| evidence | what it is |
|---|---|
| `probe_service.py` | drives the REAL `forecast_series` over the real series truncated to short lengths |
| `baseline_champions.txt` | what the champion was BEFORE the fix, 21 (brand, length) cases |
| `after_champions.txt` | the same 21 cases AFTER |

## The premise, checked before anything was built

**Do short series exist in PROD at all?** No — every real brand × region cell is
**164 months** (12 cells, `2013-01` → `2026-08`). So the length-driven half of this
issue is a guard against future data, a newly launched brand, not a live defect
today. The issue's *second* source — a model whose origins FAIL being scored on a
subset — needs no short series and is the live-reachable half.

## The defect, measured through the real path

Each model is backtested from its own `min_observations` (seasonal 24, trend and
TimesFM 8), so their MAPEs average **different months**:

```
L= 36  seasonal  7 origins [24..30]   trend 23 origins [ 8..30]
L= 42  seasonal 13 origins [24..36]   trend 24 origins [13..36]
L= 48  seasonal 19 origins [24..42]   trend 24 origins [19..42]
L>=54  every model 24 origins, identical cutoffs
L=164  (live) every model 24 origins [135..158]
```

**9 of 21 served cases compared unequal origin sets.**

A first probe called `score_model` directly and reported a flip at L=30. That was
**wrong and is recorded as wrong**: `service.py` applies `MIN_BACKTEST_ORIGINS`
**per model**, so at L=30 the seasonal models (1 origin) are skipped entirely and
only `trend` is scored — no comparison, no defect. The probe had bypassed the
production gate. Everything above is from the faithful path.

## Before / after — exactly one champion changes

```
- Remibrutinib  L=42  champion=holt_winters_seasonal_mul  origins_used=13
+ Remibrutinib  L=42  champion=holt_winters_trend         origins_used=24
```

That is the case the measurement predicted. `seasonal_mul` won at 8.33% over its 13
origins while `trend` showed 8.63% over 24 — but on the **13 origins they share**,
`trend` is **8.17%** and is the better model.

**All other 20 cases are byte-identical, including all three live 164-month series**
— the issue's third "done when". At the live length every model has the same 24
cutoffs, so the new rule is a provable no-op there.

## Scope of the empirical counts — stated, not implied

**Every number above was measured with `include_timesfm=False`**, i.e. the 3-model
Holt-Winters contest. The forecast worker ships `replicas: 0` and is unreachable in
this environment, so a 4-model contest could not be run end to end here.

What that does and does not limit:

* The **mechanism** is model-count-agnostic — `common_origins` is a set intersection
  over whatever scores it is handed, and TimesFM's batched path records cutoffs
  through the same loop as the per-origin path (asserted by
  `test_a_mid_batch_failure_drops_its_cutoff_on_the_BATCHED_path_too`).
* The **live no-op claim still holds a fortiori**: at 164 months TimesFM's
  `min_observations` is 8, so it gets the same 24 cutoffs as everyone else and cannot
  create an asymmetry that was not already there.
* The specific counts — *9 of 21 unequal, 1 of 21 flips* — are proven for the
  3-model contest only. With TimesFM in the field a short-series contest has one more
  competitor and the flip count could differ.

Raised by review as a caveat worth writing down rather than leaving implicit.

## Structural fact that makes the fix cheap

All models share `last_cutoff = n - horizon` and differ only in `first_cutoff`, so
absent failures the origin sets are **nested suffixes** and the intersection is just
the smallest. A failed origin punches a hole in the middle and breaks the nesting,
which is why `common_origins` is a real set intersection and not "the shortest
range".

## A defect my own adversarial self-check found, after the first green

Two questions I had asserted but not directly proved, checked on the real series:

**Q1 — is the restricted MAPE the SAME statistic?** `mape_on_origins(score, its own
cutoffs)` vs `score.monthly_mape` over **36 real scores: max difference 0.0**. Identical.
So the fix changes only the comparison *set*, never the metric — which is what makes
"no-op when every model shares every origin" exact rather than approximate.

**Q2 — can the champion be the worst-looking model?** **Yes, and it happens.** At
Remibrutinib L=42 the champion is **3rd of 3 by its own reported MAPE**. The payload
sorted models by full MAPE ascending, so a reader saw:

```
holt_winters_seasonal_mul  8.33
holt_winters_seasonal_add  8.55
holt_winters_trend         8.63   <-- crowned champion, listed LAST
```

Correct under the new rule, and indistinguishable from a bug. `selection_origins: 13`
was present but the rows carried no per-model number for those 13, so *why* `trend` won
was invisible.

Fixed by ordering the rows by the number that actually decided the contest and putting
that number in each row:

```
#  model                       own MAPE   on shared  origins
1  holt_winters_trend              8.63        8.17       24  <-- CHAMPION
2  holt_winters_seasonal_mul       8.33        8.33       13
3  holt_winters_seasonal_add       8.55        8.55       13
```

The seasonal models' own MAPE **equals** their shared MAPE because they were graded on
exactly those 13 origins — a free internal-consistency signal in every payload.

`selection_origins` and `selection_fell_back` are now **derived** from `shared_cutoffs`
rather than stored beside it, so no cache round trip or future call site can leave a
count that disagrees with the set it counts.

## Teeth

Both plants landed, each failing **only** its own test:

| planted bug | test that caught it |
|---|---|
| `select_champion` ranks on full MAPE, ignoring shared ground | `test_a_model_graded_on_an_easier_early_stretch_does_not_win_on_that_advantage` |
| cutoff recorded outside the success branch, so failed origins are kept | `test_a_failed_origin_is_dropped_from_the_cutoffs_not_just_counted` |

## Deliberately not done

Reporting is **not** moved to the shared ground. Each model's MAPE and origin count
stay its own, because that is the only signal of how much evidence stands behind
its number. The payload gains `selection_origins` so a ranking decided on 13 shared
origins is explainable when the champion's own number is over 24.
