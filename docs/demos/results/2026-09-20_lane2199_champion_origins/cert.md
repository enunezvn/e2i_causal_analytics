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

## Structural fact that makes the fix cheap

All models share `last_cutoff = n - horizon` and differ only in `first_cutoff`, so
absent failures the origin sets are **nested suffixes** and the intersection is just
the smallest. A failed origin punches a hole in the middle and breaks the nesting,
which is why `common_origins` is a real set intersection and not "the shortest
range".

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
