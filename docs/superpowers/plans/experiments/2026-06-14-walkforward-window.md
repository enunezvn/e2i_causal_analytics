# EXPERIMENT — Walk-forward window mode (Task 8)

**Date:** 2026-06-14
**Branch:** `feat/gse-p1-initiation`
**Decision recorded into:** `src/mlops/gold_standard_eval/walk_forward.py`
(`WalkForwardRunner.window_mode` default = `"expanding"`)

## Assumption + falsifier

- **Assumption:** expanding-origin walk-forward gives a **stable, real trend**
  across the brand's full (~37-month) journey timeline.
- **Falsifier:** degenerate/NaN months, OR wild month-to-month AUC swings
  (`>~0.2`) attributable to sample noise rather than real signal change.

The decision is the **measured** per-month AUC trend and its stability, not theory.

## Method (REAL DB, memory-guarded)

- `free -h` before the sweep: **4.6 GiB available** (≥ 2 GiB floor). ~37 serial
  `LogisticRegression` fits over ~13k rows — light, no concurrent heavy steps.
- Loaded the FULL brand frame once via
  `FeatureBuilder(INITIATION).load_frame(db, splits=None)` (all months,
  `brand='Remibrutinib'`, `is_synthetic=True`).
  - `frame.shape = (13288, 7)`
  - `journey_start range = 2023-06-?? .. 2026-06-??`, **37 distinct months**.
- Ran the real `WalkForwardRunner` with the production default fit/predict:
  `FeatureBuilder` fit/transform + `LogisticRegression(class_weight='balanced',
  max_iter=1000)` + `scorer.score`. Strict OOS: train rows have
  `journey_start_date < first-day-of(M)`; eval = rows in month `M`.
- Compared **expanding** vs **rolling-3-month** at modest guards
  (`min_train_n=30`, `n_min=10`) so thin early months are visible to reason about,
  not silently pre-filtered. (Production defaults are stricter: `min_train_n=50`,
  `n_min=20`.)
- Script (throwaway, not committed): `scripts/exp_walk_forward_window.py`.

### Real rows-per-month distribution (the shape that drives the noise)

Most months carry **~67–117 rows**; the two tail months cluster the recent /
holdout mass: **2026-05 = 3424**, **2026-06 = 1615**. The thin ~70–110-row
months are where AUC sampling error is largest (an AUC SE of ≈0.05–0.06 at these
n and a ~0.35 base rate → month-to-month deltas of ≈0.10–0.17 are the EXPECTED
noise band, not model instability).

## Measured per-month AUC (pasted from the real run)

### EXPANDING (min_train_n=30, n_min=10)

```
month         n_eval   auc_roc  Δ vs prev
------------------------------------------
2023-07-01        95    0.6653         --
2023-08-01        89    0.6177    -0.0476
2023-09-01        94    0.6229    +0.0052
2023-10-01       110    0.6959    +0.0731
2023-11-01        88    0.6770    -0.0189
2023-12-01       116    0.6604    -0.0166
2024-01-01        98    0.7445    +0.0841
2024-02-01        92    0.6846    -0.0599
2024-03-01        86    0.5806    -0.1040
2024-04-01        91    0.7097    +0.1290
2024-05-01        88    0.6381    -0.0716
2024-06-01       100    0.6474    +0.0093
2024-07-01       103    0.6368    -0.0106
2024-08-01       109    0.6464    +0.0096
2024-09-01       100    0.6017    -0.0447
2024-10-01       111    0.5697    -0.0320
2024-11-01       102    0.6203    +0.0506
2024-12-01       106    0.6261    +0.0059
2025-01-01        93    0.7198    +0.0936
2025-02-01        77    0.7653    +0.0455
2025-03-01        99    0.6084    -0.1569
2025-04-01       100    0.6921    +0.0837
2025-05-01       117    0.6616    -0.0305
2025-06-01        91    0.6526    -0.0090
2025-07-01       110    0.7411    +0.0885
2025-08-01        99    0.5804    -0.1606
2025-09-01       113    0.6986    +0.1182
2025-10-01       100    0.7490    +0.0504
2025-11-01        93    0.6316    -0.1173
2025-12-01       102    0.5631    -0.0686
2026-01-01       107    0.7305    +0.1675
2026-02-01        67    0.6172    -0.1134
2026-03-01        82    0.6546    +0.0375
2026-04-01        87    0.6369    -0.0177
2026-05-01      3424    0.6799    +0.0430
2026-06-01      1615    0.6496    -0.0303
------------------------------------------
qualifying months emitted : 36
skipped months            : 1
    SKIP 2023-06-01  train_n=0     n_eval=66    :: train guard: train_n=0 < min_train_n=30
auc mean / median         : 0.6577 / 0.6511
auc min / max             : 0.5631 / 0.7653
auc stdev                 : 0.0515
max month-to-month swing  : 0.1675
```

### ROLLING-3 (min_train_n=30, n_min=10)

```
month         n_eval   auc_roc  Δ vs prev
------------------------------------------
2023-07-01        95    0.6653         --
2023-08-01        89    0.6177    -0.0476
2023-09-01        94    0.6229    +0.0052
2023-10-01       110    0.7181    +0.0952
2023-11-01        88    0.6770    -0.0411
2023-12-01       116    0.6620    -0.0150
2024-01-01        98    0.7175    +0.0555
2024-02-01        92    0.6754    -0.0421
2024-03-01        86    0.5977    -0.0777
2024-04-01        91    0.7081    +0.1104
2024-05-01        88    0.6119    -0.0962
2024-06-01       100    0.6321    +0.0202
2024-07-01       103    0.6312    -0.0009
2024-08-01       109    0.6637    +0.0325
2024-09-01       100    0.6068    -0.0569
2024-10-01       111    0.5690    -0.0378
2024-11-01       102    0.5991    +0.0301
2024-12-01       106    0.5670    -0.0321
2025-01-01        93    0.7054    +0.1385
2025-02-01        77    0.7952    +0.0898
2025-03-01        99    0.6341    -0.1611
2025-04-01       100    0.7178    +0.0838
2025-05-01       117    0.6305    -0.0873
2025-06-01        91    0.6217    -0.0088
2025-07-01       110    0.7400    +0.1182
2025-08-01        99    0.5778    -0.1622
2025-09-01       113    0.6942    +0.1163
2025-10-01       100    0.7338    +0.0396
2025-11-01        93    0.6233    -0.1105
2025-12-01       102    0.5650    -0.0584
2026-01-01       107    0.7252    +0.1603
2026-02-01        67    0.5848    -0.1404
2026-03-01        82    0.6199    +0.0350
2026-04-01        87    0.6634    +0.0435
2026-05-01      3424    0.6550    -0.0084
2026-06-01      1615    0.6471    -0.0079
------------------------------------------
qualifying months emitted : 36
skipped months            : 1
    SKIP 2023-06-01  train_n=0     n_eval=66    :: train guard: train_n=0 < min_train_n=30
auc mean / median         : 0.6521 / 0.6406
auc min / max             : 0.5650 / 0.7952
auc stdev                 : 0.0554
max month-to-month swing  : 0.1622
```

### Decision input (side by side)

| metric    | expanding | rolling-3 |
|-----------|-----------|-----------|
| n_points  | 36        | 36        |
| mean      | 0.6577    | 0.6521    |
| stdev     | **0.0515**| 0.0554    |
| max_swing | 0.1675    | 0.1622    |
| min       | 0.5631    | 0.5650    |
| max       | 0.7653    | 0.7952    |

## Falsifier verdict: ASSUMPTION SURVIVES

- **No degenerate / NaN months.** Every one of the 36 qualifying months produced
  a finite AUC; only 2023-06 was skipped (zero training history — the train guard
  working as designed, not a degeneracy).
- **No wild swings beyond the falsifier band.** Max month-to-month swing is
  **0.1675 (expanding) / 0.1622 (rolling)**, both `< ~0.2`. The swings present are
  concentrated on the thin ~70–110-row eval months and are consistent with the
  AUC sampling SE at those sample sizes (≈0.05–0.06), i.e. estimation noise, not
  model instability. The trend itself is stable around AUC ≈ 0.65 — consistent
  with the Task-3 holdout lock (0.6709) and with validation/test (0.685 / 0.643).

## Decision: **expanding** (default), rolling-3 available via param

- Expanding is the **more stable** mode by the measured per-month stdev
  (**0.0515 < 0.0554**), and its mean is marginally higher (0.6577 vs 0.6521).
- Expanding is the leakage-free, finer-grained counterpart to the static
  chronological `data_split`: it trains on ALL prior history, matching how a
  deployed model actually accumulates data month over month — the most faithful
  "real performance trend" for the Time-Series page.
- Rolling-3 was **not** more stable; it trades away early-history signal for a
  fixed window without reducing the noise that comes from small eval months, so
  there is no stability case for making it the default.
- **Both modes are retained.** `WalkForwardRunner(window_mode="rolling",
  rolling_months=N)` remains available for callers who need a fixed-width window.

The runner's constructor already defaults to `window_mode="expanding"`; this
experiment confirms that default is the right one. `test_walk_forward_default_
window_mode_is_expanding` pins it so a future edit can't silently flip it.

## Thin-month note (for downstream tasks)

The brand yields **36 qualifying walk-forward points** at modest guards
(`min_train_n=30, n_min=10`). At the stricter production defaults
(`min_train_n=50, n_min=20`) the first 1–2 months would additionally drop on the
eval/train guards, but the bulk of the timeline (≈34 months) still qualifies, so
the Time-Series trend is well populated. The two tail months (2026-05/06) carry
the large recent mass and anchor the most recent, highest-confidence points.
