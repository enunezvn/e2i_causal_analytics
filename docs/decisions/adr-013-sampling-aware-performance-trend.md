# ADR-013: Model-performance trend is classified against sampling noise, never a fixed ±5% fold rule

**Date**: 2026-09-07 | **Status**: Accepted | **Implemented by**: PR #1916 (commits 79f3a02f5, 83ae6419e, a84c1bd37, 8b9b8d4eb; no migration — `positive_rate` reuses migration 017's nullable column)

## Context

The `/model-performance` "Performance trend" label was a single relative-change rule: the newest walk-forward fold against the mean of the older folds in the window, `degrading` below −5%, `improving` above +5%, with no notion of sample size. Two defects compounded:

- A walk-forward fold is **one calendar month** of rows (≈80–230 patients, ≈130 HCPs). The Hanley-McNeil standard error of an AUC at that n is 0.04–0.07 — larger than the 5% threshold (≈0.04). Simulated under a perfectly stationary model the rule reported something other than "stable" 38% of the time at n=130 and 56% at n=54.
- The newest fold was the **still-open** calendar month, whose row set keeps growing with every weekly frontier append (n=54 in week 1 vs ≈230 at month end); its metric moved 0.749 → 0.790 → 0.766 across three consecutive weekly runs.

On 2026-09-07 this flagged four gold-standard models `degrading` (`hcp_adoption_{fabhalta,remibrutinib}`, `persistence_fabhalta`, `discontinuation_fabhalta`) on folds 0.8–2.2 standard errors below baseline with no slope. Their holdout AUCs were unchanged week to week — the models were fine; the rule was not.

## Decision

1. **An open month is never scored.** `WalkForwardRunner` skips the calendar month containing `as_of` (default now, injectable) and anything later, recording it in `.skipped` — a partial fold is never emitted. `PerformanceMetricRepository.get_metric_trend` carries the same rule read-side (`_is_open_month_backtest_row`), dropping `backtest_wf` rows dated in the current UTC month so folds written by an older runner or a manual mid-month run cannot become "current". Daily-window rows (`mlflow` / legacy-null source) are not month folds and are untouched.
2. **Classification is sampling-aware** (`src/services/performance_trend_stats.py`, pure, no I/O). `degrading` / `improving` require the change to be BOTH statistically distinguishable from noise — level `|z| > 2.5` **or** OLS slope `|t|` beyond the Student-t critical value at `n_points − 2` df — AND material (the historical ±5%, kept only as a **materiality floor**). Two noise scales, and the classifier uses the **larger** available one:
   - `analytic` — Hanley-McNeil SE for `auc_roc`, binomial SE for `accuracy` (on n) and `recall` (on the actual-positive count). Formed only when the newest fold **and every baseline fold** have one; a baseline fold of unknown precision is never treated as exact.
   - `empirical` — the fold-to-fold sd of the baseline (≥6 folds), widened by √(1+1/k) for a new observation and inflated by `t_{k−1}/z` because the sd is itself estimated from k folds.
3. **No standard error is invented.** `precision` (denominator is the *predicted*-positive count, which the fold rows do not carry) and `f1` (no binomial variance) get no analytic SE; they fall to the empirical spread or, failing that, to the legacy ±5% rule labelled `basis="legacy_relative"` with the reason — never silently.
4. **The label says which rule produced it.** `basis` ∈ `level` / `slope` / `within_noise` / `immaterial` / `legacy_relative` / `insufficient_history` / `no_data`, with `noise_source` (`analytic` | `empirical`) and a prose `reason`; all reach the page as additive response fields and render as a visible "Trend basis: …" line. "Sampling noise" is claimed only for the analytic scale — the empirical band is named "historical fold variation", because it also carries month-composition shifts and any earlier decline.
5. **`alert_threshold` is an alert FLOOR, not a classification boundary.** The chart's red line is `max(baseline × (1 − degradation_threshold), absolute_min_accuracy)`. It is necessary, not sufficient: a fold under it still needs `trend == "degrading"` to breach. Only the absolute floor (`absolute_min_accuracy`, 0.5) alerts unconditionally. `alert_threshold_breached = (degrading AND change < −10%) OR value < 0.5` — note the alert's 10% depth is `degradation_threshold`, stricter than the classifier's 5% materiality floor.
6. **The insight reads the card's metric.** `/insights/model-performance` takes `metric_name` (default `auc_roc`, the page default; the page passes its selector) instead of a hard-coded `"accuracy"`, and the grounding carries the tracker's `reason`.

## Consequences

- (+) The four 2026-09-07 false alarms resolve to `stable` on the real windows (hcp/remibrutinib −13% on n=134 → z ≈ −2.1, `within_noise`), while a 0.15 AUC drop at n=230 is still `degrading` and a steady 0.01/month slide is caught by the slope test that a level test cannot see.
- (+) Expected false-`degrading` load across the page's 12 models × 5 metrics ≈ 0.4 per run from the level test at ±2.5 and ≈0.7 jointly with the slope test (union bound), before the materiality floor removes the shallow ones. Shewhart's ±3 would miss a genuine 0.10 AUC drop on a normal month; 2.5 catches it (z ≈ −2.7 at n=230).
- (−) The page now shows one fewer fold than the raw table has rows — the open month is deliberately absent, not missing data.
- (−) A classifier that needs a **new** column reads on the old scale until the backfill: rows written before `positive_rate` was persisted have no analytic SE for AUC/recall and take the empirical or legacy path. Expect a transient window after any such schema catch-up.
- (−) `student_t_quantile` is a dependency-free Cornish-Fisher expansion (checked against `scipy.stats.t.ppf`: within 0.3% for df ≥ 5, 0.7% at df = 4) so the module stays import-light. It is not a general-purpose t table.

## References

- `src/services/performance_trend_stats.py` — `assess_trend`, `metric_standard_error`, `slope_t_stat`, `student_t_quantile`
- `src/services/performance_tracking.py` — `PerformanceTrackingConfig` thresholds, `get_performance_trend`, the `alert_threshold` floor
- `src/mlops/gold_standard_eval/walk_forward.py` — open-month guard; `src/repositories/drift_monitoring.py` — `_is_open_month_backtest_row`
