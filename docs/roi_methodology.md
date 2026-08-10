# E2I ROI Methodology

**Status:** Derived from the implementation, 2026-08-09.
**Authoritative source:** `src/services/roi_calculation.py` (`methodology_version = "1.0"`).

This document was written because six code sites referenced `docs/roi_methodology.md`
as their specification and the file had never existed (confirmed against full git
history — it was a dangling reference, not a deletion). Every constant and formula
below is transcribed from the code rather than from an external design document.
**Where the two disagree, the code is correct and this document is stale** — it is a
description of the implementation, not a specification it must satisfy.

Referencing sites: `src/services/roi_calculation.py:12`,
`src/agents/gap_analyzer/nodes/roi_calculator.py:6,13,814`,
`src/agents/gap_analyzer/state.py:56`, `src/agents/gap_analyzer/graph.py:18`.

---

## 1. Scope: two different ROI questions

The platform answers two ROI questions by two unrelated code paths. Conflating them
is the most common error when reading ROI numbers out of this system.

| | **Projected ROI** | **Observed ROI** |
|---|---|---|
| Question | "What return should we expect if we fund this initiative?" | "What ROI did we record over the last 30 days?" |
| Path | `ROICalculationService` (this document) | KPI `WS3-BI-010` → `src/kpi/calculators/business_impact.py:557` |
| Method | Monte Carlo over input distributions | `AVG(roi)` over `business_metrics` rows |
| Output | Point estimate + 95% CI + P(ROI > target) | A single scalar |
| Uncertainty | Yes — see §4 | None — see §8 |

**This document specifies the projected-ROI path only.** The observed-ROI KPI is a
pooled average, not an estimate of a single quantity, and §8 explains why it carries
no interval.

---

## 2. Pipeline

```
value drivers (§3) ──┐
                     ├──> total value ──> attribution (§5) ──> base ROI ──> risk adjustment (§6) ──> risk-adjusted ROI
cost inputs (§7) ────┘                                              │
                                                                    └──> Monte Carlo (§4) ──> 95% CI, P(ROI>1), P(ROI>target)
                                                                    └──> sensitivity (§9) ──> tornado diagram
                                                                    └──> NPV (§10, optional)
```

Entry point: `ROICalculationService.calculate_roi()` (`roi_calculation.py:854`).
Returns an `ROIResult` carrying every intermediate above, so a caller can always
show its work.

---

## 3. Value drivers

Seven driver types (`ValueDriverType`). Each unit-economics constant is a documented
pharma assumption, **not** a measured value — treat them as calibration parameters
subject to review, and override them per brand where the calculator supports it.

| Driver | Constant | Formula |
|---|---|---|
| `TRX_LIFT` | $850 / incremental TRx | `trx × unit_value` |
| `PATIENT_IDENTIFICATION` | $1,200 / patient | `patients × 1200` |
| `ACTION_RATE` | $45 / pp / 1,000 triggers | `pp × 45 × triggers × 12` |
| `INTENT_TO_PRESCRIBE` | $320 / HCP / pp | `pp × 320 × hcp_count` |
| `DATA_QUALITY` | $200 / FP, $650 / FN avoided | `(fp×200 + fn×650) × 12` |
| `DRIFT_PREVENTION` | 2.0× multiplier | `auc_drop × baseline_value × 2` |
| `UPLIFT_TARGETING` | $125 / targeted individual, AUUC ×2.5 | `auuc × baseline × 2.5 + efficiency × pop × 125` |

Notes carried from the code:

- **$850/TRx is a generic anchor** calibrated to a Kisqali-scale oncology brand. A flat
  rate misvalues brands with very different per-script economics (an ultra-orphan PNH
  script is worth several times more). Callers with brand context **must** pass a
  brand-scoped `unit_value`; only `TRX_LIFT` consumes this override today.
- **$320/HCP/pp** is derived: 1pp ITP → +0.4 TRx/HCP/yr → 0.4 × $850 = $340, discounted
  ~6% for survey noise → $320.
- **$1,200/patient** is a lifetime figure: diagnostic confirmation, HCP engagement,
  support-program enrollment, 60% conversion to TRx, downstream adherence/refill.
- **FN ($650) is priced above FP ($200)** — a missed patient forfeits the opportunity and
  may cede it to a competitor; a false positive costs rep time and channel fatigue.

## 4. Monte Carlo confidence intervals

`BootstrapSimulator` (`roi_calculation.py:433`), 1,000 simulations by default,
seedable for reproducibility. Each simulation draws every input independently, then
recomputes ROI end-to-end (`_simulate_roi`, `roi_calculation.py:977`).

| Input class | Distribution | Rationale |
|---|---|---|
| Value drivers | Normal, σ = 0.15 μ, clamped at ≥ 0 | Symmetric estimation error; value cannot go negative |
| Costs | Gamma, shape = 2.0, scale = μ/2 | Right-skewed — costs overrun more often than they underrun |
| Acceptance rates | Beta (method of moments, 30% uncertainty) | Correct support for a proportion |

The asymmetry is deliberate and is the main reason this is a simulation rather than a
closed form: a normal value term over a gamma cost term has no convenient analytic
ROI distribution.

Reported from the resulting sample (`compute_confidence_interval`):

- `lower_bound` / `median` / `upper_bound` — 2.5th / 50th / 97.5th percentiles
- `probability_positive` — P(ROI > 1.0)
- `probability_target` — P(ROI > `target_roi`), default target 5.0

`probability_target` is usually the more decision-relevant number than the interval
itself: it answers "what are the odds this clears our bar", which is what a funding
conversation actually turns on.

**Interpretation limit.** The CI propagates *parameter* uncertainty under an assumed
σ = 15%. It is not a sampling distribution over observed outcomes, and it does not
account for model misspecification or for the unit-economics constants in §3 being
wrong. A narrow interval means the inputs were assumed precise, not that the estimate
is validated.

## 5. Causal attribution

Guards against overclaiming — the initiative gets credit only for the share of value
it plausibly caused. `attributed_value = total_value × rate`.

| Level | Rate | When |
|---|---|---|
| `FULL` | 1.00 | RCT-validated, sole driver |
| `PARTIAL` | 0.65 | Primary driver, some confounding (midpoint of 50–80%) |
| `SHARED` | 0.35 | Multiple initiatives contribute (midpoint of 20–50%) |
| `MINIMAL` | 0.10 | Minor contributor, correlation only (midpoint of <20%) |

Default is `PARTIAL`. Because the rates are band midpoints, attribution is the
coarsest step in the pipeline — moving one level shifts value by ~2×, far more than
the ±20% sensitivity sweep in §9. Choose it deliberately.

## 6. Risk adjustment

Four factors, combined multiplicatively so that risks compound without ever exceeding
100%:

```
total_adjustment = 1 − Π(1 − factorᵢ)
risk_adjusted_roi = base_roi × (1 − total_adjustment)
```

| Factor | Low | Medium | High |
|---|---|---|---|
| Technical complexity | 0% | 15% | 30% |
| Organizational change | 0% | 20% | 40% |
| Data dependencies | 0% | 25% | 50% |
| Timeline uncertainty | 0% | 10% | 25% |

All-`HIGH` yields a total adjustment of ~84%, not 145% — which is the point of the
multiplicative form. Default assessment is all-`LOW` (zero adjustment), so an
un-assessed initiative reports its *un-derisked* ROI.

## 7. Costs

`CostCalculator.calculate_total_cost` sums five buckets and returns the breakdown:

- **Engineering** — `engineering_days × engineering_day_rate` (default $2,500/day)
- **Data acquisition** — `incremental_data_cost` + sum of per-source costs
- **Change management** — training + change-management cost
- **Infrastructure** — `monthly_infrastructure_cost × infrastructure_months` (default 12)
- **Opportunity cost** — `(delayed_initiative_annual_value / 12) × delay_months`, counted
  only when both are positive

## 8. Why the observed-ROI KPI carries no confidence interval

The KPI path (`WS3-BI-010`) runs, in effect:

```sql
SELECT AVG(roi) FROM business_metrics
WHERE metric_date >= (frontier) - INTERVAL '30 days' AND roi IS NOT NULL
```

`business_metrics` is keyed by `metric_id` and carries `metric_type`, `metric_name`,
`brand`, and `region` (`database/core/e2i_ml_complete_v3_schema.sql:642`). Rows are
therefore **heterogeneous units** — different metrics, brands, and regions pooled
together.

A standard error over that pool would measure cross-brand/region/metric spread, not
uncertainty about any single ROI. Wrapping it in ±1.96·SE would produce a tight,
authoritative-looking interval around a quantity that is not a single estimand — a
plausible-but-wrong number, which is worse than reporting no interval at all.

If an interval is wanted on observed ROI, the correct move is to **condition first**
(group by brand / region / metric_name) and report dispersion within a homogeneous
slice, together with `n`. Do not add `STDDEV` to the pooled query.

Contrast `causal_metrics_ate`, which does return `ate_std` and `n_samples` and builds
a CI (`src/kpi/calculators/causal_metrics.py:124`): there the rows are repeated
estimates of one estimand, so the interval is meaningful. That precedent does not
transfer to ROI.

## 9. Sensitivity analysis (tornado diagram)

Optional (`run_sensitivity=True`). Varies each value driver ±20% while holding all
others at base, recording `roi_at_low` / `roi_at_base` / `roi_at_high` and an
`impact_range`. Ranking drivers by `impact_range` gives the tornado ordering and
identifies which assumption is worth the effort of tightening.

This is a one-at-a-time sweep: it does not capture interactions between drivers. The
Monte Carlo in §4 does, and the two answer different questions — §9 says *which input
matters most*, §4 says *how uncertain the answer is overall*.

## 10. NPV (multi-year)

`NPVCalculator`, 10% annual corporate discount rate:

```
NPV = Σ_t  value_t / (1 + 0.10)^t        (t = 1, 2, … years)
```

Exposed via `calculate_npv_roi()` for initiatives whose value accrues over multiple
years. Single-year initiatives should use `calculate_roi()` and ignore NPV.

---

## 11. Where this runs

- **`gap_analyzer` agent** — `ROICalculatorNode` (`nodes/roi_calculator.py:234`) calls the
  service per detected gap; results land in `ROIEstimate` records
  (`gap_analyzer/state.py:47`) carrying `confidence_interval`, attribution, and risk fields.
- **Gaps API / frontend** — `src/api/routes/gaps.py:195` exposes `confidence_interval` through
  to `frontend/src/types/gaps.ts`.
- **Chat** — the narrative in `gap_analyzer/nodes/formatter.py` renders the band via
  `_format_uncertainty_clause`, e.g.
  `"…at 4.0x ROI (risk-adjusted 2.4x, 95% CI 1.1x-4.0x; P(ROI>1x)=78%)"`.

  **Scale contract.** The simulated samples are
  `(value − cost)/cost × risk_multiplier` (§4), so the interval brackets
  `risk_adjusted_roi` — *not* the `expected_roi` the sentence leads with. The two are
  printed as separate, labelled quantities; splicing the band onto `expected_roi` would
  let the headline fall outside its own interval whenever a risk assessment is
  non-default. `P(ROI > 1x)` is reported with its literal meaning (returns more than
  double the outlay) and is **not** a break-even probability — break-even on this scale
  is ROI > 0. When no interval is present the clause is omitted entirely, so an absent
  CI reads as silence rather than a default band.

## 12. Known gaps

1. **Unit-economics constants (§3) are unvalidated assumptions.** Only `TRX_LIFT` accepts a
   brand-scoped override; the other six are flat platform-wide.
2. **σ = 15% default uncertainty (§4) is itself an assumption**, applied uniformly across
   drivers of very different measurement quality.
3. **`tool_composer.roi_estimator`** (`tool_registrations.py`) remains a separate, much
   simpler estimator that does **not** use this methodology. Its range is measured
   (a leave-one-out sensitivity band over `entity_values`, `[]` when fewer than 3 entity
   values make it unmeasurable) rather than a fixed ±25%, and the field carrying it is
   named `sensitivity_band` (renamed from `confidence_interval`, issue #1526) — it is a
   sensitivity band, not a sampling CI, and `assumptions` spells out the semantics.
   Note the deliberate contrast: `gap_analyzer`'s `ROIEstimate.confidence_interval`
   genuinely is a bootstrap CI and keeps its name.
4. **Observed-ROI intervals are still unavailable** (§8). Conditioning the KPI query by
   brand / region / metric_name is the path, and it needs a look at real row counts per
   slice before it is worth building.

### Closed

- ~~Chat drops the CI~~ — the band now renders in the gap_analyzer narrative (§11).
- ~~`roi_estimator` reports a hardcoded ±25% band~~ — replaced with measured dispersion.
- ~~`roi_estimator`'s band is named `confidence_interval`~~ — renamed `sensitivity_band` (#1526).
