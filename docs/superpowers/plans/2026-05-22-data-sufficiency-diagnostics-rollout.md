# Data-Sufficiency Diagnostics Rollout

**Status:** Phase 0 merged (PR #460). Phase 1 (PR #462) blocked on CI Integration Tests failure.
**Owner:** etn3724@gmail.com
**Started:** 2026-05-22
**Source PDF:** `de3b5738-Should_you_gather_more_data.pdf` (learning-curve diagnostic)

## Goal

Add a two-stage data-sufficiency capability to the Tier 0 ML Foundation pipeline:

1. **Pre-flight sufficiency check** in DataPreparer (post-load) — tiered HARD_FAIL / SOFT_FAIL / PASS verdict from data characteristics + resolved thresholds; gates execution via the existing QC plumbing; advises operators on detectable effect size given their actual data.
2. **Post-training learning-curve diagnostic** in ModelTrainer (the technique from the source PDF) — runs when `success_criteria_met=False`; fits a power-law to the curve; recommends additional samples to close the gap.
3. **INSUFFICIENT → synthetic-data preview** wiring — produces a preview cohort via existing `synthetic_v2` generator without auto-mixing into training.

All magic numbers flow through `src/utils/sufficiency_resolver.resolve_*` with the hierarchy `user_override > computed_from_data > literature_default`, every value attributed in the audit chain with its citation.

## Non-goals

- Replacing `success_criteria_met` as the production-deploy gate
- Auto-augmenting training data with synthetic rows in production runs (opt-in only)
- Implementing the Yang 2025 full causal-power decomposition (deferred Phase 2 v2)
- Running learning-curve on every pipeline run (conditional on `success_criteria_met=False` by default)
- Rewriting `PowerAnalysisNode` — only extracting its pure functions into a reusable library

## Locked decisions

| ID | Decision | Status |
|---|---|---|
| D1 | `scope_spec.target_mde` — optional with data-driven default + loud warning when defaulted | LOCKED in PR #462 |
| D2 | `PowerAnalysisNode` 500-default error fallback — replaced with explicit raise + failure flag (anti-mocking discipline) | DONE in PR #460 |
| D3 | `_calculate_minimum_samples()` — kept as advisory `_initial_min_samples_estimate()` (rename + backward-compat alias) | DONE in PR #462 |
| D4 | Refactor scope — extract `PowerAnalysisNode` pure methods to `power_analysis_lib`; Tier 3 thin adapter | DONE in PR #460 |
| D5 | `force_low_power_run` default — `False` (safe-by-default; pharma regulatory context). **Scope: applies to causal_inference SOFT_FAIL ONLY; HARD_FAIL is non-overridable.** Hotfix F8 clarified semantics: HARD_FAIL means the data is structurally insufficient (n < absolute_floor; EPV < 2; zero events) and accepting a single-flag bypass is medically dangerous in pharma regulatory contexts. The flag flips a causal SOFT_FAIL from blocking→warning AND surfaces `override_applied=True` + `original_verdict='SOFT_FAIL'` on the report so regulators/auditors can detect the bypass (F7). Predictive SOFT_FAIL warns by default regardless of this flag. | DONE in PR #462 + PR #462 hotfix |
| D6 | Pre-flight always runs — no skip flag (cheap formula evaluation; exercises invariants). **SKIPPED verdict added per hotfix F10/F11** for the three deliberate-skip cases (synthetic QC sample via `use_sample_data`, missing train_df, unknown problem_type) — each emits a SKIPPED-verdict report with a distinct rationale so audit chain consumers can tell them apart. The gate does NOT block on SKIPPED. INCONCLUSIVE (separate verdict) is reserved for the case where the diagnostic itself crashed; that path DOES block (F6). | DONE in PR #462 + PR #462 hotfix |

## PR sequence

### PR #460 — Phase 0: Foundations (MERGED 2026-05-23)

Foundational utility modules + Tier 3 PowerAnalysisNode refactor. Zero behavior change to existing pipelines.

**Files created:**
- `src/utils/power_analysis_lib.py` — pure power-analysis primitives (continuous, binary, cluster RCT, time-to-event) + reverse calc (`mde_for_sample_size`) + sensitivity grid
- `src/utils/sufficiency_defaults.py` — literature-grounded constants (Vergouwe 2007, Riley 2020, Cohen 1988, ICH E9, Hyndman/Kostenko 2007, Yang 2025) with citation registry
- `src/utils/sufficiency_schemas.py` — `SufficiencyConfig`, `DataSufficiencyReport`, `ThresholdResolution` pydantic models
- `src/utils/sufficiency_resolver.py` — three-tier resolution hierarchy: `user_override > computed_from_data > literature_default`

**Files refactored:**
- `src/agents/experiment_designer/nodes/power_analysis.py` — thin adapter over `power_analysis_lib`; public interface preserved; removes plausible-fake 500 fallback (D2)

**Tests:** 88 new + 22 existing PowerAnalysisNode + 395 experiment_designer suite — all green.

### PR #462 — Phase 1: Pre-flight check in DataPreparer (IN PROGRESS — CI BLOCKED)

**Status:** Unit tests all green locally (29 new), but CI Integration Tests failing reproducibly. Diagnosis blocked on log access.

**Files created:**
- `src/agents/ml_foundation/data_preparer/nodes/sufficiency_check.py` — new node runs post-`compute_baseline_metrics`, pre-`kg_role_enrichment`; computes verdict; writes into qc_report plumbing
- `tests/unit/test_data_preparer/test_sufficiency_check.py` — 21 tests covering every problem-type × every verdict tier + overrides + edge cases
- `tests/unit/test_agents/test_tier_0/test_handoff_protocols.py` — 8 tests for the latent-bug fix

**Files modified:**
- `src/agents/ml_foundation/data_preparer/graph.py` — wire sufficiency_check between baseline_computer and kg_role_enrichment
- `src/agents/ml_foundation/data_preparer/state.py` — add `sufficiency_report`, `power_warnings` fields
- `src/agents/ml_foundation/data_preparer/agent.py` — emit `sufficiency_report` + `power_warnings` in output dict
- `src/agents/ml_foundation/data_preparer/nodes/__init__.py` — export `run_sufficiency_check`
- `src/agents/ml_foundation/scope_definer/nodes/scope_builder.py` — `_calculate_minimum_samples` → `_initial_min_samples_estimate` (advisory; D3); backward-compat alias retained
- `src/agents/tier_0/handoff_protocols.py` — enforce the declared-but-never-checked `minimum_samples > 0` rule (latent-bug fix; classified REWIRE per CLAUDE.md REASON-BEFORE-RULES)
- `src/agents/tier_0/pipeline.py` — `PipelineConfig.force_low_power_run` + `sufficiency_strictness_preset`; `PipelineResult.sufficiency_report`; inject pipeline-level overrides into `scope_spec.sufficiency`

**Verdict semantics (post hotfix; see PR #462 hotfix brief F6-F11):**

| Verdict | Trigger | Effect |
|---|---|---|
| HARD_FAIL | `n < absolute_floor` or `EPV < 2` or zero positive events (F12) | Appends to `blocking_issues` → halts at `finalize_output`. **Non-overridable** (F8). |
| SOFT_FAIL (causal) | `n < recommended_n` and not `force_low_power_run` | Blocks (regulatory safety). Override sets `override_applied=True` + `original_verdict='SOFT_FAIL'` on report (F7). |
| SOFT_FAIL (predictive) | `n < recommended_n` | Appends to `power_warnings` only — proceeds |
| PASS | `n >= recommended_n` | Report attached, no gating action |
| SKIPPED (F10/F11) | `use_sample_data=True` OR `train_df` missing OR unknown problem_type | Report attached with distinct rationale; gate does NOT block |
| INCONCLUSIVE (F6) | Diagnostic itself crashed (uncaught exception) | Constructs valid report + appends blocking entry + sets `qc_status='failed'` → halts |

**Report contents:** verdict + verdict_rationale + resolved_thresholds (each with `source` + `citation`) + detectable_mde_at_current_n (Strategy A) + sensitivity_grid across 3 candidate MDEs (Strategy C) + mde_assumption_used (Strategy B) + human_readable_summary.

### PR #463 — Phase 2: Post-training learning curve in ModelTrainer (NOT STARTED)

New `learning_curve.py` node, triggered after training when `success_criteria_met=False` (or always-on via `PipelineConfig.always_run_learning_curve=True`). The technique from the source PDF:

1. Split training data into k=7 cumulative buckets
2. Train proxy model (LightGBM at default HPs) on each bucket, eval on validation
3. Fit power-law `score(n) = a - b·n^(-c)` via `scipy.optimize.curve_fit`
4. Slope-significance test on last 3 points
5. If still rising: extrapolate to target_score → `recommended_additional_samples`
6. Causal variant: track ATE CI-width vs n, fit `k/√n`

**Cost controls:** k=7 (not 12), single-fit per bucket (no nested HPO), 3-min walltime cap, INCONCLUSIVE verdict on cap-out.

**Causal v2:** uses `synthetic_v2` with `TRUE_ATE` for bootstrap-style CI-width estimation.

### PR #464 — Phase 3: Synthetic preview wiring (NOT STARTED)

`data_preparer/adapters/synthetic_preview.py` — triggered when `adaptive_verdict="INSUFFICIENT_TRAINING_DATA"` AND `PipelineConfig.synthetic_preview_on_insufficient=True` (opt-in, default False).

Routes:
- Predictive → `E2IDataGenerator.generate_all(n=recommended)` from `src/ml/data_generator.py`
- Causal → `generate_scenario(scenario, n=recommended)` from `src/ml/synthetic_v2/api.py`

**Critical invariant:** does NOT auto-mix into training. Saves preview to `pipeline_artifacts/synthetic_preview_<workflow_id>/` and attaches metadata to `PipelineResult.synthetic_preview`. Operator must explicitly pass it via `PipelineConfig.augmentation_data_path` to use.

## Threshold catalog

Every magic number flows through the resolver. Sources:

| Threshold | Default | Citation |
|---|---|---|
| `ABSOLUTE_FLOORS` | 50/100/200 per problem type, OR `max(2 * n_features / minority_prevalence, literature_floor)` when data known | Vergouwe 2007 severe-problems zone |
| `EPV_FLOORS` | linear=5, tree=10, NN=20 | Vergouwe 2007 / Riley 2020 / pmsampsize |
| `REGRESSION_RATIOS` | linear=5, tree=10, NN=15 | Standard sample-to-feature ratios |
| `DEFAULT_ALPHA` | 0.05 | ICH E9 |
| `DEFAULT_POWER` | 0.80 | ICH E9 |
| `DEFAULT_MDE_CONTINUOUS_COHENS_D` | 0.5 (medium) | Cohen 1988 |
| `DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR` | 0.05 (5pp ARD) | MCID pharma convention |
| `DEFAULT_MDE_BINARY_RELATIVE` | 0.20 (20% relative shift) | MCID convention |
| `DEFAULT_MDE_HAZARD_RATIO` | 0.75 | Conventional 25% risk reduction |
| `DEFAULT_OBSERVATIONAL_INFLATION` | 2.0 (good-overlap assumption); refined from observed PS overlap when available | Yang 2025 arxiv:2501.11181 |
| `TIMESERIES_CYCLES_HEADROOM` | 2 cycles + ARIMA params + features | Hyndman & Kostenko 2007 |
| `STRICTNESS_MULTIPLIERS` | conservative=0.5, moderate=1.0, strict=2.0 | Pragmatic |

## User-facing config (`scope_spec.sufficiency`)

```python
class SufficiencyConfig(BaseModel):
    epv_floor: int | None = None
    absolute_floor: int | None = None
    observational_inflation: float | None = None
    # Hotfix F5: tightened to gt=0, lt=1 (was unbounded → accepted NaN/negative/>=1)
    target_mde: float | None = None
    baseline_rate: float | None = None
    event_rate: float | None = None
    power_target: float | None = None
    alpha: float | None = None
    seasonal_period: int | None = None
    cv_outcome: float | None = None
    strictness_preset: Literal["conservative", "moderate", "strict"] | None = None
    # Hotfix F2: was missing from the schema; typed callers got ValidationError
    # under extra="forbid". Default False (safe-by-default; D5).
    force_low_power_run: bool = False
    # Hotfix F4 / D1: producer (scope_builder) stamps the source so the audit
    # chain records target_mde provenance at the scope boundary.
    target_mde_source: Literal["user_override", "computed_from_data",
                               "literature_default"] | None = None
```

Pipeline-level overrides via `PipelineConfig.force_low_power_run` and `PipelineConfig.sufficiency_strictness_preset` propagate into `scope_spec.sufficiency` at pipeline init time (per-key caller values win).

## Open follow-ups

- **PR #462 CI failure** (current blocker): Integration Tests reproducibly failing on commits `07bf8faf` and `14903a2b` (empty retrigger). Local CI-equivalent sweep ran clean — failure mode not identified yet. Need access to failing log to diagnose.
- **PR 3+4 sequencing:** wait for PR 2 merge before starting; PR 2 modules are the foundation for PR 3's learning curve and PR 4's synthetic preview wiring.

## References

- Source PDF: `Should_you_gather_more_data.pdf` (Daily Dose of DS)
- Vergouwe 2007 — *Am J Epidemiol*, "Relaxing the Rule of Ten EPV"
- van Smeden 2019 — *Stat Methods Med Res*, "Beyond EPV criteria"
- Riley 2020 — *BMJ*, "Calculating the sample size required for developing a clinical prediction model" (pmsampsize R package)
- Cohen 1988 — *Statistical Power Analysis for the Behavioral Sciences*
- ICH E9 — "Statistical Principles for Clinical Trials"
- Hyndman & Kostenko 2007 — "Minimum Sample Size Requirements for Seasonal Forecasting Models"
- Yang et al. 2025 — arxiv:2501.11181, "Sample size and power calculations for causal inference of observational studies"
