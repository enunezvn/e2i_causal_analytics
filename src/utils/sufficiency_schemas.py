"""Pydantic schemas for data-sufficiency configuration and reports.

These schemas are consumed by both Tier 0 DataPreparer (pre-flight check) and
ModelTrainer (post-training learning curve). Keeping them in src/utils/ avoids
a circular dependency between the two agent packages.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ProblemType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "causal_inference",
    "time_series",
]
StrictnessPreset = Literal["conservative", "moderate", "strict"]
# F11 (PR #462 hotfix): SKIPPED is a distinct, audit-visible verdict for the
# three deliberate-skip cases the pre-flight check has (synthetic QC sample
# via scope_spec.use_sample_data, missing train_df, unknown problem_type).
# Before this fix all three returned an empty {} update, collapsing skip vs.
# silent failure into the same audit signal.
SufficiencyVerdict = Literal["PASS", "SOFT_FAIL", "HARD_FAIL", "INCONCLUSIVE", "SKIPPED"]
ThresholdSource = Literal["user_override", "computed_from_data", "literature_default"]


class SufficiencyConfig(BaseModel):
    """User-facing overrides nested in scope_spec.sufficiency.

    Every field is optional. Unset fields flow through the resolution hierarchy
    to either a computed-from-data value or a literature-grounded default.
    """

    model_config = ConfigDict(extra="forbid")

    # Threshold overrides
    epv_floor: int | None = Field(default=None, ge=1)
    absolute_floor: int | None = Field(default=None, ge=1)
    observational_inflation: float | None = Field(default=None, gt=0.0)
    # F5 (PR #462 hotfix) + R2.2 (round-2): the F5 round-1 fix tightened
    # from `Field(default=None)` (which accepted NaN / negative / >=1 values
    # silently) to `Field(default=None, gt=0.0, lt=1.0)`. That bound is
    # CORRECT for binary outcomes (absolute risk difference is in (0, 1))
    # but WRONG for continuous outcomes: both ``scope_builder._build_scope_sufficiency``
    # and ``resolver.resolve_target_mde`` compute ``target_mde = 0.5 * sigma_outcome``
    # — RAW EFFECT SIZE in outcome units, NOT dimensionless Cohen's d. For
    # any ``sigma_outcome >= 2.0`` the value is >= 1.0 and gets rejected by
    # the schema, silently breaking the continuous-outcome MDE path.
    #
    # R2.2 fix (Option B in the round-2 brief): loosen the upper bound to
    # ``lt=1e6`` (effectively unbounded so absolute-effect-size semantics
    # for continuous work, but still catches gross typos like ``1e9``).
    # Per the dual semantics:
    #   * binary outcomes: ``target_mde`` is an ABSOLUTE RISK DIFFERENCE,
    #     should be in (0, 1). The resolver's runtime guard at
    #     ``resolve_target_mde`` re-validates against (0, 1) for binary and
    #     emits a WARN-and-fallback on invalid input — so the binary
    #     contract is still enforced where it matters.
    #   * continuous outcomes: ``target_mde`` is in OUTCOME UNITS (raw
    #     effect-size, e.g. mean difference in mmol/L). Can legitimately
    #     exceed 1.0 when sigma_outcome > 2.0.
    #   * time-to-event outcomes: ``target_mde`` is a hazard ratio,
    #     dimensionless, typically in (0.5, 2.0). The literature default
    #     uses ``DEFAULT_MDE_HAZARD_RATIO``.
    target_mde: float | None = Field(default=None, gt=0.0, lt=1e6)
    baseline_rate: float | None = Field(default=None, gt=0.0, lt=1.0)
    event_rate: float | None = Field(default=None, gt=0.0, le=1.0)
    power_target: float | None = Field(default=None, gt=0.0, lt=1.0)
    alpha: float | None = Field(default=None, gt=0.0, lt=1.0)

    # Time-series specific
    seasonal_period: int | None = Field(default=None, ge=2)
    cv_outcome: float | None = Field(
        default=None,
        ge=0.0,
        description="Coefficient of variation of outcome; inflates time-series floor",
    )

    # Convenience preset
    strictness_preset: StrictnessPreset | None = None

    # F2 (PR #462 hotfix): D5 of the rollout plan declared this flag (default
    # False — safe-by-default for pharma regulatory contexts) and pipeline.py
    # writes it into scope_spec.sufficiency, but the schema didn't declare it.
    # With `extra="forbid"`, any typed caller constructing
    # `SufficiencyConfig(force_low_power_run=True)` got a ValidationError —
    # the override only worked when the caller wrote a raw dict. Declaring
    # the field closes the typed-caller path. The sufficiency_check node
    # honors this flag for causal_inference SOFT_FAIL ONLY (HARD_FAIL is
    # non-overridable; see F8 in the hotfix brief).
    force_low_power_run: bool = False

    # F4 (PR #462 hotfix): D1 of the rollout plan requires the producer
    # (scope_builder) to set `target_mde` AND `target_mde_source` so the audit
    # chain can answer "did the user set this, did we compute it from data, or
    # did we fall back to literature?" without grep archaeology. Without this
    # field, every defaulted target_mde looked identical to a user-supplied one
    # and the "loud warning when defaulted" requirement was unsatisfiable.
    target_mde_source: ThresholdSource | None = None


class ThresholdResolution(BaseModel):
    """Audit-friendly record of how a single threshold value was derived."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | int
    source: ThresholdSource
    citation: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class DataSufficiencyReport(BaseModel):
    """Output of pre-flight sufficiency check and/or post-training learning curve.

    Same schema is reused by Phase 1 (pre-flight) and Phase 2 (learning curve)
    so downstream consumers see one shape. Phase 1 populates pre-flight fields;
    Phase 2 populates learning-curve fields. Empty fields are unset, not zero.
    """

    model_config = ConfigDict(extra="forbid")

    # Top-line verdict
    verdict: SufficiencyVerdict
    verdict_rationale: str

    # Inputs from data
    n_rows: int
    n_features: int
    problem_type: ProblemType
    minority_prevalence: float | None = None
    baseline_rate: float | None = None
    sigma_outcome: float | None = None

    # Thresholds (each with source + citation)
    resolved_thresholds: list[ThresholdResolution] = Field(default_factory=list)

    # Required n from formulas
    required_n: int | None = None
    required_n_rationale: str | None = None

    # Reverse calc (Strategy A)
    detectable_mde_at_current_n: float | None = None
    detectable_mde_units: str | None = None
    # F14 (PR #462 hotfix): for binary outcomes with very small n + small
    # baseline_rate, the asymptotic normal-approximation MDE formula can
    # return values larger than baseline_rate (e.g. 0.61 vs 0.05). That
    # number is statistically nonsensical — we cannot "detect" a 0.61
    # absolute risk difference when the baseline event rate is only 0.05.
    # The node clamps to `min(baseline_rate, 1 - baseline_rate)` and surfaces
    # the clamp via this flag so the report consumer can tell "honest MDE"
    # from "clamped MDE; n is below the asymptotic-regime threshold and the
    # actual answer is 'cannot detect anything meaningful'".
    detectable_mde_at_n_capped: bool | None = None

    # Sensitivity grid (Strategy C)
    sensitivity_grid: dict[str, Any] | None = None

    # MDE assumption (Strategy B)
    mde_assumption_used: dict[str, Any] | None = None

    # F7 (PR #462 hotfix): D5 override audit. When `force_low_power_run` flips
    # a causal SOFT_FAIL from blocking to warning, the report previously
    # reported the post-override verdict as if it were the genuine answer —
    # auditors had no way to detect the bypass. These fields preserve the
    # pre-override verdict alongside the override-applied flag so the audit
    # chain can answer "did this run actually meet the gate, or was the gate
    # bypassed?" The verdict_rationale field also gets a
    # ` [OVERRIDDEN via force_low_power_run]` suffix in that branch.
    override_applied: bool = False
    original_verdict: SufficiencyVerdict | None = None

    # Learning-curve fields (populated by Phase 2 only)
    learning_curve: list[tuple[int, float, float]] | None = None
    proxy_model: str | None = None
    slope_at_max_n: float | None = None
    slope_pvalue: float | None = None
    power_law_fit: dict[str, float] | None = None
    extrapolated_n_for_target: int | None = None
    extrapolated_n_ci: tuple[int, int] | None = None
    fit_quality_r2: float | None = None
    # PR #463 R2.5 / R2.7: the learning-curve node sets ``fit_trustworthy``
    # whenever the power-law fit's R² exceeds the gate (and to ``False``
    # explicitly on every non-success branch — walltime cap, all-failures,
    # partial-failure). Without this field declared in the schema,
    # ``extra="forbid"`` would reject the runtime dict produced by the node.
    fit_trustworthy: bool | None = None
    recommended_additional_samples: int | None = None

    # Causal-specific (populated by Phase 2 causal branch only)
    ate_ci_width_curve: list[tuple[int, float]] | None = None
    ate_target_ci_width: float | None = None
    # PR #463 R2.5: the causal branch of the learning-curve node infers an
    # outcome type from ``y_train`` (binary {0,1} vs. continuous) and
    # surfaces it so downstream consumers know which causal estimand the
    # diagnostic used. Schema-declared here to keep ``extra="forbid"``.
    outcome_type: Literal["binary", "continuous", "time_to_event", "unknown"] | None = None

    # Operational
    diagnostic_runtime_s: float | None = None
    human_readable_summary: str | None = None
