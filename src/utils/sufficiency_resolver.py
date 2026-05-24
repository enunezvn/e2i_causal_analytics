"""Threshold-resolution hierarchy for data sufficiency checks.

Implements the three-tier resolution: user_override > computed_from_data >
literature_default. Every resolution returns a ThresholdResolution record so
the audit chain can answer "why did we use 0.10 for MDE on this run?" without
code archaeology.

Per CLAUDE.md REASON-BEFORE-RULES: no magic numbers in algorithm code. Every
threshold flows through this module.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.utils.sufficiency_defaults import (
    ABSOLUTE_FLOORS,
    DEFAULT_ALPHA,
    DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR,
    DEFAULT_MDE_BINARY_RELATIVE,
    DEFAULT_MDE_CONTINUOUS_COHENS_D,
    DEFAULT_MDE_HAZARD_RATIO,
    DEFAULT_OBSERVATIONAL_INFLATION,
    DEFAULT_POWER,
    EPV_FLOORS,
    REGRESSION_RATIOS,
    STRICTNESS_MULTIPLIERS,
    TIMESERIES_CYCLES_HEADROOM,
    citation_for,
)
from src.utils.sufficiency_schemas import ThresholdResolution

logger = logging.getLogger(__name__)


def _user_override(config: dict[str, Any] | None, name: str) -> Any:
    """Extract a single override from the user's SufficiencyConfig dict."""
    if not config:
        return None
    return config.get(name)


def _strictness_multiplier(config: dict[str, Any] | None) -> float:
    """Return preset multiplier for EPV/ratio thresholds. 1.0 if not set."""
    preset = (config or {}).get("strictness_preset")
    if preset and preset in STRICTNESS_MULTIPLIERS:
        return STRICTNESS_MULTIPLIERS[preset]
    return 1.0


def resolve_epv_floor(
    *,
    user_config: dict[str, Any] | None,
    algorithm_family: str = "unknown",
) -> ThresholdResolution:
    """EPV floor for classification problems.

    Hierarchy:
    1. user_config.epv_floor — explicit override
    2. EPV_FLOORS[algorithm_family] × strictness_multiplier — algorithm-aware
    3. EPV_FLOORS["unknown"] — last-resort default
    """
    override = _user_override(user_config, "epv_floor")
    if override is not None:
        return ThresholdResolution(
            name="epv_floor",
            value=int(override),
            source="user_override",
            citation="scope_spec.sufficiency.epv_floor",
            inputs={"algorithm_family": algorithm_family},
        )

    family = algorithm_family if algorithm_family in EPV_FLOORS else "unknown"
    base = EPV_FLOORS[family]
    multiplier = _strictness_multiplier(user_config)
    value = max(1, int(round(base * multiplier)))

    if algorithm_family != "unknown":
        return ThresholdResolution(
            name="epv_floor",
            value=value,
            source="computed_from_data",
            citation=citation_for(f"EPV_FLOORS.{family}"),
            inputs={
                "algorithm_family": family,
                "base_value": base,
                "strictness_multiplier": multiplier,
            },
        )
    return ThresholdResolution(
        name="epv_floor",
        value=value,
        source="literature_default",
        citation=citation_for(f"EPV_FLOORS.{family}"),
        inputs={
            "algorithm_family": family,
            "base_value": base,
            "strictness_multiplier": multiplier,
        },
    )


def resolve_regression_ratio(
    *,
    user_config: dict[str, Any] | None,
    algorithm_family: str = "unknown",
) -> ThresholdResolution:
    """Sample-to-feature ratio floor for regression problems."""
    override = _user_override(user_config, "epv_floor")  # reuse field for regression
    if override is not None:
        return ThresholdResolution(
            name="regression_ratio",
            value=int(override),
            source="user_override",
            citation="scope_spec.sufficiency.epv_floor",
            inputs={"algorithm_family": algorithm_family},
        )

    family = algorithm_family if algorithm_family in REGRESSION_RATIOS else "unknown"
    base = REGRESSION_RATIOS[family]
    multiplier = _strictness_multiplier(user_config)
    value = max(1, int(round(base * multiplier)))

    if algorithm_family != "unknown":
        return ThresholdResolution(
            name="regression_ratio",
            value=value,
            source="computed_from_data",
            citation=citation_for(f"REGRESSION_RATIOS.{family}"),
            inputs={
                "algorithm_family": family,
                "base_value": base,
                "strictness_multiplier": multiplier,
            },
        )
    return ThresholdResolution(
        name="regression_ratio",
        value=value,
        source="literature_default",
        citation=citation_for(f"REGRESSION_RATIOS.{family}"),
        inputs={
            "algorithm_family": family,
            "base_value": base,
            "strictness_multiplier": multiplier,
        },
    )


def resolve_absolute_floor(
    *,
    user_config: dict[str, Any] | None,
    problem_type: str,
    n_features: int | None = None,
    minority_prevalence: float | None = None,
) -> ThresholdResolution:
    """HARD-FAIL absolute floor below which formulas don't apply.

    Hierarchy:
    1. user_config.absolute_floor — explicit override
    2. Computed from data: max(2 * n_features / minority_prevalence, literature_floor)
       — EPV=2 disaster threshold expressed in actual data terms
    3. ABSOLUTE_FLOORS[problem_type] — literature-grounded floor
    """
    override = _user_override(user_config, "absolute_floor")
    if override is not None:
        return ThresholdResolution(
            name="absolute_floor",
            value=int(override),
            source="user_override",
            citation="scope_spec.sufficiency.absolute_floor",
            inputs={"problem_type": problem_type},
        )

    literature_floor = ABSOLUTE_FLOORS.get(problem_type, 100)

    if (
        problem_type in ("binary_classification", "multiclass_classification")
        and n_features is not None
        and minority_prevalence is not None
        and minority_prevalence > 0
    ):
        epv_two_floor = int((2 * n_features) / minority_prevalence)
        value = max(literature_floor, epv_two_floor)
        return ThresholdResolution(
            name="absolute_floor",
            value=value,
            source="computed_from_data",
            citation=citation_for("ABSOLUTE_FLOORS"),
            inputs={
                "problem_type": problem_type,
                "n_features": n_features,
                "minority_prevalence": minority_prevalence,
                "epv_two_floor": epv_two_floor,
                "literature_floor": literature_floor,
            },
        )

    return ThresholdResolution(
        name="absolute_floor",
        value=literature_floor,
        source="literature_default",
        citation=citation_for("ABSOLUTE_FLOORS"),
        inputs={"problem_type": problem_type},
    )


def resolve_observational_inflation(
    *,
    user_config: dict[str, Any] | None,
    observed_overlap: float | None = None,
) -> ThresholdResolution:
    """Observational-study power inflation factor.

    Hierarchy:
    1. user_config.observational_inflation — explicit override
    2. 1 / observed_overlap (when overlap can be measured)
    3. DEFAULT_OBSERVATIONAL_INFLATION = 2.0 (good-overlap assumption)
    """
    override = _user_override(user_config, "observational_inflation")
    if override is not None:
        return ThresholdResolution(
            name="observational_inflation",
            value=float(override),
            source="user_override",
            citation="scope_spec.sufficiency.observational_inflation",
            inputs={},
        )

    if observed_overlap is not None and 0.0 < observed_overlap <= 1.0:
        value = 1.0 / observed_overlap
        return ThresholdResolution(
            name="observational_inflation",
            value=value,
            source="computed_from_data",
            citation=citation_for("DEFAULT_OBSERVATIONAL_INFLATION"),
            inputs={"observed_overlap": observed_overlap},
        )

    return ThresholdResolution(
        name="observational_inflation",
        value=DEFAULT_OBSERVATIONAL_INFLATION,
        source="literature_default",
        citation=citation_for("DEFAULT_OBSERVATIONAL_INFLATION"),
        inputs={},
    )


def resolve_target_mde(
    *,
    user_config: dict[str, Any] | None,
    outcome_type: str,
    sigma_outcome: float | None = None,
    baseline_rate: float | None = None,
) -> ThresholdResolution:
    """Target minimum detectable effect.

    Hierarchy:
    1. user_config.target_mde — explicit override (or scope-builder-stamped value)
    2. Computed from data characteristics:
       - continuous: 0.5 * sigma_outcome (Cohen "medium" anchored to data,
         IN OUTCOME UNITS — raw effect size, NOT dimensionless Cohen's d)
       - binary: max(absolute floor, relative shift × baseline_rate)
         (in absolute-risk-difference units, must be in (0, 1))
       - hazard_ratio: literature default (no data-driven path yet)
    3. Literature default from Cohen 1988 / MCID conventions

    Source-attribution contract (R2.3 / R2.4 round-2):
        ``user_config['target_mde_source']``, when set, is treated as the
        authoritative provenance label. scope_builder writes BOTH
        ``target_mde`` and ``target_mde_source`` together (per the D1
        audit-chain contract); the resolver MUST preserve that stamp so
        ``computed_from_data`` / ``literature_default`` values pre-stamped
        upstream don't get silently re-labelled as ``user_override``. If
        ``target_mde_source`` is absent, the resolver assumes the value
        came directly from the user (the historical "raw scope_spec dict"
        path) and stamps ``user_override`` itself.
    """
    override = _user_override(user_config, "target_mde")
    if override is not None:
        # F5 (PR #462 hotfix) + R2.2 (round-2): validate the override
        # BEFORE accepting it. ``SufficiencyConfig.target_mde`` carries
        # ``gt=0, lt=1e6`` (R2.2: widened so continuous-outcome raw effect
        # sizes don't get schema-rejected). The resolver re-validates with
        # outcome-type-aware bounds because raw dicts can bypass the schema:
        #   * binary: must be in (0, 1) — absolute risk difference
        #   * continuous: must be positive and finite — raw effect size in
        #     outcome units, no upper bound
        #   * time_to_event / unknown: must be positive and finite
        # NaN/inf are always rejected.
        try:
            override_float = float(override)
        except (TypeError, ValueError):
            override_float = math.nan
        if not math.isfinite(override_float) or override_float <= 0.0:
            valid_override = False
        elif outcome_type == "binary":
            valid_override = override_float < 1.0
        else:
            # continuous / time_to_event / unknown: positive + finite only.
            valid_override = True
        if valid_override:
            # R2.3 / R2.4: preserve scope_builder's provenance stamp if
            # present (target_mde_source). Without this, scope_builder's
            # pre-stamped ``computed_from_data`` / ``literature_default``
            # values would always get re-labelled as ``user_override`` —
            # breaking the audit chain that D1 of the rollout plan
            # explicitly built.
            stamped_source = _user_override(user_config, "target_mde_source")
            if stamped_source in (
                "user_override",
                "computed_from_data",
                "literature_default",
            ):
                source = stamped_source
                citation = (
                    "scope_spec.sufficiency.target_mde "
                    f"(source pre-stamped by upstream: {stamped_source})"
                )
            else:
                source = "user_override"
                citation = "scope_spec.sufficiency.target_mde"
            return ThresholdResolution(
                name="target_mde",
                value=override_float,
                source=source,
                citation=citation,
                inputs={"outcome_type": outcome_type},
            )
        # Outcome-type-aware warning text.
        if outcome_type == "binary":
            bound_msg = "must be in (0, 1) and finite (absolute risk difference)"
        else:
            bound_msg = "must be positive and finite"
        logger.warning(
            "scope_spec.sufficiency.target_mde=%r is invalid for "
            "outcome_type=%r (%s); falling back to data-driven / "
            "literature default.",
            override,
            outcome_type,
            bound_msg,
        )

    if outcome_type == "continuous":
        if sigma_outcome is not None and sigma_outcome > 0:
            value = 0.5 * sigma_outcome  # Cohen medium scaled to data
            return ThresholdResolution(
                name="target_mde",
                value=value,
                source="computed_from_data",
                citation=citation_for("DEFAULT_MDE_CONTINUOUS_COHENS_D"),
                inputs={"sigma_outcome": sigma_outcome, "cohens_d_anchor": 0.5},
            )
        return ThresholdResolution(
            name="target_mde",
            value=DEFAULT_MDE_CONTINUOUS_COHENS_D,
            source="literature_default",
            citation=citation_for("DEFAULT_MDE_CONTINUOUS_COHENS_D"),
            inputs={"outcome_type": outcome_type},
        )

    if outcome_type == "binary":
        if baseline_rate is not None and 0.0 < baseline_rate < 1.0:
            relative_mde = DEFAULT_MDE_BINARY_RELATIVE * baseline_rate
            value = max(DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR, relative_mde)
            return ThresholdResolution(
                name="target_mde",
                value=value,
                source="computed_from_data",
                citation=citation_for("DEFAULT_MDE_BINARY_RELATIVE"),
                inputs={
                    "baseline_rate": baseline_rate,
                    "relative_mde": relative_mde,
                    "absolute_floor": DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR,
                },
            )
        return ThresholdResolution(
            name="target_mde",
            value=DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR,
            source="literature_default",
            citation=citation_for("DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR"),
            inputs={"outcome_type": outcome_type},
        )

    if outcome_type == "time_to_event":
        return ThresholdResolution(
            name="target_mde",
            value=DEFAULT_MDE_HAZARD_RATIO,
            source="literature_default",
            citation=citation_for("DEFAULT_MDE_HAZARD_RATIO"),
            inputs={"outcome_type": outcome_type},
        )

    return ThresholdResolution(
        name="target_mde",
        value=DEFAULT_MDE_CONTINUOUS_COHENS_D,
        source="literature_default",
        citation=citation_for("DEFAULT_MDE_CONTINUOUS_COHENS_D"),
        inputs={"outcome_type": outcome_type, "note": "unknown outcome_type fallback"},
    )


def resolve_alpha(*, user_config: dict[str, Any] | None) -> ThresholdResolution:
    """Type-I error rate."""
    override = _user_override(user_config, "alpha")
    if override is not None:
        return ThresholdResolution(
            name="alpha",
            value=float(override),
            source="user_override",
            citation="scope_spec.sufficiency.alpha",
            inputs={},
        )
    return ThresholdResolution(
        name="alpha",
        value=DEFAULT_ALPHA,
        source="literature_default",
        citation=citation_for("DEFAULT_ALPHA"),
        inputs={},
    )


def resolve_power(*, user_config: dict[str, Any] | None) -> ThresholdResolution:
    """Statistical power target (1 - beta)."""
    override = _user_override(user_config, "power_target")
    if override is not None:
        return ThresholdResolution(
            name="power_target",
            value=float(override),
            source="user_override",
            citation="scope_spec.sufficiency.power_target",
            inputs={},
        )
    return ThresholdResolution(
        name="power_target",
        value=DEFAULT_POWER,
        source="literature_default",
        citation=citation_for("DEFAULT_POWER"),
        inputs={},
    )


def resolve_timeseries_min_n(
    *,
    user_config: dict[str, Any] | None,
    seasonal_period: int | None = None,
    n_features: int = 0,
    cv_outcome: float | None = None,
) -> ThresholdResolution:
    """Time-series minimum n via Hyndman/Kostenko formula.

    n_min = ceil(headroom_cycles × m × noise_factor) + n_features + 1
    where noise_factor = 1 + cv_outcome (1.0 if cv_outcome unknown).
    Falls back to absolute floor when seasonal_period unknown.
    """
    override = _user_override(user_config, "absolute_floor")
    if override is not None:
        return ThresholdResolution(
            name="timeseries_min_n",
            value=int(override),
            source="user_override",
            citation="scope_spec.sufficiency.absolute_floor",
            inputs={},
        )

    user_period = _user_override(user_config, "seasonal_period")
    period = user_period if user_period is not None else seasonal_period

    if period is None:
        floor = ABSOLUTE_FLOORS["time_series"]
        return ThresholdResolution(
            name="timeseries_min_n",
            value=floor,
            source="literature_default",
            citation=citation_for("ABSOLUTE_FLOORS"),
            inputs={"reason": "seasonal_period unknown; using absolute floor"},
        )

    user_cv = _user_override(user_config, "cv_outcome")
    cv = user_cv if user_cv is not None else cv_outcome
    noise_factor = 1.0 + cv if cv is not None and cv >= 0 else 1.0

    raw = TIMESERIES_CYCLES_HEADROOM * period * noise_factor + n_features + 1
    value = max(ABSOLUTE_FLOORS["time_series"], int(round(raw)))

    inputs = {
        "seasonal_period": period,
        "n_features": n_features,
        "cv_outcome": cv,
        "noise_factor": noise_factor,
        "headroom_cycles": TIMESERIES_CYCLES_HEADROOM,
    }
    if user_period is not None:
        return ThresholdResolution(
            name="timeseries_min_n",
            value=value,
            source="user_override",
            citation="scope_spec.sufficiency.seasonal_period",
            inputs=inputs,
        )
    return ThresholdResolution(
        name="timeseries_min_n",
        value=value,
        source="computed_from_data",
        citation=citation_for("TIMESERIES_CYCLES_HEADROOM"),
        inputs=inputs,
    )
