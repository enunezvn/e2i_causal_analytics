"""Literature-grounded default thresholds for data sufficiency checks.

Single source of truth. Every constant carries a citation explaining where it
came from so the rationale survives the next refactor. Per the
REASON-BEFORE-RULES discipline in CLAUDE.md, a change here is a change to the
literature stance, not a casual numeric tweak — update the citation when
updating the value.

Override path: every constant here is the LAST resort in the resolution
hierarchy (user_override > computed_from_data > literature_default). See
src/utils/sufficiency_resolver.py.
"""

from __future__ import annotations

from typing import Literal

ProblemType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "causal_inference",
    "time_series",
]
AlgorithmFamily = Literal["linear", "tree_based", "neural_network", "unknown"]
StrictnessPreset = Literal["conservative", "moderate", "strict"]


# ---------------------------------------------------------------------------
# Absolute floors per problem type.
# Below these row counts the asymptotic formulas underlying every other check
# stop being meaningful (CLT breaks down, EPV bounds are dominated by edge
# effects). Treated as a HARD-FAIL floor — no override.
#
# Calibrated to the Vergouwe 2007 "severe problems" zone (EPV 2-4) plus
# headroom for cross-validation splits.
# ---------------------------------------------------------------------------
ABSOLUTE_FLOORS: dict[str, int] = {
    "binary_classification": 100,
    "multiclass_classification": 200,
    "regression": 50,
    "causal_inference": 200,
    "time_series": 100,
}


# ---------------------------------------------------------------------------
# Events-per-variable (EPV) floors by algorithm complexity.
#
# Citations:
# - Vergouwe et al. 2007, Am J Epidemiol — "EPV 5-9 often adequate; severe
#   problems mainly in EPV 2-4 range"
# - van Smeden et al. 2019, Stat Methods Med Res — "EPV does not have a
#   strong relation with predictive performance; consider predictors,
#   total n, and event fraction jointly"
# - Riley et al. 2020, BMJ — full three-criterion formula in pmsampsize;
#   tree models implicitly need higher EPV due to complexity
# ---------------------------------------------------------------------------
EPV_FLOORS: dict[str, int] = {
    "linear": 5,  # Vergouwe consensus floor for stable linear models
    "tree_based": 10,  # Riley 2020 / pmsampsize default for moderately complex models
    "neural_network": 20,  # Riley 2020 + stepwise-equivalent complexity penalty
    "unknown": 5,  # Conservative default before ModelSelector picks an algorithm
}


# ---------------------------------------------------------------------------
# Sample-to-feature ratio floors for regression by algorithm complexity.
# Same logic as EPV but applied to continuous outcomes (no minority class).
# ---------------------------------------------------------------------------
REGRESSION_RATIOS: dict[str, int] = {
    "linear": 5,
    "tree_based": 10,
    "neural_network": 15,
    "unknown": 5,
}


# ---------------------------------------------------------------------------
# Statistical conventions.
#
# Citations:
# - alpha=0.05: ICH E9 "Statistical Principles for Clinical Trials"
# - power=0.80: ICH E9; long-standing convention in pharma trial design
#
# These are NOT derivable from data. They remain configurable via
# scope_spec.sufficiency overrides.
# ---------------------------------------------------------------------------
DEFAULT_ALPHA: float = 0.05
DEFAULT_POWER: float = 0.80


# ---------------------------------------------------------------------------
# Default minimum detectable effects when target_mde is not specified by user.
#
# Citations:
# - Cohen 1988 — d=0.2/0.5/0.8 (small/medium/large) "for use when no others
#   suggest themselves"; we use d=0.5 (medium) as the "data-driven default"
#   anchor
# - MCID conventions — 5-10pp absolute risk difference is the typical
#   "minimally meaningful" threshold in pharma; we floor binary at 0.05
# ---------------------------------------------------------------------------
DEFAULT_MDE_CONTINUOUS_COHENS_D: float = 0.5  # Cohen "medium"
DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR: float = 0.05  # 5pp ARD minimum
DEFAULT_MDE_BINARY_RELATIVE: float = 0.20  # 20% relative shift
DEFAULT_MDE_HAZARD_RATIO: float = 0.75  # 25% risk reduction


# ---------------------------------------------------------------------------
# Observational causal-inference inflation factor.
#
# Citations:
# - Yang et al. 2025, arxiv 2501.11181 — required n increases as overlap
#   shrinks; applying RCT formulas to observational data yields severely
#   underpowered studies
#
# At pre-flight time, observed overlap is unknown. Default = 2.0 (good-overlap
# assumption). Phase 2 refines this from observed propensity score support.
# ---------------------------------------------------------------------------
DEFAULT_OBSERVATIONAL_INFLATION: float = 2.0


# ---------------------------------------------------------------------------
# Time-series sample-size floor.
#
# Citations:
# - Hyndman & Kostenko 2007 — minimum 6 obs for quarterly (m=4), 14 for
#   monthly (m=12). General rule: n_min ≈ m + p_arima + 1.
# - Hyndman blog — "30 observations for ARIMA" rule has no theoretical basis;
#   practical guidance is 2-3 full seasonal cycles for noisy data.
#
# Our formula: n_min = 2 * m + n_features + 1 (two full cycles + ARIMA
# parameter headroom + feature degrees of freedom).
# ---------------------------------------------------------------------------
TIMESERIES_CYCLES_HEADROOM: int = 2  # Hyndman/Kostenko + practical noise margin


# ---------------------------------------------------------------------------
# Strictness preset multipliers.
# Applied to EPV_FLOORS / REGRESSION_RATIOS for the corresponding preset.
# ---------------------------------------------------------------------------
STRICTNESS_MULTIPLIERS: dict[str, float] = {
    "conservative": 0.5,  # cheap-diagnostic / exploratory mode
    "moderate": 1.0,  # default
    "strict": 2.0,  # regulatory-submission mode
}


# ---------------------------------------------------------------------------
# Citation registry — programmatic access to "where did this number come from?"
# Used by the audit chain to attach citations to each threshold resolution.
# ---------------------------------------------------------------------------
CITATIONS: dict[str, str] = {
    "ABSOLUTE_FLOORS": ("Vergouwe 2007 (severe-problems EPV<5 zone) + headroom for k-fold splits"),
    "EPV_FLOORS.linear": "Vergouwe 2007 (EPV>=5 consensus floor)",
    "EPV_FLOORS.tree_based": "Riley 2020 / pmsampsize — tree-complexity penalty",
    "EPV_FLOORS.neural_network": "Riley 2020 + stepwise-selection-equivalent complexity",
    "EPV_FLOORS.unknown": "Conservative default pre-algorithm-selection",
    "REGRESSION_RATIOS.linear": "Standard 5:1 sample-to-feature ratio for linear regression",
    "REGRESSION_RATIOS.tree_based": "10:1 for tree-based regressors (complexity penalty)",
    "REGRESSION_RATIOS.neural_network": "15:1 for NN regressors (parameter inflation)",
    "REGRESSION_RATIOS.unknown": "Conservative default pre-algorithm-selection",
    "DEFAULT_ALPHA": "ICH E9 Statistical Principles for Clinical Trials",
    "DEFAULT_POWER": "ICH E9 / standard pharma trial design convention",
    "DEFAULT_MDE_CONTINUOUS_COHENS_D": "Cohen 1988 — medium effect anchor",
    "DEFAULT_MDE_BINARY_ABSOLUTE_FLOOR": "MCID pharma convention — 5pp ARD minimum",
    "DEFAULT_MDE_BINARY_RELATIVE": "MCID convention — 20% relative shift",
    "DEFAULT_MDE_HAZARD_RATIO": "Conventional trial design — 25% risk reduction",
    "DEFAULT_OBSERVATIONAL_INFLATION": (
        "Yang 2025 arxiv:2501.11181 — RCT formulas underpower observational studies"
    ),
    "TIMESERIES_CYCLES_HEADROOM": "Hyndman & Kostenko 2007 + practical-noise margin",
}


def citation_for(name: str) -> str:
    """Return the citation for a named constant, or 'uncited' if missing."""
    return CITATIONS.get(name, "uncited")
