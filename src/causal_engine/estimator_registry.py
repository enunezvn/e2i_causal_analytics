"""Canonical metadata for estimators served by the energy-score selector.

An estimator used to be described independently by the selector, causal-impact
agent, refutation node, API catalog, interval aggregator, and UI.  That made a
"registered" estimator only partially discoverable and made exact estimator
reconstruction depend on duplicated string switches.  This module is the cheap
(no sklearn/econml imports) source of truth for those contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EstimatorType(str, Enum):
    """Estimator identities understood by :class:`EstimatorSelector`."""

    CAUSAL_FOREST = "causal_forest"
    LINEAR_DML = "linear_dml"
    DML_LEARNER = "dml_learner"
    DRLEARNER = "drlearner"
    ORTHO_FOREST = "ortho_forest"
    S_LEARNER = "s_learner"
    T_LEARNER = "t_learner"
    X_LEARNER = "x_learner"
    OLS = "ols"


@dataclass(frozen=True)
class EstimatorSpec:
    """Cross-surface contract for one selectable estimator."""

    estimator_type: EstimatorType
    wrapper_symbol: str
    speed_rank: int
    default_priority: Optional[int] = None
    aliases: tuple[str, ...] = ()
    result_method: Optional[str] = None
    dowhy_method: Optional[str] = None
    refutation_labels: tuple[str, ...] = ()
    reconstruction_key: Optional[str] = None
    forceable_alias: Optional[str] = None
    sampling_interval: bool = False
    produces_cate: bool = False
    confounding_blind: bool = False
    empty_backdoor_capable: bool = False
    public_name: Optional[str] = None
    public_library: str = "econml"
    public_estimator_type: str = "CATE"
    description: str = ""
    best_for: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    supports_confidence_intervals: bool = False


ESTIMATOR_SPECS: tuple[EstimatorSpec, ...] = (
    EstimatorSpec(
        EstimatorType.CAUSAL_FOREST,
        "CausalForestWrapper",
        speed_rank=4,
        default_priority=1,
        aliases=("CausalForestDML", "causal_forest"),
        result_method="CausalForestDML",
        dowhy_method="backdoor.econml.dml.CausalForestDML",
        refutation_labels=("causal_forest", "CausalForestDML"),
        forceable_alias="CausalForestDML",
        sampling_interval=True,
        produces_cate=True,
        description="Causal Forest for heterogeneous treatment effects",
        best_for=("Effect heterogeneity", "Feature importance"),
        parameters=("n_estimators", "min_samples_leaf", "max_depth"),
        supports_confidence_intervals=True,
    ),
    EstimatorSpec(
        EstimatorType.LINEAR_DML,
        "LinearDMLWrapper",
        speed_rank=2,
        default_priority=2,
        aliases=("LinearDML", "linear_dml"),
        result_method="LinearDML",
        dowhy_method="backdoor.econml.dml.LinearDML",
        refutation_labels=("linear_dml", "LinearDML"),
        reconstruction_key="linear_dml",
        forceable_alias="LinearDML",
        sampling_interval=True,
        produces_cate=True,
        description="Double Machine Learning with a linear final stage",
        best_for=("High-dimensional confounders", "Linear effect modification"),
        parameters=("model_y", "model_t", "cv"),
        supports_confidence_intervals=True,
    ),
    EstimatorSpec(
        EstimatorType.DML_LEARNER,
        "DMLLearnerWrapper",
        speed_rank=3,
        # Deliberately opt-in: the 2026-09-20 production-shaped benchmark
        # added ~18 s without changing the winner or substantive conclusion.
        # The benchmark report records the evidence and promotion criteria.
        aliases=("dml_learner",),
        result_method="dml_learner",
        dowhy_method="backdoor.econml.dml.DML",
        refutation_labels=("dml_learner",),
        reconstruction_key="dml_learner",
        forceable_alias="dml_learner",
        sampling_interval=True,
        produces_cate=True,
        description="DML with a rank-safe quadratic effect-modification stage",
        best_for=("Non-linear effect modification", "Expert sensitivity analysis"),
        parameters=("model_y", "model_t", "featurizer", "cv"),
        supports_confidence_intervals=True,
    ),
    EstimatorSpec(
        EstimatorType.DRLEARNER,
        "DRLearnerWrapper",
        speed_rank=3,
        default_priority=3,
        aliases=("drlearner",),
        result_method="linear_regression",  # legacy response compatibility
        dowhy_method="backdoor.econml.dr.DRLearner",
        refutation_labels=("drlearner",),
        reconstruction_key="drlearner",
        forceable_alias="drlearner",
        sampling_interval=True,
        produces_cate=True,
        public_name="dr_learner",
        description="Doubly Robust Learner",
        best_for=("Robustness to nuisance-model misspecification",),
        parameters=("model_propensity", "model_regression", "model_final"),
        supports_confidence_intervals=True,
    ),
    EstimatorSpec(
        EstimatorType.ORTHO_FOREST,
        "OrthoForestWrapper",
        speed_rank=4,
        aliases=("ortho_forest",),
        result_method="ortho_forest",
        produces_cate=True,
        description="Orthogonal Random Forest for CATE",
        best_for=("Non-linear effects",),
        parameters=("n_trees", "subsample_ratio", "max_depth"),
        supports_confidence_intervals=True,
    ),
    EstimatorSpec(
        EstimatorType.S_LEARNER,
        "SLearnerWrapper",
        speed_rank=1,
        aliases=("s_learner",),
        result_method="s_learner",
        produces_cate=True,
        public_estimator_type="Meta-Learner",
        description="Single-model meta-learner",
        best_for=("Limited data",),
        parameters=("overall_model",),
    ),
    EstimatorSpec(
        EstimatorType.T_LEARNER,
        "TLearnerWrapper",
        speed_rank=1,
        aliases=("t_learner",),
        result_method="t_learner",
        produces_cate=True,
        public_estimator_type="Meta-Learner",
        description="Two-model meta-learner",
        best_for=("Simple heterogeneous-effect modeling",),
        parameters=("models",),
    ),
    EstimatorSpec(
        EstimatorType.X_LEARNER,
        "XLearnerWrapper",
        speed_rank=2,
        aliases=("x_learner",),
        result_method="x_learner",
        produces_cate=True,
        public_estimator_type="Meta-Learner",
        description="X-Learner for heterogeneous effects",
        best_for=("Imbalanced treatment groups",),
        parameters=("models", "propensity_model"),
        supports_confidence_intervals=True,
    ),
    EstimatorSpec(
        EstimatorType.OLS,
        "OLSWrapper",
        speed_rank=0,
        default_priority=4,
        aliases=("linear_regression", "ols"),
        result_method="linear_regression",
        dowhy_method="backdoor.linear_regression",
        refutation_labels=("ols", "linear_regression"),
        forceable_alias="ols",
        sampling_interval=True,
        confounding_blind=True,
        empty_backdoor_capable=True,
        public_estimator_type="ATE",
        description="Ordinary least squares ATE baseline",
        best_for=("Randomized or empty-backdoor designs", "Fast baseline"),
        parameters=("fit_intercept",),
        supports_confidence_intervals=True,
    ),
)

ESTIMATOR_SPEC_BY_TYPE = {spec.estimator_type: spec for spec in ESTIMATOR_SPECS}
ESTIMATOR_SPEC_BY_VALUE = {spec.estimator_type.value: spec for spec in ESTIMATOR_SPECS}
ESTIMATOR_TYPE_BY_ALIAS = {
    alias: spec.estimator_type for spec in ESTIMATOR_SPECS for alias in spec.aliases
}
DEFAULT_ESTIMATOR_SPECS = tuple(
    sorted(
        (spec for spec in ESTIMATOR_SPECS if spec.default_priority is not None),
        key=lambda spec: int(spec.default_priority or 0),
    )
)
ESTIMATOR_SPEED_RANK = {spec.estimator_type: spec.speed_rank for spec in ESTIMATOR_SPECS}
CONFOUNDING_BLIND_ESTIMATORS = frozenset(
    spec.estimator_type for spec in ESTIMATOR_SPECS if spec.confounding_blind
)
EMPTY_BACKDOOR_CAPABLE = frozenset(
    spec.estimator_type for spec in ESTIMATOR_SPECS if spec.empty_backdoor_capable
)
SAMPLING_INTERVAL_ESTIMATOR_VALUES = frozenset(
    spec.estimator_type.value for spec in ESTIMATOR_SPECS if spec.sampling_interval
)
AGENT_FORCEABLE_ESTIMATORS = tuple(
    spec.forceable_alias for spec in ESTIMATOR_SPECS if spec.forceable_alias is not None
)
FORCEABLE_ESTIMATOR_TYPE_BY_ALIAS = {
    alias: spec.estimator_type
    for spec in ESTIMATOR_SPECS
    if spec.forceable_alias is not None
    for alias in spec.aliases
}
DOWHY_METHOD_BY_LABEL = {
    label: spec.dowhy_method
    for spec in ESTIMATOR_SPECS
    if spec.dowhy_method is not None
    for label in spec.refutation_labels
}
ESTIMATOR_SPEC_BY_DOWHY_METHOD = {
    spec.dowhy_method: spec for spec in ESTIMATOR_SPECS if spec.dowhy_method is not None
}


def get_estimator_spec(estimator: EstimatorType | str) -> EstimatorSpec:
    """Resolve a canonical estimator value or an exact registered alias."""

    if isinstance(estimator, EstimatorType):
        return ESTIMATOR_SPEC_BY_TYPE[estimator]
    spec = ESTIMATOR_SPEC_BY_VALUE.get(estimator)
    if spec is not None:
        return spec
    estimator_type = ESTIMATOR_TYPE_BY_ALIAS.get(estimator)
    if estimator_type is None:
        raise KeyError(estimator)
    return ESTIMATOR_SPEC_BY_TYPE[estimator_type]
