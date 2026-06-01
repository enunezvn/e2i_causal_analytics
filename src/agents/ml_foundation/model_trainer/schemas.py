"""Pydantic v2 schemas for ``model_trainer`` agent outputs.

Formalises the ``validation_metrics`` / ``test_metrics`` payloads that
the trainer emits to downstream consumers. Per Decision 2a in the
migration plan, this module hosts the canonical ``MetricsSchema`` —
``model_deployer`` and ``feature_analyzer`` will import from here in
Shard C.

Shard B adds ``OptunaDistributionSchema`` — the typed encoding of
``hyperparameter_search_space`` entries passed between model_selector
and model_trainer.

D2.1 (2026-05-05) wires the typed schema into State annotations:
- ``ModelTrainerState.hyperparameter_search_space: Optional[Dict[str, OptunaDistribution]]``
- ``ModelSelectorState.hyperparameter_search_space: Optional[Dict[str, OptunaDistribution]]``

Producer dict literals from ``algorithm_registry.py`` validate cleanly
into the discriminated union (``int`` | ``float`` | ``categorical``);
consumer access via ``config["low"]`` works through the dict-shim on
``_OptunaDistributionBase`` (which now extends ``BaseAgentSchema``).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, ConfigDict, Field, model_validator

from src.agents.ml_foundation._pydantic_utils import BaseAgentSchema

logger = logging.getLogger(__name__)


class MetricsSchema(BaseAgentSchema):
    """Unified validation/test metrics surface for any problem type.

    Per migration plan Hotspot #2, this currently surfaces as
    ``Dict[str, float]`` with problem-type-conditional keys (e.g.
    classification fills ``auc_roc``/``f1_score``; regression fills
    ``rmse``/``r2``/``mae``). The schema declares every metric as
    ``Optional[float] = None`` so downstream agents can read whichever
    subset their logic needs.

    The ``problem_type`` discriminator is retained as a hint — the
    ``model_validator(mode="after")`` enforces a soft invariant that
    at least one metric is set per problem-type subset, which catches
    the regression where a trainer node forgets to populate metrics
    for a given problem type.
    """

    problem_type: Optional[
        Literal[
            "binary_classification",
            "multiclass_classification",
            "regression",
            "causal_inference",
            "time_series",
        ]
    ] = None

    # Classification metrics
    # D2.5 (2026-05-05): naming drift fix. The producer at
    # ``model_trainer/nodes/evaluator.py::_compute_split_classification_metrics``
    # emits the key ``roc_auc`` (model_trainer convention), but the
    # original schema declared ``auc_roc`` (transposed). The deployer
    # at ``registry_manager.py:358`` was reading the schema-name-form
    # and silently returning 0.0 for every promotion (PR #59 D2.0 fixed
    # the deployer to read ``roc_auc``).
    #
    # AliasChoices accepts BOTH names at construction time per PR #55
    # precedent. Read-old / write-new asymmetry: the canonical name is
    # ``auc_roc`` (kept for backward compat with legacy checkpoints),
    # but the modern producer key ``roc_auc`` resolves to the same field.
    # Serialization uses the python field name (``auc_roc``).
    auc_roc: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("auc_roc", "roc_auc"),
    )
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    accuracy: Optional[float] = None
    log_loss: Optional[float] = None

    # D2.5: classification-extras emitted by
    # ``_compute_split_classification_metrics`` (per Phase-1 D2 audit).
    f1_macro: Optional[float] = None
    f1_weighted: Optional[float] = None
    precision_class_0: Optional[float] = None
    precision_class_1: Optional[float] = None
    recall_class_0: Optional[float] = None
    recall_class_1: Optional[float] = None
    mcc: Optional[float] = None
    pr_auc: Optional[float] = None
    brier_score: Optional[float] = None

    # D2.5: threshold-selection metadata (evaluator post-eval at
    # ``evaluator.py:1156-1175`` adds these to validation_metrics).
    chosen_threshold: Optional[float] = None
    chosen_threshold_source: Optional[str] = None

    # D2.5: calibration metrics (test_metrics adds these from
    # calibration analysis; emitted by evaluator's calibration step).
    calibration_slope: Optional[float] = None
    calibration_intercept: Optional[float] = None
    calibration_intercept_magnitude: Optional[float] = None
    calibration_slope_deviation: Optional[float] = None
    calibration_error: Optional[float] = None
    net_benefit_grid: Optional[Dict[str, float]] = None

    # D2.5: lift / baseline comparison (test_metrics-only fields per
    # ``test_lift_metric.py`` references).
    baseline_test_auc: Optional[float] = None
    train_val_auc_delta: Optional[float] = None
    train_val_delta: Optional[float] = None

    # D2.5b: evaluator-injected fields surfaced by codex P1+P5+P8 on
    # PR #80 review. Binary path: ``minimum_lift_over_baseline``
    # (evaluator.py:1226, lift gate input) + ``calibrated_ece``
    # (evaluator.py:347-384, calibration gate input + read at 2432).
    # Multiclass path: ``precision_macro`` / ``recall_macro`` / ``roc_auc_ovr``
    # (evaluator.py:1573-1584, _compute_multiclass_metrics output).
    # Without these declarations, BaseAgentSchema's extra="ignore" silently
    # drops them on coercion — causing latent data loss on checkpoint
    # restart and silent gate-pass for the calibration / lift criteria.
    minimum_lift_over_baseline: Optional[float] = None
    calibrated_ece: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    roc_auc_ovr: Optional[float] = None

    # Regression metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None

    # Business utility (when scope_spec.cost_matrix is present)
    business_utility: Optional[float] = None

    # Gate N1 (plan v4 §2): regulatory-eligibility audit.
    #
    # Append-only audit trail with two list-typed sub-fields:
    #
    #   * ``gate_history`` — every gate evaluation (timestamp, gate_name,
    #     threshold, value, outcome).
    #   * ``adaptation_history`` — every adaptive threshold relaxation
    #     (commit_sha, justification_doc, gate_name, before_threshold,
    #     after_threshold, timestamp). Empty == clean lifecycle.
    #
    # The runtime guard at ``model_deployer/regulatory_audit.py`` rejects
    # ``__setitem__`` so an entry, once landed, cannot be silently rewritten.
    # The dict shape declared here lets the field flow through the typed
    # MetricsSchema contract; deployer code wraps reads/writes through
    # ``RegulatoryEligibilityAudit.from_dict`` / ``.to_dict()``.
    regulatory_eligibility_audit: Optional[Dict[str, List[Dict[str, Any]]]] = None

    @model_validator(mode="after")
    def _check_metrics_subset_for_problem_type(self) -> "MetricsSchema":
        """Soft invariant: a populated MetricsSchema must have at least
        one metric in the subset relevant to its problem_type.

        If ``problem_type`` is None this validator is a no-op — schemas
        constructed at scaffolding time without a problem_type pass
        through. The migration's transition window may need that
        flexibility; once Shard B lands, the trainer always sets
        ``problem_type`` so the invariant becomes load-bearing.

        EXPLICIT NOTE (per codex review M2, 2026-05-05): this validator
        is currently NON-ENFORCED. Every code path returns ``self``
        without raising, including the documented "violation" cases
        below. The body exists as a placeholder for future tightening:
        when downstream consumers stop tolerating empty metric bags,
        flip the ``return self`` lines under "no metrics for stated
        problem_type" to ``raise ValueError(...)``. Until then, the
        permissive behavior is intentional and pinned by
        ``test_metrics_schema_permits_empty_metrics_for_stated_problem_type``
        in the unit-test suite.
        """
        classification_metrics = (
            self.auc_roc,
            self.f1_score,
            self.precision,
            self.recall,
            self.accuracy,
            self.log_loss,
        )
        regression_metrics = (self.rmse, self.mae, self.r2, self.mape)
        any_classification = any(m is not None for m in classification_metrics)
        any_regression = any(m is not None for m in regression_metrics)

        # Permissive: empty MetricsSchema (no problem_type, no metrics)
        # is valid during scaffolding.
        if self.problem_type is None:
            return self

        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if not any_classification and not any_regression:
                # Soft warning surface: empty metrics for a stated
                # problem_type is suspicious. We allow it (returning
                # self) because trainer nodes may populate metrics
                # in a later phase. A future tightening can convert
                # this to a raise.
                #
                # Gap G14: emit a NON-FATAL warning (no raise, behavior
                # unchanged) so the #594 "validation_metrics emptied upstream"
                # shape surfaces at the schema boundary instead of only at a
                # downstream band assertion.
                logger.warning(
                    "MetricsSchema has problem_type=%s but NO metrics populated "
                    "— the #594 empty-metrics shape. Permitted (scaffolding-"
                    "permissive, not enforced) but surfaced here (gap G14).",
                    self.problem_type,
                )
                return self

        if self.problem_type == "regression":
            if not any_regression and not any_classification:
                logger.warning(
                    "MetricsSchema has problem_type=regression but NO metrics "
                    "populated — the #594 empty-metrics shape. Permitted "
                    "(scaffolding-permissive, not enforced) but surfaced here "
                    "(gap G14)."
                )
            # Same permissive stance for regression.
            return self

        # causal_inference and time_series do not yet have a canonical
        # metric subset declared in this schema; allow through.
        return self


# --------------------------------------------------------------------------- #
# OptunaDistributionSchema — typed hyperparameter_search_space entries        #
# --------------------------------------------------------------------------- #
#
# Shard B deliverable. The plan (Hotspot #3) flagged this as non-trivial
# because Optuna distributions don't have a single canonical pydantic
# representation. The discriminated-union pattern below covers the three
# distribution kinds the project uses today:
#
# - "int":         {type, low, high, [step], [log]}
# - "float":       {type, low, high, [step], [log]}
# - "categorical": {type, choices: [...]}
#
# A model_selector entry today might look like:
#   {"learning_rate": {"type": "float", "low": 1e-4, "high": 0.1, "log": True}}
# After opt-in migration:
#   OptunaFloatDistribution(type="float", low=1e-4, high=0.1, log=True)
#
# D2.1 (2026-05-05): The schemas extend ``BaseAgentSchema`` (rather than
# plain ``BaseModel``) so the dict-shim ``__getitem__`` / ``get`` /
# ``__contains__`` is available. The ``hyperparameter_search_space``
# consumer in ``model_trainer/nodes/hyperparameter_tuner.py`` and
# ``mlops/optuna_optimizer.py`` reads these as ``config["low"]`` /
# ``config["high"]`` etc. — the shim makes those reads work transparently.
#
# We intentionally OVERRIDE BaseAgentSchema's ``extra="allow"`` with
# ``extra="forbid"`` (pydantic v2 merges ``model_config`` in subclasses,
# so this is safe; ``arbitrary_types_allowed``, ``populate_by_name``, and
# ``validate_assignment`` continue to inherit from BaseAgentSchema). The
# strict ``forbid`` is load-bearing here: producer dict literals in
# ``algorithm_registry.py`` could otherwise smuggle typo'd keys into
# ``model_extra`` silently. Discriminated-union strictness must catch the
# typo at construction time.


class _OptunaDistributionBase(BaseAgentSchema):
    """Common config for Optuna distribution variants.

    Extends ``BaseAgentSchema`` for the dict-shim accessors but keeps
    ``extra="forbid"`` for typed-shape strictness — see module-level
    comment block above the class for the D2.1 rationale.
    """

    model_config = ConfigDict(extra="forbid")  # strict — typed shape


class OptunaIntDistribution(_OptunaDistributionBase):
    """Optuna integer distribution (``suggest_int``).

    Encoded today as ``{"type": "int", "low": int, "high": int,
    "step": Optional[int], "log": Optional[bool]}``.
    """

    type: Literal["int"]
    low: int
    high: int
    step: Optional[int] = None
    log: Optional[bool] = None


class OptunaFloatDistribution(_OptunaDistributionBase):
    """Optuna float distribution (``suggest_float``).

    Encoded today as ``{"type": "float", "low": float, "high": float,
    "step": Optional[float], "log": Optional[bool]}``.
    """

    type: Literal["float"]
    low: float
    high: float
    step: Optional[float] = None
    log: Optional[bool] = None


class OptunaCategoricalDistribution(_OptunaDistributionBase):
    """Optuna categorical distribution (``suggest_categorical``).

    Encoded today as ``{"type": "categorical", "choices": [...]}``.
    Choices are kept as ``List[Any]`` because they include int/float/str
    values across different hyperparameters.
    """

    type: Literal["categorical"]
    choices: List[Any] = Field(min_length=1)


# Discriminated union — pydantic v2 picks the variant by the ``type`` tag.
OptunaDistribution = Annotated[
    Union[
        OptunaIntDistribution,
        OptunaFloatDistribution,
        OptunaCategoricalDistribution,
    ],
    Field(discriminator="type"),
]
