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

from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import ConfigDict, Field, model_validator

from src.agents.ml_foundation._pydantic_utils import BaseAgentSchema


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
    auc_roc: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    accuracy: Optional[float] = None
    log_loss: Optional[float] = None

    # Regression metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None

    # Business utility (when scope_spec.cost_matrix is present)
    business_utility: Optional[float] = None

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
                return self

        if self.problem_type == "regression":
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
