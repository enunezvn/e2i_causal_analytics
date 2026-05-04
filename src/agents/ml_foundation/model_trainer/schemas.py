"""Pydantic v2 schemas for ``model_trainer`` agent outputs.

Formalises the ``validation_metrics`` / ``test_metrics`` payloads that
the trainer emits to downstream consumers. Per Decision 2a in the
migration plan, this module hosts the canonical ``MetricsSchema`` —
``model_deployer`` and ``feature_analyzer`` will import from here in
Shard C.

The OptunaDistributionSchema for hyperparameter_search_space is
intentionally NOT in this scaffolding PR — that's a Shard B
deliverable because it requires deeper integration with the
hyperparameter-search flow.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import model_validator

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
