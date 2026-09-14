"""#2067: the CausalML stage payload must name the estimand its `ate` belongs to.

The executor now warns when CausalML's internal ``y = (y > 0)`` recodes a
non-collapsing outcome, but ``warnings`` reaches only the raw JSON of
``POST /causal/pipeline/{sequential,parallel}`` — no page renders it. The
cheap seam is to put the estimand next to the number that needs it: the
causalml branch of ``_extract_library_payload`` carries
``identified_estimand`` (as the dowhy branch already does) and the
``data_provenance`` honesty marker the executor sets but the route dropped.

Both land in ``PipelineStageResult.additional_results`` (``Dict[str, Any]``)
and in the parallel response's free-form ``library_results``, so no pydantic
field is added and the generated ``api.ts`` does not move.

The payloads under test come from a REAL ``CausalMLExecutor.execute`` run —
only the forest fit is stood in for, since the estimand is decided before the
fit and does not depend on its numbers.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.api.routes.causal import pipelines as causal_pipelines
from src.api.schemas.causal import AnalysisStatus
from src.causal_engine.pipeline.executors.causalml import CausalMLExecutor
from src.causal_engine.pipeline.state import PipelineConfig, PipelineOutput, PipelineState
from src.causal_engine.uplift.base import (
    PROVENANCE_MODEL_PREDICTED_UPLIFT,
    UpliftModelType,
    UpliftResult,
)

BINARIZED_ESTIMAND = "risk_difference_on_indicator_y_gt_0"


def _frame(pattern: list[float], reps: int, seed: int = 2067) -> pd.DataFrame:
    y = np.array(pattern * reps, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "treatment": np.arange(n) % 2,
            "outcome": y,
            "age": rng.normal(50.0, 10.0, size=n),
            "income": rng.normal(60000.0, 15000.0, size=n),
        }
    )


def _fake_fit(*args: Any, **kwargs: Any) -> tuple:
    x_df = kwargs.get("X_df", args[0] if args else None)
    return (
        UpliftResult(
            model_type=UpliftModelType.UPLIFT_RANDOM_FOREST,
            success=True,
            uplift_scores=np.linspace(-0.1, 0.1, len(x_df)),
            ate=0.02,
            att=0.02,
            atc=0.02,
            ate_std=0.01,
            ate_ci_lower=0.0,
            ate_ci_upper=0.04,
            treatment_groups=["1"],
            feature_importances={"age": 0.5, "income": 0.5},
        ),
        UpliftModelType.UPLIFT_RANDOM_FOREST.value,
    )


def _config() -> PipelineConfig:
    return cast(
        PipelineConfig,
        {
            "mode": "sequential",
            "libraries_enabled": ["causalml"],
            "primary_library": "causalml",
            "stage_timeout_ms": 30000,
            "total_timeout_ms": 120000,
            "cross_validate": False,
            "min_agreement_threshold": 0.85,
            "max_parallel_libraries": 4,
            "fail_fast": False,
            "segment_by_uplift": False,
            "nested_ci_level": 0.95,
        },
    )


def _run_causalml(pattern: list[float], reps: int) -> tuple:
    """Execute the real CausalML executor and return ``(output, state)``."""
    state = cast(
        PipelineState,
        {
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "confounders": ["age", "income"],
            "effect_modifiers": None,
            "filters": {"dataframe": _frame(pattern, reps)},
            "config": _config(),
            "stage_latencies": {},
            "warnings": [],
        },
    )
    with (
        patch(
            "src.causal_engine.pipeline.executors.causalml._fit_uplift_model",
            side_effect=_fake_fit,
        ),
        patch(
            "src.causal_engine.pipeline.executors.causalml._compute_uplift_metrics_safe",
            return_value={"auuc": 0.51, "qini": 0.02},
        ),
    ):
        result = asyncio.run(CausalMLExecutor().execute(state, _config()))

    assert result["success"] is True, result.get("error")
    cast(Dict[str, Any], state)["causalml_result"] = result
    output = cast(
        PipelineOutput,
        {"libraries_used": ["causalml"], "errors": [], "total_latency_ms": 12},
    )
    return output, state


def _stage(output: PipelineOutput, state: PipelineState):
    return causal_pipelines._build_stage_result_from_output(
        stage_number=1,
        stage_config_library="causalml",
        stage_config_estimator="uplift_random_forest",
        output=output,
        state=state,
        successful_libraries=["causalml"],
        error_by_library={},
    )


def test_binarized_outcome_stage_names_the_binarized_estimand() -> None:
    output, state = _run_causalml([-1.0, 2.0, -3.0, 4.0, 0.0], reps=12)
    assert state["causalml_result"]["result"]["outcome_binarized"] is True

    stage = _stage(output, state)

    assert stage.status == AnalysisStatus.COMPLETED
    assert stage.effect_estimate == 0.02
    assert stage.additional_results["identified_estimand"] == BINARIZED_ESTIMAND
    assert stage.additional_results["data_provenance"] == PROVENANCE_MODEL_PREDICTED_UPLIFT


def test_genuinely_binary_outcome_stage_names_the_ate() -> None:
    output, state = _run_causalml([0.0, 1.0], reps=30)
    assert state["causalml_result"]["result"]["outcome_binarized"] is False

    stage = _stage(output, state)

    assert stage.additional_results["identified_estimand"] == "ate"
    assert stage.additional_results["data_provenance"] == PROVENANCE_MODEL_PREDICTED_UPLIFT


def test_payload_without_the_discriminator_claims_no_estimand() -> None:
    # A causalml payload that carries no `outcome_binarized` says nothing
    # about which estimand its `ate` is; naming one would be a guess.
    state = cast(
        PipelineState,
        {
            "config": _config(),
            "causalml_result": {"success": True, "result": {"ate": 0.02}},
        },
    )
    output = cast(PipelineOutput, {"libraries_used": ["causalml"], "errors": []})

    payload = causal_pipelines._extract_library_payload("causalml", output, state=state)

    assert payload["effect_estimate"] == 0.02
    assert "identified_estimand" not in payload
    assert "data_provenance" not in payload
