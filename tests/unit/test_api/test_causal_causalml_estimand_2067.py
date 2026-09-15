"""#2067 / #2106: the CausalML stage payload must name the estimand its `ate` belongs to.

The executor warns when CausalML's internal ``y = (y > 0)`` recodes a
non-collapsing outcome, but ``warnings`` reaches only the raw JSON of
``POST /causal/pipeline/{sequential,parallel}`` — no page renders it. The
cheap seam is to put the estimand next to the number that needs it: the
causalml branch of ``_extract_library_payload`` carries the estimand and the
``data_provenance`` honesty marker the executor sets but the route dropped.

#2106: #2067 wrote that estimand under ``identified_estimand`` — the key the
dowhy branch uses for DoWhy's own identification label (``str()`` of DoWhy's
``EstimandType`` enum, ``EstimandType.NONPARAMETRIC_ATE`` on the real
executor: the estimand it IDENTIFIED from the graph, a step CausalML never
performs).
One key, two vocabularies, two meanings. The stage payload now separates them:
``identified_estimand`` stays DoWhy's identification label, DoWhy only;
``estimand`` names what the reported ``effect_estimate`` ESTIMATES, in one
vocabulary for every branch that can say (``ate`` /
``risk_difference_on_indicator_y_gt_0``). DoWhy derives ``estimand`` only from
the one label whose mapping is certain; CausalML never writes
``identified_estimand`` again.

Both land in ``PipelineStageResult.additional_results`` (``Dict[str, Any]``)
and in the parallel response's free-form ``library_results``, so no pydantic
field is added and the generated ``api.ts`` does not move.

The causalml payloads under test come from a REAL ``CausalMLExecutor.execute``
run — only the forest fit is stood in for, since the estimand is decided before
the fit and does not depend on its numbers. The dowhy payloads are the
executor's result-dict shape (``causal_effect`` + ``identified_estimand``)
placed in state directly.
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
    assert stage.additional_results["estimand"] == BINARIZED_ESTIMAND
    # #2106: CausalML performs no identification step, so the stage must not
    # carry the key that means "the estimand DoWhy identified from the graph".
    assert "identified_estimand" not in stage.additional_results
    assert stage.additional_results["data_provenance"] == PROVENANCE_MODEL_PREDICTED_UPLIFT


def test_genuinely_binary_outcome_stage_names_the_ate() -> None:
    output, state = _run_causalml([0.0, 1.0], reps=30)
    assert state["causalml_result"]["result"]["outcome_binarized"] is False

    stage = _stage(output, state)

    assert stage.additional_results["estimand"] == "ate"
    assert "identified_estimand" not in stage.additional_results
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
    assert "estimand" not in payload
    assert "identified_estimand" not in payload
    assert "data_provenance" not in payload


def test_two_valued_non_binary_outcome_stage_names_the_binarized_estimand() -> None:
    # codex r1 HIGH: {0, 2} passes the collapse gate and has 2 distinct
    # values, so a `distinct > 2` discriminator labelled its risk difference
    # "ate". The recoding, not the distinct count, decides the estimand.
    output, state = _run_causalml([0.0, 2.0], reps=30)
    assert state["causalml_result"]["result"]["outcome_binarized"] is True

    stage = _stage(output, state)

    assert stage.additional_results["estimand"] == BINARIZED_ESTIMAND
    assert stage.additional_results["outcome_distinct_values"] == 2


# =============================================================================
# #2106: the dowhy branch keeps its identification label and gains `estimand`
# =============================================================================


def _dowhy_payload(identified_estimand: str) -> Dict[str, Any]:
    """Run ``_extract_library_payload`` over a DoWhy result dict in state.

    The dict mirrors the executor's result shape (``dowhy.py`` builds
    ``causal_effect`` and ``identified_estimand``, the latter from
    ``_extract_estimand_label`` — ``str()`` of DoWhy's ``estimand_type``
    enum, else the estimand's class name).
    """
    state = cast(
        PipelineState,
        {
            "config": _config(),
            "dowhy_result": {
                "success": True,
                "result": {
                    "causal_effect": 0.25,
                    "identified_estimand": identified_estimand,
                },
            },
        },
    )
    output = cast(PipelineOutput, {"libraries_used": ["dowhy"], "errors": []})
    return causal_pipelines._extract_library_payload("dowhy", output, state=state)


def test_dowhy_real_executor_label_is_kept_verbatim_and_derives_estimand_ate() -> None:
    # The label the REAL executor emits: `_extract_estimand_label` returns
    # `str(estimand_type)` (dowhy.py:427), and DoWhy's `EstimandType` is a
    # plain Enum, so that is "EstimandType.NONPARAMETRIC_ATE" — the string
    # the deployed api's dowhy stage carried in the #2067 live cert — not the
    # enum's value "nonparametric-ate". The test follows the library, not a
    # literal, so a DoWhy change to the enum's `__str__` is caught here.
    from dowhy.causal_identifier import EstimandType

    real_label = str(EstimandType.NONPARAMETRIC_ATE)
    assert real_label != EstimandType.NONPARAMETRIC_ATE.value, real_label

    payload = _dowhy_payload(real_label)

    assert payload["effect_estimate"] == 0.25
    assert payload["identified_estimand"] == real_label
    assert payload["estimand"] == "ate"


def test_dowhy_enum_value_label_also_derives_estimand_ate() -> None:
    # The enum's value spelling is the same certain identification; a
    # payload that carries it (e.g. a caller that stored `.value`) maps too.
    payload = _dowhy_payload("nonparametric-ate")

    assert payload["identified_estimand"] == "nonparametric-ate"
    assert payload["estimand"] == "ate"


def test_dowhy_other_identification_label_claims_no_estimand() -> None:
    # Any other DoWhy label (here the pre-real-wiring placeholder, "backdoor")
    # is kept verbatim as the identification label, and no `estimand` is
    # guessed from it.
    payload = _dowhy_payload("backdoor")

    assert payload["identified_estimand"] == "backdoor"
    assert "estimand" not in payload
