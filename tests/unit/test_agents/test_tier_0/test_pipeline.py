"""Hop-4 regression tests for the post-PR-30 wiring bug.

``MLFoundationPipeline._run_model_training`` (`src/agents/tier_0/pipeline.py`)
is the API-surface terminal hop of the v3 adaptive overlay propagation
chain. After Edits 1-5 propagate ``success_criteria`` into the trainer
output, Edit 7 must copy the overlaid dict onto
``PipelineResult.success_criteria``.

Without this hop, ``PipelineResult.success_criteria`` keeps the
scope_definer's pre-overlay stash from line 545 even though the JSON
artifact (hops 1-3) is correct. API/UI consumers of ``PipelineResult``
see stale criteria.

The empty-dict guard (mirrors Edit 6 in run_tier0_test.py) preserves the
scope_definer stash if the trainer (or a stub) returns no overlay.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)


def _build_pipeline_result_at_model_training_stage(
    success_criteria: dict,
) -> PipelineResult:
    """Build a PipelineResult pre-filled with scope_definer stash output.

    Mirrors what ``_run_scope_definition`` writes (line 545) plus the
    minimal model_candidate / qc_report / scope_spec required for
    ``_run_model_training`` to construct its trainer input.
    """
    result = PipelineResult(
        pipeline_run_id="pipe_test",
        status="running",
        current_stage=PipelineStage.MODEL_SELECTION,
        experiment_id="exp_test",
    )
    result.success_criteria = dict(success_criteria)  # ← scope_definer's stash
    result.model_candidate = {
        "algorithm_name": "logistic_regression",
        "algorithm_class": "sklearn.linear_model.LogisticRegression",
        "hyperparameter_search_space": {},
        "default_hyperparameters": {},
    }
    result.qc_report = {"qc_passed": True}
    result.scope_spec = {"problem_type": "binary_classification"}
    return result


@pytest.mark.asyncio
async def test_pipeline_propagates_trainer_overlay_onto_result_success_criteria() -> None:
    """After model_trainer returns the v3-overlaid success_criteria,
    PipelineResult.success_criteria must reflect the overlay (not the
    scope_definer stash at line 545).

    FAILS on current main: ``_run_model_training`` writes
    ``result.training_result = trainer_output`` at line 829 but never
    updates ``result.success_criteria``. The attribute keeps the
    scope_definer stash that ``_run_scope_definition`` set at line 545.
    """
    config = PipelineConfig(skip_mlflow=True, enable_hpo=False)
    pipeline = MLFoundationPipeline(config=config)

    scope_stash = {
        "minimum_auc": 0.75,
        "minimum_precision": 0.70,
        "minimum_f1": 0.70,
        "criteria_source": "adaptive",
        "_adaptive_inputs": {
            "n_samples": 900,
            "prevalence": 0.50,
            "feature_count": 8,
            "regime": "clean",
        },
    }
    result = _build_pipeline_result_at_model_training_stage(scope_stash)

    overlaid_sc = {
        "minimum_auc": 0.95,  # regime override
        "minimum_recall": 0.65,
        "minimum_lift_over_baseline": 0.10,
        "minimum_net_benefit_at_p_t": 0.0,  # v3 active gate
        "minimum_mcc": 0.45,  # v3 active gate
        "maximum_calibration_slope_deviation": 0.15,
        "maximum_calibration_intercept_magnitude": 0.30,
        "criteria_source": "adaptive",
        # minimum_precision / minimum_f1 popped by overlay.
    }

    fake_trainer = MagicMock()
    fake_trainer.run = AsyncMock(
        return_value={
            "success_criteria_met": True,
            "success_criteria_results": {"minimum_auc": True, "minimum_recall": True},
            "success_criteria": overlaid_sc,  # ← Edits 4-5 contract
            "test_metrics": {"auc_roc": 0.96},
            "training_run_id": "run_test",
            "hpo_trials_run": 0,
        }
    )

    with patch.object(pipeline, "_get_agent", return_value=fake_trainer):
        await pipeline._run_model_training(input_data={}, result=result, obs_context=None)

    # Edit 7 contract: PipelineResult.success_criteria reflects the trainer overlay.
    assert result.success_criteria == overlaid_sc, (
        "PipelineResult.success_criteria still has the scope_definer stash; "
        "Edit 7 in pipeline.py did not propagate the trainer's overlay."
    )
    # Sanity: deprecated keys are gone, v3 active gates are present.
    assert "minimum_precision" not in result.success_criteria
    assert "minimum_f1" not in result.success_criteria
    assert result.success_criteria["minimum_net_benefit_at_p_t"] == 0.0
    # Trainer output is also stored.
    assert result.training_result is not None
    assert result.training_result["success_criteria_met"] is True


@pytest.mark.asyncio
async def test_pipeline_keeps_scope_stash_if_trainer_returns_no_success_criteria() -> None:
    """Empty-dict guard semantics: if the trainer (or a stub) returns no
    success_criteria, ``PipelineResult.success_criteria`` keeps the
    scope_definer stash. Mirrors Edit 6's runner-side guard.

    Forward-looking guard: the unguarded form
    ``result.success_criteria = trainer_output.get("success_criteria", {})``
    would clobber the stash with ``{}`` and silently break audit-trail
    rendering under stub-style trainer outputs.
    """
    config = PipelineConfig(skip_mlflow=True, enable_hpo=False)
    pipeline = MLFoundationPipeline(config=config)

    scope_stash = {"minimum_auc": 0.75, "criteria_source": "fixed"}
    result = _build_pipeline_result_at_model_training_stage(scope_stash)

    fake_trainer = MagicMock()
    fake_trainer.run = AsyncMock(
        return_value={
            "success_criteria_met": True,
            "success_criteria_results": {},
            # No "success_criteria" key — stub-style return.
            "test_metrics": {"auc_roc": 0.85},
            "training_run_id": "run_test",
            "hpo_trials_run": 0,
        }
    )

    with patch.object(pipeline, "_get_agent", return_value=fake_trainer):
        await pipeline._run_model_training(input_data={}, result=result, obs_context=None)

    assert result.success_criteria == scope_stash, (
        "Edit 7's empty-dict guard failed: scope_definer stash got clobbered "
        "by a trainer that didn't return success_criteria."
    )


# ---------------------------------------------------------------------------
# PR #462 hotfix F3: pipeline.py merge of pipeline-level sufficiency overrides
# into scope_spec.sufficiency must preserve typed SufficiencyConfig user
# fields AND let PipelineConfig.force_low_power_run win over caller value.
# ---------------------------------------------------------------------------


def _make_data_prep_result() -> PipelineResult:
    """Build a PipelineResult at the DATA_PREPARATION stage with a
    scope_spec that already carries user sufficiency overrides.
    """
    result = PipelineResult(
        pipeline_run_id="pipe_test",
        status="running",
        current_stage=PipelineStage.DATA_PREPARATION,
        experiment_id="exp_test",
    )
    return result


@pytest.mark.asyncio
async def test_pipeline_merge_preserves_typed_sufficiency_config_user_fields() -> None:
    """F3: typed SufficiencyConfig user fields (target_mde, epv_floor, etc.)
    must SURVIVE pipeline-level override merge.

    Pre-fix: the merge replaced a typed SufficiencyConfig with `{}` (the
    `isinstance(dict)` branch silently dropped it), losing every user-
    supplied calibration knob. The fix calls model_dump() first then
    merges per-key.
    """
    from src.utils.sufficiency_schemas import SufficiencyConfig

    config = PipelineConfig(
        skip_mlflow=True,
        enable_hpo=False,
        force_low_power_run=True,
        sufficiency_strictness_preset="strict",
    )
    pipeline = MLFoundationPipeline(config=config)
    result = _make_data_prep_result()
    result.scope_spec = {
        "problem_type": "causal_inference",
        "sufficiency": SufficiencyConfig(
            target_mde=0.05,
            epv_floor=12,
            absolute_floor=500,
        ),
    }

    captured = {}

    fake_dp = MagicMock()

    async def _capture_input(input_data):
        captured.update(input_data)
        return {"qc_report": {}, "baseline_metrics": {}, "gate_passed": True}

    fake_dp.run = AsyncMock(side_effect=_capture_input)

    with (
        patch.object(pipeline, "_get_agent", return_value=fake_dp),
        patch.object(pipeline.config, "enable_feast", False),
    ):
        await pipeline._run_data_preparation(
            input_data={"data_source": "test"}, result=result, obs_context=None
        )

    merged_suff = captured["scope_spec"]["sufficiency"]
    # User fields survive (F3 bug 1: typed shape was being dropped).
    assert merged_suff["target_mde"] == 0.05
    assert merged_suff["epv_floor"] == 12
    assert merged_suff["absolute_floor"] == 500
    # PipelineConfig overrides wrote into the merged dict.
    assert merged_suff["force_low_power_run"] is True
    assert merged_suff["strictness_preset"] == "strict"


@pytest.mark.asyncio
async def test_pipeline_force_low_power_run_wins_over_caller_value() -> None:
    """F3: PipelineConfig.force_low_power_run is a safety-critical flag
    (D5); the orchestrator's value MUST win over a caller-supplied
    scope_spec.sufficiency.force_low_power_run. Pre-fix, the caller-
    supplied value shadowed the pipeline-level default — letting any
    scope_spec author quietly bypass the pharma-safety gate.
    """
    config = PipelineConfig(
        skip_mlflow=True,
        enable_hpo=False,
        force_low_power_run=True,  # pipeline-level: safety knob ON
    )
    pipeline = MLFoundationPipeline(config=config)
    result = _make_data_prep_result()
    # Caller tries to override via scope_spec — must NOT succeed for the
    # safety-critical force_low_power_run flag.
    result.scope_spec = {
        "problem_type": "causal_inference",
        "sufficiency": {
            "force_low_power_run": False,  # caller tries to override OFF
            "target_mde": 0.05,
        },
    }

    captured = {}

    fake_dp = MagicMock()

    async def _capture_input(input_data):
        captured.update(input_data)
        return {"qc_report": {}, "baseline_metrics": {}, "gate_passed": True}

    fake_dp.run = AsyncMock(side_effect=_capture_input)

    with (
        patch.object(pipeline, "_get_agent", return_value=fake_dp),
        patch.object(pipeline.config, "enable_feast", False),
    ):
        await pipeline._run_data_preparation(
            input_data={"data_source": "test"}, result=result, obs_context=None
        )

    merged_suff = captured["scope_spec"]["sufficiency"]
    # Pipeline-level wins for force_low_power_run (safety contract).
    assert merged_suff["force_low_power_run"] is True
    # Calibration knob (target_mde) is NOT overridden by pipeline.
    assert merged_suff["target_mde"] == 0.05


@pytest.mark.asyncio
async def test_pipeline_strictness_preset_caller_wins() -> None:
    """F3: strictness_preset is a calibration knob (not a safety gate);
    a user that took the trouble to set `strict` should not be silently
    downgraded by an orchestrator default of `moderate`.
    """
    config = PipelineConfig(
        skip_mlflow=True,
        enable_hpo=False,
        force_low_power_run=False,
        sufficiency_strictness_preset="moderate",  # pipeline default
    )
    pipeline = MLFoundationPipeline(config=config)
    result = _make_data_prep_result()
    result.scope_spec = {
        "problem_type": "binary_classification",
        "sufficiency": {"strictness_preset": "strict"},  # caller insists
    }

    captured = {}

    fake_dp = MagicMock()

    async def _capture_input(input_data):
        captured.update(input_data)
        return {"qc_report": {}, "baseline_metrics": {}, "gate_passed": True}

    fake_dp.run = AsyncMock(side_effect=_capture_input)

    # force_low_power_run is False and sufficiency_strictness_preset is set, so
    # the merge block enters and we'll see the resolution.
    with (
        patch.object(pipeline, "_get_agent", return_value=fake_dp),
        patch.object(pipeline.config, "enable_feast", False),
    ):
        await pipeline._run_data_preparation(
            input_data={"data_source": "test"}, result=result, obs_context=None
        )

    merged_suff = captured["scope_spec"]["sufficiency"]
    # User-supplied strictness_preset wins for a calibration knob.
    assert merged_suff["strictness_preset"] == "strict"
