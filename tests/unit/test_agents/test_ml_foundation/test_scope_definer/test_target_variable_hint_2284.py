"""#2284: a caller can pin a physical target column, bypassing the NL rewrite.

``_infer_target_variable`` invents a canonical target name from a natural-language
objective — ``"likely to adopt"`` -> ``will_adopt``. That was the point of the node
and it is still wanted (``test_problem_classifier.py`` pins it).

It aged into a defect when a caller started passing a *physical column name*:
since #2207/#2241 the retrain path reads the registry's ``cohort_target_outcome``
("the physical label", ``drift_monitoring_tasks.py``) and hands it over as
``target_outcome``, so a contract target of ``adopted`` came back out as
``prediction_target='will_adopt'`` and the data_preparer's target guard
(``data_loader._require_target_in_columns``) refused the load. No table anywhere
defines a ``will_adopt`` column.

``classify_problem`` already had an escape hatch for the problem *type*
(``problem_type_hint``, "If hint provided, trust it") and none for the target —
that asymmetry is the bug. ``target_variable_hint`` closes it, and because it
bypasses ``_infer_target_variable`` wholesale it covers all six rewrite families
(prescribe / churn / convert / adopt / abandon / trx|nrx / time-to), not just
``adopt``.

The legacy ``target_variable`` input (declared on the state since the initial
commit, documented in ``agent.run`` as "Target variable name if known", and
paired with a descriptive ``target_outcome`` in ``scripts/sample_ml_pipeline.py``)
was never read by any node. It is accepted here as the hint's alias rather than
left as a decoy beside a near-identical live field.

This does NOT make the 3 HCP-adoption champions retrainable — migration 151 names
three blockers and this is one of them. See the PR body.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.agents.ml_foundation.scope_definer import ScopeDefinerAgent
from src.agents.ml_foundation.scope_definer.nodes.problem_classifier import (
    classify_problem,
)
from src.agents.ml_foundation.scope_definer.state import ScopeDefinerState
from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)

# The contract target on the 3 ``hcp_adoption_<brand>_goldstd_lr_v1`` registry rows.
PHYSICAL_TARGET = "adopted"


# ---------------------------------------------------------------------------
# classify_problem: the hint wins, the NL path is untouched
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_pinned_physical_target_survives_the_adopt_rewrite():
    """The exact #2284 case: contract target ``adopted`` must not become ``will_adopt``."""
    result = await classify_problem(
        {
            "business_objective": "Triggered model refresh (drift/manual)",
            "target_outcome": PHYSICAL_TARGET,
            "target_variable_hint": PHYSICAL_TARGET,
        }
    )

    assert result["inferred_target_variable"] == PHYSICAL_TARGET


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_natural_language_path_still_invents_will_adopt():
    """No hint -> the rewrite stands. ``test_problem_classifier.py:90`` pins this too."""
    result = await classify_problem(
        {
            "business_objective": "Identify HCPs likely to adopt the brand",
            "target_outcome": "Predict whether an HCP is likely to adopt in 90 days",
        }
    )

    assert result["inferred_target_variable"] == "will_adopt"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_outcome", "rewritten_to", "physical_column"),
    [
        ("Predict whether the HCP will prescribe", "will_prescribe", "rx_written_90d"),
        ("Predict patient churn", "will_churn", "discontinued_30d"),
        ("Predict conversion", "will_convert", "converted_90d"),
        ("Predict adoption", "will_adopt", "adopted"),
        ("Predict whether they abandon therapy", "will_abandon", "abandoned_flag"),
        ("Forecast TRx per HCP", "prescription_count", "trx_90d"),
        ("Measure time to first fill", "time_to_event_days", "days_to_first_fill"),
    ],
)
async def test_the_hint_covers_every_rewrite_family(
    target_outcome: str, rewritten_to: str, physical_column: str
):
    """The hint bypasses ``_infer_target_variable`` wholesale, not just the adopt branch."""
    without_hint = await classify_problem({"target_outcome": target_outcome})
    assert without_hint["inferred_target_variable"] == rewritten_to

    with_hint = await classify_problem(
        {"target_outcome": target_outcome, "target_variable_hint": physical_column}
    )
    assert with_hint["inferred_target_variable"] == physical_column


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_tier0_runners_physical_column_is_no_longer_rewritten():
    """The symptom in a checked-in run of ``scripts/run_tier0_test.py``.

    ``docs/reports/synthetic_csu_e2e_validation_20260610/tier0_hcp_adoption/
    rwd_pipeline_run_20260610_183649.md`` records target ``adopted_target_brand``
    coming back out of step 1 as ``prediction_target: will_adopt`` — persisted to
    ml_experiments and the knowledge graph before step 2 repaired the local spec.
    ``CONFIG.target_outcome`` is a physical DataFrame column on that runner, so it
    now pins the hint.
    """
    column = "adopted_target_brand"

    without_hint = await classify_problem({"target_outcome": column})
    assert without_hint["inferred_target_variable"] == "will_adopt"

    with_hint = await classify_problem({"target_outcome": column, "target_variable_hint": column})
    assert with_hint["inferred_target_variable"] == column


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_hint_is_taken_verbatim_not_sanitised():
    """A physical column name is case- and punctuation-sensitive; only whitespace is trimmed."""
    result = await classify_problem(
        {"target_outcome": "adoption", "target_variable_hint": "  Adopted_180D  "}
    )

    assert result["inferred_target_variable"] == "Adopted_180D"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", "   ", None])
async def test_a_blank_hint_falls_back_to_inference(blank: Any):
    """An absent/empty hint must not blank the target — fall back to the NL path."""
    result = await classify_problem(
        {"target_outcome": "likely to adopt", "target_variable_hint": blank}
    )

    assert result["inferred_target_variable"] == "will_adopt"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_hint_does_not_touch_problem_type_inference():
    """Pinning the target column says nothing about the problem type."""
    result = await classify_problem(
        {
            "business_objective": "Forecast monthly TRx volume per HCP",
            "target_outcome": "prescription volume",
            "target_variable_hint": "trx_90d",
        }
    )

    assert result["inferred_problem_type"] == "regression"
    assert result["inferred_target_variable"] == "trx_90d"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_legacy_target_variable_input_is_honoured_as_the_hints_alias():
    """``target_variable`` was declared and documented but read by no node."""
    result = await classify_problem(
        {"target_outcome": "hcp_conversion", "target_variable": "converted_90d"}
    )

    assert result["inferred_target_variable"] == "converted_90d"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_explicit_hint_wins_over_the_legacy_alias():
    result = await classify_problem(
        {
            "target_outcome": "adoption",
            "target_variable": "legacy_column",
            "target_variable_hint": PHYSICAL_TARGET,
        }
    )

    assert result["inferred_target_variable"] == PHYSICAL_TARGET


# ---------------------------------------------------------------------------
# The field must be declared: BaseAgentSchema keeps extras out of attribute access
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_state_declares_the_hint_so_it_is_not_dropped():
    state = ScopeDefinerState(
        audit_workflow_id="00000000-0000-0000-0000-000000000000",
        target_variable_hint=PHYSICAL_TARGET,
    )

    assert "target_variable_hint" in ScopeDefinerState.model_fields
    assert state.get("target_variable_hint") == PHYSICAL_TARGET


# ---------------------------------------------------------------------------
# End-to-end through the agent: the hint reaches scope_spec.prediction_target,
# the key every data_preparer node (and the loader's target guard) reads.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_agent_threads_the_hint_onto_scope_spec_prediction_target():
    output = await ScopeDefinerAgent().run(
        {
            "problem_description": "Retrain model on cohort hcp_brand_adoption",
            "business_objective": "Triggered model refresh (drift/manual)",
            "target_outcome": PHYSICAL_TARGET,
            "target_variable_hint": PHYSICAL_TARGET,
            "brand": "Kisqali",
        }
    )

    assert output.get("error") is None, output.get("error")
    assert output["scope_spec"]["prediction_target"] == PHYSICAL_TARGET


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_agent_without_a_hint_still_rewrites():
    output = await ScopeDefinerAgent().run(
        {
            "problem_description": "Find HCPs likely to adopt",
            "business_objective": "Grow adoption",
            "target_outcome": "likely to adopt",
            "brand": "Kisqali",
        }
    )

    assert output["scope_spec"]["prediction_target"] == "will_adopt"


# ---------------------------------------------------------------------------
# Wiring: retrain contract -> MLFoundationPipeline -> scope_definer
# ---------------------------------------------------------------------------


class _StopAfterScopeInput(Exception):
    pass


def _result() -> PipelineResult:
    return PipelineResult(
        pipeline_run_id="run-2284",
        status="running",
        current_stage=PipelineStage.SCOPE_DEFINITION,
    )


async def _scope_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = MLFoundationPipeline(config=PipelineConfig(enable_feast=False))
    captured: Dict[str, Any] = {}

    async def _capture(scope_input):
        captured.update(scope_input)
        raise _StopAfterScopeInput()

    fake_scope = MagicMock()
    fake_scope.run = _capture
    base = {
        "problem_description": "p",
        "business_objective": "b",
        "target_outcome": PHYSICAL_TARGET,
    }
    with patch.object(pipeline, "_get_agent", return_value=fake_scope):
        with pytest.raises(_StopAfterScopeInput):
            await pipeline._run_scope_definition({**base, **input_data}, _result(), None)
    return captured


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_pipeline_forwards_the_hint_to_scope_definer():
    captured = await _scope_input(
        {"data_source": "business_metrics", "target_variable_hint": PHYSICAL_TARGET}
    )

    assert captured["target_variable_hint"] == PHYSICAL_TARGET


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_pipeline_invents_no_hint_when_the_caller_gives_none():
    captured = await _scope_input({"data_source": "business_metrics"})

    assert captured.get("target_variable_hint") is None


@pytest.mark.unit
def test_the_retrain_contract_pins_its_physical_label_as_the_hint():
    """``cohort_target_outcome`` is the physical label — the retrain must pin it."""
    from src.tasks.drift_monitoring_tasks import _cohort_input_from_training_config

    input_data = _cohort_input_from_training_config(
        {
            "data_source": {
                "type": "table",
                "table": "hcp_brand_adoption",
                "filters": {"brand": "Kisqali"},
            },
            "target_outcome": PHYSICAL_TARGET,
        }
    )

    assert input_data["target_outcome"] == PHYSICAL_TARGET
    assert input_data["target_variable_hint"] == PHYSICAL_TARGET


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_retrain_contract_reaches_scope_spec_unrewritten():
    """The whole chain the bug ran through, end to end."""
    from src.tasks.drift_monitoring_tasks import _cohort_input_from_training_config

    input_data = _cohort_input_from_training_config(
        {"data_source": "hcp_brand_adoption", "target_outcome": PHYSICAL_TARGET}
    )
    output = await ScopeDefinerAgent().run(
        {
            "problem_description": input_data["problem_description"],
            "business_objective": input_data["business_objective"],
            "target_outcome": input_data["target_outcome"],
            "target_variable_hint": input_data["target_variable_hint"],
        }
    )

    assert output["scope_spec"]["prediction_target"] == PHYSICAL_TARGET
