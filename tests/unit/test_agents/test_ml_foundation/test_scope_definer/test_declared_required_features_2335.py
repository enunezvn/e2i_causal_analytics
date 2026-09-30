"""#2335: ``scope_spec.required_features`` is always DECLARED, never invented.

Before: ``scope_builder._define_required_features`` returned the caller's
``candidate_features`` and otherwise a hard-coded scaffold list (hcp_specialty,
patient_count, prescription_history, brand_affinity_score + 2 per problem type). Four of
those columns exist in no table, so every caller without candidates got a phantom
"Missing required features" blocker, and ``detect_leakage`` / the baseline stats / the
Feast registrar were scoped to columns the frame does not have.

After (owner-approved design):
- the placeholder list is gone;
- the requirement resolves from declared sources in ONE place
  (``src.services.cohort_contract.resolve_required_features``): explicit
  ``candidate_features`` first, else a table cohort contract's ``columns`` minus the
  target;
- the provenance is recorded as ``scope_spec["required_features_source"]`` and the
  data_preparer's readiness blocker names it;
- no declared source fails closed with an actionable error, never a list and never an
  empty list.

The manifest branch (``feature_manifest_source`` admissible features) is deliberately NOT
a source: measured against the real host frames it is not a subset of their columns
(PR body has the numbers), so as a requirement it would produce false "missing" blockers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.graph import finalize_output
from src.agents.ml_foundation.scope_definer.agent import ScopeDefinerAgent
from src.agents.ml_foundation.scope_definer.nodes.scope_builder import build_scope_spec
from src.agents.tier_0.pipeline import (
    MLFoundationPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)
from src.services.cohort_contract import (
    UNDECLARED_REQUIRED_FEATURES,
    UndeclaredRequiredFeaturesError,
    resolve_required_features,
)

pytestmark = pytest.mark.unit

# Every name the deleted scaffold list could emit.
PLACEHOLDER = {
    "hcp_specialty",
    "patient_count",
    "prescription_history",
    "brand_affinity_score",
    "engagement_score",
    "channel_response_rate",
    "historical_prescription_volume",
    "market_share",
}

TABLE_CONTRACT = {
    "type": "table",
    "table": "hcp_adoption_goldstd_v",
    "filters": {"brand": "Kisqali", "is_synthetic": True},
    "columns": ["peer_influence_score", "years_experience", "specialty", "adopted"],
}


@pytest.fixture(autouse=True)
def _no_external_writes():
    """``ScopeDefinerAgent.run`` persists unconditionally (#2300) — stub its writers."""
    hooks = MagicMock()
    hooks.store_experiment_pattern = AsyncMock(return_value=True)
    hooks.store_scope_definition = AsyncMock(return_value=True)
    base = "src.agents.ml_foundation.scope_definer.agent"
    with (
        patch(f"{base}._get_experiment_repository", new=AsyncMock(return_value=None)),
        patch(f"{base}.ScopeDefinerMemoryHooks", return_value=hooks),
        patch(f"{base}._get_procedural_memory", return_value=None),
        patch(f"{base}._get_opik_connector", return_value=None),
    ):
        yield


def _builder_state(**extra: Any) -> Dict[str, Any]:
    return {
        "business_objective": "Grow new prescribers",
        "target_outcome": "adopted",
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "adopted",
        "brand": "Kisqali",
        "region": "US",
        **extra,
    }


def _pipeline_input(**extra: Any) -> Dict[str, Any]:
    return {
        "problem_description": "Predict which HCPs adopt Kisqali",
        "business_objective": "Grow new prescribers",
        "target_outcome": "adopted",
        "target_variable_hint": "adopted",
        "brand": "Kisqali",
        **extra,
    }


def _result() -> PipelineResult:
    return PipelineResult(
        pipeline_run_id="run-2335-declared",
        status="running",
        current_stage=PipelineStage.SCOPE_DEFINITION,
    )


# --------------------------------------------------------------------------- resolver


def test_explicit_candidates_win_over_the_contract_columns() -> None:
    assert resolve_required_features(["specialty"], TABLE_CONTRACT, targets=["adopted"]) == (
        ["specialty"],
        "explicit",
    )


def test_a_table_contract_declares_its_columns_minus_the_target() -> None:
    assert resolve_required_features(None, TABLE_CONTRACT, targets=["adopted"]) == (
        ["peer_influence_score", "years_experience", "specialty"],
        "contract",
    )


def test_a_padded_target_hint_is_still_excluded_from_the_contract_columns() -> None:
    features, _ = resolve_required_features(None, TABLE_CONTRACT, targets=["  adopted  ", None])
    assert "adopted" not in features


def test_a_json_encoded_table_contract_is_read_like_the_dict() -> None:
    import json

    encoded = json.dumps(TABLE_CONTRACT, sort_keys=True)
    features, source = resolve_required_features(None, encoded, targets=["adopted"])
    assert source == "contract"
    assert set(features) == {"peer_influence_score", "years_experience", "specialty"}


@pytest.mark.parametrize(
    "candidates, data_source",
    [
        (None, None),
        ([], None),
        (None, "patient_journeys"),  # bare table name: no declared column list
        (None, {"type": "file_dir", "path": "data/rwd/optum/initiation"}),
        (None, {"type": "table", "table": "patient_journeys", "filters": {"brand": "K"}}),
        (None, {"type": "table", "table": "patient_journeys", "columns": ["adopted"]}),
        ([], {"type": "table", "table": "patient_journeys", "columns": []}),
    ],
)
def test_no_declared_source_fails_closed(candidates: Any, data_source: Any) -> None:
    with pytest.raises(UndeclaredRequiredFeaturesError, match=UNDECLARED_REQUIRED_FEATURES):
        resolve_required_features(candidates, data_source, targets=["adopted"])


def test_a_bare_string_is_not_a_candidate_list() -> None:
    # "a,b" must not be iterated character by character into a requirement.
    with pytest.raises(UndeclaredRequiredFeaturesError):
        resolve_required_features("peer_influence_score", None, targets=["adopted"])


# --------------------------------------------------------------------------- scope_builder


@pytest.mark.asyncio
@pytest.mark.parametrize("problem_type", ["binary_classification", "regression", "multiclass"])
async def test_the_builder_never_invents_a_requirement(problem_type: str) -> None:
    state = _builder_state(inferred_problem_type=problem_type)
    with pytest.raises(UndeclaredRequiredFeaturesError, match=UNDECLARED_REQUIRED_FEATURES):
        await build_scope_spec(state)


@pytest.mark.asyncio
async def test_the_builder_refuses_an_empty_candidate_list() -> None:
    with pytest.raises(UndeclaredRequiredFeaturesError):
        await build_scope_spec(_builder_state(candidate_features=[]))


@pytest.mark.asyncio
async def test_the_builder_records_explicit_provenance() -> None:
    out = await build_scope_spec(_builder_state(candidate_features=["specialty"]))
    spec = out["scope_spec"]
    assert spec["required_features"] == ["specialty"]
    assert spec["required_features_source"] == "explicit"


@pytest.mark.asyncio
async def test_the_builder_keeps_a_resolved_contract_provenance() -> None:
    out = await build_scope_spec(
        _builder_state(candidate_features=["specialty"], required_features_source="contract")
    )
    assert out["scope_spec"]["required_features_source"] == "contract"


@pytest.mark.asyncio
async def test_an_unknown_provenance_label_is_not_trusted() -> None:
    out = await build_scope_spec(
        _builder_state(candidate_features=["specialty"], required_features_source="made_up")
    )
    assert out["scope_spec"]["required_features_source"] == "explicit"


# --------------------------------------------------------------------------- agent


@pytest.mark.asyncio
async def test_the_agent_returns_an_actionable_error_without_a_declared_source() -> None:
    out = await ScopeDefinerAgent().run(
        {
            "problem_description": "Predict which HCPs adopt Kisqali",
            "business_objective": "Grow new prescribers",
            "target_outcome": "adopted",
            "target_variable_hint": "adopted",
        }
    )
    assert out.get("error_type") == "undeclared_required_features", out
    assert UNDECLARED_REQUIRED_FEATURES in out["error"]
    assert "scope_spec" not in out


@pytest.mark.asyncio
async def test_the_agent_threads_the_provenance_onto_scope_spec() -> None:
    out = await ScopeDefinerAgent().run(
        {
            "problem_description": "Predict which HCPs adopt Kisqali",
            "business_objective": "Grow new prescribers",
            "target_outcome": "adopted",
            "target_variable_hint": "adopted",
            "candidate_features": ["specialty", "years_experience"],
            "required_features_source": "contract",
        }
    )
    assert out.get("error") is None, out
    assert out["scope_spec"]["required_features"] == ["specialty", "years_experience"]
    assert out["scope_spec"]["required_features_source"] == "contract"


# --------------------------------------------------------------------------- pipeline stage


async def _pipeline_scope(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = MLFoundationPipeline(config=PipelineConfig(enable_feast=False))
    result = _result()
    await pipeline._run_scope_definition(input_data, result, None)
    assert result.scope_spec, result.errors
    return dict(result.scope_spec)


@pytest.mark.asyncio
async def test_the_pipeline_resolves_a_table_contract_as_the_requirement() -> None:
    spec = await _pipeline_scope(_pipeline_input(data_source=TABLE_CONTRACT))
    assert spec["required_features"] == ["peer_influence_score", "years_experience", "specialty"]
    assert spec["required_features_source"] == "contract"
    assert not PLACEHOLDER & set(spec["required_features"])


@pytest.mark.asyncio
async def test_the_pipeline_keeps_explicit_candidates_over_the_contract() -> None:
    spec = await _pipeline_scope(
        _pipeline_input(data_source=TABLE_CONTRACT, candidate_features=["specialty"])
    )
    assert spec["required_features"] == ["specialty"]
    assert spec["required_features_source"] == "explicit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data_source",
    [
        "patient_journeys",
        {"type": "file_dir", "path": "data/rwd/mart/hcp_adoption"},
        {"type": "table", "table": "hcp_adoption_goldstd_v", "columns": ["adopted"]},
    ],
)
async def test_the_pipeline_fails_closed_before_scope_definer_runs(data_source: Any) -> None:
    pipeline = MLFoundationPipeline(config=PipelineConfig(enable_feast=False))
    with patch.object(pipeline, "_get_agent") as get_agent:
        with pytest.raises(UndeclaredRequiredFeaturesError, match=UNDECLARED_REQUIRED_FEATURES):
            await pipeline._run_scope_definition(
                _pipeline_input(data_source=data_source), _result(), None
            )
    get_agent.assert_not_called()


# --------------------------------------------------------------------------- data_preparer


def _finalize_state(scope_spec: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    frame = pd.DataFrame({c: [0, 1] for c in columns})
    return {
        "experiment_id": "exp-2335",
        "scope_spec": scope_spec,
        "qc_status": "passed",
        "overall_score": 1.0,
        "blocking_issues": [],
        "train_df": frame,
        "validation_df": frame,
        "test_df": frame,
        "holdout_df": frame,
    }


@pytest.mark.asyncio
async def test_the_readiness_blocker_names_the_declared_source() -> None:
    spec = {
        "required_features": ["peer_influence_score", "specialty"],
        "required_features_source": "contract",
    }
    out = await finalize_output(_finalize_state(spec, ["specialty", "adopted"]))

    assert out["is_ready"] is False
    assert out["missing_required_features"] == ["peer_influence_score"]
    assert (
        "Missing required features (declared by contract): peer_influence_score"
        in (out["blockers"])
    )


@pytest.mark.asyncio
async def test_a_spec_without_provenance_says_so_in_the_blocker() -> None:
    spec = {"required_features": ["peer_influence_score"]}
    out = await finalize_output(_finalize_state(spec, ["adopted"]))
    assert (
        "Missing required features (declared by unrecorded source): peer_influence_score"
        in (out["blockers"])
    )


# --------------------------------------------------------------------------- run_tier0_test


def _import_runner():
    root = Path(__file__).resolve().parents[5]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import scripts.run_tier0_test as runner

    return runner


@pytest.mark.asyncio
async def test_the_tier0_runner_declares_its_frame_columns_up_front() -> None:
    """Step 1 must pass the real columns as candidate_features, or scope fails closed."""
    runner = _import_runner()
    seen: Dict[str, Any] = {}

    class _Recorder:
        async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            seen.update(input_data)
            return {"scope_spec": {"problem_type": "binary_classification"}}

    with patch("src.agents.ml_foundation.scope_definer.ScopeDefinerAgent", _Recorder):
        await runner.step_1_scope_definer("exp-2335", candidate_features=["a", "b"])

    assert seen["candidate_features"] == ["a", "b"]
