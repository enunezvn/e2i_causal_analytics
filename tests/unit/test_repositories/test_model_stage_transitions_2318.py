"""#2318: `candidate` is a first-class model stage, and only activation may promote it."""

import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mlops.mlflow_connector import MLflowConnector
from src.mlops.mlflow_connector import ModelStage as ConnectorStage
from src.repositories.ml_experiment import (
    STAGE_TRANSITIONS,
    MLModelRegistryRepository,
    ModelStage,
    StageTransitionRefused,
)
from tests.unit._fakes.async_supabase import FakeAsyncSupabase

MODEL_ID = "00000000-0000-0000-0000-000000000001"


def test_candidate_is_a_model_stage_in_both_enums():
    assert ModelStage.CANDIDATE.value == "candidate"
    assert ConnectorStage.CANDIDATE.value == "candidate"
    # Both enums mirror model_stage_enum (migration 159) exactly.
    assert (
        {s.value for s in ModelStage}
        == {s.value for s in ConnectorStage}
        == {
            "development",
            "staging",
            "shadow",
            "production",
            "archived",
            "deprecated",
            "candidate",
        }
    )


def test_normalize_stage_accepts_candidate():
    assert MLModelRegistryRepository.normalize_stage("candidate") == "candidate"
    assert MLModelRegistryRepository.normalize_stage("Candidate") == "candidate"


def _repo_with_current(stage: str):
    repo = MLModelRegistryRepository.__new__(MLModelRegistryRepository)
    repo.client = MagicMock()
    repo.table_name = "ml_model_registry"
    current = MagicMock(stage=stage, model_name="m_goldstd_lr_v1", training_provenance="real")
    repo.get_by_id = AsyncMock(return_value=current)
    return repo


def test_candidate_may_only_be_archived_generically():
    assert STAGE_TRANSITIONS["candidate"] == frozenset({"archived"})


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["staging", "production", "shadow", "development"])
async def test_transition_stage_refuses_promoting_a_candidate(target):
    repo = _repo_with_current("candidate")
    with pytest.raises(StageTransitionRefused, match="promote-candidate"):
        await repo.transition_stage("00000000-0000-0000-0000-000000000001", target)
    repo.client.table.assert_not_called()  # refused before any write


@pytest.mark.asyncio
async def test_transition_stage_still_archives_a_candidate():
    repo = _repo_with_current("candidate")
    query = MagicMock()
    query.eq.return_value = query
    query.execute = AsyncMock(return_value=MagicMock(data=[{"id": "x"}]))
    repo.client.table.return_value.update.return_value = query
    assert await repo.transition_stage(MODEL_ID, "archived")
    # The rule is repeated on the write: it lands only while the row is still a candidate.
    query.eq.assert_any_call("stage", "candidate")


@pytest.mark.asyncio
async def test_nobody_transitions_into_candidate_generically():
    repo = _repo_with_current("staging")
    with pytest.raises(StageTransitionRefused):
        await repo.transition_stage("00000000-0000-0000-0000-000000000001", "candidate")
    repo.client.table.assert_not_called()


@pytest.mark.parametrize(
    "from_stage, new_stage, refused",
    [
        ("candidate", "archived", False),
        ("candidate", "staging", True),
        ("candidate", "candidate", True),
        ("staging", "candidate", True),
        ("", "candidate", True),
        ("staging", "production", False),
        ("", "staging", False),
    ],
)
def test_stage_transition_refusal_is_the_one_statement_of_the_rule(from_stage, new_stage, refused):
    reason = MLModelRegistryRepository.stage_transition_refusal(MODEL_ID, from_stage, new_stage)
    assert (reason is not None) is refused


# ---------------------------------------------------------------------------
# codex r1: the rule is repeated on the write (stale read), over the in-memory fake
# ---------------------------------------------------------------------------


def _fake_db(stage):
    return FakeAsyncSupabase(
        {
            "ml_model_registry": [
                {
                    "id": MODEL_ID,
                    "model_name": "m_goldstd_lr_v1",
                    "model_version": "1.0_retrained_x",
                    "stage": stage,
                    "training_provenance": "real",
                    "is_synthetic": False,
                }
            ]
        }
    )


@pytest.mark.asyncio
async def test_a_row_that_became_a_candidate_after_the_read_is_not_promoted():
    """The check reads a snapshot; the write must not land on a row that is a candidate now."""
    db = _fake_db("candidate")
    repo = MLModelRegistryRepository(supabase_client=db)
    real_get = repo.get_by_id
    stale = dataclasses.replace(await real_get(MODEL_ID), stage="staging")
    reads = iter([stale])

    async def get_by_id(model_id, *a, **k):
        return next(reads, None) or await real_get(model_id, *a, **k)

    repo.get_by_id = get_by_id
    with pytest.raises(StageTransitionRefused, match="moved from 'staging' to 'candidate'"):
        await repo.transition_stage(MODEL_ID, "production")
    assert db.rows("ml_model_registry")[0]["stage"] == "candidate"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["staging", None])
async def test_the_write_lands_while_the_stage_is_unchanged(stage):
    db = _fake_db(stage)
    repo = MLModelRegistryRepository(supabase_client=db)
    assert await repo.transition_stage(MODEL_ID, "Shadow") is True
    assert db.rows("ml_model_registry")[0]["stage"] == "shadow"


@pytest.mark.asyncio
async def test_a_write_to_a_row_that_is_gone_is_not_reported_as_done():
    """codex r2: a zero-row write used to return True for any non-production target."""
    db = _fake_db("staging")
    repo = MLModelRegistryRepository(supabase_client=db)
    snapshot = await repo.get_by_id(MODEL_ID)
    db.store["ml_model_registry"].clear()  # deleted between the read and the write
    reads = iter([snapshot])
    real_get = MLModelRegistryRepository.get_by_id

    async def get_by_id(model_id, *a, **k):
        return next(reads, None) or await real_get(repo, model_id, *a, **k)

    repo.get_by_id = get_by_id
    with pytest.raises(StageTransitionRefused, match="the row is gone"):
        await repo.transition_stage(MODEL_ID, "Shadow")


@pytest.mark.asyncio
async def test_a_zero_row_write_at_an_unchanged_stage_is_not_reported_as_done():
    """E.g. a row the write cannot see (grants/RLS): nothing was written, so no True."""
    repo = _repo_with_current("staging")
    query = MagicMock()
    query.eq.return_value = query
    query.execute = AsyncMock(return_value=MagicMock(data=[]))
    repo.client.table.return_value.update.return_value = query
    with pytest.raises(StageTransitionRefused, match="its stage is unchanged"):
        await repo.transition_stage(MODEL_ID, "shadow")


# ---------------------------------------------------------------------------
# codex r1: promote_stage refuses a candidate BEFORE MLflow moves
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["Staging", "Production", "Shadow"])
async def test_promote_stage_refuses_a_candidate_before_mlflow_moves(target):
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager

    db = _fake_db("candidate")
    mlflow_move = AsyncMock(return_value=True)
    with (
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
        patch.object(registry_manager, "_transition_stage_mlflow", mlflow_move),
    ):
        result = await registry_manager.promote_stage(
            {
                "registered_model_name": "m_goldstd_lr_v1",
                "model_version": 8,
                "promotion_target_stage": target,
                "current_stage": "None",
                "model_registry_id": MODEL_ID,
            }
        )
    assert result["promotion_successful"] is False
    assert result["error_type"] == "promotion_refused"
    assert "promote-candidate" in result["promotion_refused_reason"]
    mlflow_move.assert_not_called()
    assert db.rows("ml_model_registry")[0]["stage"] == "candidate"


@pytest.mark.asyncio
async def test_promote_stage_of_a_non_candidate_still_reaches_mlflow():
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager

    db = _fake_db("staging")
    mlflow_move = AsyncMock(return_value=True)
    with (
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
        patch.object(registry_manager, "_transition_stage_mlflow", mlflow_move),
    ):
        result = await registry_manager.promote_stage(
            {
                "registered_model_name": "m_goldstd_lr_v1",
                "model_version": 8,
                "promotion_target_stage": "Shadow",
                "current_stage": "Staging",
                "model_registry_id": MODEL_ID,
            }
        )
    assert result.get("error_type") != "promotion_refused"
    mlflow_move.assert_awaited_once()


# ---------------------------------------------------------------------------
# codex r1: MLflow stages cannot express 'candidate'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connector_refuses_candidate_as_an_mlflow_stage():
    connector = MLflowConnector.__new__(MLflowConnector)
    connector._enabled = True
    connector._client = MagicMock()
    connector.circuit_breaker = MagicMock()
    with pytest.raises(ValueError, match="e2i.role=candidate"):
        await connector.transition_model_stage("m", "8", ConnectorStage.CANDIDATE)
    with pytest.raises(ValueError, match="e2i.role=candidate"):
        await connector.get_latest_model_version("m", stage=ConnectorStage.CANDIDATE)
    connector._client.transition_model_version_stage.assert_not_called()
    connector._client.search_model_versions.assert_not_called()
