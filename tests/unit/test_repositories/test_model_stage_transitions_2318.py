"""#2318: `candidate` is a first-class model stage, and only activation may promote it."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mlops.mlflow_connector import ModelStage as ConnectorStage
from src.repositories.ml_experiment import (
    STAGE_TRANSITIONS,
    MLModelRegistryRepository,
    ModelStage,
    StageTransitionRefused,
)


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
    repo.client.table.return_value.update.return_value.eq.return_value.execute = AsyncMock(
        return_value=MagicMock(data=[{"id": "x"}])
    )
    assert await repo.transition_stage("00000000-0000-0000-0000-000000000001", "archived")


@pytest.mark.asyncio
async def test_nobody_transitions_into_candidate_generically():
    repo = _repo_with_current("staging")
    with pytest.raises(StageTransitionRefused):
        await repo.transition_stage("00000000-0000-0000-0000-000000000001", "candidate")
    repo.client.table.assert_not_called()
