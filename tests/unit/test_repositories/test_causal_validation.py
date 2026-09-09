"""Tests for CausalValidationRepository.

get_by_ids backs the expert-review assessment endpoint (mig 097): a review row
links its refutation evidence via related_validation_ids, and the endpoint
fetches those rows to ground the advisory assessment.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.causal_validation import CausalValidationRepository


class TestGetByIds:
    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def repo(self, mock_client):
        repo = CausalValidationRepository()
        repo.client = mock_client
        return repo

    @pytest.mark.asyncio
    async def test_fetches_rows_for_ids(self, repo, mock_client):
        rows = [
            {"validation_id": "v1", "test_type": "random_common_cause", "status": "passed"},
            {"validation_id": "v2", "test_type": "data_subset", "status": "failed"},
        ]
        mock_execute = AsyncMock(return_value=MagicMock(data=rows))
        (mock_client.table.return_value.select.return_value.in_.return_value.execute) = mock_execute

        result = await repo.get_by_ids(["v1", "v2"])

        assert result == rows
        mock_client.table.assert_called_with("causal_validations")
        in_call = mock_client.table.return_value.select.return_value.in_
        assert in_call.call_args[0] == ("validation_id", ["v1", "v2"])

    @pytest.mark.asyncio
    async def test_empty_ids_short_circuits(self, repo, mock_client):
        result = await repo.get_by_ids([])
        assert result == []
        mock_client.table.assert_not_called()

    @pytest.mark.asyncio
    async def test_without_client_returns_empty(self):
        repo = CausalValidationRepository()
        repo.client = None
        assert await repo.get_by_ids(["v1"]) == []

    @pytest.mark.asyncio
    async def test_error_returns_empty_not_raise(self, repo, mock_client):
        (mock_client.table.return_value.select.return_value.in_.return_value.execute) = AsyncMock(
            side_effect=Exception("boom")
        )
        assert await repo.get_by_ids(["v1"]) == []


from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)


class TestEvidenceRowsAreJsonObjects:
    """Lane 1 (owner decision 2026-09-09): evidence is written as JSON OBJECTS,
    not JSON strings inside the jsonb column, so it can be queried and tested in
    one shape; non-finite floats become null (the transport encodes with
    allow_nan=False, so a NaN would otherwise fail the whole write)."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def repo(self, mock_client):
        repo = CausalValidationRepository()
        repo.client = mock_client
        return repo

    @staticmethod
    def _suite() -> RefutationSuite:
        test = RefutationResult(
            test_name=RefutationTestType.BOOTSTRAP,
            status=RefutationStatus.PASSED,
            original_effect=0.15,
            refuted_effect=0.151,
            p_value=0.4,
            details={
                "message": "ok",
                "bootstrap_effects": [0.14, 0.16],
                "ci_ratio": float("nan"),
                "config": {"n": 2},
            },
            execution_time_ms=12.5,
        )
        return RefutationSuite(
            passed=True,
            confidence_score=0.9,
            tests=[test],
            gate_decision=GateDecision.PROCEED,
            estimate_id="est-1",
            brand="Kisqali",
        )

    @pytest.mark.asyncio
    async def test_save_suite_writes_objects_not_strings(self, repo, mock_client):
        insert = mock_client.table.return_value.insert
        insert.return_value.execute = AsyncMock(
            return_value=MagicMock(data=[{"validation_id": "v1"}])
        )
        ids = await repo.save_suite(self._suite(), estimate_id="e1")
        assert ids == ["v1"]
        row = insert.call_args[0][0][0]
        assert isinstance(row["details_json"], dict)
        assert isinstance(row["test_config"], dict)
        assert row["details_json"]["bootstrap_effects"] == [0.14, 0.16]
        assert row["details_json"]["ci_ratio"] is None  # NaN -> null, never a transport failure
        assert row["test_config"] == {"execution_time_ms": 12.5}

    @pytest.mark.asyncio
    async def test_save_single_test_writes_objects_not_strings(self, repo, mock_client):
        insert = mock_client.table.return_value.insert
        insert.return_value.execute = AsyncMock(
            return_value=MagicMock(data=[{"validation_id": "v2"}])
        )
        suite = self._suite()
        vid = await repo.save_single_test(
            suite.tests[0],
            estimate_id="e1",
            gate_decision=GateDecision.PROCEED,
            confidence_score=0.9,
        )
        assert vid == "v2"
        row = insert.call_args[0][0]
        assert isinstance(row["details_json"], dict)
        assert row["details_json"]["ci_ratio"] is None
        assert row["test_config"] == {"n": 2}
