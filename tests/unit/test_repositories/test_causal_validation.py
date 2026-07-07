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
