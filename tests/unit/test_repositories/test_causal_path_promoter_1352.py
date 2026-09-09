"""CausalPathRepository promoter surface (#1352 item 3).

The RefutationNode — SOLE promoter of ``causal_paths.validation_status``
(migration 119 semantics pin) — reads/writes paths through three helpers:

* ``get_path_row``: id lookup that INCLUDES synthetic rows (the promoter must
  SEE a synthetic row to refuse it, not treat it as absent);
* ``find_real_paths_for_pair``: real-mode default-exclude auto-linkage read;
* ``set_validation_status``: a CONDITIONAL server-side transition — the UPDATE
  matches only rows whose current status is in ``allowed_current``, so a
  concurrent writer or operator adjudication is never silently overwritten.

Mock style mirrors test_causal_path.py (self-chaining query mocks tolerate the
provenance predicate appended by real-mode reads).
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.causal_path import CausalPathRepository


def _chain_query(data):
    query = MagicMock()
    query.eq.return_value = query
    query.in_.return_value = query
    query.limit.return_value = query
    query.offset.return_value = query
    result = MagicMock()
    result.data = data
    query.execute = AsyncMock(return_value=result)
    return query


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def repo(mock_client):
    return CausalPathRepository(supabase_client=mock_client)


@pytest.mark.unit
class TestGetPathRow:
    @pytest.mark.asyncio
    async def test_returns_row_including_synthetic(self, repo, mock_client):
        row = {"path_id": "scp_abc", "is_synthetic": True, "validation_status": "validated"}
        query = _chain_query([row])
        mock_client.table.return_value.select.return_value = query
        got = await repo.get_path_row("scp_abc")
        assert got == row
        # include_synthetic=True ⇒ the provenance default-exclude predicate
        # must NOT fire — no .eq("is_synthetic", False) call.
        assert ("is_synthetic", False) not in [c.args for c in query.eq.call_args_list]

    @pytest.mark.asyncio
    async def test_returns_none_when_absent(self, repo, mock_client):
        mock_client.table.return_value.select.return_value = _chain_query([])
        assert await repo.get_path_row("nope") is None


@pytest.mark.unit
class TestFindRealPathsForPair:
    @pytest.mark.asyncio
    async def test_filters_pair_and_brand_real_mode(self, repo, mock_client):
        row = {"path_id": "cp_1", "is_synthetic": False}
        query = _chain_query([row])
        mock_client.table.return_value.select.return_value = query
        got = await repo.find_real_paths_for_pair("rep_visits", "trx", brand="Kisqali")
        assert got == [row]
        eq_args = [c.args for c in query.eq.call_args_list]
        assert ("start_node", "rep_visits") in eq_args
        assert ("end_node", "trx") in eq_args
        assert ("brand", "Kisqali") in eq_args

    @pytest.mark.asyncio
    async def test_brand_omitted_when_none(self, repo, mock_client):
        query = _chain_query([])
        mock_client.table.return_value.select.return_value = query
        await repo.find_real_paths_for_pair("t", "y")
        assert not any(c.args[0] == "brand" for c in query.eq.call_args_list)


@pytest.mark.unit
class TestSetValidationStatus:
    def _install_update(self, mock_client, data):
        query = MagicMock()
        query.eq.return_value = query
        query.in_.return_value = query
        result = MagicMock()
        result.data = data
        query.execute = AsyncMock(return_value=result)
        mock_client.table.return_value.update.return_value = query
        return query

    @pytest.mark.asyncio
    async def test_conditional_transition_returns_true_on_update(self, repo, mock_client):
        query = self._install_update(mock_client, [{"path_id": "cp_1"}])
        moved = await repo.set_validation_status("cp_1", "validated", ("pending", "needs_review"))
        assert moved is True
        mock_client.table.return_value.update.assert_called_once_with(
            {"validation_status": "validated"}
        )
        query.eq.assert_called_once_with("path_id", "cp_1")
        query.in_.assert_called_once_with("validation_status", ["pending", "needs_review"])

    @pytest.mark.asyncio
    async def test_no_matching_current_status_returns_false(self, repo, mock_client):
        self._install_update(mock_client, [])
        moved = await repo.set_validation_status("cp_1", "validated", ("pending",))
        assert moved is False

    @pytest.mark.asyncio
    async def test_no_client_returns_false(self):
        repo = CausalPathRepository(supabase_client=None)
        assert await repo.set_validation_status("cp_1", "validated", ("pending",)) is False


@pytest.mark.unit
class TestGuardedPromoteRpc:
    """Lane 1 (spec §4.3): with a DAG hash the transition runs through
    ``promote_causal_path_guarded`` (migration 134); without one the plain
    conditional update is kept."""

    def _install_rpc(self, mock_client, payload):
        call = MagicMock()
        call.execute = AsyncMock(return_value=MagicMock(data=payload))
        mock_client.rpc.return_value = call
        return call

    @pytest.mark.asyncio
    async def test_hash_routes_through_the_guarded_rpc(self, repo, mock_client):
        self._install_rpc(mock_client, {"moved": 1, "rejected": False})
        moved = await repo.set_validation_status(
            "cp_1",
            "validated",
            ("pending", "needs_review"),
            dag_version_hash="h" * 64,
            brand="Kisqali",
        )
        assert moved is True
        name, params = mock_client.rpc.call_args.args
        assert name == "promote_causal_path_guarded"
        assert params == {
            "p_path_id": "cp_1",
            "p_new_status": "validated",
            "p_allowed_current": ["pending", "needs_review"],
            "p_dag_version_hash": "h" * 64,
            "p_brand": "Kisqali",
        }
        mock_client.table.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejected_structure_moves_nothing(self, repo, mock_client):
        self._install_rpc(mock_client, {"moved": 0, "rejected": True})
        moved = await repo.set_validation_status(
            "cp_1", "validated", ("pending",), dag_version_hash="h" * 64
        )
        assert moved is False

    @pytest.mark.asyncio
    async def test_list_wrapped_payload_is_read(self, repo, mock_client):
        self._install_rpc(mock_client, [{"moved": 1, "rejected": False}])
        assert (
            await repo.set_validation_status(
                "cp_1", "validated", ("pending",), dag_version_hash="h" * 64
            )
            is True
        )

    @pytest.mark.asyncio
    async def test_no_hash_keeps_the_plain_conditional_update(self, repo, mock_client):
        query = MagicMock()
        query.eq.return_value = query
        query.in_.return_value = query
        query.execute = AsyncMock(return_value=MagicMock(data=[{"path_id": "cp_1"}]))
        mock_client.table.return_value.update.return_value = query
        assert await repo.set_validation_status("cp_1", "validated", ("pending",)) is True
        mock_client.rpc.assert_not_called()

    @pytest.mark.asyncio
    async def test_rpc_error_propagates(self, repo, mock_client):
        call = MagicMock()
        call.execute = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_client.rpc.return_value = call
        with pytest.raises(RuntimeError):
            await repo.set_validation_status(
                "cp_1", "validated", ("pending",), dag_version_hash="h" * 64
            )

    @pytest.mark.asyncio
    async def test_unexpected_payload_returns_false_and_warns(self, repo, mock_client, caplog):
        # postgrest falls back to ``data = response.text`` on a non-JSON 2xx body:
        # a str payload must read as "no transition" AND leave a trace.
        self._install_rpc(mock_client, "not json")
        caplog.set_level(logging.WARNING, logger="src.repositories.causal_path")
        moved = await repo.set_validation_status(
            "cp_1", "validated", ("pending",), dag_version_hash="h" * 64
        )
        assert moved is False
        assert "unexpected payload" in caplog.text


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
