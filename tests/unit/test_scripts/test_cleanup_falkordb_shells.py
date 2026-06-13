"""Unit tests for scripts/cleanup_falkordb_shells.py (#890).

The cleanup script removes the empty FalkorDB graph shells identified in
issue #890. Deletion is deliberately guarded:

- dry-run by default; ``--execute`` is required to delete anything
- only graphs on the hardcoded shell allowlist may ever be dropped
- a graph is only dropped after a live re-verification that it is EMPTY
- ``e2i_causal`` (the populated production graph) is never touchable

No FalkorDB connection is made in these tests: the client is injected.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.cleanup_falkordb_shells import (
    SHELL_GRAPH_ALLOWLIST,
    CleanupResult,
    cleanup_shells,
)


class _FakeGraph:
    """Minimal FalkorDB Graph stand-in: counts nodes, records deletion."""

    def __init__(self, name: str, node_count: int, deleted: list[str]) -> None:
        self.name = name
        self._node_count = node_count
        self._deleted = deleted

    def query(self, q: str) -> Any:
        class _Result:
            result_set: list[list[int]] = []

        r = _Result()
        r.result_set = [[self._node_count]]
        return r

    def delete(self) -> None:
        self._deleted.append(self.name)


class _FakeClient:
    def __init__(self, graphs: dict[str, int]) -> None:
        # graph name -> node count
        self._graphs = graphs
        self.deleted: list[str] = []

    def list_graphs(self) -> list[str]:
        return list(self._graphs)

    def select_graph(self, name: str) -> _FakeGraph:
        return _FakeGraph(name, self._graphs.get(name, 0), self.deleted)


UUID_SHELL = "f789fbc0-9779-4ae2-9fb6-4d962f7f3da1"


@pytest.fixture
def client() -> _FakeClient:
    return _FakeClient(
        {
            "e2i_causal": 637,
            "e2i_semantic": 0,
            "e2i_knowledge": 0,
            UUID_SHELL: 0,
        }
    )


class TestAllowlist:
    def test_allowlist_contains_the_ten_shells_and_never_e2i_causal(self) -> None:
        assert "e2i_semantic" in SHELL_GRAPH_ALLOWLIST
        assert "e2i_knowledge" in SHELL_GRAPH_ALLOWLIST
        assert UUID_SHELL in SHELL_GRAPH_ALLOWLIST
        assert len(SHELL_GRAPH_ALLOWLIST) == 10
        assert "e2i_causal" not in SHELL_GRAPH_ALLOWLIST


class TestDryRunDefault:
    def test_dry_run_deletes_nothing(self, client: _FakeClient) -> None:
        result = cleanup_shells(client, execute=False)
        assert client.deleted == []
        assert isinstance(result, CleanupResult)
        # The three present shells are reported as would-delete
        assert set(result.would_delete) == {"e2i_semantic", "e2i_knowledge", UUID_SHELL}
        assert result.deleted == []


class TestExecuteGuards:
    def test_execute_deletes_only_empty_allowlisted_graphs(self, client: _FakeClient) -> None:
        result = cleanup_shells(client, execute=True)
        assert set(client.deleted) == {"e2i_semantic", "e2i_knowledge", UUID_SHELL}
        assert set(result.deleted) == {"e2i_semantic", "e2i_knowledge", UUID_SHELL}
        assert "e2i_causal" not in client.deleted

    def test_execute_refuses_non_empty_allowlisted_graph(self) -> None:
        client = _FakeClient({"e2i_semantic": 5, "e2i_knowledge": 0})
        result = cleanup_shells(client, execute=True)
        assert "e2i_semantic" not in client.deleted
        assert "e2i_semantic" in result.skipped_non_empty
        assert client.deleted == ["e2i_knowledge"]

    def test_execute_ignores_graphs_not_on_allowlist(self) -> None:
        client = _FakeClient({"some_other_graph": 0, "e2i_causal": 637})
        result = cleanup_shells(client, execute=True)
        assert client.deleted == []
        assert result.deleted == []

    def test_missing_allowlisted_graphs_are_reported_absent(self, client: _FakeClient) -> None:
        result = cleanup_shells(client, execute=False)
        # 10 allowlisted, only 3 exist on this fake server
        assert len(result.absent) == 7
