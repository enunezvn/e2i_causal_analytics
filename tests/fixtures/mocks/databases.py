"""Centralized database mocking for E2I tests.

Provides consistent mock implementations for database clients:
- MockSupabaseClient: Full mock for Supabase operations
- MockSupabaseQuery: Chainable query builder mock
- MockRedisClient: Redis operations mock
- MockFalkorDBClient: Graph database mock

Usage:
    from tests.fixtures.mocks.databases import mock_supabase_client

    async def test_db_operation(mock_supabase_client):
        mock_supabase_client.set_query_result([{"id": 1, "name": "test"}])
        result = await repository.get_all()
        assert len(result) == 1
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

import pytest


class MockDatabaseClient:
    """Base class for database client mocks.

    Provides common functionality for tracking calls and setting responses.
    """

    def __init__(self):
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []
        self._connected = True

    def record_call(self, method: str, *args, **kwargs) -> None:
        """Record a call for inspection."""
        self.call_count += 1
        self.calls.append(
            {
                "method": method,
                "args": args,
                "kwargs": kwargs,
            }
        )

    def get_calls_for_method(self, method: str) -> List[Dict[str, Any]]:
        """Get all calls to a specific method."""
        return [c for c in self.calls if c["method"] == method]

    def reset(self) -> None:
        """Reset call tracking."""
        self.call_count = 0
        self.calls = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self._connected = False

    def connect(self) -> None:
        self._connected = True


class MockSupabaseQuery:
    """Chainable mock for Supabase query builder.

    Supports the fluent interface pattern used by Supabase:
        client.table("users").select("*").eq("id", 1).execute()
    """

    def __init__(
        self,
        table_name: str = "",
        result_data: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ):
        self.table_name = table_name
        self._result_data = result_data or []
        self._error = error
        self._filters: List[Dict[str, Any]] = []
        self._select_columns: str = "*"
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._order_by: Optional[str] = None

    def select(self, columns: str = "*") -> "MockSupabaseQuery":
        """Mock select operation."""
        self._select_columns = columns
        return self

    def insert(self, data: Union[Dict, List[Dict]]) -> "MockSupabaseQuery":
        """Mock insert operation."""
        if isinstance(data, dict):
            self._result_data = [data]
        else:
            self._result_data = data
        return self

    def update(self, data: Dict[str, Any]) -> "MockSupabaseQuery":
        """Mock update operation."""
        return self

    def delete(self) -> "MockSupabaseQuery":
        """Mock delete operation."""
        return self

    def upsert(self, data: Union[Dict, List[Dict]]) -> "MockSupabaseQuery":
        """Mock upsert operation."""
        if isinstance(data, dict):
            self._result_data = [data]
        else:
            self._result_data = data
        return self

    def eq(self, column: str, value: Any) -> "MockSupabaseQuery":
        """Mock equality filter."""
        self._filters.append({"type": "eq", "column": column, "value": value})
        return self

    def neq(self, column: str, value: Any) -> "MockSupabaseQuery":
        """Mock not-equal filter."""
        self._filters.append({"type": "neq", "column": column, "value": value})
        return self

    def gt(self, column: str, value: Any) -> "MockSupabaseQuery":
        """Mock greater-than filter."""
        self._filters.append({"type": "gt", "column": column, "value": value})
        return self

    def lt(self, column: str, value: Any) -> "MockSupabaseQuery":
        """Mock less-than filter."""
        self._filters.append({"type": "lt", "column": column, "value": value})
        return self

    def gte(self, column: str, value: Any) -> "MockSupabaseQuery":
        """Mock greater-than-or-equal filter."""
        self._filters.append({"type": "gte", "column": column, "value": value})
        return self

    def lte(self, column: str, value: Any) -> "MockSupabaseQuery":
        """Mock less-than-or-equal filter."""
        self._filters.append({"type": "lte", "column": column, "value": value})
        return self

    def contains(self, column: str, value: Any) -> "MockSupabaseQuery":
        """Mock JSONB contains filter."""
        self._filters.append({"type": "contains", "column": column, "value": value})
        return self

    def in_(self, column: str, values: List[Any]) -> "MockSupabaseQuery":
        """Mock IN filter."""
        self._filters.append({"type": "in", "column": column, "value": values})
        return self

    def is_(self, column: str, value: Any) -> "MockSupabaseQuery":
        """Mock IS filter (for NULL checks)."""
        self._filters.append({"type": "is", "column": column, "value": value})
        return self

    def order(self, column: str, *, desc: bool = False) -> "MockSupabaseQuery":
        """Mock ORDER BY."""
        self._order_by = f"{column} {'DESC' if desc else 'ASC'}"
        return self

    def limit(self, count: int) -> "MockSupabaseQuery":
        """Mock LIMIT."""
        self._limit = count
        return self

    def offset(self, count: int) -> "MockSupabaseQuery":
        """Mock OFFSET."""
        self._offset = count
        return self

    def single(self) -> "MockSupabaseQuery":
        """Mock single() to return first result."""
        if self._result_data:
            self._result_data = [self._result_data[0]]
        return self

    def maybe_single(self) -> "MockSupabaseQuery":
        """Mock maybe_single() to return first result or None."""
        if self._result_data:
            self._result_data = [self._result_data[0]]
        return self

    def execute(self) -> MagicMock:
        """Execute the query and return mock response."""
        response = MagicMock()
        if self._error:
            response.data = None
            response.error = {"message": self._error}
        else:
            response.data = self._result_data
            response.error = None
        response.count = len(self._result_data) if self._result_data else 0
        return response


class MockSupabaseClient(MockDatabaseClient):
    """Full mock for Supabase client operations.

    Supports:
    - Table operations (CRUD)
    - RPC calls
    - Storage operations
    - Auth operations (mocked)
    """

    def __init__(
        self,
        default_data: Optional[List[Dict[str, Any]]] = None,
        rpc_responses: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._default_data = default_data or []
        self._table_data: Dict[str, List[Dict[str, Any]]] = {}
        self._rpc_responses = rpc_responses or {}
        self._current_query: Optional[MockSupabaseQuery] = None

    def table(self, table_name: str) -> MockSupabaseQuery:
        """Get a query builder for the specified table."""
        self.record_call("table", table_name)
        data = self._table_data.get(table_name, self._default_data)
        self._current_query = MockSupabaseQuery(table_name=table_name, result_data=data)
        return self._current_query

    def from_(self, table_name: str) -> MockSupabaseQuery:
        """Alias for table() - some Supabase versions use this."""
        return self.table(table_name)

    def rpc(self, function_name: str, params: Optional[Dict[str, Any]] = None) -> MagicMock:
        """Mock RPC function call."""
        self.record_call("rpc", function_name, params=params)
        response = MagicMock()
        if function_name in self._rpc_responses:
            response.data = self._rpc_responses[function_name]
            response.error = None
        else:
            response.data = None
            response.error = None
        return response

    def set_table_data(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        """Set mock data for a specific table."""
        self._table_data[table_name] = data

    def set_rpc_response(self, function_name: str, response: Any) -> None:
        """Set mock response for an RPC function."""
        self._rpc_responses[function_name] = response

    def set_query_result(self, data: List[Dict[str, Any]]) -> None:
        """Set the default query result data."""
        self._default_data = data

    @property
    def auth(self) -> MagicMock:
        """Mock auth interface."""
        auth = MagicMock()
        auth.get_user.return_value = MagicMock(
            user=MagicMock(id="test-user-id", email="test@example.com")
        )
        return auth

    @property
    def storage(self) -> MagicMock:
        """Mock storage interface."""
        storage = MagicMock()
        bucket = MagicMock()
        bucket.upload.return_value = {"path": "test/file.txt"}
        bucket.download.return_value = b"file content"
        bucket.get_public_url.return_value = "https://example.com/test/file.txt"
        storage.from_.return_value = bucket
        return storage


class MockRedisClient(MockDatabaseClient):
    """Mock for Redis client operations."""

    def __init__(self):
        super().__init__()
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, int] = {}

    async def get(self, key: str) -> Optional[str]:
        """Mock GET operation."""
        self.record_call("get", key)
        return self._data.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
        px: Optional[int] = None,
    ) -> bool:
        """Mock SET operation."""
        self.record_call("set", key, value, ex=ex, px=px)
        self._data[key] = value
        if ex:
            self._expiry[key] = ex
        return True

    async def delete(self, *keys: str) -> int:
        """Mock DELETE operation."""
        self.record_call("delete", *keys)
        deleted = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                deleted += 1
        return deleted

    async def exists(self, *keys: str) -> int:
        """Mock EXISTS operation."""
        self.record_call("exists", *keys)
        return sum(1 for key in keys if key in self._data)

    async def ping(self) -> bool:
        """Mock PING operation."""
        self.record_call("ping")
        return self._connected

    async def hget(self, name: str, key: str) -> Optional[str]:
        """Mock HGET operation."""
        self.record_call("hget", name, key)
        hash_data = self._data.get(name, {})
        return hash_data.get(key) if isinstance(hash_data, dict) else None

    async def hset(self, name: str, key: str, value: str) -> int:
        """Mock HSET operation."""
        self.record_call("hset", name, key, value)
        if name not in self._data:
            self._data[name] = {}
        self._data[name][key] = value
        return 1

    async def hgetall(self, name: str) -> Dict[str, str]:
        """Mock HGETALL operation."""
        self.record_call("hgetall", name)
        return self._data.get(name, {})

    def set_data(self, data: Dict[str, Any]) -> None:
        """Set mock data directly."""
        self._data = data


class MockFalkorDBClient(MockDatabaseClient):
    """Mock for FalkorDB (graph database) operations."""

    def __init__(self):
        super().__init__()
        self._nodes: List[Dict[str, Any]] = []
        self._relationships: List[Dict[str, Any]] = []
        self._query_results: List[Any] = []

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> MagicMock:
        """Mock Cypher query execution."""
        self.record_call("query", cypher, params=params)
        result = MagicMock()
        result.result_set = self._query_results
        return result

    def set_query_results(self, results: List[Any]) -> None:
        """Set mock query results."""
        self._query_results = results

    def add_node(self, node: Dict[str, Any]) -> None:
        """Add a mock node to the graph."""
        self._nodes.append(node)

    def add_relationship(self, rel: Dict[str, Any]) -> None:
        """Add a mock relationship to the graph."""
        self._relationships.append(rel)


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_supabase_client() -> MockSupabaseClient:
    """Fixture providing a mock Supabase client."""
    return MockSupabaseClient()


@pytest.fixture
def mock_supabase_with_data(request) -> MockSupabaseClient:
    """Fixture providing a mock Supabase client with pre-configured data.

    Use with pytest.mark.parametrize to pass table data:
        @pytest.mark.parametrize("mock_supabase_with_data", [
            {"users": [{"id": 1, "name": "test"}]}
        ], indirect=True)
    """
    client = MockSupabaseClient()
    if hasattr(request, "param") and isinstance(request.param, dict):
        for table, data in request.param.items():
            client.set_table_data(table, data)
    return client


@pytest.fixture
def mock_redis_client() -> MockRedisClient:
    """Fixture providing a mock Redis client."""
    return MockRedisClient()


@pytest.fixture
def mock_falkordb_client() -> MockFalkorDBClient:
    """Fixture providing a mock FalkorDB client."""
    return MockFalkorDBClient()
