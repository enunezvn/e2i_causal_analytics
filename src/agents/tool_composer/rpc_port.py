"""Transport for the tool-composer learning loop's database functions.

The registry sync (``registry_sync``) and the composition recorder call SQL functions from
ml/040 and ml/041 (``sync_tool_registry``, ``composer_record_*``, ``get_tool_reliability``) by
name with named arguments. They depend on this small port instead of a client, so production
calls PostgREST through the service-role Supabase client while the real-database tests call the
same functions through psycopg (``tests/unit/test_database/learning_loop/_pg.PsycopgRpcPort``).
Every function is executable by ``service_role`` only, so the anon client cannot be used.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol


class RpcPort(Protocol):
    """Calls a database function by name; returns its JSON result."""

    async def call(self, name: str, params: Dict[str, Any]) -> Any: ...


class SupabaseRpcPort:
    """``POST /rpc/<name>`` through the cached service-role async Supabase client."""

    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        # Function-local: the factories module pulls in the memory stack, which importers of
        # this port (the composer, the admin route) do not otherwise need at import time.
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        response = await client.rpc(name, params).execute()
        return response.data
