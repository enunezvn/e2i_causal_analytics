"""Keep the DB tool registry equal to the running code (spec §4, ml/040).

At API startup ``learning_loop_startup()`` builds the payload from the live registry and calls
``sync_tool_registry``, which upserts every registered tool, deprecates tools the code no longer
registers and makes ``tool_dependencies`` exactly ``DEPENDENCY_FIELD_MAPPINGS``. This replaces
generating a migration whenever a tool's declared inputs or output model change (#2003).

The sync runs at most once per process: the startup task, and the composition recorder when a
step write reports tools the DB does not know, share one ``RegistrySync`` whose lock makes
concurrent callers wait for a single RPC. A failure is logged and never raised; the API keeps
serving and the recorder reports the missing tools. The two API workers run the same image, and
the SQL function serialises them with an advisory lock.

The same startup task fetches the column allowlist the recorder's serializer uses to decide
whether a string parameter is a public column name (spec §5.5).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .rpc_port import RpcPort, SupabaseRpcPort

logger = logging.getLogger(__name__)

# sync_tool_registry refuses a payload that would deprecate more active tools than this
# before any write: a partially imported registry must not wipe the table.
MAX_DEPRECATIONS = 3
# The public catalog changes only with migrations, which redeploy the API.
ALLOWLIST_TTL_S = 3600.0

SYNC_COUNT_KEYS = frozenset(
    {"inserted", "updated", "deprecated", "dependencies_upserted", "dependencies_deleted"}
)


def build_sync_payload() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Tool rows and dependency rows for every live tool, from the live registry.

    Raises ``LookupError`` (from ``create_default_tools``) when a tool in ``TOOL_METADATA`` is
    not registered, so a partially imported registry is never synced.
    """
    # Function-local: registering the tools imports the causal engine.
    from src.tool_registry.registry import get_registry

    from .tool_registry import DEPENDENCY_FIELD_MAPPINGS, create_default_tools

    live = get_registry()
    tools: List[Dict[str, Any]] = []
    for tool in sorted(create_default_tools(), key=lambda t: t.name):
        registered = live.get(tool.name)
        if registered is None:  # create_default_tools already checked; the registry is shared
            raise LookupError(f"tool '{tool.name}' is not registered in the live tool registry")
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "source_agent": tool.source_agent,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "avg_latency_ms": tool.avg_latency_ms,
                "version": registered.schema.version,
            }
        )
    dependencies = [
        {
            "consumer": consumer,
            "producer": producer,
            "output_field": output_field,
            "input_field": input_field,
        }
        for (consumer, producer), (output_field, input_field) in sorted(
            DEPENDENCY_FIELD_MAPPINGS.items()
        )
    ]
    return tools, dependencies


class RegistrySync:
    """Once-per-process registry sync and the cached column allowlist."""

    def __init__(
        self,
        port: Optional[RpcPort] = None,
        *,
        max_deprecations: int = MAX_DEPRECATIONS,
        allowlist_ttl_s: float = ALLOWLIST_TTL_S,
    ):
        self._port = port
        self._max_deprecations = max_deprecations
        self._allowlist_ttl_s = allowlist_ttl_s
        self._sync_lock = asyncio.Lock()
        self._allowlist_lock = asyncio.Lock()
        self._allowlist: Optional[frozenset[str]] = None
        self._allowlist_at: Optional[float] = None
        self.synced = False

    @property
    def port(self) -> RpcPort:
        if self._port is None:
            self._port = SupabaseRpcPort()
        return self._port

    async def sync_once(self) -> Optional[Dict[str, int]]:
        """Sync the DB registry unless this process already did; returns the counts if it ran.

        ``None`` means no sync happened now: it already succeeded earlier, or it failed (logged
        at WARNING; a later call tries again).
        """
        if self.synced:
            return None
        async with self._sync_lock:
            if self.synced:
                return None
            try:
                tools, dependencies = await asyncio.to_thread(build_sync_payload)
                counts = await self.port.call(
                    "sync_tool_registry",
                    {
                        "p_tools": tools,
                        "p_dependencies": dependencies,
                        "p_max_deprecations": self._max_deprecations,
                    },
                )
                if not isinstance(counts, dict) or set(counts) != SYNC_COUNT_KEYS:
                    raise TypeError(f"unexpected receipt {counts!r}")
            except Exception as exc:
                logger.warning(
                    "sync_tool_registry failed (%s: %s); the DB tool registry keeps its previous rows",
                    type(exc).__name__,
                    exc,
                )
                return None
            self.synced = True
            logger.info("sync_tool_registry: %s", counts)
            return counts

    async def column_allowlist(self) -> Optional[frozenset[str]]:
        """Column names of public relations, refreshed hourly.

        ``None`` while no fetch has ever succeeded: the serializer then keeps no string as a
        name. A failed refresh keeps the last good set (catalog names, developer-authored).
        """
        if self._fresh():
            return self._allowlist
        async with self._allowlist_lock:
            if self._fresh():
                return self._allowlist
            try:
                names = await self.port.call("composer_public_column_names", {})
                if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                    raise TypeError(f"unexpected column list of type {type(names).__name__}")
            except Exception as exc:
                logger.warning(
                    "composer_public_column_names failed (%s: %s); %s",
                    type(exc).__name__,
                    exc,
                    "keeping the previous allowlist"
                    if self._allowlist is not None
                    else "no column names will be recorded until it succeeds",
                )
                return self._allowlist
            self._allowlist = frozenset(names)
            self._allowlist_at = time.monotonic()
            return self._allowlist

    def _fresh(self) -> bool:
        return (
            self._allowlist is not None
            and self._allowlist_at is not None
            and time.monotonic() - self._allowlist_at < self._allowlist_ttl_s
        )


_default: Optional[RegistrySync] = None


def default_registry_sync() -> RegistrySync:
    """The process-wide instance the startup task and the recorder share."""
    global _default
    if _default is None:
        _default = RegistrySync()
    return _default


async def sync_tool_registry_once() -> Optional[Dict[str, int]]:
    return await default_registry_sync().sync_once()


async def fetch_column_allowlist() -> Optional[frozenset[str]]:
    return await default_registry_sync().column_allowlist()


async def learning_loop_startup(sync: Optional[RegistrySync] = None) -> None:
    """API startup task: sync the registry, fetch the allowlist. Never raises."""
    sync = sync or default_registry_sync()
    try:
        counts = await sync.sync_once()
        allowlist = await sync.column_allowlist()
        logger.info(
            "tool-composer learning loop startup: registry sync %s; column allowlist %s",
            counts if counts is not None else ("already synced" if sync.synced else "failed"),
            f"{len(allowlist)} names" if allowlist is not None else "unavailable",
        )
    except Exception as exc:  # the lifespan task must never crash the app
        logger.warning(
            "tool-composer learning loop startup failed (%s: %s)", type(exc).__name__, exc
        )
