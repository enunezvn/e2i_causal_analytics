"""Live champion-model registry adapter for prediction_synthesizer (#840).

The ``model_orchestrator`` expects a ``model_registry`` implementing::

    async def get_models_for_target(target: str, entity_type: str) -> List[str]

The concrete data lives in the live ``ml_model_registry`` table, reachable only
via the ASYNC Supabase client. But the agent ``factory`` constructs agents in a
SYNC context, and acquiring the async client there is unsafe (it may run inside
a FastAPI event loop). This adapter bridges the gap:

* It is constructed synchronously (factory-friendly) with no I/O.
* It acquires the async client LAZILY, on first query, inside the agent's async
  context — and caches it.
* It FAILS CLOSED: when no client is available (or no deployable champion is
  registered for the target), it returns ``[]`` — it never fabricates a model
  name, and it never silently no-ops at construction (the #845 client-less
  trap that masked the missing wiring).

Acquiring the client lazily (rather than passing ``None`` at construction) is
what distinguishes this from the dormant #845 repos.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.memory.services.factories import get_async_supabase_client
from src.repositories.ml_experiment import MLModelRegistryRepository


class LiveChampionModelRegistry:
    """Resolve deployable production model names for a target from the live DB."""

    def __init__(self, repo: Optional[Any] = None) -> None:
        # ``repo`` is an injection seam for tests; production leaves it None and
        # the real repo (with the async client) is resolved lazily.
        self._repo = repo
        self._resolved = repo is not None
        self._lock = asyncio.Lock()

    async def _ensure_repo(self) -> Any:
        if self._resolved:
            return self._repo
        # Serialize the first-call resolution so concurrent callers acquire the
        # async client exactly once (double-checked under the lock).
        async with self._lock:
            if self._resolved:
                return self._repo
            client = await get_async_supabase_client()
            # A None client yields a repo whose methods fail closed (return []).
            self._repo = MLModelRegistryRepository(supabase_client=client)
            self._resolved = True
            return self._repo

    async def get_models_for_target(self, target: str, entity_type: str = "") -> List[str]:
        repo = await self._ensure_repo()
        if repo is None:
            return []
        return await repo.get_models_for_target(target, entity_type)

    async def get_model_performance_for_target(
        self, target: str, entity_type: str = ""
    ) -> Dict[str, Dict[str, Any]]:
        """Measured registry metrics per deployable serving model (#883 PR B).

        Pass-through to
        ``MLModelRegistryRepository.get_model_performance_for_target`` — the
        source of the model-performance working-memory key the
        prediction_synthesizer's ``get_context`` reads. Fails closed: ``{}``.
        """
        repo = await self._ensure_repo()
        if repo is None:
            return {}
        result = await repo.get_model_performance_for_target(target, entity_type)
        return dict(result or {})
