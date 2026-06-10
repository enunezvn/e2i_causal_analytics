"""Real Supabase-backed knowledge stores for the feedback learner (issue #837).

The ``KnowledgeUpdaterNode`` proposes updates of four ``knowledge_type``s
(``baseline``/``agent_config``/``prompt``/``threshold``), each carrying a
free-form ``proposed_change`` suggestion string. Until #837 there was NO real
backend implementing ``store.update(...)`` for those types, so ``applied_updates``
was structurally empty and ``update_effectiveness`` was honestly reported as
``None`` (``update_backend_wired=False``, the F15 fix #838).

This module is that real backend. Each :class:`SupabaseKnowledgeStore` durably
upserts the CURRENT recorded value per ``(knowledge_type, key)`` into the
``agent_knowledge_store`` table (migration 065), bumps a version, and READS BACK
to confirm persistence before reporting success — so a proposed update counts as
"applied" only when it durably persisted. ``build_knowledge_stores`` wires the
four typed stores into the agent.

Honesty (REASON-BEFORE-RULES): ``update_effectiveness`` derived from this is a
real measure of durable PERSISTENCE of the recorded learning (read-back
confirmed), NOT of downstream behavioural impact. The recorded values are
queryable via :meth:`SupabaseKnowledgeStore.get`; agent-side CONSUMPTION of them
(reading these back to change runtime behaviour) is a separate, future loop and
is not claimed here. ``proposed_change`` is a free-form suggestion string, so it
is recorded as-is — it is NOT, e.g., an installable DSPy prompt bundle (those go
through the distinct ``prompt_bundles`` mechanism).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# The four knowledge types KnowledgeUpdaterNode emits (one per recommendation
# category: data_update->baseline, config_change->agent_config,
# prompt_update->prompt, threshold->threshold).
KNOWLEDGE_TYPES = ("baseline", "agent_config", "prompt", "threshold")

_TABLE = "agent_knowledge_store"


def _is_meaningless(value: Any) -> bool:
    """A value carrying no real recorded learning to persist.

    ``proposed_change`` is ``Optional[str]`` in practice, so this rejects ``None``
    and blank/whitespace strings — an empty learning must NEVER be counted
    "applied", or it would inflate ``update_effectiveness`` with a false positive.
    Empty collections are also rejected; scalars like ``0``/``False`` are kept
    (they are meaningful values).
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    return False


class SupabaseKnowledgeStore:
    """Durable per-``(knowledge_type, key)`` store backing KnowledgeUpdaterNode.

    ``update`` upserts the recorded value, bumps ``version``, then reads the row
    back and confirms the persisted value matches before returning ``True``.
    FAIL-CLOSED: an empty value, a DB error, or a read-back mismatch returns
    ``False`` (the update is then NOT counted as applied) and never raises into
    the node — a raise would mark the whole learning cycle failed.
    """

    def __init__(self, client: Any, knowledge_type: str) -> None:
        self._client = client
        self._knowledge_type = knowledge_type

    async def update(self, key: str, value: Any, justification: Optional[str] = None) -> bool:
        """Persist ``value`` for ``key`` and confirm via read-back.

        Returns ``True`` only when the value is durably persisted and reads back
        equal. A ``None`` value is not a real recorded learning, so it returns
        ``False`` WITHOUT touching the DB (it can never inflate effectiveness).
        """
        if _is_meaningless(value):
            logger.debug(
                "knowledge_store[%s]: no update for %s — empty/meaningless value",
                self._knowledge_type,
                key,
            )
            return False

        try:
            # version + updated_at are bumped ATOMICALLY by the DB trigger
            # (migration 065) — the store never read-then-bumps (which would race
            # two concurrent writers onto the same version). Upsert only the real
            # columns; on INSERT the defaults apply, on ON CONFLICT DO UPDATE the
            # trigger increments version and refreshes updated_at.
            row = {
                "knowledge_type": self._knowledge_type,
                "key": key,
                "value": value,
                "justification": justification,
            }
            await self._client.table(_TABLE).upsert(row, on_conflict="knowledge_type,key").execute()

            # Read back: a truthy upsert response is not proof the row is durable
            # and correct. Confirm the persisted value equals what we wrote.
            persisted = await self._get_row(key)
            if persisted is None or persisted.get("value") != value:
                logger.warning(
                    "knowledge_store[%s]: read-back mismatch for %s — not applied",
                    self._knowledge_type,
                    key,
                )
                return False
            return True
        except Exception as e:  # noqa: BLE001 - fail closed, never crash the node
            logger.warning(
                "knowledge_store[%s]: update failed for %s: %s",
                self._knowledge_type,
                key,
                e,
            )
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Return the current persisted value for ``key`` (None if absent)."""
        row = await self._get_row(key)
        return row.get("value") if row else None

    async def _get_row(self, key: str) -> Optional[Dict[str, Any]]:
        result = await (
            self._client.table(_TABLE)
            .select("*")
            .eq("knowledge_type", self._knowledge_type)
            .eq("key", key)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None


def build_knowledge_stores(client: Any) -> Dict[str, SupabaseKnowledgeStore]:
    """Build the four real knowledge stores keyed by ``knowledge_type``.

    Returns ``{}`` when ``client`` is ``None`` (e.g. SUPABASE_URL unset in CI /
    offline). KnowledgeUpdaterNode then reports ``update_backend_wired=False`` so
    ``update_effectiveness`` stays ``None`` — the F15 honest fail-closed path,
    never a fabricated 0.0.
    """
    if client is None:
        return {}
    return {kt: SupabaseKnowledgeStore(client, kt) for kt in KNOWLEDGE_TYPES}
