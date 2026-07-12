"""Aggregation for GET /api/admin/observability/llm-usage (spec 2026-07-12).

Reads llm_usage_events and computes cost at READ time via llm_pricing, so
pricing corrections apply retroactively. Chat rows (user_id set) roll up per
user and per session; NULL-user rows aggregate into the platform section.
Unpriced models contribute tokens/calls but no cost and are listed in
unpriced_models — never silently costed. Sync methods by design: the route
runs them via asyncio.to_thread like its admin.py siblings.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from src.services.llm_pricing import PRICING_VERSION, cost_usd

logger = logging.getLogger(__name__)

_PAGE = 1000
_IN_CHUNK = 100  # keep .in_() URL length bounded

_EVENT_COLUMNS = (
    "created_at, provider, model, input_tokens, output_tokens, "
    "surface, component, user_id, session_id"
)


class LLMObservabilityService:
    def __init__(self, client: Optional[Any] = None) -> None:
        if client is None:
            from src.api.dependencies.supabase_client import get_supabase

            client = get_supabase()
        if client is None:
            # get_supabase() returns None when Supabase is unavailable. Raise
            # instead of caching a dead client so the route-level singleton
            # getter stays uncached and self-heals on the next request
            # (matching AdminUserService semantics).
            raise RuntimeError("Supabase client unavailable for LLM observability")
        self.client = client

    # ------------------------------------------------------------- fetch ----

    def _fetch_events(self, since_iso: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        offset = 0
        while True:
            page = (
                self.client.table("llm_usage_events")
                .select(_EVENT_COLUMNS)
                .gte("created_at", since_iso)
                .order("id", desc=False)
                .range(offset, offset + _PAGE - 1)
                .execute()
                .data
                or []
            )
            events.extend(page)
            if len(page) < _PAGE:
                return events
            offset += _PAGE

    def _tracking_since(self) -> Optional[str]:
        rows = (
            self.client.table("llm_usage_events")
            .select("created_at")
            .order("id", desc=False)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0]["created_at"] if rows else None

    def _conversations(self, session_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(session_ids), _IN_CHUNK):
            chunk = session_ids[i : i + _IN_CHUNK]
            rows = (
                self.client.table("chatbot_conversations")
                .select("session_id, title, created_at")
                .in_("session_id", chunk)
                .execute()
                .data
                or []
            )
            for row in rows:
                out[row["session_id"]] = row
        return out

    # --------------------------------------------------------- aggregate ----

    def llm_usage(self, days: int, users: List[Dict[str, Any]]) -> Dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        events = self._fetch_events(since.isoformat())
        emails = {u.get("id"): u.get("email") for u in users}

        unpriced: Set[str] = set()

        def _cost(event: Dict[str, Any], input_t: int, output_t: int) -> Optional[float]:
            cost = cost_usd(event.get("model") or "", input_t, output_t)
            if cost is None:
                unpriced.add(event.get("model") or "")
            return cost

        summary: Dict[str, Any] = {
            "total_cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "calls": 0,
            "distinct_users": 0,
            "days": days,
            "tracking_since": self._tracking_since(),
        }
        summary_priced_calls = 0
        daily: Dict[str, Dict[str, Any]] = {}
        per_user: Dict[str, Dict[str, Any]] = {}
        per_session: Dict[str, Dict[str, Any]] = {}
        platform: Dict[Tuple[str, Optional[str], str], Dict[str, Any]] = {}

        for event in events:
            input_t = int(event.get("input_tokens") or 0)
            output_t = int(event.get("output_tokens") or 0)
            cost = _cost(event, input_t, output_t)
            model = event.get("model") or "unknown"

            summary["calls"] += 1
            summary["input_tokens"] += input_t
            summary["output_tokens"] += output_t
            if cost is not None:
                summary_priced_calls += 1
            if cost:
                summary["total_cost_usd"] += cost

            day = (event.get("created_at") or "")[:10]
            bucket = daily.setdefault(
                day,
                {"date": day, "chat_cost_usd": 0.0, "platform_cost_usd": 0.0, "tokens": 0},
            )
            bucket["tokens"] += input_t + output_t

            user_id = event.get("user_id")
            if user_id:
                if cost:
                    bucket["chat_cost_usd"] += cost
                user_row = per_user.setdefault(
                    user_id,
                    {
                        "user_id": user_id,
                        "email": emails.get(user_id),
                        "session_ids": set(),
                        "calls": 0,
                        "priced_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "models": set(),
                    },
                )
                session_id = event.get("session_id") or "unknown"
                user_row["session_ids"].add(session_id)
                user_row["calls"] += 1
                user_row["input_tokens"] += input_t
                user_row["output_tokens"] += output_t
                if cost is not None:
                    user_row["priced_calls"] += 1
                if cost:
                    user_row["cost_usd"] += cost
                user_row["models"].add(model)

                session_row = per_session.setdefault(
                    session_id,
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "first_event_at": event.get("created_at"),
                        "calls": 0,
                        "priced_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "models": set(),
                    },
                )
                session_row["calls"] += 1
                session_row["input_tokens"] += input_t
                session_row["output_tokens"] += output_t
                if cost is not None:
                    session_row["priced_calls"] += 1
                if cost:
                    session_row["cost_usd"] += cost
                session_row["models"].add(model)
            else:
                if cost:
                    bucket["platform_cost_usd"] += cost
                key = (event.get("surface") or "other", event.get("component"), model)
                platform_row = platform.setdefault(
                    key,
                    {
                        "surface": key[0],
                        "component": key[1],
                        "model": model,
                        "calls": 0,
                        "priced_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    },
                )
                platform_row["calls"] += 1
                platform_row["input_tokens"] += input_t
                platform_row["output_tokens"] += output_t
                if cost is not None:
                    platform_row["priced_calls"] += 1
                if cost:
                    platform_row["cost_usd"] += cost

        summary["distinct_users"] = len(per_user)
        if summary["calls"] and not summary_priced_calls:
            # Every call in the window was unpriced — "$0" would be a lie.
            # A genuinely empty window keeps 0.0: zero spend is true there.
            summary["total_cost_usd"] = None
        else:
            summary["total_cost_usd"] = round(summary["total_cost_usd"], 6)

        conversations = self._conversations(sorted(per_session))

        by_user = []
        for row in per_user.values():
            by_user.append(
                {
                    "user_id": row["user_id"],
                    "email": row["email"],
                    "sessions": len(row["session_ids"]),
                    "calls": row["calls"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "cost_usd": round(row["cost_usd"], 6) if row["priced_calls"] else None,
                    "models": sorted(row["models"]),
                }
            )
        by_user.sort(key=lambda r: r["cost_usd"] or 0.0, reverse=True)

        sessions: Dict[str, List[Dict[str, Any]]] = {}
        for row in per_session.values():
            conv = conversations.get(row["session_id"], {})
            sessions.setdefault(row["user_id"], []).append(
                {
                    "session_id": row["session_id"],
                    "title": conv.get("title"),
                    "started_at": conv.get("created_at") or row["first_event_at"],
                    "calls": row["calls"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "cost_usd": round(row["cost_usd"], 6) if row["priced_calls"] else None,
                    "models": sorted(row["models"]),
                }
            )
        for rows in sessions.values():
            rows.sort(key=lambda r: r["started_at"] or "", reverse=True)

        platform_rows = [
            {
                "surface": row["surface"],
                "component": row["component"],
                "model": row["model"],
                "calls": row["calls"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cost_usd": round(row["cost_usd"], 6) if row["priced_calls"] else None,
            }
            for row in platform.values()
        ]
        platform_rows.sort(key=lambda r: r["cost_usd"] or 0.0, reverse=True)

        return {
            "summary": summary,
            "daily": [daily[d] for d in sorted(daily)],
            "by_user": by_user,
            "sessions": sessions,
            "platform": platform_rows,
            "pricing_version": PRICING_VERSION,
            "unpriced_models": sorted(m for m in unpriced if m),
        }
