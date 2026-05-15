"""
Data-driven sentinel watchers.

Replaces hardcoded Celery beat tasks for the common case of "if condition X
about table Y happens, fire action Z." Operators register sentinels via
``POST /api/sentinels``; a single Celery beat task ``sentinel_dispatcher``
runs every 5 minutes and evaluates each enabled sentinel.

Patterns:
    threshold_breach   - {"table": "causal_paths", "column": "causal_effect_size",
                          "op": "<", "value": 0.05}
    freshness          - {"table": "triggers", "ts_column": "updated_at",
                          "max_age_hours": 24}
    drift_score        - {"max_drift_score": 0.3}
    new_causal_path    - {"since": "<iso>"} (auto-bumped on fire)

Actions:
    invalidate         - {"source_type": "causal_path"} — passes matched row id
                          as source_id to cascade_invalidate with sentinel's brand
                          as scope_brand
    dispatch_agent     - {"agent_name": "drift_monitor", "input": {...}}
                          (placeholder — emits a signal on InsightSignalBus;
                          actual dispatch is the orchestrator's job)
    notify             - {"channel": "slack#alerts", "template": "..."}
                          (placeholder — logs only in v1)

Brand scoping is enforced at every layer:
- register_sentinel rejects NULL brand
- ADMIN role required to register brand='all' (enforced at the API layer
  in src/api/routes/sentinels.py)
- evaluate_sentinel restricts pattern evaluation to rows in the sentinel's
  brand
- invalidate action passes the sentinel's brand to cascade_invalidate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.memory.coordination.signals import get_insight_signal_bus
from src.memory.lifecycle.invalidator import cascade_invalidate
from src.memory.services.factories import get_supabase_client

logger = logging.getLogger(__name__)


VALID_PATTERN_TYPES = {"threshold_breach", "freshness", "drift_score", "new_causal_path"}
VALID_ACTION_TYPES = {"invalidate", "dispatch_agent", "notify"}
VALID_OPS = {">", ">=", "<", "<=", "==", "!="}


class SentinelEvaluationError(RuntimeError):
    """Raised when a sentinel's pattern_config is malformed."""


@dataclass
class SentinelDispatchResult:
    """Summary of one dispatcher pass."""

    examined: int = 0
    fired: int = 0
    actions_taken: int = 0
    errors: List[str] = field(default_factory=list)
    by_sentinel: Dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None


# ============================================================================
# REGISTRATION
# ============================================================================


async def register_sentinel(
    *,
    name: str,
    pattern_type: str,
    pattern_config: Dict[str, Any],
    action_type: str,
    action_config: Dict[str, Any],
    brand: str,
    region: Optional[str] = None,
    created_by_user_id: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Register a new sentinel. Returns sentinel_id.

    Validation here is structural (pattern type known, brand non-null);
    callers (e.g. ``POST /api/sentinels``) MUST additionally enforce that
    the user has access to the requested brand, and that ``brand='all'``
    requires ADMIN role.
    """
    if not brand:
        raise ValueError("brand is required")
    if pattern_type not in VALID_PATTERN_TYPES:
        raise ValueError(f"unknown pattern_type {pattern_type}")
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"unknown action_type {action_type}")
    _validate_pattern_config(pattern_type, pattern_config)
    _validate_action_config(action_type, action_config)

    client = get_supabase_client()
    record = {
        "name": name,
        "description": description,
        "pattern_type": pattern_type,
        "pattern_config": pattern_config,
        "action_type": action_type,
        "action_config": action_config,
        "brand": brand,
        "region": region,
        "created_by_user_id": created_by_user_id,
        "enabled": True,
    }
    result = client.table("sentinels").insert(record).execute()
    rows = result.data or []
    if not rows:
        raise RuntimeError("sentinel insert returned no rows")
    sentinel_id = rows[0]["sentinel_id"]
    logger.info(
        f"registered sentinel {sentinel_id} name={name!r} brand={brand} "
        f"pattern={pattern_type} action={action_type}"
    )
    return sentinel_id


def _validate_pattern_config(pattern_type: str, cfg: Dict[str, Any]) -> None:
    if pattern_type == "threshold_breach":
        for key in ("table", "column", "op", "value"):
            if key not in cfg:
                raise ValueError(f"threshold_breach requires '{key}'")
        if cfg["op"] not in VALID_OPS:
            raise ValueError(f"unsupported op {cfg['op']!r}; allowed: {VALID_OPS}")
    elif pattern_type == "freshness":
        for key in ("table", "ts_column", "max_age_hours"):
            if key not in cfg:
                raise ValueError(f"freshness requires '{key}'")
    elif pattern_type == "drift_score":
        if "max_drift_score" not in cfg:
            raise ValueError("drift_score requires 'max_drift_score'")
    elif pattern_type == "new_causal_path":
        # 'since' defaults to "epoch" on first fire; no requirement.
        pass


def _validate_action_config(action_type: str, cfg: Dict[str, Any]) -> None:
    if action_type == "invalidate":
        if "source_type" not in cfg:
            raise ValueError("invalidate requires 'source_type' in action_config")
    elif action_type == "dispatch_agent":
        if "agent_name" not in cfg:
            raise ValueError("dispatch_agent requires 'agent_name'")
    # notify has no required fields in v1


# ============================================================================
# EVALUATION (called per-sentinel by the dispatcher)
# ============================================================================


async def evaluate_sentinel(sentinel: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Evaluate a single sentinel's pattern. Returns a list of "matches" —
    one per row that breached the pattern. Each match contains at least
    {row_id} and an optional payload describing the breach.

    Brand scoping: every query is constrained by sentinel['brand'] unless
    brand=='all' (admin-only).
    """
    pattern_type = sentinel["pattern_type"]
    cfg = sentinel.get("pattern_config", {}) or {}
    brand = sentinel["brand"]

    if pattern_type == "threshold_breach":
        return await _eval_threshold_breach(cfg, brand)
    if pattern_type == "freshness":
        return await _eval_freshness(cfg, brand)
    if pattern_type == "drift_score":
        return await _eval_drift_score(cfg, brand)
    if pattern_type == "new_causal_path":
        return await _eval_new_causal_path(cfg, brand, sentinel.get("last_fired_at"))
    raise SentinelEvaluationError(f"unknown pattern_type {pattern_type}")


_OP_TO_METHOD = {
    ">": "gt",
    ">=": "gte",
    "<": "lt",
    "<=": "lte",
    "==": "eq",
    "!=": "neq",
}


async def _eval_threshold_breach(cfg: Dict[str, Any], brand: str) -> List[Dict[str, Any]]:
    """Select rows where ``column op value`` AND ``brand=brand``."""
    client = get_supabase_client()
    table = cfg["table"]
    column = cfg["column"]
    op = cfg["op"]
    value = cfg["value"]

    method_name = _OP_TO_METHOD[op]
    pk_col = _pk_column_for_table(table)
    query = client.table(table).select(f"{pk_col}, brand, {column}")
    query = getattr(query, method_name)(column, value)
    if brand != "all":
        query = query.eq("brand", brand)
    rows = (query.execute().data) or []
    return [
        {"row_id": r[pk_col], "brand": r.get("brand", brand), "value": r.get(column)} for r in rows
    ]


async def _eval_freshness(cfg: Dict[str, Any], brand: str) -> List[Dict[str, Any]]:
    """Select rows whose ts_column is older than max_age_hours."""
    client = get_supabase_client()
    table = cfg["table"]
    ts_column = cfg["ts_column"]
    max_age_hours = float(cfg["max_age_hours"])
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

    pk_col = _pk_column_for_table(table)
    query = client.table(table).select(f"{pk_col}, brand, {ts_column}").lt(ts_column, cutoff)
    if brand != "all":
        query = query.eq("brand", brand)
    rows = (query.execute().data) or []
    return [{"row_id": r[pk_col], "brand": r.get("brand", brand)} for r in rows]


async def _eval_drift_score(cfg: Dict[str, Any], brand: str) -> List[Dict[str, Any]]:
    """
    Look up the latest drift_monitor outputs and check max drift_score.

    Placeholder for v1: drift_monitor's persisted alerts table isn't part
    of the migration we're shipping. We return [] unless a downstream
    integration provides it; the action plumbing is still exercised by
    threshold_breach and freshness patterns.
    """
    return []


async def _eval_new_causal_path(
    cfg: Dict[str, Any], brand: str, last_fired_at: Optional[str]
) -> List[Dict[str, Any]]:
    """Select causal_paths created since last_fired_at."""
    client = get_supabase_client()
    since = cfg.get("since") or last_fired_at or "1970-01-01T00:00:00+00:00"
    query = client.table("causal_paths").select("path_id, brand, created_at").gte(
        "created_at", since
    )
    if brand != "all":
        query = query.eq("brand", brand)
    rows = (query.execute().data) or []
    return [{"row_id": r["path_id"], "brand": r.get("brand", brand)} for r in rows]


def _pk_column_for_table(table: str) -> str:
    """Best-known primary-key column for E2I tables sentinels watch."""
    return {
        "causal_paths": "path_id",
        "triggers": "trigger_id",
        "ml_predictions": "prediction_id",
        "episodic_memories": "memory_id",
        "executive_insights": "insight_id",
    }.get(table, "id")


# ============================================================================
# DISPATCH (run by Celery beat)
# ============================================================================


async def dispatch_sentinels() -> SentinelDispatchResult:
    """
    Single pass: fetch enabled sentinels, evaluate each, fire matching actions.

    Called every 5 minutes by the ``sentinel_dispatcher`` Celery beat task.
    Errors in one sentinel never block others.
    """
    client = get_supabase_client()
    result = SentinelDispatchResult()

    rows = (
        client.table("sentinels")
        .select("*")
        .eq("enabled", True)
        .execute()
        .data
    ) or []
    result.examined = len(rows)

    for sentinel in rows:
        sentinel_id = sentinel.get("sentinel_id")
        try:
            matches = await evaluate_sentinel(sentinel)
            if not matches:
                continue
            result.fired += 1
            result.by_sentinel[str(sentinel_id)] = len(matches)
            await _fire_action(sentinel, matches, result)
        except Exception as exc:
            logger.exception(f"sentinel {sentinel_id} evaluation failed")
            result.errors.append(f"{sentinel_id}: {exc}")

    result.finished_at = datetime.now(timezone.utc)
    return result


async def _fire_action(
    sentinel: Dict[str, Any],
    matches: List[Dict[str, Any]],
    result: SentinelDispatchResult,
) -> None:
    """Execute the sentinel's action_type for each match."""
    action_type = sentinel["action_type"]
    action_cfg = sentinel.get("action_config", {}) or {}
    brand = sentinel["brand"]
    sentinel_id = sentinel.get("sentinel_id")
    name = sentinel.get("name", "unnamed")

    for match in matches:
        try:
            if action_type == "invalidate":
                # cascade_invalidate uses the sentinel's brand as scope_brand,
                # making cross-brand bleed structurally impossible regardless
                # of what the match row's own brand is.
                await cascade_invalidate(
                    source_type=action_cfg["source_type"],
                    source_id=match["row_id"],
                    reason=f"sentinel:{name}",
                    scope_brand=brand,
                )
            elif action_type == "dispatch_agent":
                # v1: emit a signal; the orchestrator decides whether to act.
                bus = get_insight_signal_bus()
                await bus.publish(
                    topic="sentinel:dispatch",
                    brand=brand,
                    payload={
                        "sentinel_id": str(sentinel_id),
                        "agent_name": action_cfg["agent_name"],
                        "match": match,
                        "action_input": action_cfg.get("input", {}),
                    },
                )
            elif action_type == "notify":
                logger.info(
                    f"sentinel:notify {name} brand={brand} match={match} "
                    f"cfg={action_cfg}"
                )
            else:
                continue
            result.actions_taken += 1
        except Exception as exc:
            logger.exception(f"sentinel {sentinel_id}: action {action_type} failed on {match}")
            result.errors.append(f"{sentinel_id} action: {exc}")

    # Bump fire_count + last_fired_at after the loop.
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        new_count = (sentinel.get("fire_count") or 0) + 1
        client = get_supabase_client()
        client.table("sentinels").update(
            {"last_fired_at": now_iso, "fire_count": new_count}
        ).eq("sentinel_id", sentinel_id).execute()
    except Exception as exc:
        logger.warning(f"sentinel {sentinel_id}: failed to bump fire_count: {exc}")
