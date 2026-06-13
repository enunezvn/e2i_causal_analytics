"""
Data-driven sentinel watchers.

Replaces hardcoded Celery beat tasks for the common case of "if condition X
about table Y happens, fire action Z." Operators register sentinels via
``POST /api/sentinels``; a single Celery beat task ``sentinel_dispatcher``
runs every 5 minutes and evaluates each enabled sentinel.

Patterns (shipped vocabulary — this is the storage layer):
    threshold_breach    - {"table": "causal_paths", "column": "causal_effect_size",
                           "op": "<", "value": 0.05}
    freshness           - {"table": "triggers", "ts_column": "updated_at",
                           "max_age_hours": 24}
    drift_score         - {"max_drift_score": 0.3}
    new_causal_path     - {"since": "<iso>"} (auto-bumped on fire)
    invalidation_count  - {"table": "executive_insights", "tier": "semantic"}
                           Decision 3 = KEEP BINARY (plan §"DECISIONS ADOPTED"
                           2026-05-19): selects rows where
                           ``invalidated_at IS NOT NULL`` AND brand matches.
                           The shipped analog of the plan's
                           ``staleness_threshold`` trigger vocabulary; see
                           ``src.memory.sentinels.config_loader.
                           PLAN_TRIGGER_TO_INTERNAL_PATTERN``.

Pattern-vocabulary divergence vs the plan
-----------------------------------------
The plan (``.claude/plans/e2i_memory_subsystems_implementation_plan.md``
§3.6) names triggers with operator-friendly nouns
(``data_drop``, ``staleness_threshold``, ``cohort_drift``, ``schedule``);
this module ships the internal/mechanistic vocab above. Plan→shipped
mapping lives in
:data:`src.memory.sentinels.config_loader.PLAN_TRIGGER_TO_INTERNAL_PATTERN`
and is the SINGLE translation point — the registry itself never sees the
plan vocabulary. Rationale: renaming the shipped enum would break the
PR #250 audit trail and the existing API contract; keeping the plan vocab
in YAML lets operators write what they mean while the storage layer stays
stable.

Actions:
    invalidate         - {"source_type": "causal_path"} — passes matched row id
                          as source_id to cascade_invalidate with sentinel's brand
                          as scope_brand
    dispatch_agent     - {"agent_name": "drift_monitor", "input": {...}}
                          Always emits an ``InsightSignalBus`` signal.
                          For the four plan-specced agent names
                          (``rerun_all_active_cohorts``,
                          ``notify_and_queue_reanalysis``, ``flag_for_review``,
                          ``run_full_consolidation``), ADDITIONALLY enqueues
                          the corresponding Celery task in
                          ``src.tasks.sentinel_actions`` so a worker runs
                          the handler. Mapping lives in
                          :data:`PLAN_ACTION_TO_CELERY_TASK` (#375 iter-1 H1).
    notify             - {"channel": "slack#alerts", "template": "..."}
                          Logs the match AND publishes to the Redis
                          ``e2i:alerts`` channel (via
                          ``src.tasks.sentinel_actions.publish_alert``) for
                          CopilotKit real-time delivery (#375 item 3).

Brand scoping is enforced at every layer:
- register_sentinel rejects NULL brand
- ADMIN role required to register brand='all' (enforced at the API layer
  in src/api/routes/sentinels.py)
- evaluate_sentinel restricts pattern evaluation to rows in the sentinel's
  brand
- invalidate action passes the sentinel's brand to cascade_invalidate

Cooldown (#375 item 2)
----------------------
``register_sentinel`` accepts an optional ``cooldown_minutes`` argument.
``dispatch_sentinels`` skips a sentinel if
``now - last_fired_at < cooldown_minutes``. NULL or 0 means "no cooldown".
The column is persisted by migration ``023_sentinel_cooldown.sql`` with
defense-in-depth CHECK constraints (non-negative, ≤ 365 days).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.memory.coordination.signals import get_insight_signal_bus
from src.memory.lifecycle.invalidator import cascade_invalidate
from src.memory.services.factories import get_supabase_client
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


VALID_PATTERN_TYPES = {
    "threshold_breach",
    "freshness",
    "drift_score",
    "new_causal_path",
    # invalidation_count: enumerates rows with invalidated_at IS NOT NULL.
    # Shipped analog of the plan's ``staleness_threshold`` trigger vocab,
    # specifically aligned with Decision 3 = KEEP BINARY (no graded
    # staleness_score; each match degrades to staleness_score=1.0).
    "invalidation_count",
}
VALID_ACTION_TYPES = {"invalidate", "dispatch_agent", "notify"}
VALID_OPS = {">", ">=", "<", "<=", "==", "!="}

# ----------------------------------------------------------------------------
# Tables that carry an ``invalidated_at`` column. Used by the
# ``invalidation_count`` evaluator below to validate the pattern_config (the
# query only makes sense for tables in this set). New invalidation-bearing
# tables should be added here together with their migration.
#
# Source of truth: database/memory/021_insight_lifecycle.sql (the migration
# that introduced the invalidated_at column on these tables).
# ----------------------------------------------------------------------------
INVALIDATION_AWARE_TABLES = {"triggers", "ml_predictions", "executive_insights"}

# ----------------------------------------------------------------------------
# Tables a ``threshold_breach`` / ``freshness`` sentinel may watch. Each one
# carries a ``brand`` column (so the evaluator's ``.eq("brand", brand)`` tenant
# scoping holds) and a known primary key (see ``_pk_column_for_table``). This
# mirrors the ``INVALIDATION_AWARE_TABLES`` allowlist and closes review finding
# M8: ``table``/``column`` are operator-supplied and interpolated into a
# PostgREST projection, so an off-allowlist table could read PHI tables or
# escape brand scoping.
# ----------------------------------------------------------------------------
THRESHOLD_WATCHABLE_TABLES = {
    "causal_paths",
    "triggers",
    "ml_predictions",
    "episodic_memories",
    "executive_insights",
}

# A column/ts_column must be a plain SQL identifier. PostgREST's projection
# mini-language treats ``*``, ``,``, ``(``, ``)``, ``:`` specially (resource
# embedding ``fk(colA,colB)``, aliasing, casts); restricting to this pattern
# blocks projection-widening / foreign-table exfiltration (review finding M8).
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_watch_target(table: str, column: str, *, column_field: str = "column") -> None:
    """Reject unknown watch tables and unsafe column identifiers.

    Used by both ``_validate_pattern_config`` (registration time) and the
    ``threshold_breach``/``freshness`` evaluators (evaluation time, so sentinels
    persisted before this guard existed are still constrained). See finding M8.
    """
    if table not in THRESHOLD_WATCHABLE_TABLES:
        raise ValueError(
            f"table {table!r} is not an allowed sentinel watch target; "
            f"allowed tables: {sorted(THRESHOLD_WATCHABLE_TABLES)}"
        )
    if not isinstance(column, str) or not _SAFE_IDENTIFIER_RE.match(column):
        raise ValueError(
            f"{column_field} {column!r} is not a plain SQL identifier "
            f"(letters/digits/underscore, not starting with a digit)"
        )


# ----------------------------------------------------------------------------
# Plan-specced action name → Celery task path mapping (#375 iter-1 H1).
#
# When a ``dispatch_agent`` action's ``agent_name`` is one of the four
# plan-specced action names, the dispatcher additionally calls
# ``celery_app.send_task(...)`` so a Celery worker enqueues the corresponding
# handler in ``src.tasks.sentinel_actions``. The bus event still fires (it's
# the non-Celery-subscriber contract).
#
# Single source of truth (#375 iter-1 M1): this dict is the canonical mapping
# for the plan-specced action vocabulary. ``src.memory.sentinels.config_loader.
# PLAN_ACTION_TASK_NAMES`` is DERIVED from this dict's keys via
# ``frozenset(PLAN_ACTION_TO_CELERY_TASK)`` — adding a plan action here
# automatically extends the YAML loader's accept-list. The lockstep invariant
# is locked by ``test_plan_action_constants_are_in_lockstep``.
# ----------------------------------------------------------------------------
PLAN_ACTION_TO_CELERY_TASK: Dict[str, str] = {
    "rerun_all_active_cohorts": "src.tasks.sentinel_actions.rerun_all_active_cohorts",
    "notify_and_queue_reanalysis": "src.tasks.sentinel_actions.notify_and_queue_reanalysis",
    "flag_for_review": "src.tasks.sentinel_actions.flag_for_review",
    "run_full_consolidation": "src.tasks.sentinel_actions.run_full_consolidation",
}


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
    cooldown_minutes: Optional[int] = None,
) -> str:
    """
    Register a new sentinel. Returns sentinel_id.

    Validation here is structural (pattern type known, brand non-null);
    callers (e.g. ``POST /api/sentinels``) MUST additionally enforce that
    the user has access to the requested brand, and that ``brand='all'``
    requires ADMIN role.

    ``cooldown_minutes`` is optional (NULL = no cooldown gate, dispatcher
    always re-evaluates). When set, the dispatcher (Celery beat
    ``sentinel_dispatcher``) only re-fires the sentinel after
    ``now - last_fired_at >= cooldown_minutes``. Persisted to the
    ``sentinels.cooldown_minutes`` column (migration 023).
    """
    if not brand:
        raise ValueError("brand is required")
    if pattern_type not in VALID_PATTERN_TYPES:
        raise ValueError(f"unknown pattern_type {pattern_type}")
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"unknown action_type {action_type}")
    if cooldown_minutes is not None:
        # bool exclusion before numeric check — Python's True/False are int
        # subclasses; isinstance(False, int) is True. Without this guard a
        # caller passing cooldown_minutes=False would silently coerce to 0
        # and disable the cooldown gate. (Load-bearing pattern from
        # max_staleness filter, PR #374 / memory feedback.)
        if isinstance(cooldown_minutes, bool) or not isinstance(cooldown_minutes, (int, float)):
            raise ValueError(
                f"cooldown_minutes must be a non-negative number, "
                f"got {type(cooldown_minutes).__name__}"
            )
        if cooldown_minutes < 0:
            raise ValueError(f"cooldown_minutes must be non-negative, got {cooldown_minutes}")
    _validate_pattern_config(pattern_type, pattern_config)
    _validate_action_config(action_type, action_config)

    client = get_supabase_client()
    record: Dict[str, Any] = {
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
    if cooldown_minutes is not None:
        # int() coerces 0.0 / 60.5 to nearest int; values are stored as INTEGER
        # by migration 023.
        record["cooldown_minutes"] = int(cooldown_minutes)
    result = client.table("sentinels").insert(record).execute()
    rows = result.data or []
    if not rows:
        raise RuntimeError("sentinel insert returned no rows")
    sentinel_id = rows[0]["sentinel_id"]
    logger.info(
        f"registered sentinel {sentinel_id} name={name!r} brand={brand} "
        f"pattern={pattern_type} action={action_type}"
    )
    return str(sentinel_id)


def _validate_pattern_config(pattern_type: str, cfg: Dict[str, Any]) -> None:
    if pattern_type == "threshold_breach":
        for key in ("table", "column", "op", "value"):
            if key not in cfg:
                raise ValueError(f"threshold_breach requires '{key}'")
        if cfg["op"] not in VALID_OPS:
            raise ValueError(f"unsupported op {cfg['op']!r}; allowed: {VALID_OPS}")
        _validate_watch_target(cfg["table"], cfg["column"])
    elif pattern_type == "freshness":
        for key in ("table", "ts_column", "max_age_hours"):
            if key not in cfg:
                raise ValueError(f"freshness requires '{key}'")
        _validate_watch_target(cfg["table"], cfg["ts_column"], column_field="ts_column")
    elif pattern_type == "drift_score":
        if "max_drift_score" not in cfg:
            raise ValueError("drift_score requires 'max_drift_score'")
    elif pattern_type == "new_causal_path":
        # 'since' defaults to "epoch" on first fire; no requirement.
        pass
    elif pattern_type == "invalidation_count":
        if "table" not in cfg:
            raise ValueError("invalidation_count requires 'table'")
        table = cfg["table"]
        if table not in INVALIDATION_AWARE_TABLES:
            # Fail loudly rather than emit an unfilterable query — the
            # invalidation_count semantic is only well-defined on tables that
            # carry an invalidated_at column (see migration 021).
            raise ValueError(
                f"invalidation_count table {table!r} does not carry an "
                f"invalidated_at column; allowed tables: "
                f"{sorted(INVALIDATION_AWARE_TABLES)}"
            )


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


def _apply_sentinel_provenance(query: Any, table: str, cfg: Dict[str, Any]) -> Any:
    """Default-exclude synthetic rows from a sentinel watch query (#894).

    The dispatcher evaluates sentinels every 5 minutes against tables the
    synthetic loader populates (live ``causal_paths`` was 250/250 synthetic
    at filing time), so an unfiltered scan fires reanalysis actions from
    planted ground-truth test rows. Real mode default-excludes; a sentinel's
    ``pattern_config`` may opt in with ``include_synthetic`` (strictly parsed
    — ``"false"``/ambiguous values stay real-mode). Untagged watch tables
    (``executive_insights``) are left unfiltered to avoid a 42703.
    """
    from src.repositories.provenance import (
        apply_provenance_filter_for_table,
        coerce_provenance_flag,
    )

    return apply_provenance_filter_for_table(
        query, table, include_synthetic=coerce_provenance_flag(cfg.get("include_synthetic"))
    )


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
    if pattern_type == "invalidation_count":
        return await _eval_invalidation_count(cfg, brand)
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

    # Defense in depth: re-validate at evaluation time so sentinels persisted
    # before the registration-time guard existed cannot inject (finding M8).
    _validate_watch_target(table, column)

    method_name = _OP_TO_METHOD[op]
    pk_col = _pk_column_for_table(table)
    query = client.table(table).select(f"{pk_col}, brand, {column}")
    query = getattr(query, method_name)(column, value)
    if brand != "all":
        query = query.eq("brand", brand)
    query = _apply_sentinel_provenance(query, table, cfg)
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
    # Defense in depth: re-validate at evaluation time (finding M8).
    _validate_watch_target(table, ts_column, column_field="ts_column")
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

    pk_col = _pk_column_for_table(table)
    query = client.table(table).select(f"{pk_col}, brand, {ts_column}").lt(ts_column, cutoff)
    if brand != "all":
        query = query.eq("brand", brand)
    query = _apply_sentinel_provenance(query, table, cfg)
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
    query = (
        client.table("causal_paths").select("path_id, brand, created_at").gte("created_at", since)
    )
    if brand != "all":
        query = query.eq("brand", brand)
    query = _apply_sentinel_provenance(query, "causal_paths", cfg)
    rows = (query.execute().data) or []
    return [{"row_id": r["path_id"], "brand": r.get("brand", brand)} for r in rows]


async def _eval_invalidation_count(cfg: Dict[str, Any], brand: str) -> List[Dict[str, Any]]:
    """Enumerate rows whose ``invalidated_at IS NOT NULL`` (binary staleness).

    M2 (#381): shipped analog of the plan's ``staleness_threshold`` trigger.
    Per Decision 3 = KEEP BINARY (plan §"DECISIONS ADOPTED" 2026-05-19),
    staleness collapses to ``invalidated_at IS NOT NULL`` — every match is
    treated with ``staleness_score=1.0``.

    pattern_config shape::

        {"table": "<invalidation-aware table>",
         "tier": "<optional human label>"}

    Currently supported tables: those in :data:`INVALIDATION_AWARE_TABLES`
    (validated up-front by :func:`_validate_pattern_config`).

    Returned match shape (finding-shaped so the dispatcher can package these
    as ``stale_findings`` for the ``notify_and_queue_reanalysis`` handler)::

        {"row_id":           <pk>,
         "finding_id":       <pk>,       # alias kept for the handler
         "brand":            <row brand>,
         "table":            <source table>,
         "invalidated_at":   <iso str>,
         "staleness_score":  1.0}        # binary per Decision 3
    """
    client = get_supabase_client()
    table = cfg["table"]
    pk_col = _pk_column_for_table(table)
    select_cols = f"{pk_col}, brand, invalidated_at"
    query = client.table(table).select(select_cols).not_.is_("invalidated_at", "null")
    if brand != "all":
        query = query.eq("brand", brand)
    query = _apply_sentinel_provenance(query, table, cfg)
    rows = (query.execute().data) or []
    matches: List[Dict[str, Any]] = []
    for r in rows:
        row_id = r.get(pk_col)
        matches.append(
            {
                "row_id": row_id,
                # ``finding_id`` is the key the notify_and_queue_reanalysis
                # handler uses when logging top-5 findings; we expose both
                # so the dispatcher can package these as stale_findings
                # without further translation.
                "finding_id": row_id,
                "brand": r.get("brand", brand),
                "table": table,
                "invalidated_at": r.get("invalidated_at"),
                "staleness_score": 1.0,
            }
        )
    return matches


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

    Cooldown gate (#375 item 2)
    ---------------------------
    A sentinel with ``cooldown_minutes`` set is SKIPPED at evaluation time
    if ``now - last_fired_at < cooldown_minutes``. The gate is per-sentinel,
    counted from the last successful fire (not from the last evaluation —
    so a sentinel that hasn't matched in 24h with a 6h cooldown still
    evaluates every dispatcher tick).

    Semantics:
        * ``cooldown_minutes IS NULL``       → no cooldown (always evaluate)
        * ``cooldown_minutes == 0``          → no cooldown (always evaluate)
        * ``last_fired_at IS NULL``          → never fired; evaluate
        * ``now - last_fired_at >= cooldown_minutes`` → cooldown elapsed; evaluate
        * otherwise                          → SKIP (cooldown in effect)
    """
    client = get_supabase_client()
    result = SentinelDispatchResult()

    rows = (client.table("sentinels").select("*").eq("enabled", True).execute().data) or []
    result.examined = len(rows)

    now = datetime.now(timezone.utc)

    for sentinel in rows:
        sentinel_id = sentinel.get("sentinel_id")
        if _is_in_cooldown(sentinel, now=now):
            # Logged at INFO so operators can see why a sentinel is quiet.
            logger.info(
                f"sentinel {sentinel_id} in cooldown "
                f"(cooldown_minutes={sentinel.get('cooldown_minutes')}, "
                f"last_fired_at={sentinel.get('last_fired_at')}); skipping"
            )
            continue
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

    result.finished_at = now
    return result


def _is_in_cooldown(sentinel: Dict[str, Any], *, now: datetime) -> bool:
    """Return True if the sentinel was fired within ``cooldown_minutes`` of now.

    Defensive against:
    * bool-as-int (``cooldown_minutes=False`` or ``=True`` smuggled past the
      registration gate by direct DB write) — both treated as "no cooldown"
      since the gate is no longer trustworthy under bool semantics
    * NaN / non-numeric cooldown values — treated as "no cooldown"
    * Unparseable ``last_fired_at`` strings — treated as "never fired"
    """
    cooldown = sentinel.get("cooldown_minutes")
    # NULL / 0 / missing → no gate.
    if cooldown is None:
        return False
    # bool exclusion before numeric check; same load-bearing pattern as
    # register_sentinel — Python's True/False are int subclasses.
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)):
        return False
    # NaN-safe: not (x > 0) is the PostgreSQL-parity pattern (in PG, NaN
    # comparisons are False; we mirror that by inverting the guard).
    if not (cooldown > 0):
        return False
    last_fired_raw = sentinel.get("last_fired_at")
    if not last_fired_raw:
        return False
    try:
        # Supabase serializes timestamptz as ISO 8601 string; tolerate Z suffix.
        if isinstance(last_fired_raw, str):
            iso = last_fired_raw.replace("Z", "+00:00")
            last_fired = datetime.fromisoformat(iso)
        elif isinstance(last_fired_raw, datetime):
            last_fired = last_fired_raw
        else:
            return False
    except ValueError:
        # Unparseable; treat as never-fired so the sentinel can evaluate.
        return False
    if last_fired.tzinfo is None:
        last_fired = last_fired.replace(tzinfo=timezone.utc)
    elapsed = now - last_fired
    return elapsed < timedelta(minutes=float(cooldown))


def _trigger_data_for_dispatch(
    *,
    agent_name: str,
    match: Dict[str, Any],
    matches: List[Dict[str, Any]],
    action_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the ``trigger_data`` payload for a ``dispatch_agent`` Celery enqueue.

    M2 (#381): the ``notify_and_queue_reanalysis`` handler at
    :mod:`src.tasks.sentinel_actions` reads ``trigger_data["stale_findings"]``
    (sorting by ``staleness_score`` and capping at top-5). The dispatcher's
    per-match shape pre-fix only carried ``{"match": ..., "action_input": ...}``
    so the handler always saw an empty list. This helper centralises the
    payload construction: for the ``notify_and_queue_reanalysis`` path it
    packages the FULL matches list under ``stale_findings`` (the handler caps
    internally); for all other agent_names the original per-match shape is
    preserved (back-compat with PR #250).
    """
    base: Dict[str, Any] = {
        "match": match,
        "action_input": action_cfg.get("input", {}),
    }
    if agent_name == "notify_and_queue_reanalysis":
        # Single-fire-with-list semantics: the staleness handler iterates the
        # full matches list. Per Decision 3 = KEEP BINARY each match already
        # carries staleness_score=1.0 (set by _eval_invalidation_count).
        base["stale_findings"] = list(matches)
    return base


async def _fire_action(
    sentinel: Dict[str, Any],
    matches: List[Dict[str, Any]],
    result: SentinelDispatchResult,
) -> None:
    """Execute the sentinel's action_type for each match.

    Per-match iteration is the default (each match → one bus publish + one
    Celery enqueue). Single-fire-with-list semantics apply only to the
    ``notify_and_queue_reanalysis`` agent path, where the handler expects the
    full list of stale_findings in one invocation (M2 of #381) — we still
    iterate ``matches`` for the bus publish but enqueue a single Celery task
    with the aggregated list to keep the back-compat contract for the bus
    subscriber while honoring the handler's expected shape.
    """
    action_type = sentinel["action_type"]
    action_cfg = sentinel.get("action_config", {}) or {}
    brand = sentinel["brand"]
    sentinel_id = sentinel.get("sentinel_id")
    name = sentinel.get("name", "unnamed")

    # M2 (#381): for notify_and_queue_reanalysis we enqueue a SINGLE Celery
    # task carrying the full matches list; the per-match bus events still
    # fire (preserving the PR #250 bus contract). The flag lets us early-out
    # of the per-match Celery enqueue inside the loop body.
    is_staleness_dispatch = (
        action_type == "dispatch_agent"
        and action_cfg.get("agent_name") == "notify_and_queue_reanalysis"
    )

    # Track successes for THIS sentinel only (result.actions_taken is cumulative
    # across the whole dispatcher pass, so we can't use it to decide cooldown).
    actions_succeeded = 0

    for match_index, match in enumerate(matches):
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
                # #375 iter-1 H1: for the four plan-specced action names,
                # ALSO enqueue the corresponding Celery task so a worker
                # actually runs the handler. The bus event above is
                # complementary — non-Celery subscribers (e.g. an in-process
                # orchestrator) still see the dispatch.
                #
                # The mapping is intentionally narrow (whitelist via
                # ``PLAN_ACTION_TO_CELERY_TASK``); ``agent_name`` values
                # outside the map continue to flow bus-only, preserving
                # back-compat with the PR #250 contract.
                #
                # M2 (#381): notify_and_queue_reanalysis is the single
                # exception to per-match enqueue — the handler iterates the
                # full matches list internally, so we enqueue exactly once
                # (on match_index == 0).
                agent_name = action_cfg["agent_name"]
                celery_task_name = PLAN_ACTION_TO_CELERY_TASK.get(agent_name)
                should_enqueue = celery_task_name is not None and (
                    not is_staleness_dispatch or match_index == 0
                )
                if should_enqueue:
                    try:
                        celery_app.send_task(
                            celery_task_name,
                            kwargs={
                                "sentinel_id": str(sentinel_id),
                                "brands": [brand],
                                "trigger_data": _trigger_data_for_dispatch(
                                    agent_name=agent_name,
                                    match=match,
                                    matches=matches,
                                    action_cfg=action_cfg,
                                ),
                            },
                        )
                        logger.info(
                            f"sentinel {sentinel_id}: enqueued "
                            f"{celery_task_name} for agent={agent_name}"
                        )
                    except Exception:
                        # Best-effort: a broker outage MUST NOT crash the
                        # dispatcher loop. The bus event already fired so
                        # local subscribers still get the dispatch.
                        logger.exception(
                            f"sentinel {sentinel_id}: send_task failed for {celery_task_name}"
                        )
            elif action_type == "notify":
                # #375 item 3: wire ``notify`` to Redis pub/sub on
                # ``e2i:alerts``. Falls back to log-only if the alerts
                # publisher is unreachable; the original log line is
                # preserved for operator continuity with prior behaviour.
                logger.info(f"sentinel:notify {name} brand={brand} match={match} cfg={action_cfg}")
                try:
                    # Lazy import: ``src.tasks.sentinel_actions`` depends on
                    # ``celery_app`` which transitively pulls in this module
                    # at worker boot. Importing inside the action keeps the
                    # cycle from forming at module-load time.
                    from src.tasks.sentinel_actions import publish_alert

                    await publish_alert(
                        {
                            "type": "sentinel_notify",
                            "sentinel_id": str(sentinel_id),
                            "sentinel_name": name,
                            "brand": brand,
                            "match": match,
                            "action_config": action_cfg,
                        }
                    )
                except Exception:
                    # publish_alert is itself best-effort; we wrap defensively
                    # so any further unexpected issue here still doesn't break
                    # the dispatcher loop. Narrow class would be preferable but
                    # the import itself can raise on misconfigured deployments.
                    logger.exception(f"sentinel:notify {name} alert publication crashed")
            else:
                continue
            result.actions_taken += 1
            actions_succeeded += 1
        except Exception as exc:
            logger.exception(f"sentinel {sentinel_id}: action {action_type} failed on {match}")
            result.errors.append(f"{sentinel_id} action: {exc}")

    # Only enter cooldown if at least one action actually succeeded. A sentinel
    # whose every action FAILED must stay eligible so the next dispatch pass can
    # retry, rather than being silently suppressed by a premature cooldown.
    if actions_succeeded == 0:
        return

    # Bump fire_count + last_fired_at after the loop.
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        new_count = (sentinel.get("fire_count") or 0) + 1
        client = get_supabase_client()
        client.table("sentinels").update({"last_fired_at": now_iso, "fire_count": new_count}).eq(
            "sentinel_id", sentinel_id
        ).execute()
    except Exception as exc:
        logger.warning(f"sentinel {sentinel_id}: failed to bump fire_count: {exc}")
