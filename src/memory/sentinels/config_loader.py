"""
YAML sentinel config + startup loader (issue #375, plan §Phase 3 Step 3.7+3.10).

Reads ``config/sentinels.yaml`` (or operator-supplied path) and registers the
plan-specced sentinels via :func:`src.memory.sentinels.registry.register_sentinel`.

Pattern-vocabulary divergence
-----------------------------
The plan §3.6 trigger vocabulary
(``data_drop``, ``staleness_threshold``, ``cohort_drift``, ``schedule``)
diverges from the shipped registry vocabulary
(``freshness``, ``threshold_breach``, ``drift_score``, ``new_causal_path``).

This loader maintains the plan vocabulary in the YAML for operator clarity
and translates to internal pattern types via
:data:`PLAN_TRIGGER_TO_INTERNAL_PATTERN`. The mapping is intentionally
narrow — each plan trigger has a single shipped analog:

* ``data_drop``           → ``freshness``       (age of a tracked table row)
* ``staleness_threshold`` → ``threshold_breach`` (numeric metric crossed)
* ``cohort_drift``        → ``drift_score``      (drift_monitor output)
* ``schedule``            → ``new_causal_path``  (time-windowed new-rows watcher)

The renames are NOT applied at the registry layer (that would break PR #250
audit trails and existing test fixtures); the registry continues to speak its
shipped vocabulary. This loader is the only translation point.

Idempotency
-----------
Re-loading the same YAML must not duplicate sentinels. Identity is determined
by ``(name, brand)`` — both are YAML-stable and easy to compare. The loader
queries existing sentinels with the same ``(name, brand)`` before insert and
skips if a match exists.

Cooldown
--------
``cooldown_minutes`` is persisted to the ``sentinels.cooldown_minutes`` column
added in migration ``023_sentinel_cooldown.sql``. The dispatcher enforces it
(see :func:`src.tasks.insight_lifecycle_tasks.sentinel_dispatcher`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Final, List

import yaml

from src.memory.sentinels.registry import (
    PLAN_ACTION_TO_CELERY_TASK,
    VALID_ACTION_TYPES,
    VALID_PATTERN_TYPES,
    register_sentinel,
)
from src.memory.services.factories import get_supabase_client

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Default config path. The loader resolves to
# ``<repo_root>/config/sentinels.yaml`` based on this module's location.
#
# ``Path(__file__).resolve()`` lands in
# ``src/memory/sentinels/config_loader.py``; we ascend four parents to reach
# the repo root, then descend into ``config/sentinels.yaml``.
# ----------------------------------------------------------------------------
DEFAULT_CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "config" / "sentinels.yaml"


# ----------------------------------------------------------------------------
# Plan-vocab → internal-vocab translation. Documented in this file's docstring.
#
# M2 (#381): the plan's ``staleness_threshold`` was previously aliased to the
# shipped ``threshold_breach`` pattern, but that resulted in a three-way name
# mismatch (sentinel name promises staleness, condition evaluates effect size,
# handler reads ``stale_findings`` from trigger_data that the evaluator never
# set). The shipped ``invalidation_count`` pattern (added 2026-05-19) is the
# binary-staleness analog per Decision 3 = KEEP BINARY (plan §"DECISIONS
# ADOPTED" 2026-05-19): enumerates rows with ``invalidated_at IS NOT NULL``
# in the configured invalidation-aware table.
# ----------------------------------------------------------------------------
PLAN_TRIGGER_TO_INTERNAL_PATTERN: Final[Dict[str, str]] = {
    "data_drop": "freshness",
    "staleness_threshold": "invalidation_count",
    "cohort_drift": "drift_score",
    "schedule": "new_causal_path",
}


# ----------------------------------------------------------------------------
# Plan-action → shipped action_type. The 4 plan-specified actions
# (rerun_all_active_cohorts, notify_and_queue_reanalysis, flag_for_review,
# run_full_consolidation) are implemented as Celery tasks under
# ``src.tasks.sentinel_actions``. The registry's existing action_type vocabulary
# only knows ``invalidate | dispatch_agent | notify`` — we route all four
# plan-actions through ``dispatch_agent`` with the Celery task name supplied
# via ``action_config['agent_name']``. The dispatcher's _fire_action path
# already wires dispatch_agent → signal-bus, so the Celery task fires off
# the signal-bus subscriber side.
#
# Single-source-of-truth (#375 iter-1 M1): the canonical mapping lives in
# ``src.memory.sentinels.registry.PLAN_ACTION_TO_CELERY_TASK`` (a dict from
# plan-action-name → full Celery task path). This frozenset is DERIVED from
# that dict so any future addition propagates automatically — the loader's
# accept-list and the dispatcher's enqueue-map cannot drift apart. The
# invariant is locked by ``test_plan_action_constants_are_in_lockstep``.
# ----------------------------------------------------------------------------
PLAN_ACTION_TASK_NAMES: Final[frozenset[str]] = frozenset(PLAN_ACTION_TO_CELERY_TASK)


class SentinelConfigLoadError(RuntimeError):
    """Raised when a sentinel YAML config is unreadable or malformed."""


def _coerce_brand_list(brands: List[str] | str | None) -> str:
    """The shipped registry stores a single ``brand`` per sentinel. Plan-YAML
    supports ``brands: [a, b]`` lists; we coerce ``[\"*\"]`` and ``[\"all\"]``
    to ``\"all\"`` (admin scope), and ``[\"X\"]`` (single brand) to that brand.

    A multi-brand list like ``[a, b]`` is rejected; operators must register
    one sentinel per brand. We document this in the YAML.
    """
    if brands is None:
        raise SentinelConfigLoadError("sentinel 'brands' field is required")
    if isinstance(brands, str):
        # Allow scalar string form for convenience.
        brands = [brands]
    if not isinstance(brands, list) or not brands:
        raise SentinelConfigLoadError("sentinel 'brands' must be a non-empty list or string")
    if len(brands) == 1 and brands[0] in {"*", "all"}:
        return "all"
    if len(brands) == 1:
        return str(brands[0])
    # Multi-brand: out of scope for this loader; admin can register one
    # sentinel per brand or use ``brands: [all]``.
    raise SentinelConfigLoadError(
        f"multi-brand sentinel registration not supported in YAML "
        f"(got brands={brands}); use ['all'] for cross-brand or register "
        f"separately via POST /api/sentinels"
    )


def _validate_yaml_entry(entry: Dict[str, Any]) -> None:
    """Validate a single YAML sentinel entry's required fields."""
    required = {"name", "trigger_type", "condition", "action"}
    missing = required - set(entry.keys())
    if missing:
        raise SentinelConfigLoadError(f"sentinel entry missing required fields: {sorted(missing)}")
    trigger = entry["trigger_type"]
    if trigger not in PLAN_TRIGGER_TO_INTERNAL_PATTERN:
        raise SentinelConfigLoadError(
            f"unknown trigger_type {trigger!r}; allowed: {sorted(PLAN_TRIGGER_TO_INTERNAL_PATTERN)}"
        )
    action = entry["action"]
    if action not in PLAN_ACTION_TASK_NAMES and action not in VALID_ACTION_TYPES:
        raise SentinelConfigLoadError(
            f"unknown action {action!r}; allowed plan-actions: "
            f"{sorted(PLAN_ACTION_TASK_NAMES)} or shipped: {sorted(VALID_ACTION_TYPES)}"
        )
    cooldown = entry.get("cooldown_minutes")
    if cooldown is not None:
        # bool exclusion before numeric check — True is a subclass of int and
        # would falsely pass `isinstance(x, int)` (load-bearing pattern).
        if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)):
            raise SentinelConfigLoadError(
                f"cooldown_minutes must be a non-negative number, "
                f"got {type(cooldown).__name__}={cooldown!r}"
            )
        if cooldown < 0:
            raise SentinelConfigLoadError(f"cooldown_minutes must be non-negative, got {cooldown}")


def _build_pattern_config(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the plan-YAML ``condition`` block into the shipped registry's
    expected pattern_config shape, per pattern_type."""
    internal = PLAN_TRIGGER_TO_INTERNAL_PATTERN[entry["trigger_type"]]
    condition = entry.get("condition") or {}
    if internal == "freshness":
        # freshness needs (table, ts_column, max_age_hours). YAML may omit
        # any of these; supply sane fallbacks for the plan-specced sentinels.
        return {
            "table": condition.get("table", "triggers"),
            "ts_column": condition.get("ts_column", "updated_at"),
            "max_age_hours": float(condition.get("max_age_hours", 24)),
        }
    if internal == "threshold_breach":
        # threshold_breach needs (table, column, op, value).
        return {
            "table": condition.get("table", "causal_paths"),
            "column": condition.get("column", "causal_effect_size"),
            "op": condition.get("op", "<"),
            "value": condition.get("value", 0.05),
        }
    if internal == "drift_score":
        return {"max_drift_score": float(condition.get("max_drift_score", 0.30))}
    if internal == "new_causal_path":
        cfg: Dict[str, Any] = {}
        if condition.get("since"):
            cfg["since"] = condition["since"]
        return cfg
    if internal == "invalidation_count":
        # M2 (#381): invalidation_count needs ``table``; we default to
        # ``executive_insights`` because it is the semantic-tier table that
        # carries ``invalidated_at`` (matches plan intent "promoted findings
        # with staleness"). ``tier`` is a human-readable label preserved for
        # operator clarity in alerts; the evaluator does not key off it.
        cfg = {
            "table": condition.get("table", "executive_insights"),
        }
        if condition.get("tier"):
            cfg["tier"] = condition["tier"]
        return cfg
    raise SentinelConfigLoadError(f"internal pattern type {internal!r} not handled")


def _build_action_config(entry: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Return (action_type, action_config) the registry will store.

    Plan actions (rerun_all_active_cohorts, notify_and_queue_reanalysis,
    flag_for_review, run_full_consolidation) are routed through the registry's
    ``dispatch_agent`` action with ``agent_name`` set to the Celery task name.
    The dispatcher emits an ``InsightSignalBus`` event, which the action
    handler subscribes to and translates into ``celery.send_task``.
    """
    plan_action = entry["action"]
    if plan_action in PLAN_ACTION_TASK_NAMES:
        return "dispatch_agent", {
            "agent_name": plan_action,
            "input": entry.get("action_config", {}),
        }
    # Shipped vocab passthrough (caller used invalidate/notify/dispatch_agent
    # directly in the YAML — non-default but supported).
    return plan_action, entry.get("action_config", {}) or {}


async def _sentinel_exists(*, name: str, brand: str) -> bool:
    """Return True if a sentinel with the same (name, brand) is already
    persisted. Used to keep the loader idempotent."""
    client = get_supabase_client()
    rows = (
        client.table("sentinels")
        .select("sentinel_id, name, brand")
        .eq("name", name)
        .eq("brand", brand)
        .limit(1)
        .execute()
        .data
    ) or []
    return bool(rows)


async def load_sentinels_from_yaml(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> int:
    """
    Load sentinels from ``path`` and register them via :func:`register_sentinel`.

    Returns the count of newly-registered sentinels (existing entries with the
    same ``(name, brand)`` are skipped). Inactive entries (``active: false``)
    are not registered.

    Raises :class:`SentinelConfigLoadError` if the file is unreadable or
    malformed.
    """
    p = Path(path)
    if not p.exists():
        raise SentinelConfigLoadError(f"sentinel config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise SentinelConfigLoadError(f"failed to parse YAML at {p}: {exc}") from exc

    if not isinstance(config, dict) or "sentinels" not in config:
        raise SentinelConfigLoadError(f"YAML at {p} missing required top-level key 'sentinels'")
    entries = config["sentinels"]
    if not isinstance(entries, list):
        raise SentinelConfigLoadError(
            f"YAML 'sentinels' must be a list, got {type(entries).__name__}"
        )

    registered = 0
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise SentinelConfigLoadError(
                f"sentinel entry must be a mapping, got {type(raw_entry).__name__}"
            )
        if raw_entry.get("active") is False:
            logger.info(f"sentinel-loader: skipping inactive entry {raw_entry.get('name')!r}")
            continue
        _validate_yaml_entry(raw_entry)
        brand = _coerce_brand_list(raw_entry.get("brands"))
        name = str(raw_entry["name"])
        if await _sentinel_exists(name=name, brand=brand):
            logger.info(f"sentinel-loader: {name!r} already registered for brand={brand}; skipping")
            continue
        internal_pattern = PLAN_TRIGGER_TO_INTERNAL_PATTERN[raw_entry["trigger_type"]]
        pattern_config = _build_pattern_config(raw_entry)
        action_type, action_config = _build_action_config(raw_entry)
        cooldown = raw_entry.get("cooldown_minutes")
        # ``cooldown_minutes`` goes on a column added by migration 023.
        # We pass it through register_sentinel below; the registry persists
        # it on the row. If a deployment hasn't run migration 023 yet, the
        # insert will simply drop the unknown column at the Postgres layer
        # (this codepath is defensive — operators should run all migrations
        # before bringing up the API).
        try:
            await register_sentinel(
                name=name,
                description=raw_entry.get("description"),
                pattern_type=internal_pattern,
                pattern_config=pattern_config,
                action_type=action_type,
                action_config=action_config,
                brand=brand,
                region=raw_entry.get("region"),
                cooldown_minutes=cooldown,
            )
        except TypeError:
            # Backward-compat: register_sentinel may not yet accept
            # ``cooldown_minutes``. Retry without it; the row will land with
            # cooldown_minutes NULL (DB default), and the dispatcher gate will
            # treat NULL as "no cooldown" → always-fire (its prior behavior).
            await register_sentinel(
                name=name,
                description=raw_entry.get("description"),
                pattern_type=internal_pattern,
                pattern_config=pattern_config,
                action_type=action_type,
                action_config=action_config,
                brand=brand,
                region=raw_entry.get("region"),
            )
        registered += 1
        # Bookkeeping for unused validations that mypy might otherwise flag —
        # we deliberately do NOT validate internal_pattern against
        # VALID_PATTERN_TYPES here because register_sentinel does that itself
        # (single-source-of-truth).
        _ = VALID_PATTERN_TYPES  # noqa: F841

    logger.info(
        f"sentinel-loader: registered {registered} new sentinel(s) from {p} "
        f"(total entries processed: {len(entries)})"
    )
    return registered
