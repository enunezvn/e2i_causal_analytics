"""The per-model COHORT CONTRACT of record (#2207 follow-up, owner decision 2026-09-22).

A live retrain (``execute_model_retraining`` -> ``MLFoundationPipeline.run``) must name
the committed cohort it retrains on — ``data_source`` (the Supabase table or file bundle
``data_loader`` loads) and ``target_outcome`` (the prediction target column) — and fails
closed without them. Migration 150 persists that identity on the entity the daily sweep
iterates, ``ml_model_registry``:

    cohort_data_source TEXT             table name, or the JSON of a file-source dict
    cohort_target_outcome TEXT          prediction target
    cohort_feature_manifest_source TEXT resolved Layer-5 manifest source (optional)

This module is the one place that encodes / decodes those columns, loads a row's
contract by any model handle the monitoring API accepts (registry uuid, model_version,
model_name — ``_resolve_model_id``), merges an explicit request over the persisted row
(explicit wins), and heals NULL columns from a complete contract (never overwrites).

Writers of the contract: the model_deployer's registry writer at training time
(``registry_manager._persist_model_registry_row``) and ``execute_model_retraining`` on a
COMPLETED, promotable retrain (``heal_registry_cohort_contract`` — the manual route's
self-heal, deliberately not at trigger time). Reader: the drift-monitor connector
projection -> ``check_retraining_for_all_models`` -> ``evaluate_retraining_need``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

REGISTRY_CONTRACT_COLUMNS: Tuple[str, str, str] = (
    "cohort_data_source",
    "cohort_target_outcome",
    "cohort_feature_manifest_source",
)

# contract key -> registry column
_KEY_TO_COLUMN: Dict[str, str] = {
    "data_source": "cohort_data_source",
    "target_outcome": "cohort_target_outcome",
    "feature_manifest_source": "cohort_feature_manifest_source",
}


def encode_data_source(value: Any) -> Optional[str]:
    """A table name stays a string; a file-source dict is stored as canonical JSON."""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def decode_data_source(text: Optional[str]) -> Any:
    """Inverse of :func:`encode_data_source`; a non-JSON string is returned verbatim."""
    if not text:
        return None
    if text.startswith("{"):
        try:
            decoded = json.loads(text)
        except ValueError:
            return text
        return decoded if isinstance(decoded, dict) else text
    return text


def contract_from_registry_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The None-free cohort contract a registry row carries."""
    if not row:
        return {}
    contract: Dict[str, Any] = {}
    data_source = decode_data_source(row.get("cohort_data_source"))
    if data_source:
        contract["data_source"] = data_source
    for key in ("target_outcome", "feature_manifest_source"):
        value = row.get(_KEY_TO_COLUMN[key])
        if value:
            contract[key] = value
    return contract


def merge_contracts(
    explicit: Optional[Dict[str, Any]], fallback: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """``fallback`` (the persisted row) overlaid with every non-None ``explicit`` value."""
    merged: Dict[str, Any] = {k: v for k, v in (fallback or {}).items() if v is not None}
    merged.update({k: v for k, v in (explicit or {}).items() if v is not None})
    return merged


async def load_registry_cohort_contract(
    client: Any, model_handle: Optional[str]
) -> Tuple[Optional[str], Dict[str, Any]]:
    """``(registry row id, contract)`` for a model handle; ``(None, {})`` when unregistered.

    Tolerates a pre-migration-150 schema (the select 42703s): the id still resolves and
    the contract is ``{}`` — a trigger must never fail because the columns are not there
    yet.
    """
    if client is None or not model_handle:
        return None, {}
    from src.repositories.drift_monitoring import _resolve_model_id

    model_id = await _resolve_model_id(client, model_handle)
    if not model_id:
        return None, {}
    try:
        result = await (
            client.table("ml_model_registry")
            .select("id, " + ", ".join(REGISTRY_CONTRACT_COLUMNS))
            .eq("id", model_id)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001 — pre-150 schema or transport; contract unknown
        logger.warning(
            "Cohort contract for %s could not be read (%s); treating as none", model_id, e
        )
        return model_id, {}
    rows = getattr(result, "data", None) or []
    return model_id, contract_from_registry_row(rows[0] if rows else None)


def contract_from_training_config(training_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The None-free cohort contract a retraining job's ``training_config`` carries."""
    if not training_config:
        return {}
    return {k: training_config[k] for k in _KEY_TO_COLUMN if training_config.get(k) is not None}


def _encoded(key: str, value: Any) -> str:
    return (encode_data_source(value) if key == "data_source" else str(value)) or ""


async def heal_registry_cohort_contract(
    client: Any, model_id: Optional[str], contract: Dict[str, Any]
) -> Dict[str, Any]:
    """Fill the row's NULL contract columns from ``contract`` — as a CONSISTENT unit.

    Called only with a contract that has just produced a promotable model
    (``execute_model_retraining`` on completion) or that a re-deploy of the SAME model
    carries (``registry_manager``); never at trigger time (codex r1 HIGH-2: a contract
    persisted before the job ran could heal wrongly and the sweep would then enqueue
    failing jobs).

    Rules: (1) if any column the row already carries disagrees with the contract's
    value for it, NOTHING is written — a half-filled pair ``{row's target, contract's
    source}`` would be a contract nobody ever ran; (2) every NULL column the contract
    can fill is written in ONE compare-and-set statement (``UPDATE ... SET <all> WHERE
    id = ? AND <each column> IS NULL``) — all or nothing, so two concurrent healers with
    different contracts cannot interleave into a mixed pair (codex r2 HIGH-4); (3) a
    persisted value is never overwritten. Returns the ``{column: value}`` actually
    written (``{}`` when nothing was, including the conflict and lost-race cases, which
    are logged).
    """
    if client is None or not model_id or not contract:
        return {}
    try:
        current = await (
            client.table("ml_model_registry")
            .select("id, " + ", ".join(REGISTRY_CONTRACT_COLUMNS))
            .eq("id", model_id)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Cohort contract for %s could not be read for healing (%s)", model_id, e)
        return {}
    rows = getattr(current, "data", None) or []
    if not rows:
        return {}
    row = rows[0]
    candidates: Dict[str, str] = {}
    for key, column in _KEY_TO_COLUMN.items():
        value = contract.get(key)
        if value is None:
            continue
        encoded = _encoded(key, value)
        existing = row.get(column)
        if existing is not None:
            if str(existing) != encoded:
                logger.warning(
                    "Cohort contract for %s NOT healed: %s is %r on the row but %r in the "
                    "contract that ran — refusing to compose a mixed contract",
                    model_id,
                    column,
                    existing,
                    encoded,
                )
                return {}
            continue
        candidates[column] = encoded
    if not candidates:
        return {}
    try:
        query = client.table("ml_model_registry").update(dict(candidates)).eq("id", model_id)
        for column in candidates:
            query = query.is_(column, "null")
        result = await query.execute()
    except Exception as e:  # noqa: BLE001 — healing is best-effort
        logger.warning("Cohort contract for %s not persisted (%s)", model_id, e)
        return {}
    if not getattr(result, "data", None):
        logger.warning(
            "Cohort contract for %s NOT healed: a concurrent writer filled %s first",
            model_id,
            sorted(candidates),
        )
        return {}
    logger.info("Healed cohort contract on ml_model_registry %s: %s", model_id, sorted(candidates))
    return dict(candidates)


__all__ = [
    "REGISTRY_CONTRACT_COLUMNS",
    "contract_from_registry_row",
    "decode_data_source",
    "encode_data_source",
    "contract_from_training_config",
    "heal_registry_cohort_contract",
    "load_registry_cohort_contract",
    "merge_contracts",
]
