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
(``registry_manager._persist_model_registry_row``) and ``RetrainingTriggerService
.trigger_retraining`` (the manual route's self-heal). Reader: the drift-monitor
connector projection -> ``check_retraining_for_all_models`` -> ``evaluate_retraining_need``.
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


async def persist_registry_cohort_contract_if_missing(
    client: Any, model_id: Optional[str], contract: Dict[str, Any]
) -> Dict[str, Any]:
    """Write the contract's values into the row's NULL contract columns only.

    Returns the ``{column: value}`` actually written (``{}`` when nothing was NULL, the
    contract has nothing to give, or the row/client is absent). Never overwrites a
    persisted value — the row is the contract of record once set.
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
    updates: Dict[str, Any] = {}
    for key, column in _KEY_TO_COLUMN.items():
        value = contract.get(key)
        if value is None or row.get(column) is not None:
            continue
        updates[column] = encode_data_source(value) if key == "data_source" else str(value)
    if not updates:
        return {}
    try:
        await client.table("ml_model_registry").update(updates).eq("id", model_id).execute()
    except Exception as e:  # noqa: BLE001 — healing is best-effort; the trigger proceeds
        logger.warning("Cohort contract for %s could not be persisted (%s)", model_id, e)
        return {}
    logger.info(
        "Persisted cohort contract onto ml_model_registry %s: %s", model_id, sorted(updates)
    )
    return updates


__all__ = [
    "REGISTRY_CONTRACT_COLUMNS",
    "contract_from_registry_row",
    "decode_data_source",
    "encode_data_source",
    "load_registry_cohort_contract",
    "merge_contracts",
    "persist_registry_cohort_contract_if_missing",
]
