"""Retrain -> registry linkage for the ``register_model`` node (#2242).

A retrain's candidate is a new version of the model being retrained: registered under
ITS name (MLflow numbers the version) and written to ``ml_model_registry`` as
``(model_name, retrain_of.new_model_version)`` inside ITS experiment — the pair
``ml_retraining_history`` records. ``retrain_of`` is built by the retraining trigger
(``RetrainingTriggerService.trigger_retraining``). Everything here fails closed: a
candidate that cannot be attached to the retrained model is never registered under a
generated name (the orphan this replaces).

Split out of ``registry_manager`` (module-size ratchet); behaviour unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("model_name", "new_model_version", "experiment_id")


async def resolve_retrain_registration(
    state: Dict[str, Any],
    experiment_id: Optional[str],
    deployment_name: str,
    get_client: Callable[[], Awaitable[Optional[Any]]],
) -> Tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, Any]]]:
    """``(retrain_of, registry_name, error)`` for the register node.

    Not a retrain -> ``(None, deployment_name, None)``. A partial identity, or a
    pipeline experiment that is not the retrained model's (checked BEFORE MLflow, so no
    MLflow version is created under its name — codex r1), returns the node's error dict.
    """
    retrain_of = state.get("retrain_of") or None
    if not retrain_of:
        return None, deployment_name, None
    missing = [k for k in _REQUIRED_KEYS if not retrain_of.get(k)]
    if missing:
        return (
            retrain_of,
            deployment_name,
            {
                "error": f"retrain_of is missing {missing} — candidate not registered",
                "error_type": "incomplete_retrain_identity",
                "registration_successful": False,
            },
        )
    client = await get_client()
    resolved = None
    if client is not None and experiment_id:
        from src.repositories.ml_experiment import MLExperimentRepository

        resolved = await MLExperimentRepository(supabase_client=client).get_by_mlflow_id(
            experiment_id
        )
    if not (resolved and str(resolved.id) == str(retrain_of["experiment_id"])):
        return (
            retrain_of,
            retrain_of["model_name"],
            {
                "error": (
                    f"experiment {experiment_id!r} is not the retrained model's "
                    f"experiment {retrain_of['experiment_id']} — candidate not registered"
                ),
                "error_type": "retrain_experiment_mismatch",
                "registration_successful": False,
            },
        )
    return retrain_of, retrain_of["model_name"], None


def retrain_persist_kwargs(retrain_of: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``_persist_model_registry_row`` kwargs a retrain adds (None when not one)."""
    return {
        "version_label": (retrain_of or {}).get("new_model_version"),
        "expected_experiment_id": (retrain_of or {}).get("experiment_id"),
    }


def retrain_experiment_mismatch(
    registered_model_name: str,
    row_version: str,
    experiment_id_str: str,
    resolved_experiment_id: Any,
    expected_experiment_id: Optional[str],
) -> bool:
    """The registry writer's backstop: True (logged) when the resolved experiment is not
    the retrained model's — the row must then NOT be written."""
    if not expected_experiment_id or str(resolved_experiment_id) == str(expected_experiment_id):
        return False
    logger.error(
        "ml_model_registry NOT written for '%s' v%s: experiment %r resolved to %s, "
        "not the retrained model's experiment %s (db_persisted=False)",
        registered_model_name,
        row_version,
        experiment_id_str,
        resolved_experiment_id,
        expected_experiment_id,
    )
    return True


def retrain_not_persisted_error(
    retrain_of: Dict[str, Any], registered_model_name: Optional[str]
) -> Dict[str, Any]:
    """codex r1: a retrain's deliverable IS the linked registry row; an MLflow version
    without it must not be promoted as that model."""
    return {
        "error": (
            f"retrain candidate {registered_model_name} "
            f"v{retrain_of['new_model_version']} was not written to "
            "ml_model_registry (see the fail-closed log above)"
        ),
        "error_type": "retrain_candidate_not_persisted",
        "registration_successful": False,
        "registered_model_name": registered_model_name,
        "model_registry_id": None,
    }
