"""Retrain -> registry linkage for the ``register_model`` node (#2242).

A retrain's candidate is a new version of the model being retrained: registered under
ITS name (MLflow numbers the version) and written to ``ml_model_registry`` as
``(model_name, retrain_of.new_model_version)`` inside ITS experiment — the pair
``ml_retraining_history`` records. ``retrain_of`` is built by the retraining trigger
(``RetrainingTriggerService.trigger_retraining``). Everything here fails closed: a
candidate that cannot be attached to the retrained model is never registered under a
generated name (the orphan this replaces).

#2310: the candidate row is inserted at stage ``'candidate'`` with ``retrain_of_id`` = the
parent (lineage: immutable, never a filter); its MLflow version stays at stage None and is
tagged ``e2i.role=candidate`` / ``e2i.retrain_of=<parent>`` (:func:`promote_candidate`) --
registered as a candidate, not promoted, no endpoint. #2311: the row records the exact
MLflow version, and a retry reuses it instead of registering an orphan
(:func:`register_or_reuse_version`). :func:`reuse_refusal` is the one statement of when an
existing row may be reused as this registration's row (both reuse paths apply it).

Split out of ``registry_manager`` (module-size ratchet).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ``model_id`` is the parent row the candidate's ``retrain_of_id`` points at (#2310).
_REQUIRED_KEYS = ("model_id", "model_name", "new_model_version", "experiment_id")

#: The deployer's environment / stage label for a retrain candidate, and the DB stage. The
#: label is never sent to MLflow: a candidate's MLflow version keeps stage None.
CANDIDATE_ENVIRONMENT = "candidate"
CANDIDATE_STAGE = "Candidate"
CANDIDATE_DB_STAGE = "candidate"
CANDIDATE_NOTE = "registered as candidate, not promoted, no endpoint"


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
        "retrain_of_id": (retrain_of or {}).get("model_id"),
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


def reuse_refusal(
    existing: Any,
    *,
    experiment_id: Any,
    run_id: Optional[str],
    retrain_of_id: Optional[str],
    mlflow_version: Optional[int],
) -> Optional[str]:
    """Why ``existing`` (same name + version) may NOT be reused as this registration's row.

    None when it may. Applied by BOTH reuse paths of the registry writer (the pre-check and
    the unique-race re-read) and, for a retrain, before MLflow is touched.
      * another experiment: a name+version collision;
      * another source run (both run ids known): a different artifact;
      * #2310 lineage: the row's ``retrain_of_id`` must equal the requested parent -- NULL
        or another parent fails closed and is never healed (a non-retrain never adopts a
        candidate row either);
      * #2311: a retrain row that records another MLflow version than the one this
        delivery registered would leave that version unlinked.
    """
    if str(existing.experiment_id) != str(experiment_id):
        return (
            f"an existing row belongs to experiment {existing.experiment_id}, not "
            f"{experiment_id} -- name+version collision"
        )
    existing_run = (existing.mlflow_run_id or "").strip() or None
    if run_id and existing_run and existing_run != run_id:
        return (
            f"the existing row was registered from run {existing_run}, this deployment "
            f"references run {run_id} -- different source run"
        )
    have = str(existing.retrain_of_id) if existing.retrain_of_id else None
    want = str(retrain_of_id) if retrain_of_id else None
    if have != want:
        return (
            f"the existing row's lineage is retrain_of_id={have}, this registration's is "
            f"{want} -- lineage mismatch (fails closed, never healed)"
        )
    if want and existing.mlflow_model_version and mlflow_version is not None:
        if int(existing.mlflow_model_version) != int(mlflow_version):
            return (
                f"the existing row records MLflow version {existing.mlflow_model_version}, "
                f"this delivery registered version {mlflow_version} -- that version is an "
                "unlinked duplicate of the same candidate"
            )
    return None


async def register_or_reuse_version(
    state: Dict[str, Any],
    retrain_of: Optional[Dict[str, Any]],
    registry_name: str,
    register: Callable[[str, str], Awaitable[Tuple[Optional[str], Optional[int], Optional[str]]]],
    get_client: Callable[[], Awaitable[Optional[Any]]],
) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[Dict[str, Any]]]:
    """``(name, version, stage, error)``: register in MLflow, or reuse a retry's version.

    Not a retrain: ``register`` as before. A retrain first reads its candidate row
    (name, new_model_version): a row this registration may not reuse fails closed BEFORE
    MLflow (no orphan version is created); a reusable row that records its MLflow version
    reuses that version (#2311 -- a redelivered job used to register a second one).
    """
    model_uri = state.get("model_uri") or ""
    if not retrain_of:
        return (*(await register(model_uri, registry_name)), None)
    from src.agents.ml_foundation.model_deployer.nodes.training_provenance import (
        _parse_mlflow_run_id,
        pinned_training_run_id,
    )
    from src.repositories.ml_experiment import MLModelRegistryRepository

    client = await get_client()
    existing = (
        await MLModelRegistryRepository(supabase_client=client).get_by_name_version(
            registry_name, retrain_of["new_model_version"]
        )
        if client is not None
        else None
    )
    if existing is not None and existing.id:
        run_id = pinned_training_run_id(_parse_mlflow_run_id(model_uri), state.get("mlflow_run_id"))
        refusal = reuse_refusal(
            existing,
            experiment_id=retrain_of["experiment_id"],
            run_id=run_id,
            retrain_of_id=retrain_of["model_id"],
            mlflow_version=None,
        )
        if refusal:
            logger.error("retrain candidate %s NOT registered: %s", registry_name, refusal)
            return (
                None,
                None,
                None,
                {
                    "error": f"retrain candidate {registry_name} "
                    f"v{retrain_of['new_model_version']} not registered: {refusal}",
                    "error_type": "retrain_candidate_reuse_refused",
                    "registration_successful": False,
                    "model_registry_id": None,
                },
            )
        if existing.mlflow_model_version:
            logger.warning(
                "retrain candidate %s v%s: reusing MLflow version %s recorded on row %s "
                "(an earlier delivery registered it); no new MLflow version created",
                registry_name,
                retrain_of["new_model_version"],
                existing.mlflow_model_version,
                existing.id,
            )
            return registry_name, int(existing.mlflow_model_version), "None", None
    return (*(await register(model_uri, registry_name)), None)


async def promote_candidate(
    state: Dict[str, Any], tag: Callable[[str, int, Dict[str, str]], Awaitable[bool]]
) -> Dict[str, Any]:
    """``promote_stage`` for the candidate target (#2310): tag, never transition.

    The MLflow version keeps stage None; it is tagged with its role and parent. Success is
    the tag landing -- a retrain whose MLflow version is not marked as a candidate has not
    delivered one. The target requires a retrain (a parent to point at).
    """
    from datetime import datetime

    name, version = state.get("registered_model_name"), state.get("model_version")
    parent = (state.get("retrain_of") or {}).get("model_id")
    current = state.get("current_stage", "None")
    tagged = False
    if parent and name and version:
        tagged = await tag(
            name, int(version), {"e2i.role": CANDIDATE_DB_STAGE, "e2i.retrain_of": str(parent)}
        )
    reason = (
        f"{name} v{version} {CANDIDATE_NOTE}"
        if tagged
        else f"candidate {name} v{version} not tagged in MLflow (parent={parent!r})"
    )
    if not tagged:
        logger.error("Candidate registration incomplete: %s", reason)
    return {
        "previous_stage": current,
        "current_stage": CANDIDATE_STAGE if tagged else current,
        "promotion_successful": tagged,
        "promotion_simulated": False,
        "promotion_reason": reason,
        "promotion_timestamp": datetime.now(tz=None).isoformat(),
        "mlflow_transition_success": False,
    }
