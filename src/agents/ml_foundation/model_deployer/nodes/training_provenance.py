"""What a pipeline candidate was trained on, as the ``register_model`` node records it.

- ``cohort_contract_from_state``: the #2207 cohort contract (migration 150 columns).
- ``candidate_training_provenance`` / ``heal_training_provenance``:
  ``ml_model_registry.training_provenance`` (#2255). The #968 promotion gate
  (``MLModelRegistryRepository.transition_stage``) refuses ``synthetic_gold ->
  production``; before #2255 the node never set the column, so a retrain on the
  synthetic-gold cohort landed NULL and passed the gate. Allowed values (migration
  083): ``synthetic_gold`` | ``real`` | ``mixed``; NULL = unknown.

- ``production_gate``: ``promote_stage`` applies the same gate BEFORE MLflow moves (#2259).

Split out of ``registry_manager`` (module-size ratchet).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.services.cohort_contract import (
    heal_registry_cohort_contract,
    training_provenance_from_contract,
)

logger = logging.getLogger(__name__)


def candidate_training_provenance(state: Any) -> Optional[str]:
    """What the candidate was trained on (#2255), for the #968 promotion gate.

    Derived from the training data only: the load (``training_provenance_from_contract``
    — a table contract pinning ``is_synthetic``) plus the trainer's opt-in synthetic
    augmentation (``real`` rows + synthetic rows = ``mixed``). An unpinned load stays
    ``None`` (unknown) — never inherited from a retrain's parent (its label says nothing
    about an unpinned load's rows: real in strict mode, both on a showcase instance), so
    a retrain of a ``synthetic_gold`` parent on a REAL cohort (the remedy #968
    prescribes) is ``real``.
    """
    loaded = training_provenance_from_contract(state.get("data_source"))
    if loaded == "real" and state.get("training_augmentation_applied"):
        return "mixed"
    return loaded


async def heal_training_provenance(client: Any, row_id: str, provenance: str) -> None:
    """NULL-only heal of a reused row's ``training_provenance`` (#2255, codex r1).

    Only when the row's STORED cohort contract derives the same provenance: a redeploy
    claiming a different cohort than the artifact was trained on must not relabel it
    (a synthetic cohort healed ``real`` would pass the #968 gate). A row with no
    derivable stored contract is left NULL.
    """
    res = await (
        client.table("ml_model_registry")
        .select("cohort_data_source, training_provenance")
        .eq("id", row_id)
        .limit(1)
        .execute()
    )
    rows = getattr(res, "data", None) or []
    if not rows or rows[0].get("training_provenance"):
        return
    stored = training_provenance_from_contract(rows[0].get("cohort_data_source"))
    if stored != provenance:
        logger.warning(
            "ml_model_registry %s: training_provenance %r not healed — the stored cohort "
            "contract derives %r",
            row_id,
            provenance,
            stored,
        )
        return
    await (
        client.table("ml_model_registry")
        .update({"training_provenance": provenance})
        .eq("id", row_id)
        .is_("training_provenance", "null")
        .execute()
    )


def cohort_contract_from_state(state: Any) -> Dict[str, Any]:
    """The #2207 cohort contract the deployer state carries (None-free): the pipeline's
    ``data_source`` / ``target_outcome`` (via ``ModelDeployerAgent.run``) and the
    RESOLVED manifest source (flat field, else ``scope_spec``)."""
    scope_spec = state.get("scope_spec") or {}
    manifest = state.get("feature_manifest_source") or scope_spec.get("feature_manifest_source")
    fields = {
        "data_source": state.get("data_source"),
        "target_outcome": state.get("target_outcome"),
        "feature_manifest_source": manifest,
    }
    return {k: v for k, v in fields.items() if v is not None}


async def heal_reused_row(
    client: Any, row_id: str, cohort: Optional[Dict[str, Any]], provenance: Optional[str]
) -> None:
    """NULL-only heals of a reused registry row: the cohort contract first (#2207), then
    the provenance, which is validated against that stored contract. ``cohort`` is None
    when the caller passed no contract (``_persist_model_registry_row``'s default) —
    nothing to heal from, the provenance heal still runs."""
    if cohort:
        await heal_registry_cohort_contract(client, row_id, cohort)
    if provenance:
        await heal_training_provenance(client, row_id, provenance)


async def production_gate(state: Any, target_stage: str) -> Optional[Dict[str, Any]]:
    """The #968/#2259 gate for ``promote_stage``, applied BEFORE the MLflow transition.

    MLflow's stage is read on its own (e.g. the KPI calculator's
    ``get_latest_versions(stages=["Production", ...])``), so a promotion the registry
    would refuse must not move MLflow either. Returns the node's refusal update, or None
    when the target is not production or the registry row's provenance is promotable.
    The predicate is ``MLModelRegistryRepository.production_refusal``, the same one
    ``transition_stage`` enforces. No registry row (or no client to read it) means nothing
    proves the training data: refused.
    """
    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
        _get_async_supabase_client_or_none,
    )
    from src.repositories.ml_experiment import MLModelRegistryRepository

    if MLModelRegistryRepository.normalize_stage(target_stage) != "production":
        return None
    model_id = state.get("model_registry_id")
    client = await _get_async_supabase_client_or_none() if model_id else None
    row = await MLModelRegistryRepository(client).get_by_id(str(model_id)) if client else None
    if row is None:
        reason: Optional[str] = (
            f"Refusing to promote to production: no readable ml_model_registry row "
            f"(model_registry_id={model_id!r}), so its training_provenance is unproven (#2259)."
        )
    else:
        reason = MLModelRegistryRepository.production_refusal(model_id, row.training_provenance)
    if reason is None:
        return None
    logger.error("promote_stage REFUSED before the MLflow transition: %s", reason)
    return {
        "promotion_successful": False,
        "promotion_refused_reason": reason,
        "error": reason,
        "error_type": "promotion_refused",
        "current_stage": state.get("current_stage", "None"),
    }


def pinned_training_run_id(
    uri_run_id: Optional[str], trainer_run_id: Optional[str]
) -> Optional[str]:
    """The MLflow run the registry row must be sourced from (#2296).

    A ``runs:/`` URI's run and the trainer's own run are two views of one fact; when
    both are known they must agree — a contradiction raises (the caller fails closed)
    rather than silently preferring one. Either alone pins; neither -> ``None``.
    """
    if uri_run_id and trainer_run_id and uri_run_id != trainer_run_id:
        raise ValueError(
            f"model_uri pins run {uri_run_id!r} but the trainer reported run "
            f"{trainer_run_id!r}: contradictory provenance"
        )
    return uri_run_id or trainer_run_id
