"""What a pipeline candidate was trained on, as the ``register_model`` node records it.

- ``cohort_contract_from_state``: the #2207 cohort contract (migration 150 columns).
- ``candidate_training_provenance`` / ``heal_training_provenance``:
  ``ml_model_registry.training_provenance`` (#2255). The #968 promotion gate
  (``MLModelRegistryRepository.transition_stage``) refuses ``synthetic_gold ->
  production``; before #2255 the node never set the column, so a retrain on the
  synthetic-gold cohort landed NULL and passed the gate. Allowed values (migration
  083): ``synthetic_gold`` | ``real`` | ``mixed``; NULL = unknown.

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
    client: Any, row_id: str, cohort: Dict[str, Any], provenance: Optional[str]
) -> None:
    """NULL-only heals of a reused registry row: the cohort contract first (#2207), then
    the provenance, which is validated against that stored contract."""
    if cohort:
        await heal_registry_cohort_contract(client, row_id, cohort)
    if provenance:
        await heal_training_provenance(client, row_id, provenance)
