"""Phase 6 KG role enrichment node for data_preparer.

Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §6.3.

This node runs AFTER ``compute_baseline_metrics`` and BEFORE
``finalize_output``. It reconciles each LLM-source ``RoleAttribution``
(produced by Phase 1's ``derive_role_attributions`` and persisted to
the sidecar via ``write_adaptive_verdicts_sidecar``) with a per-feature
Phase-6 KG signal (``ensemble_voter.layer_2_kg_signal``):

  - LLM attribution + ``evaluator_satisfied=True``:
      * KG corroborates (same role) → promote ``source="kg"`` and stamp
        the KG provenance into ``evaluator_model="kg:falkordb"``.
      * KG contradicts (different role) → keep ``source="llm"`` but
        set ``evaluator_satisfied=False``. Phase 2's ``_should_act``
        predicate then gates the attribution out.
      * KG silent (no Feature node) → leave attribution unchanged.

  - LLM attribution + ``evaluator_satisfied=False``: leave unchanged
    (the evaluator already rejected the LLM; KG corroboration is moot
    because C1 already excludes it).

  - Manifest-source attribution: NEVER query the KG (manifest is
    already verification-grade per the C1 trust boundary). This is the
    codex-2 §6.3 fix and the falsifiability anchor for Phase 6 case 5.

  - KG-source attribution: NEVER re-query the KG (idempotency; would be
    a tautology).

The node is non-blocking: any exception (graph unavailable, malformed
state) is logged at WARNING and the input ``role_attributions`` list
flows through unchanged. Trust-source promotion is additive — never
something the QC gate should fail on.

Falsifiability anchors (see ``tests/integration/test_falkordb_role_persistence.py``):
  - Revert the corroborate branch → case 2 trips (``source`` stays
    ``"llm"``).
  - Revert the contradict branch → case 3 trips
    (``evaluator_satisfied`` stays ``True``).
  - Remove the manifest short-circuit → case 5 trips (KG is queried
    for a manifest source).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from src.data.kg.ensemble_voter import layer_2_kg_signal

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


# Stamped into ``evaluator_model`` when KG corroborates and we promote
# ``source="kg"``. Mirrors the documented sentinel in
# ``src.data.role_attribution.RoleAttribution`` docstring ("``kg:falkordb``").
_KG_EVALUATOR_MODEL = "kg:falkordb"


async def kg_role_enrichment(
    state: DataPreparerState,
    *,
    _graph_override: Any = None,
) -> Dict[str, Any]:
    """Reconcile LLM-source role attributions with Phase-6 KG signals.

    Args:
        state: Current data_preparer state. Reads
            ``role_attributions`` (Phase-1 producer output) and
            ``experiment_id``. Phase-1 ``RoleAttribution`` rows have
            shape ``{feature, causal_role, source, evaluator_satisfied,
            evaluator_model}``.
        _graph_override: Test hook. When provided, used as the FalkorDB
            graph handle in place of the live ``get_graph()``
            dependency. Tests pass a ``_FakeGraph`` here.

    Returns:
        State patch with the (possibly mutated) ``role_attributions``
        list. Original list is never mutated in place — Phase-1's
        sidecar write happens in ``finalize_output``, but a defensive
        deep copy keeps this node side-effect-free on the input dict.
    """
    role_attributions = state.get("role_attributions") or []
    if not role_attributions:
        # No attributions to enrich. Common path on pre-Phase-1
        # callers, or on runs where no LLM verdict fired and the
        # manifest declared no causal_role.
        return {"role_attributions": list(role_attributions)}

    experiment_id = state.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        logger.warning(
            "kg_role_enrichment: missing/non-str experiment_id; skipping enrichment. Found type=%s",
            type(experiment_id).__name__,
        )
        return {"role_attributions": list(role_attributions)}

    graph = _graph_override
    if graph is None:
        graph = await _resolve_graph()
    if graph is None:
        logger.info(
            "kg_role_enrichment: FalkorDB graph unavailable; passing through "
            "%d role_attributions unchanged",
            len(role_attributions),
        )
        return {"role_attributions": list(role_attributions)}

    enriched: List[Dict[str, Any]] = []
    promoted = 0
    contradicted = 0
    silent = 0
    skipped_manifest = 0
    skipped_kg = 0
    skipped_unsatisfied = 0

    for attr in role_attributions:
        # Defensive copy so the input state dict is not mutated.
        new_attr = deepcopy(dict(attr))
        source = new_attr.get("source")

        # Manifest and KG sources short-circuit. Manifest is the codex-2
        # §6.3 falsifiability anchor for case 5.
        if source == "manifest":
            skipped_manifest += 1
            enriched.append(new_attr)
            continue
        if source == "kg":
            # Idempotency: do not re-query KG for kg-source rows.
            skipped_kg += 1
            enriched.append(new_attr)
            continue
        if source != "llm":
            # Unknown source label — pass through (forward-compat with
            # future enum additions; Phase-2 ``_should_act`` does its
            # own validation).
            enriched.append(new_attr)
            continue

        # LLM source path. Skip KG check when evaluator already
        # rejected the LLM — Phase 2 C1 already excludes the row.
        if not bool(new_attr.get("evaluator_satisfied")):
            skipped_unsatisfied += 1
            enriched.append(new_attr)
            continue

        feature = new_attr.get("feature")
        if not isinstance(feature, str) or not feature:
            enriched.append(new_attr)
            continue

        signal = layer_2_kg_signal(
            graph,
            feature=feature,
            experiment_id=experiment_id,
        )
        if signal is None:
            silent += 1
            enriched.append(new_attr)
            continue

        kg_role = signal["causal_role"]
        llm_role = new_attr.get("causal_role")
        if kg_role == llm_role:
            # Corroborate: promote source to kg. Preserve evaluator_model
            # as the upstream LLM's model id — KG is a corroborating
            # store, not an evaluator. Overwriting evaluator_model would
            # corrupt audit provenance ("which model produced this verdict?").
            new_attr["source"] = "kg"
            promoted += 1
            enriched.append(new_attr)
        else:
            # Contradict: keep LLM but downgrade evaluator_satisfied.
            new_attr["evaluator_satisfied"] = False
            contradicted += 1
            enriched.append(new_attr)

    logger.info(
        "kg_role_enrichment: experiment=%s in=%d promoted_kg=%d "
        "contradicted=%d silent=%d manifest=%d kg=%d unsatisfied=%d",
        experiment_id,
        len(role_attributions),
        promoted,
        contradicted,
        silent,
        skipped_manifest,
        skipped_kg,
        skipped_unsatisfied,
    )

    return {"role_attributions": enriched}


async def _resolve_graph() -> Optional[Any]:
    """Resolve the live FalkorDB graph handle.

    Indirected through this helper so unit tests can monkeypatch the
    boundary cleanly, and so the data_preparer node import does not
    eagerly drag in the FastAPI dependency module (which carries a
    Redis/tenacity import surface).
    """
    try:
        from src.api.dependencies.falkordb_client import get_graph

        return await get_graph()
    except Exception as exc:  # noqa: BLE001
        logger.warning("kg_role_enrichment: failed to resolve FalkorDB graph: %s", exc)
        return None
