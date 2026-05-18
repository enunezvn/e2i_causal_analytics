"""Phase 2 — collider/mediator exclusion policy node.

Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §2.

The node sits between ``graph_builder`` and ``estimation`` in the
causal_impact LangGraph (``src/agents/causal_impact/graph.py``). It
consumes ``state.role_attributions`` (Phase 1 producer) and, per policy,
mutates ``causal_graph.adjustment_sets`` to remove confounded-by-design
covariates that should NOT be adjusted for (colliders, mediators).

Policy semantics
================

* ``OFF`` (default): no-op. The node returns the graph unmodified and
  emits an empty log. ``adjustment_set_hash`` and
  ``adjustment_set_hash_pre_policy`` are still populated (for audit) but
  are identical.
* ``ADVISORY``: log warnings only — covariates are *kept* in
  ``adjustment_sets`` but the policy log records them with
  ``kind="warning_collider"`` / ``"warning_mediator"``.
* ``STRICT``: drop colliders + mediators from every adjustment set;
  ``kind="dropped_collider"`` / ``"dropped_mediator"`` is logged for
  each removal.

Trust-boundary gate (C1 from the plan): an attribution only *acts* when

.. code-block:: python

    attr["source"] in ("manifest", "kg")
    or (attr["source"] == "llm" and attr["evaluator_satisfied"])

Manifest sources are verification-grade (a maintainer wrote the
contract); kg sources carry FalkorDB provenance (Phase 6). LLM sources
require the Layer-4 evaluator to have independently validated the
worker's rationale.

Hash refresh contract (codex-2 B1 fix)
======================================

``dag_version_hash`` is the primary key used by
``src/repositories/expert_review.py`` (15 lookup sites). The policy node
**MUST NOT** mutate it. Instead, a separate ``adjustment_set_hash``
field is computed via :func:`compute_adjustment_set_hash` over the
post-policy ``adjustment_sets``; the pre-policy snapshot is preserved on
``adjustment_set_hash_pre_policy``. Audits prove what changed by
comparing the two; expert-review lookups remain stable.

Log cap (codex-2 fix)
=====================

The audit log is truncated via ``log[-log_cap:]`` (keep the most recent
N entries, not the first N — audit logs are temporally relevant).
``policy_log_was_truncated`` flags the eviction.

Configuration
=============

``CAUSAL_IMPACT_ADJUSTMENT_POLICY``: ``OFF`` | ``ADVISORY`` | ``STRICT``
(default ``OFF``).
``CAUSAL_IMPACT_POLICY_LOG_CAP``: positive int (default 100).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Literal, TypedDict, cast

from src.agents.causal_impact.state import CausalGraph, CausalImpactState
from src.causal_engine.dag_hash import compute_adjustment_set_hash

logger = logging.getLogger(__name__)

PolicyName = Literal["OFF", "ADVISORY", "STRICT"]

POLICY_LOG_CAP_DEFAULT: int = 100
_VALID_POLICIES: frozenset[str] = frozenset({"OFF", "ADVISORY", "STRICT"})
_DROPPABLE_ROLES: frozenset[str] = frozenset({"collider", "mediator"})

# Env var keys (per plan §2.3).
_ENV_POLICY = "CAUSAL_IMPACT_ADJUSTMENT_POLICY"
_ENV_LOG_CAP = "CAUSAL_IMPACT_POLICY_LOG_CAP"


class AdjustmentPolicyLogEntry(TypedDict, total=False):
    """One per-feature policy decision row.

    ``kind`` values:
      * ``"dropped_collider"`` / ``"dropped_mediator"`` (STRICT).
      * ``"warning_collider"`` / ``"warning_mediator"`` (ADVISORY).
    """

    kind: str
    feature: str
    causal_role: str
    source: str
    evaluator_satisfied: bool
    evaluator_model: str
    set_index: int  # which adjustment set the feature was removed from
    caveats: List[str]


def _should_act(attr: Dict[str, Any]) -> bool:
    """C1 trust-gate predicate. Mirrors ``src/data/role_attribution.py``.

    Manifest and KG sources always act; LLM sources only when the
    evaluator independently validated the worker's rationale.
    """
    source = attr.get("source")
    if source in ("manifest", "kg"):
        return True
    if source == "llm":
        return bool(attr.get("evaluator_satisfied"))
    return False


def _resolve_policy(policy: str | None) -> PolicyName:
    """Resolve an effective policy name from explicit arg, env, default.

    Order: explicit non-None arg > env var > "OFF". Unknown values fall
    back to ``"OFF"`` with a WARN log (silent demotion would hide
    config typos; loud demotion makes them debuggable).
    """
    candidate = policy if policy is not None else os.environ.get(_ENV_POLICY)
    if candidate is None:
        return "OFF"
    candidate_str = str(candidate).strip().upper()
    if candidate_str not in _VALID_POLICIES:
        logger.warning(
            "Unknown adjustment-set policy %r; falling back to OFF. Valid: OFF, ADVISORY, STRICT.",
            candidate,
        )
        return "OFF"
    return cast(PolicyName, candidate_str)


def _resolve_log_cap(log_cap: int | None) -> int:
    """Resolve effective log cap from explicit arg, env, default."""
    if log_cap is not None:
        return max(1, int(log_cap))
    raw = os.environ.get(_ENV_LOG_CAP)
    if raw is None:
        return POLICY_LOG_CAP_DEFAULT
    try:
        parsed = int(raw)
        if parsed < 1:
            raise ValueError
        return parsed
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r; falling back to %d.",
            _ENV_LOG_CAP,
            raw,
            POLICY_LOG_CAP_DEFAULT,
        )
        return POLICY_LOG_CAP_DEFAULT


def apply_role_attributions(
    causal_graph: CausalGraph,
    role_attributions: List[Dict[str, Any]],
    policy: PolicyName,
    log_cap: int = POLICY_LOG_CAP_DEFAULT,
) -> tuple[CausalGraph, List[AdjustmentPolicyLogEntry], bool]:
    """Apply the collider/mediator exclusion policy.

    Args:
        causal_graph: pre-policy ``CausalGraph``. Treated as immutable
            input; the returned graph is a fresh dict (the function
            performs a structural deep-copy of ``adjustment_sets`` before
            mutating).
        role_attributions: ``state.role_attributions``, the Phase 1
            producer output. Each row is a dict carrying ``feature``,
            ``causal_role``, ``source``, ``evaluator_satisfied``,
            ``evaluator_model``.
        policy: ``"OFF"`` | ``"ADVISORY"`` | ``"STRICT"``. Unknown
            values are silently demoted to ``"OFF"`` here (callers that
            want loud demotion go through :func:`_resolve_policy`).
        log_cap: max entries retained in the returned policy log. The
            log is truncated via ``log[-log_cap:]`` — keep the last N,
            not the first N. Codex-2 fix.

    Returns:
        ``(post_policy_graph, log, mutated)``. The graph carries new
        ``adjustment_set_hash`` and ``adjustment_set_hash_pre_policy``
        fields; ``dag_version_hash`` is NEVER touched (expert-review
        lookups depend on it remaining stable). ``mutated`` is True iff
        STRICT actually dropped at least one feature.
    """
    # Shallow-copy the dict but deep-copy adjustment_sets so we don't
    # mutate caller-shared lists.
    post: CausalGraph = cast(CausalGraph, dict(causal_graph))
    raw_sets = post.get("adjustment_sets") or []
    pre_sets: List[List[str]] = [list(s) for s in raw_sets]
    new_sets: List[List[str]] = [list(s) for s in pre_sets]

    # Always snapshot the pre-policy hash, even on OFF — audit needs it.
    pre_hash = compute_adjustment_set_hash(pre_sets)
    post["adjustment_set_hash_pre_policy"] = pre_hash

    if policy not in _VALID_POLICIES:
        logger.warning(
            "apply_role_attributions: unknown policy %r — treating as OFF.",
            policy,
        )
        policy = "OFF"

    log: List[AdjustmentPolicyLogEntry] = []
    mutated = False

    if policy == "OFF":
        post["adjustment_set_hash"] = pre_hash
        post["adjustment_sets"] = new_sets
        return post, log, mutated

    # Build a feature → attribution map for O(1) lookup. Manifest
    # producer ordering means a single feature has at most one
    # attribution (manifest|kg|llm), so a dict-keyed-by-feature is safe.
    attr_by_feature: Dict[str, Dict[str, Any]] = {}
    for attr in role_attributions:
        if not isinstance(attr, dict):
            continue
        feature = attr.get("feature")
        if not isinstance(feature, str):
            continue
        # First attribution wins (matches producer ordering: manifest
        # before llm). A duplicate manifest+llm row would have been
        # collapsed in the producer; this is defensive.
        attr_by_feature.setdefault(feature, attr)

    for set_idx, adj_set in enumerate(new_sets):
        survivors: List[str] = []
        for feature in adj_set:
            attr = attr_by_feature.get(feature)
            if attr is None:
                survivors.append(feature)
                continue
            causal_role = attr.get("causal_role")
            if causal_role not in _DROPPABLE_ROLES:
                survivors.append(feature)
                continue
            if not _should_act(attr):
                survivors.append(feature)
                continue

            kind_prefix = "dropped" if policy == "STRICT" else "warning"
            entry: AdjustmentPolicyLogEntry = {
                "kind": f"{kind_prefix}_{causal_role}",
                "feature": feature,
                "causal_role": str(causal_role),
                "source": str(attr.get("source", "")),
                "evaluator_satisfied": bool(attr.get("evaluator_satisfied")),
                "evaluator_model": str(attr.get("evaluator_model", "")),
                "set_index": set_idx,
                "caveats": [
                    # M-bias surfaces when conditioning on a collider's
                    # ancestor; dropping the collider can re-open or
                    # close M-paths. Track via S13 research spinoff.
                    "Dropping a collider can introduce M-bias depending on"
                    " upstream graph structure (S13 research)."
                ]
                if causal_role == "collider"
                else [],
            }
            log.append(entry)

            if policy == "STRICT":
                mutated = True
                # Skip the feature (drop from adjustment set).
                continue
            survivors.append(feature)
        new_sets[set_idx] = survivors

    log_was_truncated = False
    if len(log) > log_cap:
        log = log[-log_cap:]
        log_was_truncated = True

    post["adjustment_sets"] = new_sets

    if mutated:
        post["adjustment_set_hash"] = compute_adjustment_set_hash(new_sets)
    else:
        post["adjustment_set_hash"] = pre_hash

    # Surface log-truncation as a sibling key on the graph for the
    # traced node wrapper to forward into state.
    if log_was_truncated:
        post["policy_log_was_truncated"] = True  # type: ignore[typeddict-unknown-key]

    return post, log, mutated


async def apply_adjustment_set_policy(state: CausalImpactState) -> Dict[str, Any]:
    """LangGraph node: apply Phase 2 policy and emit a state update.

    Reads ``state.causal_graph`` + ``state.role_attributions``; writes
    back a partial state dict carrying the updated graph (new hashes),
    the per-feature policy log, the truncation flag, and a latency.

    Errors are isolated: a malformed input does NOT crash the workflow;
    instead an ``adjustment_set_policy_error`` is recorded and the
    pre-policy graph is passed through. This matches the existing
    pattern of other causal_impact nodes (see ``estimation.py``).
    """

    start_time = time.time()

    policy = _resolve_policy(None)
    log_cap = _resolve_log_cap(None)

    causal_graph = state.get("causal_graph")
    if not isinstance(causal_graph, dict):
        return {
            "current_phase": "graph_building",
            "adjustment_set_policy_error": "missing causal_graph in state",
            "adjustment_set_policy_latency_ms": (time.time() - start_time) * 1000.0,
            "errors": [
                {
                    "phase": "adjustment_set_policy",
                    "message": "missing causal_graph in state",
                }
            ],
        }

    raw_attrs = state.get("role_attributions") or []
    if not isinstance(raw_attrs, list):
        raw_attrs = []
    role_attributions = [a for a in raw_attrs if isinstance(a, dict)]

    try:
        post_graph, log, mutated = apply_role_attributions(
            cast(CausalGraph, causal_graph),
            role_attributions,
            policy,
            log_cap=log_cap,
        )
    except Exception as exc:  # defensive: should not raise on validated input
        logger.exception("adjustment_set_policy failed; passing through.")
        return {
            "current_phase": "graph_building",
            "adjustment_set_policy_error": f"{type(exc).__name__}: {exc}",
            "adjustment_set_policy_latency_ms": (time.time() - start_time) * 1000.0,
            "errors": [
                {
                    "phase": "adjustment_set_policy",
                    "message": str(exc),
                }
            ],
        }

    log_was_truncated = bool(post_graph.pop("policy_log_was_truncated", False))  # type: ignore[misc]

    latency_ms = (time.time() - start_time) * 1000.0

    if mutated:
        logger.info(
            "adjustment_set_policy: STRICT dropped %d features from "
            "adjustment_sets; pre=%s post=%s",
            sum(1 for entry in log if entry["kind"].startswith("dropped_")),
            post_graph["adjustment_set_hash_pre_policy"][:12],
            post_graph["adjustment_set_hash"][:12],
        )

    return {
        "causal_graph": post_graph,
        "policy_log": log,
        "policy_log_was_truncated": log_was_truncated,
        "adjustment_set_policy_latency_ms": latency_ms,
        "current_phase": "graph_building",
    }


__all__ = [
    "AdjustmentPolicyLogEntry",
    "POLICY_LOG_CAP_DEFAULT",
    "apply_adjustment_set_policy",
    "apply_role_attributions",
]
