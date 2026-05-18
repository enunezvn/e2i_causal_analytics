"""Phase 1 of the causal-role propagation contract (Issue #237 reframe).

Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §1.1 + §1.2.

Producer-side helper that converts ``adaptive_verdicts`` (the Layer-4
LLM classifier output, already persisted) into a typed
``RoleAttribution`` list that downstream Tier-2 agents (causal_impact,
heterogeneous_optimizer) can consume.

The helper is **additive**: it does not mutate verdicts and does not
change behavior for any caller that does not read
``state.role_attributions``. Phase 2 (collider/mediator exclusion
policy) is the first consumer; until Phase 2 lands, the list is purely
audit-only.

**Trust-boundary constraint C1** (from the plan): the producer emits
``source="manifest"`` and ``source="llm"`` in this phase. KG
attributions arrive in Phase 6 via a downstream enrichment node. The
predicate Phase 2 uses to decide whether to *act* on an attribution is::

    def _should_act(attr: RoleAttribution) -> bool:
        return attr["source"] in ("manifest", "kg") or (
            attr["source"] == "llm" and attr["evaluator_satisfied"]
        )

Manifest sources are verification-grade per the declared
``FeatureContract`` (a maintainer wrote them, with KG entity codes and
explicit knowable-at timing); they bypass the LLM-evaluator gate.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, TypedDict

from src.data.feature_contract import FeatureContract

__all__ = ["RoleAttribution", "RoleAttributionSource", "derive_role_attributions"]


RoleAttributionSource = Literal["manifest", "llm", "kg"]


class RoleAttribution(TypedDict):
    """One causal-role attribution row.

    ``feature``: the feature name (matches ``adaptive_verdicts[i]["feature"]``
    and the ``FeatureContract.name`` for manifest sources).

    ``causal_role``: one of ``ancestor | confounder | mediator | collider
    | descendant | instrument`` (the six-valued Literal from
    ``src.data.causal_role_classifier.CausalRole``). Surfaced as ``str``
    on this dict so a future enum addition does not require a downstream
    coupling change.

    ``source``: the trust label. ``manifest`` (always
    ``evaluator_satisfied=True``), ``llm`` (gated on
    ``evaluator_satisfied``), ``kg`` (set by Phase 6 enrichment).

    ``evaluator_satisfied``: the C1 trust-gate. True for manifest|kg
    sources unconditionally. For llm, mirrors the verdict's
    ``evaluator_audit.satisfied``.

    ``evaluator_model``: human-readable provenance. ``"n/a"`` for
    manifest sources (no model — a maintainer wrote the contract),
    ``"kg:falkordb"`` for kg sources (Phase 6), the underlying model id
    string (e.g. ``"anthropic/claude-haiku-4-5-20251001"``) for llm.
    """

    feature: str
    causal_role: str
    source: RoleAttributionSource
    evaluator_satisfied: bool
    evaluator_model: str


# Manifest sources do not have an evaluator model — the contract itself
# is the source of authority. Use a sentinel string instead of None so
# downstream consumers can rely on ``evaluator_model`` always being a
# ``str``.
_MANIFEST_EVALUATOR_MODEL_SENTINEL = "n/a"


def derive_role_attributions(
    adaptive_verdicts: Iterable[dict[str, Any]],
    feature_contracts: dict[str, FeatureContract],
) -> list[RoleAttribution]:
    """Produce ``RoleAttribution`` rows from adaptive verdicts + manifest.

    Logic (Phase 1):

    1. Build ``manifest_role_map`` = {name: contract.causal_role for name,
       contract in feature_contracts.items() if contract.causal_role is
       not None}. Contracts without a declared ``causal_role`` are
       skipped (the manifest writer did not certify a role for that
       feature, so manifest cannot claim authority).
    2. For each feature in ``manifest_role_map``: emit
       ``source="manifest", evaluator_satisfied=True, evaluator_model="n/a",
       causal_role=manifest_role_map[name]`` — **regardless** of any LLM
       verdict for the same feature. Manifest precedence is the C1
       trust-gate.
    3. For each LLM verdict ``v`` where ``v["feature"]`` NOT in
       ``manifest_role_map`` AND ``v.get("llm_role")`` is not None:
       emit ``source="llm",
       evaluator_satisfied=bool(v.get("evaluator_satisfied")),
       evaluator_model=v.get("evaluator_model") or "<unknown>",
       causal_role=v["llm_role"]``.

       Verdicts without an ``llm_role`` (Layer-4 did not fire — the
       common case per ``adaptive_validity_check.py:2944-2952`` trigger
       conditions) produce no LLM attribution. The manifest branch
       handles them in step 2 if the feature is in the manifest;
       otherwise no attribution row is produced for that feature.

    Output ordering: manifest attributions first (in dict-iteration
    order, which is insertion order in Python 3.7+), then llm
    attributions (in input-verdict order). Stable for snapshot tests.
    """
    manifest_role_map: dict[str, str] = {
        name: contract.causal_role
        for name, contract in feature_contracts.items()
        if contract.causal_role is not None
    }

    attributions: list[RoleAttribution] = []

    # Step 2: manifest sources first, ignoring any LLM verdict for the
    # same feature.
    for feature, role in manifest_role_map.items():
        attributions.append(
            RoleAttribution(
                feature=feature,
                causal_role=role,
                source="manifest",
                evaluator_satisfied=True,
                evaluator_model=_MANIFEST_EVALUATOR_MODEL_SENTINEL,
            )
        )

    # Step 3: LLM sources for features NOT in the manifest. Bind to a
    # distinct local name (``verdict_feature``) so mypy's narrowing
    # does not collide with step 2's ``feature: str`` from the
    # ``manifest_role_map.items()`` iteration above (Any|None vs str
    # rebind warning under strict checking).
    for verdict in adaptive_verdicts:
        verdict_feature = verdict.get("feature")
        if not isinstance(verdict_feature, str):
            continue
        if verdict_feature in manifest_role_map:
            continue
        llm_role = verdict.get("llm_role")
        if not isinstance(llm_role, str) or not llm_role:
            continue
        evaluator_satisfied_raw = verdict.get("evaluator_satisfied")
        # Strict bool coercion: None / unparseable → False (the C1 gate
        # is conservative; absence of evidence is evidence of absence).
        evaluator_satisfied = (
            evaluator_satisfied_raw if isinstance(evaluator_satisfied_raw, bool) else False
        )
        evaluator_model_raw = verdict.get("evaluator_model")
        evaluator_model: str = (
            evaluator_model_raw if isinstance(evaluator_model_raw, str) else "<unknown>"
        )
        attributions.append(
            RoleAttribution(
                feature=verdict_feature,
                causal_role=llm_role,
                source="llm",
                evaluator_satisfied=evaluator_satisfied,
                evaluator_model=evaluator_model,
            )
        )

    return attributions


def should_act(attr: RoleAttribution) -> bool:
    """Phase 2 trust-gate predicate (re-exported here for testability).

    Manifest and KG sources always act; LLM sources only act when the
    evaluator independently validated the worker's rationale. This is
    the C1 constraint from the plan.
    """
    if attr["source"] in ("manifest", "kg"):
        return True
    if attr["source"] == "llm":
        return bool(attr["evaluator_satisfied"])
    return False
