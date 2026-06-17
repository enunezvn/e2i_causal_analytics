"""Translate domain priors into a causal-learn ``BackgroundKnowledge``.

Guided discovery: observational structure learning only recovers a Markov
equivalence class, so edge ORIENTATION is underdetermined and unconstrained
PC/GES can emit implausibly-directed edges (e.g. ``treatment -> confounder``).
A :class:`~src.causal_engine.discovery.base.CausalPriorKnowledge` anchors what is
KNOWN (tiers / required / forbidden edges); this module converts it into the
object causal-learn's PC accepts so the DATA still selects the rest of the
structure under those constraints.

Validated on the patient_journeys gold standard: tiers
``[[confounders], [treatment], [outcome]]`` recover ``confounder -> treatment``
and ``confounder -> outcome`` (no reversed edges) and data-drivenly drop a
non-confounder covariate, whereas unconstrained PC reversed those edges.
"""

from __future__ import annotations

from typing import Any, List

from .base import CausalPriorKnowledge


def build_background_knowledge(prior: CausalPriorKnowledge, node_names: List[str]) -> Any:
    """Build a causal-learn ``BackgroundKnowledge`` from ``prior``.

    Args:
        prior: Domain priors (tiers / required / forbidden edges). Names not in
            ``node_names`` are ignored (defensive — the caller's allowlist may
            differ from the loaded frame's columns).
        node_names: Column names of the data frame PC will run on, in order. The
            same names must be passed to ``pc(..., node_names=node_names)`` so
            causal-learn matches the constraints to its internal nodes by name.

    Returns:
        A ``causallearn.utils.PCUtils.BackgroundKnowledge.BackgroundKnowledge``.

    Raises:
        ImportError: if causal-learn is not installed.
    """
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    nodes = {name: GraphNode(name) for name in node_names}
    bk = BackgroundKnowledge()

    # Tiers: a node in an earlier tier is causally prior to one in a later tier.
    for tier_idx, tier in enumerate(prior.tiers or []):
        for name in tier:
            node = nodes.get(name)
            if node is not None:
                bk.add_node_to_tier(node, tier_idx)

    # Required / forbidden directed edges.
    for src, tgt in prior.required_edges or []:
        if src in nodes and tgt in nodes:
            bk.add_required_by_node(nodes[src], nodes[tgt])
    for src, tgt in prior.forbidden_edges or []:
        if src in nodes and tgt in nodes:
            bk.add_forbidden_by_node(nodes[src], nodes[tgt])

    return bk
