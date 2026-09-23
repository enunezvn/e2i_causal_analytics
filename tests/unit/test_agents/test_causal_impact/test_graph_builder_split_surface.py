"""Lane B item 5 (PR #2230 follow-up): ``graph_builder.py`` split by concern.

The module was one 1532-line class mixing manual DAG construction, discovery
orchestration, discovery reporting, adjustment-set handling and persistence.
The split moved method bodies into sibling mixin modules; ``graph_builder``
stays the node entry (``GraphBuilderNode.execute``) and re-exports every name
importers read from it.

Pinned here, captured from ``dir()`` of the UNSPLIT module (origin/main
4974774db, before any move) so the pin is the real surface, not a guess:

* every module-level name importers could read from ``graph_builder``;
* every attribute of ``GraphBuilderNode`` (own or inherited);
* the two names tests monkeypatch AS MODULE GLOBALS of ``graph_builder``
  (``satisfies_backdoor_criterion``, ``_build_discovered_dag_repository``)
  are still resolved from ``graph_builder``'s namespace by their callers —
  a move of those callers would leave the patches pointing at the wrong
  module while the tests stay green;
* no sibling module imports ``graph_builder`` (no import cycle).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import src.agents.causal_impact.nodes.graph_builder as graph_builder
from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode

# dir(graph_builder) minus dunders, on the unsplit module.
MODULE_SURFACE = [
    "ADJUSTMENT_SEARCH_MAX_CANDIDATES",
    "ADJUSTMENT_SEARCH_TIME_BUDGET_S",
    "Any",
    "CausalGraph",
    "CausalImpactState",
    "CausalPriorKnowledge",
    "DEFAULT_DISCOVERY_ALGORITHMS",
    "DEFAULT_DISCOVERY_ALGORITHM_NAMES",
    "DISCOVERY_BOOTSTRAP_RESAMPLES",
    "DISCOVERY_MAX_COVARIATES",
    "DISCOVERY_MIN_RESAMPLES",
    "DISCOVERY_TIME_BUDGET_S",
    "Dict",
    "DiscoveryAlgorithmType",
    "DiscoveryConfig",
    "DiscoveryGate",
    "DiscoveryGateDecision",
    "DiscoveryResult",
    "DiscoveryRunner",
    "GraphBuilderNode",
    "List",
    "Literal",
    "Optional",
    "Set",
    "Tuple",
    "_build_discovered_dag_repository",
    "_persist_discovered_dag",
    "asyncio",
    "build_causal_graph",
    "cast",
    "coerce_session_uuid",
    "compute_dag_hash",
    "find_adjustment_sets",
    "logger",
    "logging",
    "nx",
    "pd",
    "preflight_discovery_frame",
    "preflight_summary",
    "satisfies_backdoor_criterion",
    "spread_safe",
    "time",
]

# vars(GraphBuilderNode) minus dunders, on the unsplit class.
NODE_SURFACE = [
    "KNOWN_CAUSAL_RELATIONSHIPS",
    "_add_curated_confounder_edges",
    "_annotate_latent_diagnostic",
    "_annotate_required_edges",
    "_apply_adjustment_guarantee",
    "_build_dag_with_discovery",
    "_compute_edge_provenance",
    "_construct_dag",
    "_discovery_honesty_warnings",
    "_find_adjustment_sets",
    "_infer_variables_from_query",
    "_latent_confounding_warning",
    "_preflight_removed",
    "_resolve_anchored_confounders",
    "_run_discovery",
    "_satisfies_backdoor_criterion",
    "_to_dot_format",
    "discovery_gate",
    "discovery_runner",
    "execute",
]

SIBLINGS = ("manual_dag.py", "discovery_orchestration.py", "discovery_reporting.py")


@pytest.mark.parametrize("name", MODULE_SURFACE)
def test_module_level_name_still_importable(name):
    assert hasattr(graph_builder, name), f"graph_builder.{name} is no longer importable"


@pytest.mark.parametrize("name", NODE_SURFACE)
def test_node_attribute_still_present(name):
    assert hasattr(GraphBuilderNode, name), f"GraphBuilderNode.{name} is gone"


def test_constants_keep_their_values():
    assert graph_builder.DISCOVERY_BOOTSTRAP_RESAMPLES == 20
    assert graph_builder.DISCOVERY_MAX_COVARIATES == 20
    assert graph_builder.DISCOVERY_TIME_BUDGET_S == 180.0
    assert graph_builder.DISCOVERY_MIN_RESAMPLES == 10
    assert len(GraphBuilderNode.KNOWN_CAUSAL_RELATIONSHIPS) == 13


def test_module_global_patch_targets_stay_in_graph_builder():
    """The callers of these two globals must be defined in ``graph_builder``
    itself: tests substitute them with ``monkeypatch.setattr(graph_builder, …)``
    and the substitution only reaches a caller whose ``__globals__`` are this
    module's."""
    node_criterion = inspect.unwrap(GraphBuilderNode._satisfies_backdoor_criterion)
    assert node_criterion.__globals__ is vars(graph_builder)
    assert "satisfies_backdoor_criterion" in node_criterion.__code__.co_names
    persist = inspect.unwrap(graph_builder._persist_discovered_dag)
    assert persist.__globals__ is vars(graph_builder)
    assert "_build_discovered_dag_repository" in persist.__code__.co_names
    # execute resolves the persistence step through this module too.
    execute = inspect.unwrap(GraphBuilderNode.execute)
    assert execute.__globals__ is vars(graph_builder)
    assert "_persist_discovered_dag" in execute.__code__.co_names


def test_siblings_do_not_import_graph_builder():
    nodes_dir = Path(graph_builder.__file__).resolve().parent
    for sibling in SIBLINGS:
        source = (nodes_dir / sibling).read_text(encoding="utf-8")
        assert "nodes.graph_builder" not in source, f"{sibling} imports graph_builder (cycle)"
        assert "import graph_builder" not in source, f"{sibling} imports graph_builder (cycle)"
