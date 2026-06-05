"""P11 / H12 + MED — gate-config DELETE, acyclicity invariant, cache integrity.

- H12: require_dag / min_algorithm_agreement were documented + defaulted GateConfig
  knobs that evaluate() NEVER read (false assurance) → DELETED. The REAL acyclicity
  guarantee (runner._remove_cycles) is pinned by an invariant test instead.
- MED: discovery cache _deserialize_result must return None (a cache miss) on a
  corrupt entry, not raise — get()'s contract is "result or None".
"""

from __future__ import annotations

import networkx as nx

from src.causal_engine.discovery.cache import DiscoveryCache
from src.causal_engine.discovery.gate import GateConfig
from src.causal_engine.discovery.runner import DiscoveryRunner


class TestGateConfigKnobsDeleted:
    def test_inert_knobs_are_gone(self):
        config = GateConfig()
        assert not hasattr(config, "require_dag")
        assert not hasattr(config, "min_algorithm_agreement")
        # The real, consumed thresholds remain.
        assert config.accept_threshold == 0.8
        assert config.min_edges == 1


class TestRemoveCyclesInvariant:
    def test_cyclic_input_yields_acyclic_output(self):
        """_remove_cycles is the real acyclicity guarantee the gate relies on."""
        runner = DiscoveryRunner()
        dag = nx.DiGraph()
        dag.add_edge("A", "B", confidence=0.9)
        dag.add_edge("B", "C", confidence=0.8)
        dag.add_edge("C", "A", confidence=0.3)  # closes a cycle; lowest confidence
        result = runner._remove_cycles(dag)
        assert nx.is_directed_acyclic_graph(result), "cycle must be broken"
        # The lowest-confidence edge in the cycle is the one removed.
        assert not result.has_edge("C", "A")


class TestCacheCorruptEntryReturnsNone:
    def test_invalid_json_returns_none(self):
        cache = DiscoveryCache()
        assert cache._deserialize_result("{ not valid json") is None

    def test_schema_drift_returns_none(self):
        cache = DiscoveryCache()
        # Edge missing the required 'source' key → reconstruction raises → miss.
        assert cache._deserialize_result('{"edges": [{"target": "B"}]}') is None

    def test_valid_entry_still_deserializes(self):
        cache = DiscoveryCache()
        out = cache._deserialize_result(
            '{"success": true, "config": {"algorithms": []}, "edges": [], "gate_confidence": 0.5}'
        )
        assert out is not None
        assert out.gate_confidence == 0.5
