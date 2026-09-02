"""Bootstrap edge stability: data model, hashing, cache round-trip, runner.

Fix 2 of the causal-DAG grading sequence (gate corroboration). The runner
tests live here too (Task 3) so the whole bootstrap seam is one file.
"""

import json

from src.causal_engine.discovery.base import (
    DiscoveredEdge,
    DiscoveryConfig,
)
from src.causal_engine.discovery.cache import DiscoveryCache
from src.causal_engine.discovery.hasher import hash_config


class TestBootstrapStabilityModel:
    def test_edge_defaults_to_no_stability(self) -> None:
        edge = DiscoveredEdge(source="a", target="b")
        assert edge.bootstrap_stability is None

    def test_edge_to_dict_carries_stability(self) -> None:
        edge = DiscoveredEdge(source="a", target="b", bootstrap_stability=0.85)
        assert edge.to_dict()["bootstrap_stability"] == 0.85

    def test_config_defaults_bootstrap_off_and_serializes(self) -> None:
        config = DiscoveryConfig()
        assert config.bootstrap_resamples == 0
        assert config.to_dict()["bootstrap_resamples"] == 0

    def test_bootstrap_resamples_changes_config_hash(self) -> None:
        assert hash_config(DiscoveryConfig()) != hash_config(
            DiscoveryConfig(bootstrap_resamples=20)
        )

    def test_cache_round_trips_stability(self) -> None:
        cache = DiscoveryCache()
        edge = DiscoveredEdge(source="a", target="b", bootstrap_stability=0.4)
        payload = json.dumps(
            {
                "success": True,
                "config": DiscoveryConfig().to_dict(),
                "edges": [edge.to_dict()],
                "gate_decision": None,
                "gate_confidence": 0.0,
                "created_at": "2026-09-01T00:00:00+00:00",
                "session_id": None,
                "metadata": {},
            }
        )
        restored = cache._deserialize_result(payload)
        assert restored is not None
        assert restored.edges[0].bootstrap_stability == 0.4
