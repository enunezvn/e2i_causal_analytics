"""
Unit tests for src/ontology/grafiti_config.py

Tests for Graphity configuration for FalkorDB including:
- Enum types (EdgeGroupingStrategy, CacheEvictionPolicy)
- Dataclasses (EdgeGroupConfig, CacheConfig, HubDetectionConfig, GraphityConfig)
- GraphityConfigBuilder builder pattern
- GraphityOptimizer runtime optimizer
- Factory function create_e2i_graphity_config
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from src.ontology.grafiti_config import (
    EdgeGroupingStrategy,
    CacheEvictionPolicy,
    EdgeGroupConfig,
    CacheConfig,
    HubDetectionConfig,
    GraphityConfig,
    GraphityConfigBuilder,
    GraphityOptimizer,
    E2I_TRAVERSAL_PATTERNS,
    create_e2i_graphity_config,
)


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestEdgeGroupingStrategyEnum:
    """Tests for EdgeGroupingStrategy enum."""

    def test_by_type_value(self):
        """Test BY_TYPE enum value."""
        assert EdgeGroupingStrategy.BY_TYPE.value == "by_type"

    def test_by_source_value(self):
        """Test BY_SOURCE enum value."""
        assert EdgeGroupingStrategy.BY_SOURCE.value == "by_source"

    def test_by_target_value(self):
        """Test BY_TARGET enum value."""
        assert EdgeGroupingStrategy.BY_TARGET.value == "by_target"

    def test_by_frequency_value(self):
        """Test BY_FREQUENCY enum value."""
        assert EdgeGroupingStrategy.BY_FREQUENCY.value == "by_frequency"

    def test_hybrid_value(self):
        """Test HYBRID enum value."""
        assert EdgeGroupingStrategy.HYBRID.value == "hybrid"

    def test_all_enum_members_exist(self):
        """Test that all expected enum members exist."""
        expected = {'BY_TYPE', 'BY_SOURCE', 'BY_TARGET', 'BY_FREQUENCY', 'HYBRID'}
        actual = {m.name for m in EdgeGroupingStrategy}
        assert expected == actual


class TestCacheEvictionPolicyEnum:
    """Tests for CacheEvictionPolicy enum."""

    def test_lru_value(self):
        """Test LRU enum value."""
        assert CacheEvictionPolicy.LRU.value == "lru"

    def test_lfu_value(self):
        """Test LFU enum value."""
        assert CacheEvictionPolicy.LFU.value == "lfu"

    def test_ttl_value(self):
        """Test TTL enum value."""
        assert CacheEvictionPolicy.TTL.value == "ttl"

    def test_adaptive_value(self):
        """Test ADAPTIVE enum value."""
        assert CacheEvictionPolicy.ADAPTIVE.value == "adaptive"

    def test_all_eviction_policies_exist(self):
        """Test that all expected eviction policies exist."""
        expected = {'LRU', 'LFU', 'TTL', 'ADAPTIVE'}
        actual = {m.name for m in CacheEvictionPolicy}
        assert expected == actual


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestEdgeGroupConfigDataclass:
    """Tests for EdgeGroupConfig dataclass."""

    def test_default_values(self):
        """Test EdgeGroupConfig default values."""
        config = EdgeGroupConfig()

        assert config.strategy == EdgeGroupingStrategy.BY_TYPE
        assert config.chunk_size == 1000
        assert config.min_edges_for_grouping == 100
        assert config.rebalance_threshold == 0.3

    def test_custom_values(self):
        """Test EdgeGroupConfig with custom values."""
        config = EdgeGroupConfig(
            strategy=EdgeGroupingStrategy.HYBRID,
            chunk_size=500,
            min_edges_for_grouping=50,
            rebalance_threshold=0.5
        )

        assert config.strategy == EdgeGroupingStrategy.HYBRID
        assert config.chunk_size == 500
        assert config.min_edges_for_grouping == 50
        assert config.rebalance_threshold == 0.5


class TestCacheConfigDataclass:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Test CacheConfig default values."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.eviction_policy == CacheEvictionPolicy.LRU
        assert config.max_cache_size_mb == 256
        assert config.ttl_seconds == 3600
        assert config.hot_path_caching is True
        assert config.prefetch_depth == 2

    def test_custom_values(self):
        """Test CacheConfig with custom values."""
        config = CacheConfig(
            enabled=False,
            eviction_policy=CacheEvictionPolicy.LFU,
            max_cache_size_mb=512,
            ttl_seconds=7200,
            hot_path_caching=False,
            prefetch_depth=3
        )

        assert config.enabled is False
        assert config.eviction_policy == CacheEvictionPolicy.LFU
        assert config.max_cache_size_mb == 512
        assert config.ttl_seconds == 7200


class TestHubDetectionConfigDataclass:
    """Tests for HubDetectionConfig dataclass."""

    def test_default_values(self):
        """Test HubDetectionConfig default values."""
        config = HubDetectionConfig()

        assert config.enabled is True
        assert config.min_degree_threshold == 100
        assert config.detection_method == "degree_centrality"
        assert config.update_interval_seconds == 3600

    def test_custom_values(self):
        """Test HubDetectionConfig with custom values."""
        config = HubDetectionConfig(
            enabled=False,
            min_degree_threshold=50,
            detection_method="betweenness",
            update_interval_seconds=1800
        )

        assert config.enabled is False
        assert config.min_degree_threshold == 50
        assert config.detection_method == "betweenness"


class TestGraphityConfigDataclass:
    """Tests for GraphityConfig dataclass."""

    def test_default_values(self, default_graphity_config):
        """Test GraphityConfig default values."""
        assert default_graphity_config.enabled is True
        assert isinstance(default_graphity_config.edge_grouping, EdgeGroupConfig)
        assert isinstance(default_graphity_config.caching, CacheConfig)
        assert isinstance(default_graphity_config.hub_detection, HubDetectionConfig)

    def test_e2i_patterns_loaded_by_default(self, default_graphity_config):
        """Test that E2I traversal patterns are loaded by default."""
        # __post_init__ should populate traversal_patterns
        assert len(default_graphity_config.traversal_patterns) == len(E2I_TRAVERSAL_PATTERNS)

    def test_to_dict_serialization(self, default_graphity_config):
        """Test to_dict serialization."""
        config_dict = default_graphity_config.to_dict()

        assert 'enabled' in config_dict
        assert 'edge_grouping' in config_dict
        assert 'caching' in config_dict
        assert 'hub_detection' in config_dict
        assert 'traversal_patterns' in config_dict
        assert 'index_hints' in config_dict
        assert 'metadata' in config_dict

    def test_to_dict_edge_grouping_values(self, default_graphity_config):
        """Test edge grouping values in serialized dict."""
        config_dict = default_graphity_config.to_dict()

        assert config_dict['edge_grouping']['strategy'] == 'by_type'
        assert config_dict['edge_grouping']['chunk_size'] == 1000

    def test_to_dict_caching_values(self, default_graphity_config):
        """Test caching values in serialized dict."""
        config_dict = default_graphity_config.to_dict()

        assert config_dict['caching']['enabled'] is True
        assert config_dict['caching']['eviction_policy'] == 'lru'
        assert config_dict['caching']['ttl_seconds'] == 3600

    def test_from_dict_roundtrip(self, graphity_config_dict):
        """Test from_dict/to_dict roundtrip."""
        config = GraphityConfig.from_dict(graphity_config_dict)
        result_dict = config.to_dict()

        assert result_dict['enabled'] == graphity_config_dict['enabled']
        assert result_dict['edge_grouping']['strategy'] == graphity_config_dict['edge_grouping']['strategy']
        assert result_dict['caching']['enabled'] == graphity_config_dict['caching']['enabled']
        assert result_dict['hub_detection']['enabled'] == graphity_config_dict['hub_detection']['enabled']

    def test_from_dict_with_defaults(self):
        """Test from_dict with minimal data uses defaults."""
        minimal_dict = {'enabled': True}
        config = GraphityConfig.from_dict(minimal_dict)

        assert config.enabled is True
        assert config.edge_grouping.strategy == EdgeGroupingStrategy.BY_TYPE
        assert config.caching.eviction_policy == CacheEvictionPolicy.LRU

    def test_apply_to_graph_disabled(self, default_graphity_config):
        """Test apply_to_graph does nothing when disabled."""
        default_graphity_config.enabled = False
        mock_client = MagicMock()

        default_graphity_config.apply_to_graph(mock_client)

        # No method calls on mock client when disabled
        mock_client.config_set.assert_not_called()

    def test_apply_to_graph_enabled(self, default_graphity_config):
        """Test apply_to_graph processes configuration."""
        mock_client = MagicMock()

        # Should not raise
        default_graphity_config.apply_to_graph(mock_client)


# =============================================================================
# BUILDER TESTS
# =============================================================================


class TestGraphityConfigBuilder:
    """Tests for GraphityConfigBuilder."""

    def test_builder_default_build(self):
        """Test building with defaults."""
        config = GraphityConfigBuilder().build()

        assert config.enabled is True
        assert isinstance(config, GraphityConfig)

    def test_builder_enabled_method(self):
        """Test enabled() method."""
        config = GraphityConfigBuilder().enabled(False).build()

        assert config.enabled is False

    def test_builder_chaining(self):
        """Test method chaining returns builder."""
        builder = GraphityConfigBuilder()

        result = builder.enabled(True)
        assert isinstance(result, GraphityConfigBuilder)

        result = builder.with_edge_grouping()
        assert isinstance(result, GraphityConfigBuilder)

        result = builder.with_caching()
        assert isinstance(result, GraphityConfigBuilder)

        result = builder.with_hub_detection()
        assert isinstance(result, GraphityConfigBuilder)

    def test_builder_edge_grouping_config(self):
        """Test with_edge_grouping configuration."""
        config = (GraphityConfigBuilder()
            .with_edge_grouping(
                strategy='by_source',
                chunk_size=500,
                min_edges=50,
                rebalance_threshold=0.4
            )
            .build())

        assert config.edge_grouping.strategy == EdgeGroupingStrategy.BY_SOURCE
        assert config.edge_grouping.chunk_size == 500
        assert config.edge_grouping.min_edges_for_grouping == 50
        assert config.edge_grouping.rebalance_threshold == 0.4

    def test_builder_caching_config(self):
        """Test with_caching configuration."""
        config = (GraphityConfigBuilder()
            .with_caching(
                enabled=True,
                eviction_policy='lfu',
                max_size_mb=512,
                ttl_seconds=1800,
                hot_path_caching=False,
                prefetch_depth=3
            )
            .build())

        assert config.caching.enabled is True
        assert config.caching.eviction_policy == CacheEvictionPolicy.LFU
        assert config.caching.max_cache_size_mb == 512
        assert config.caching.ttl_seconds == 1800
        assert config.caching.hot_path_caching is False
        assert config.caching.prefetch_depth == 3

    def test_builder_hub_detection_config(self):
        """Test with_hub_detection configuration."""
        config = (GraphityConfigBuilder()
            .with_hub_detection(
                enabled=True,
                threshold=50,
                method='betweenness',
                update_interval=1800
            )
            .build())

        assert config.hub_detection.enabled is True
        assert config.hub_detection.min_degree_threshold == 50
        assert config.hub_detection.detection_method == 'betweenness'
        assert config.hub_detection.update_interval_seconds == 1800

    def test_builder_with_e2i_patterns(self):
        """Test with_e2i_patterns adds E2I patterns."""
        config = (GraphityConfigBuilder()
            .with_e2i_patterns()
            .build())

        assert len(config.traversal_patterns) == len(E2I_TRAVERSAL_PATTERNS)
        # Check for specific E2I patterns
        pattern_names = [p['name'] for p in config.traversal_patterns]
        assert 'patient_journey' in pattern_names
        assert 'causal_chain' in pattern_names

    def test_builder_with_custom_pattern(self):
        """Test with_custom_pattern adds custom pattern."""
        config = (GraphityConfigBuilder()
            .with_custom_pattern(
                name='custom_pattern',
                pattern='(a)-[:CUSTOM]->(b)',
                frequency='high',
                description='Custom pattern'
            )
            .build())

        custom_patterns = [p for p in config.traversal_patterns if p['name'] == 'custom_pattern']
        assert len(custom_patterns) == 1
        assert custom_patterns[0]['pattern'] == '(a)-[:CUSTOM]->(b)'

    def test_builder_with_index_hints(self):
        """Test with_index_hints configuration."""
        hints = {'Patient': ['patient_id', 'region']}
        config = GraphityConfigBuilder().with_index_hints(hints).build()

        assert config.index_hints == hints

    def test_builder_with_metadata(self):
        """Test with_metadata configuration."""
        metadata = {'version': '1.0', 'author': 'test'}
        config = GraphityConfigBuilder().with_metadata(metadata).build()

        assert config.metadata == metadata

    def test_builder_full_configuration(self, custom_graphity_config):
        """Test fully configured builder."""
        # custom_graphity_config is created by fixture using builder
        assert custom_graphity_config.enabled is True
        assert custom_graphity_config.edge_grouping.chunk_size == 500
        assert custom_graphity_config.caching.ttl_seconds == 1800
        assert custom_graphity_config.hub_detection.min_degree_threshold == 50


# =============================================================================
# OPTIMIZER TESTS
# =============================================================================


class TestGraphityOptimizer:
    """Tests for GraphityOptimizer runtime optimizer."""

    @pytest.fixture
    def optimizer(self, default_graphity_config):
        """Create optimizer with mock client."""
        mock_client = MagicMock()
        return GraphityOptimizer(mock_client, default_graphity_config)

    def test_optimizer_init(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.access_patterns == {}
        assert optimizer.optimization_history == []

    def test_record_query_tracks_patterns(self, optimizer):
        """Test that record_query tracks access patterns."""
        query = "MATCH (p:Patient)-[:TREATED_BY]->(h:HCP) RETURN p, h"

        optimizer.record_query(query, 10.0)

        assert len(optimizer.access_patterns) == 1

    def test_record_query_increments_count(self, optimizer):
        """Test that repeated queries increment count."""
        query = "MATCH (p:Patient) RETURN p"

        optimizer.record_query(query, 10.0)
        optimizer.record_query(query, 15.0)
        optimizer.record_query(query, 12.0)

        # Same pattern should be counted multiple times
        assert sum(optimizer.access_patterns.values()) >= 3

    def test_extract_pattern_from_query(self, optimizer):
        """Test pattern extraction from Cypher query."""
        query = "MATCH (p:Patient)-[:TREATED_BY]->(h:HCP) WHERE p.age > 30 RETURN p"

        pattern = optimizer._extract_pattern(query)

        assert pattern is not None
        assert "Patient" in pattern
        assert "HCP" in pattern

    def test_extract_pattern_no_match(self, optimizer):
        """Test pattern extraction returns None for non-MATCH queries."""
        query = "CREATE (n:Node)"

        pattern = optimizer._extract_pattern(query)

        assert pattern is None

    def test_analyze_and_suggest_returns_list(self, optimizer):
        """Test analyze_and_suggest returns suggestions list."""
        suggestions = optimizer.analyze_and_suggest()

        assert isinstance(suggestions, list)

    def test_analyze_and_suggest_hot_patterns(self, optimizer):
        """Test suggestions for hot patterns."""
        # Record many queries to create hot pattern
        query = "MATCH (p:Patient) RETURN p"
        for _ in range(150):
            optimizer.record_query(query, 10.0)

        suggestions = optimizer.analyze_and_suggest()

        # Should have at least one suggestion
        assert len(suggestions) >= 1

    def test_auto_optimize_returns_report(self, optimizer):
        """Test auto_optimize returns report dict."""
        report = optimizer.auto_optimize()

        assert 'timestamp' in report
        assert 'patterns_analyzed' in report
        assert 'optimizations_applied' in report

    def test_auto_optimize_adds_hot_patterns(self, optimizer):
        """Test auto_optimize adds hot patterns to config."""
        # Record many queries to trigger optimization
        query = "MATCH (custom:Custom) RETURN custom"
        for _ in range(1500):
            optimizer.record_query(query, 10.0)

        initial_patterns = len(optimizer.config.traversal_patterns)
        report = optimizer.auto_optimize(threshold=1000)

        # May have added new patterns
        assert report['patterns_analyzed'] >= 1

    def test_get_stats_returns_dict(self, optimizer):
        """Test get_stats returns statistics dict."""
        stats = optimizer.get_stats()

        assert 'patterns_tracked' in stats
        assert 'total_queries' in stats
        assert 'top_patterns' in stats
        assert 'optimizations_count' in stats

    def test_get_stats_empty_initial(self, optimizer):
        """Test get_stats for fresh optimizer."""
        stats = optimizer.get_stats()

        assert stats['patterns_tracked'] == 0
        assert stats['total_queries'] == 0
        assert stats['optimizations_count'] == 0

    def test_get_stats_after_queries(self, optimizer):
        """Test get_stats after recording queries."""
        optimizer.record_query("MATCH (a) RETURN a", 10.0)
        optimizer.record_query("MATCH (b) RETURN b", 10.0)

        stats = optimizer.get_stats()

        assert stats['patterns_tracked'] >= 1
        assert stats['total_queries'] >= 2


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateE2IGraphityConfig:
    """Tests for create_e2i_graphity_config factory function."""

    def test_factory_returns_graphity_config(self):
        """Test factory returns GraphityConfig instance."""
        config = create_e2i_graphity_config()

        assert isinstance(config, GraphityConfig)

    def test_factory_default_values(self):
        """Test factory default values."""
        config = create_e2i_graphity_config()

        assert config.edge_grouping.chunk_size == 1000
        assert config.caching.ttl_seconds == 3600
        assert config.hub_detection.min_degree_threshold == 100

    def test_factory_custom_values(self):
        """Test factory with custom values."""
        config = create_e2i_graphity_config(
            edge_chunk_size=500,
            cache_ttl=1800,
            hub_threshold=50
        )

        assert config.edge_grouping.chunk_size == 500
        assert config.caching.ttl_seconds == 1800
        assert config.hub_detection.min_degree_threshold == 50

    def test_factory_includes_e2i_patterns(self):
        """Test factory includes E2I traversal patterns."""
        config = create_e2i_graphity_config()

        assert len(config.traversal_patterns) == len(E2I_TRAVERSAL_PATTERNS)

    def test_factory_includes_metadata(self):
        """Test factory includes metadata."""
        config = create_e2i_graphity_config()

        assert 'created_at' in config.metadata
        assert 'version' in config.metadata
        assert config.metadata['environment'] == 'e2i_causal_analytics'

    def test_factory_hot_path_caching_enabled(self):
        """Test factory enables hot path caching."""
        config = create_e2i_graphity_config()

        assert config.caching.hot_path_caching is True


# =============================================================================
# E2I TRAVERSAL PATTERNS TESTS
# =============================================================================


class TestE2ITraversalPatterns:
    """Tests for E2I_TRAVERSAL_PATTERNS constant."""

    def test_patterns_is_list(self):
        """Test E2I_TRAVERSAL_PATTERNS is a list."""
        assert isinstance(E2I_TRAVERSAL_PATTERNS, list)

    def test_patterns_not_empty(self):
        """Test E2I_TRAVERSAL_PATTERNS is not empty."""
        assert len(E2I_TRAVERSAL_PATTERNS) > 0

    def test_pattern_structure(self):
        """Test each pattern has required keys."""
        required_keys = {'name', 'pattern', 'frequency', 'description'}

        for pattern in E2I_TRAVERSAL_PATTERNS:
            assert all(key in pattern for key in required_keys)

    def test_pattern_frequencies_valid(self):
        """Test pattern frequencies are valid values."""
        valid_frequencies = {'high', 'medium', 'low'}

        for pattern in E2I_TRAVERSAL_PATTERNS:
            assert pattern['frequency'] in valid_frequencies

    def test_patient_journey_pattern_exists(self):
        """Test patient_journey pattern exists."""
        pattern_names = [p['name'] for p in E2I_TRAVERSAL_PATTERNS]
        assert 'patient_journey' in pattern_names

    def test_causal_chain_pattern_exists(self):
        """Test causal_chain pattern exists."""
        pattern_names = [p['name'] for p in E2I_TRAVERSAL_PATTERNS]
        assert 'causal_chain' in pattern_names

    def test_brand_analysis_pattern_exists(self):
        """Test brand_analysis pattern exists."""
        pattern_names = [p['name'] for p in E2I_TRAVERSAL_PATTERNS]
        assert 'brand_analysis' in pattern_names


# =============================================================================
# EDGE CASES
# =============================================================================


class TestGraphityConfigEdgeCases:
    """Tests for edge cases in Graphity configuration."""

    def test_from_dict_empty_nested_dicts(self):
        """Test from_dict with empty nested dictionaries."""
        data = {
            'enabled': True,
            'edge_grouping': {},
            'caching': {},
            'hub_detection': {}
        }

        config = GraphityConfig.from_dict(data)

        # Should use defaults for missing values
        assert config.edge_grouping.strategy == EdgeGroupingStrategy.BY_TYPE
        assert config.caching.eviction_policy == CacheEvictionPolicy.LRU

    def test_builder_multiple_custom_patterns(self):
        """Test builder with multiple custom patterns."""
        config = (GraphityConfigBuilder()
            .with_custom_pattern('pattern1', '(a)-[:R1]->(b)', 'high', 'First')
            .with_custom_pattern('pattern2', '(c)-[:R2]->(d)', 'medium', 'Second')
            .with_custom_pattern('pattern3', '(e)-[:R3]->(f)', 'low', 'Third')
            .build())

        custom_patterns = [p for p in config.traversal_patterns
                         if p['name'].startswith('pattern')]
        assert len(custom_patterns) == 3

    def test_optimizer_query_without_match(self, default_graphity_config):
        """Test optimizer handles queries without MATCH clause."""
        mock_client = MagicMock()
        optimizer = GraphityOptimizer(mock_client, default_graphity_config)

        # These queries don't have MATCH patterns
        optimizer.record_query("CREATE (n:Node)", 10.0)
        optimizer.record_query("RETURN 1+1", 5.0)

        # Should handle gracefully
        stats = optimizer.get_stats()
        assert stats['patterns_tracked'] == 0

    def test_config_disabled_skips_apply(self):
        """Test that disabled config skips application."""
        config = GraphityConfigBuilder().enabled(False).build()
        mock_client = MagicMock()

        config.apply_to_graph(mock_client)

        # Should not call any methods on disabled config
        assert not mock_client.method_calls
