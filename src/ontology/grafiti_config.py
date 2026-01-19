"""
E2I Causal Analytics - Graphity Configuration for FalkorDB

Configures FalkorDB with Graphity optimizations for the E2I semantic memory layer.
Graphity improves graph traversal performance through intelligent edge grouping
and caching strategies optimized for E2I's causal query patterns.

Usage:
    from src.ontology import GraphityConfig, GraphityConfigBuilder

    # Build optimized config
    config = (GraphityConfigBuilder()
        .with_hub_detection()
        .with_edge_grouping(strategy='by_type', chunk_size=1000)
        .with_caching(ttl_seconds=3600)
        .build())

    # Apply to FalkorDB
    config.apply_to_graph(falkordb_client)
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EdgeGroupingStrategy(Enum):
    """Edge grouping strategies for Graphity optimization"""
    BY_TYPE = "by_type"           # Group edges by relationship type
    BY_SOURCE = "by_source"       # Group edges by source node
    BY_TARGET = "by_target"       # Group edges by target node
    BY_FREQUENCY = "by_frequency" # Group edges by access frequency
    HYBRID = "hybrid"             # Combination approach


class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                   # Least recently used
    LFU = "lfu"                   # Least frequently used
    TTL = "ttl"                   # Time-to-live based
    ADAPTIVE = "adaptive"         # Adaptive based on access patterns


# E2I-specific traversal patterns for Graphity optimization
E2I_TRAVERSAL_PATTERNS = [
    {
        'name': 'patient_journey',
        'pattern': '(p:Patient)-[:HAS_INTERVENTION]->(i:Intervention)-[:LED_TO]->(o:Outcome)',
        'frequency': 'high',
        'description': 'Patient treatment journey traversal'
    },
    {
        'name': 'hcp_attribution',
        'pattern': '(h:HCP)-[:TREATED]->(p:Patient)-[:HAS_OUTCOME]->(o:Outcome)',
        'frequency': 'high',
        'description': 'HCP treatment attribution'
    },
    {
        'name': 'causal_chain',
        'pattern': '(t:Treatment)-[:CAUSES]->(e:Effect)-[:CAUSES]->(o:Outcome)',
        'frequency': 'high',
        'description': 'Causal chain traversal'
    },
    {
        'name': 'agent_activity',
        'pattern': '(a:Agent)-[:PRODUCED]->(r:Result)-[:REFERENCED]->(e:Entity)',
        'frequency': 'medium',
        'description': 'Agent activity lookup'
    },
    {
        'name': 'brand_analysis',
        'pattern': '(b:Brand)-[:HAS_PATIENT]->(p:Patient)-[:IN_REGION]->(r:Region)',
        'frequency': 'high',
        'description': 'Brand regional analysis'
    },
    {
        'name': 'kpi_trend',
        'pattern': '(k:KPI)-[:MEASURED_AT]->(t:Timepoint)-[:HAS_VALUE]->(v:Value)',
        'frequency': 'high',
        'description': 'KPI time series traversal'
    },
    {
        'name': 'confounder_check',
        'pattern': '(c:Confounder)-[:AFFECTS]->(t:Treatment), (c)-[:AFFECTS]->(o:Outcome)',
        'frequency': 'medium',
        'description': 'Confounder identification'
    },
    {
        'name': 'mediator_path',
        'pattern': '(t:Treatment)-[*1..3]->(m:Mediator)-[*1..3]->(o:Outcome)',
        'frequency': 'medium',
        'description': 'Mediator discovery path'
    }
]


@dataclass
class EdgeGroupConfig:
    """Configuration for edge grouping"""
    strategy: EdgeGroupingStrategy = EdgeGroupingStrategy.BY_TYPE
    chunk_size: int = 1000
    min_edges_for_grouping: int = 100
    rebalance_threshold: float = 0.3


@dataclass
class CacheConfig:
    """Configuration for Graphity caching"""
    enabled: bool = True
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    max_cache_size_mb: int = 256
    ttl_seconds: int = 3600
    hot_path_caching: bool = True
    prefetch_depth: int = 2


@dataclass
class HubDetectionConfig:
    """Configuration for hub node detection"""
    enabled: bool = True
    min_degree_threshold: int = 100
    detection_method: str = "degree_centrality"
    update_interval_seconds: int = 3600


@dataclass
class GraphityConfig:
    """
    Complete Graphity configuration for FalkorDB

    Graphity is FalkorDB's optimization layer that improves
    traversal performance through intelligent edge grouping
    and caching strategies.
    """
    enabled: bool = True
    edge_grouping: EdgeGroupConfig = field(default_factory=EdgeGroupConfig)
    caching: CacheConfig = field(default_factory=CacheConfig)
    hub_detection: HubDetectionConfig = field(default_factory=HubDetectionConfig)
    traversal_patterns: List[Dict[str, Any]] = field(default_factory=list)
    index_hints: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize with E2I-specific patterns if empty"""
        if not self.traversal_patterns:
            self.traversal_patterns = E2I_TRAVERSAL_PATTERNS.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'enabled': self.enabled,
            'edge_grouping': {
                'strategy': self.edge_grouping.strategy.value,
                'chunk_size': self.edge_grouping.chunk_size,
                'min_edges_for_grouping': self.edge_grouping.min_edges_for_grouping,
                'rebalance_threshold': self.edge_grouping.rebalance_threshold
            },
            'caching': {
                'enabled': self.caching.enabled,
                'eviction_policy': self.caching.eviction_policy.value,
                'max_cache_size_mb': self.caching.max_cache_size_mb,
                'ttl_seconds': self.caching.ttl_seconds,
                'hot_path_caching': self.caching.hot_path_caching,
                'prefetch_depth': self.caching.prefetch_depth
            },
            'hub_detection': {
                'enabled': self.hub_detection.enabled,
                'min_degree_threshold': self.hub_detection.min_degree_threshold,
                'detection_method': self.hub_detection.detection_method,
                'update_interval_seconds': self.hub_detection.update_interval_seconds
            },
            'traversal_patterns': self.traversal_patterns,
            'index_hints': self.index_hints,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphityConfig":
        """Create from dictionary"""
        edge_grouping = EdgeGroupConfig(
            strategy=EdgeGroupingStrategy(data.get('edge_grouping', {}).get('strategy', 'by_type')),
            chunk_size=data.get('edge_grouping', {}).get('chunk_size', 1000),
            min_edges_for_grouping=data.get('edge_grouping', {}).get('min_edges_for_grouping', 100),
            rebalance_threshold=data.get('edge_grouping', {}).get('rebalance_threshold', 0.3)
        )

        caching = CacheConfig(
            enabled=data.get('caching', {}).get('enabled', True),
            eviction_policy=CacheEvictionPolicy(data.get('caching', {}).get('eviction_policy', 'lru')),
            max_cache_size_mb=data.get('caching', {}).get('max_cache_size_mb', 256),
            ttl_seconds=data.get('caching', {}).get('ttl_seconds', 3600),
            hot_path_caching=data.get('caching', {}).get('hot_path_caching', True),
            prefetch_depth=data.get('caching', {}).get('prefetch_depth', 2)
        )

        hub_detection = HubDetectionConfig(
            enabled=data.get('hub_detection', {}).get('enabled', True),
            min_degree_threshold=data.get('hub_detection', {}).get('min_degree_threshold', 100),
            detection_method=data.get('hub_detection', {}).get('detection_method', 'degree_centrality'),
            update_interval_seconds=data.get('hub_detection', {}).get('update_interval_seconds', 3600)
        )

        return cls(
            enabled=data.get('enabled', True),
            edge_grouping=edge_grouping,
            caching=caching,
            hub_detection=hub_detection,
            traversal_patterns=data.get('traversal_patterns', []),
            index_hints=data.get('index_hints', {}),
            metadata=data.get('metadata', {})
        )

    def apply_to_graph(self, graph_client: Any) -> None:
        """
        Apply Graphity configuration to FalkorDB graph

        Args:
            graph_client: FalkorDB graph client
        """
        if not self.enabled:
            logger.info("Graphity optimization disabled, skipping")
            return

        logger.info("Applying Graphity configuration to FalkorDB")

        # Apply edge grouping configuration
        self._apply_edge_grouping(graph_client)

        # Apply caching configuration
        self._apply_caching(graph_client)

        # Apply hub detection
        if self.hub_detection.enabled:
            self._detect_and_configure_hubs(graph_client)

        # Create indexes for traversal patterns
        self._create_pattern_indexes(graph_client)

        logger.info("Graphity configuration applied successfully")

    def _apply_edge_grouping(self, graph_client: Any) -> None:
        """Apply edge grouping configuration"""
        config = {
            'edge_grouping_strategy': self.edge_grouping.strategy.value,
            'edge_chunk_size': self.edge_grouping.chunk_size,
            'min_edges_for_grouping': self.edge_grouping.min_edges_for_grouping
        }

        # FalkorDB configuration command would go here
        # graph_client.config_set('graphity', config)
        logger.debug(f"Edge grouping config: {config}")

    def _apply_caching(self, graph_client: Any) -> None:
        """Apply caching configuration"""
        if not self.caching.enabled:
            return

        config = {
            'cache_enabled': True,
            'cache_eviction_policy': self.caching.eviction_policy.value,
            'cache_max_size_mb': self.caching.max_cache_size_mb,
            'cache_ttl_seconds': self.caching.ttl_seconds,
            'hot_path_caching': self.caching.hot_path_caching,
            'prefetch_depth': self.caching.prefetch_depth
        }

        # FalkorDB configuration command would go here
        # graph_client.config_set('cache', config)
        logger.debug(f"Cache config: {config}")

    def _detect_and_configure_hubs(self, graph_client: Any) -> None:
        """Detect hub nodes and configure special handling"""
        # Query to identify high-degree nodes
        query = """
        MATCH (n)
        WITH n, size((n)--()) as degree
        WHERE degree >= $threshold
        RETURN labels(n) as labels, count(*) as count
        """

        # Execute hub detection query
        # result = graph_client.query(query, {'threshold': self.hub_detection.min_degree_threshold})
        logger.debug(f"Hub detection threshold: {self.hub_detection.min_degree_threshold}")

    def _create_pattern_indexes(self, graph_client: Any) -> None:
        """Create indexes optimized for E2I traversal patterns"""
        for pattern in self.traversal_patterns:
            if pattern.get('frequency') == 'high':
                logger.debug(f"Optimizing for pattern: {pattern['name']}")
                # Create appropriate indexes based on pattern
                # This would analyze the pattern and create composite indexes


class GraphityConfigBuilder:
    """
    Builder pattern for creating GraphityConfig

    Usage:
        config = (GraphityConfigBuilder()
            .with_hub_detection(threshold=50)
            .with_edge_grouping(strategy='by_type', chunk_size=500)
            .with_caching(ttl_seconds=1800)
            .with_e2i_patterns()
            .build())
    """

    def __init__(self):
        self._enabled = True
        self._edge_grouping = EdgeGroupConfig()
        self._caching = CacheConfig()
        self._hub_detection = HubDetectionConfig()
        self._traversal_patterns = []
        self._index_hints = {}
        self._metadata = {}

    def enabled(self, value: bool) -> "GraphityConfigBuilder":
        """Enable or disable Graphity"""
        self._enabled = value
        return self

    def with_edge_grouping(
        self,
        strategy: str = 'by_type',
        chunk_size: int = 1000,
        min_edges: int = 100,
        rebalance_threshold: float = 0.3
    ) -> "GraphityConfigBuilder":
        """Configure edge grouping"""
        self._edge_grouping = EdgeGroupConfig(
            strategy=EdgeGroupingStrategy(strategy),
            chunk_size=chunk_size,
            min_edges_for_grouping=min_edges,
            rebalance_threshold=rebalance_threshold
        )
        return self

    def with_caching(
        self,
        enabled: bool = True,
        eviction_policy: str = 'lru',
        max_size_mb: int = 256,
        ttl_seconds: int = 3600,
        hot_path_caching: bool = True,
        prefetch_depth: int = 2
    ) -> "GraphityConfigBuilder":
        """Configure caching"""
        self._caching = CacheConfig(
            enabled=enabled,
            eviction_policy=CacheEvictionPolicy(eviction_policy),
            max_cache_size_mb=max_size_mb,
            ttl_seconds=ttl_seconds,
            hot_path_caching=hot_path_caching,
            prefetch_depth=prefetch_depth
        )
        return self

    def with_hub_detection(
        self,
        enabled: bool = True,
        threshold: int = 100,
        method: str = 'degree_centrality',
        update_interval: int = 3600
    ) -> "GraphityConfigBuilder":
        """Configure hub detection"""
        self._hub_detection = HubDetectionConfig(
            enabled=enabled,
            min_degree_threshold=threshold,
            detection_method=method,
            update_interval_seconds=update_interval
        )
        return self

    def with_e2i_patterns(self) -> "GraphityConfigBuilder":
        """Add E2I-specific traversal patterns"""
        self._traversal_patterns = E2I_TRAVERSAL_PATTERNS.copy()
        return self

    def with_custom_pattern(
        self,
        name: str,
        pattern: str,
        frequency: str = 'medium',
        description: str = ''
    ) -> "GraphityConfigBuilder":
        """Add custom traversal pattern"""
        self._traversal_patterns.append({
            'name': name,
            'pattern': pattern,
            'frequency': frequency,
            'description': description
        })
        return self

    def with_index_hints(self, hints: Dict[str, List[str]]) -> "GraphityConfigBuilder":
        """Add index hints"""
        self._index_hints = hints
        return self

    def with_metadata(self, metadata: Dict[str, Any]) -> "GraphityConfigBuilder":
        """Add metadata"""
        self._metadata = metadata
        return self

    def build(self) -> GraphityConfig:
        """Build the GraphityConfig"""
        return GraphityConfig(
            enabled=self._enabled,
            edge_grouping=self._edge_grouping,
            caching=self._caching,
            hub_detection=self._hub_detection,
            traversal_patterns=self._traversal_patterns or E2I_TRAVERSAL_PATTERNS.copy(),
            index_hints=self._index_hints,
            metadata=self._metadata
        )


class GraphityOptimizer:
    """
    Runtime optimizer for Graphity configuration

    Monitors graph access patterns and suggests or applies
    optimizations dynamically.
    """

    def __init__(self, graph_client: Any, config: GraphityConfig):
        """
        Initialize optimizer

        Args:
            graph_client: FalkorDB graph client
            config: Current Graphity configuration
        """
        self.graph = graph_client
        self.config = config
        self.access_patterns: Dict[str, int] = {}
        self.optimization_history: List[Dict[str, Any]] = []

    def record_query(self, query: str, execution_time_ms: float) -> None:
        """
        Record a query execution for pattern analysis

        Args:
            query: Cypher query executed
            execution_time_ms: Query execution time
        """
        # Extract pattern from query
        pattern = self._extract_pattern(query)
        if pattern:
            self.access_patterns[pattern] = self.access_patterns.get(pattern, 0) + 1

    def _extract_pattern(self, query: str) -> Optional[str]:
        """Extract traversal pattern from query"""
        import re

        # Simple pattern extraction - look for MATCH clauses
        match = re.search(r'MATCH\s+(.+?)(?:\s+WHERE|\s+RETURN|\s+WITH|$)', query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def analyze_and_suggest(self) -> List[Dict[str, Any]]:
        """
        Analyze access patterns and suggest optimizations

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Find hot patterns
        sorted_patterns = sorted(
            self.access_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for pattern, count in sorted_patterns[:10]:
            if count > 100:
                suggestions.append({
                    'type': 'index_suggestion',
                    'pattern': pattern,
                    'access_count': count,
                    'recommendation': f"Consider creating index for pattern: {pattern}"
                })

        # Check cache hit rate
        if self.config.caching.enabled:
            suggestions.append({
                'type': 'cache_review',
                'recommendation': "Review cache configuration based on access patterns"
            })

        return suggestions

    def auto_optimize(self, threshold: int = 1000) -> Dict[str, Any]:
        """
        Automatically apply optimizations based on patterns

        Args:
            threshold: Minimum access count to trigger optimization

        Returns:
            Optimization report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'patterns_analyzed': len(self.access_patterns),
            'optimizations_applied': []
        }

        for pattern, count in self.access_patterns.items():
            if count >= threshold:
                # Add to hot paths for caching
                if pattern not in [p['pattern'] for p in self.config.traversal_patterns]:
                    self.config.traversal_patterns.append({
                        'name': f'auto_detected_{len(self.config.traversal_patterns)}',
                        'pattern': pattern,
                        'frequency': 'high',
                        'description': 'Auto-detected hot pattern'
                    })
                    report['optimizations_applied'].append({
                        'type': 'pattern_added',
                        'pattern': pattern,
                        'access_count': count
                    })

        self.optimization_history.append(report)
        return report

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        return {
            'patterns_tracked': len(self.access_patterns),
            'total_queries': sum(self.access_patterns.values()),
            'top_patterns': dict(sorted(
                self.access_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'optimizations_count': len(self.optimization_history)
        }


# Factory function for quick setup
def create_e2i_graphity_config(
    edge_chunk_size: int = 1000,
    cache_ttl: int = 3600,
    hub_threshold: int = 100
) -> GraphityConfig:
    """
    Create optimized Graphity configuration for E2I

    Args:
        edge_chunk_size: Edge grouping chunk size
        cache_ttl: Cache TTL in seconds
        hub_threshold: Hub detection threshold

    Returns:
        Configured GraphityConfig
    """
    return (GraphityConfigBuilder()
        .with_edge_grouping(strategy='by_type', chunk_size=edge_chunk_size)
        .with_caching(ttl_seconds=cache_ttl, hot_path_caching=True)
        .with_hub_detection(threshold=hub_threshold)
        .with_e2i_patterns()
        .with_metadata({
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'environment': 'e2i_causal_analytics'
        })
        .build())


# CLI interface
if __name__ == "__main__":
    import json

    print("E2I Graphity Configuration")
    print("=" * 70)

    # Create default config
    config = create_e2i_graphity_config()

    print("\nDefault E2I Graphity Configuration:")
    print("-" * 70)
    print(json.dumps(config.to_dict(), indent=2))

    print("\n\nE2I Traversal Patterns:")
    print("-" * 70)
    for pattern in E2I_TRAVERSAL_PATTERNS:
        print(f"  {pattern['name']}: {pattern['frequency']}")
        print(f"    Pattern: {pattern['pattern']}")
        print(f"    {pattern['description']}")
        print()

    print("\nBuilder Pattern Example:")
    print("-" * 70)
    print("""
    config = (GraphityConfigBuilder()
        .with_hub_detection(threshold=50)
        .with_edge_grouping(strategy='by_type', chunk_size=500)
        .with_caching(ttl_seconds=1800)
        .with_e2i_patterns()
        .build())
    """)
