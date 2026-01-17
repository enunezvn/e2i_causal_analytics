"""
E2I Causal Analytics - Graphity Configuration
Configures FalkorDB Graphity optimizations for semantic memory layer
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class EdgeGroupingStrategy(Enum):
    """Graphity edge grouping strategies"""
    BY_TYPE = "by_type"              # Group edges by relationship type
    BY_PROPERTY = "by_property"      # Group by property value
    BY_LABEL = "by_label"            # Group by target node label
    HYBRID = "hybrid"                # Combine multiple strategies


class CacheStrategy(Enum):
    """Caching strategies for hot paths"""
    LRU = "lru"                      # Least Recently Used
    LFU = "lfu"                      # Least Frequently Used
    TTL = "ttl"                      # Time To Live
    ADAPTIVE = "adaptive"            # Adapt based on usage patterns


@dataclass
class EdgeGroupConfig:
    """Configuration for edge grouping"""
    strategy: EdgeGroupingStrategy
    chunk_size: int = 1000           # Edges per chunk
    max_chunks_per_node: int = 100   # Max chunks before rebalancing
    rebalance_threshold: float = 0.8 # Rebalance when utilization exceeds
    compression_enabled: bool = True


@dataclass
class CacheConfig:
    """Configuration for path caching"""
    strategy: CacheStrategy
    hot_path_enabled: bool = True
    ttl_seconds: int = 3600
    max_cache_size_mb: int = 512
    eviction_policy: str = "lru"
    prefetch_enabled: bool = False


@dataclass
class TraversalPattern:
    """Frequently traversed graph pattern"""
    name: str
    cypher_pattern: str
    estimated_frequency: str  # 'high', 'medium', 'low'
    edge_types: List[str]
    optimization_priority: int = 1  # 1=highest
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HubNodeConfig:
    """Configuration for high-degree hub nodes"""
    entity_label: str
    degree_threshold: int  # Min edges to be considered hub
    partition_strategy: str = "range"  # 'range', 'hash', 'list'
    partition_count: int = 4
    index_required: bool = True


@dataclass
class GraphityConfig:
    """Complete Graphity optimization configuration"""
    enabled: bool = True
    edge_grouping: EdgeGroupConfig
    cache: CacheConfig
    hub_nodes: List[HubNodeConfig]
    traversal_patterns: List[TraversalPattern]
    performance_monitoring: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphityConfigBuilder:
    """
    Builds Graphity configuration optimized for E2I causal graph patterns.
    Graphity improves traversal performance through edge grouping and caching.
    """
    
    # E2I-specific entity labels that are likely to be hubs
    E2I_HUB_ENTITIES = [
        'HCP',              # Many patient relationships
        'Patient',          # Multiple interventions/outcomes
        'Intervention',     # Connected to many patients
        'CausalEstimate',   # Referenced by validations/experiments
        'Experiment'        # Links to many causal tests
    ]
    
    # E2I traversal patterns for optimization
    E2I_TRAVERSAL_PATTERNS = [
        {
            'name': 'patient_journey',
            'pattern': '(p:Patient)-[:HAS_INTERVENTION]->(i:Intervention)-[:LED_TO]->(o:Outcome)',
            'frequency': 'high',
            'edge_types': ['HAS_INTERVENTION', 'LED_TO']
        },
        {
            'name': 'hcp_attribution',
            'pattern': '(h:HCP)-[:TREATED]->(p:Patient)-[:HAS_OUTCOME]->(o:Outcome)',
            'frequency': 'high',
            'edge_types': ['TREATED', 'HAS_OUTCOME']
        },
        {
            'name': 'causal_validation',
            'pattern': '(e:CausalEstimate)-[:VALIDATED_BY]->(v:Validation)-[:USED_TEST]->(t:RefutationTest)',
            'frequency': 'medium',
            'edge_types': ['VALIDATED_BY', 'USED_TEST']
        },
        {
            'name': 'experiment_lineage',
            'pattern': '(e1:Experiment)-[:DERIVED_FROM]->(e2:Experiment)-[:USED_ESTIMATE]->(c:CausalEstimate)',
            'frequency': 'medium',
            'edge_types': ['DERIVED_FROM', 'USED_ESTIMATE']
        },
        {
            'name': 'agent_workflow',
            'pattern': '(a1:AgentActivity)-[:TRIGGERED]->(a2:AgentActivity)-[:PRODUCED]->(e:CausalEstimate)',
            'frequency': 'high',
            'edge_types': ['TRIGGERED', 'PRODUCED']
        }
    ]
    
    def __init__(self):
        """Initialize configuration builder"""
        self.config = self._build_default_config()
    
    def _build_default_config(self) -> GraphityConfig:
        """Build default E2I-optimized Graphity configuration"""
        
        # Edge grouping config
        edge_grouping = EdgeGroupConfig(
            strategy=EdgeGroupingStrategy.HYBRID,
            chunk_size=1000,
            max_chunks_per_node=100,
            rebalance_threshold=0.8,
            compression_enabled=True
        )
        
        # Cache config
        cache = CacheConfig(
            strategy=CacheStrategy.ADAPTIVE,
            hot_path_enabled=True,
            ttl_seconds=3600,
            max_cache_size_mb=512,
            eviction_policy="lru",
            prefetch_enabled=True
        )
        
        # Hub node configs
        hub_nodes = [
            HubNodeConfig(
                entity_label='HCP',
                degree_threshold=100,
                partition_strategy='range',
                partition_count=8,
                index_required=True
            ),
            HubNodeConfig(
                entity_label='Patient',
                degree_threshold=50,
                partition_strategy='hash',
                partition_count=4,
                index_required=True
            ),
            HubNodeConfig(
                entity_label='CausalEstimate',
                degree_threshold=30,
                partition_strategy='list',
                partition_count=4,
                index_required=True
            )
        ]
        
        # Traversal patterns
        traversal_patterns = []
        for idx, pattern_def in enumerate(self.E2I_TRAVERSAL_PATTERNS):
            pattern = TraversalPattern(
                name=pattern_def['name'],
                cypher_pattern=pattern_def['pattern'],
                estimated_frequency=pattern_def['frequency'],
                edge_types=pattern_def['edge_types'],
                optimization_priority=1 if pattern_def['frequency'] == 'high' else 2
            )
            traversal_patterns.append(pattern)
        
        # Performance monitoring
        performance_monitoring = {
            'enabled': True,
            'metrics': [
                'edge_scan_count',
                'chunk_access_count',
                'cache_hit_rate',
                'traversal_latency_p95',
                'rebalance_frequency'
            ],
            'alert_thresholds': {
                'cache_hit_rate_min': 0.7,
                'traversal_latency_p95_max_ms': 100,
                'rebalance_frequency_max_per_hour': 5
            }
        }
        
        return GraphityConfig(
            enabled=True,
            edge_grouping=edge_grouping,
            cache=cache,
            hub_nodes=hub_nodes,
            traversal_patterns=traversal_patterns,
            performance_monitoring=performance_monitoring,
            metadata={
                'version': '1.0',
                'optimized_for': 'E2I Causal Analytics',
                'graph_size_estimate': 'medium (10K-1M nodes)'
            }
        )
    
    def with_edge_grouping(
        self,
        strategy: EdgeGroupingStrategy,
        chunk_size: Optional[int] = None
    ) -> 'GraphityConfigBuilder':
        """
        Configure edge grouping strategy
        
        Args:
            strategy: Grouping strategy to use
            chunk_size: Optional custom chunk size
        """
        self.config.edge_grouping.strategy = strategy
        if chunk_size:
            self.config.edge_grouping.chunk_size = chunk_size
        return self
    
    def with_cache(
        self,
        strategy: CacheStrategy,
        ttl_seconds: Optional[int] = None,
        max_size_mb: Optional[int] = None
    ) -> 'GraphityConfigBuilder':
        """
        Configure caching strategy
        
        Args:
            strategy: Cache strategy to use
            ttl_seconds: Optional TTL for cached items
            max_size_mb: Optional max cache size in MB
        """
        self.config.cache.strategy = strategy
        if ttl_seconds:
            self.config.cache.ttl_seconds = ttl_seconds
        if max_size_mb:
            self.config.cache.max_cache_size_mb = max_size_mb
        return self
    
    def add_hub_entity(
        self,
        entity_label: str,
        degree_threshold: int,
        partition_count: int = 4
    ) -> 'GraphityConfigBuilder':
        """
        Add a hub entity configuration
        
        Args:
            entity_label: Entity label to configure as hub
            degree_threshold: Min degree to be considered hub
            partition_count: Number of partitions for the hub
        """
        hub = HubNodeConfig(
            entity_label=entity_label,
            degree_threshold=degree_threshold,
            partition_count=partition_count,
            index_required=True
        )
        self.config.hub_nodes.append(hub)
        return self
    
    def add_traversal_pattern(
        self,
        name: str,
        cypher_pattern: str,
        frequency: str,
        edge_types: List[str]
    ) -> 'GraphityConfigBuilder':
        """
        Add a traversal pattern to optimize
        
        Args:
            name: Pattern name
            cypher_pattern: Cypher pattern string
            frequency: 'high', 'medium', or 'low'
            edge_types: List of edge types in pattern
        """
        pattern = TraversalPattern(
            name=name,
            cypher_pattern=cypher_pattern,
            estimated_frequency=frequency,
            edge_types=edge_types,
            optimization_priority=1 if frequency == 'high' else 2
        )
        self.config.traversal_patterns.append(pattern)
        return self
    
    def disable_graphity(self) -> 'GraphityConfigBuilder':
        """Disable Graphity optimizations"""
        self.config.enabled = False
        return self
    
    def build(self) -> GraphityConfig:
        """
        Build and return the configuration
        
        Returns:
            Complete GraphityConfig
        """
        return self.config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary
        
        Returns:
            Configuration dictionary
        """
        return {
            'enabled': self.config.enabled,
            'edge_grouping': {
                'strategy': self.config.edge_grouping.strategy.value,
                'chunk_size': self.config.edge_grouping.chunk_size,
                'max_chunks_per_node': self.config.edge_grouping.max_chunks_per_node,
                'rebalance_threshold': self.config.edge_grouping.rebalance_threshold,
                'compression_enabled': self.config.edge_grouping.compression_enabled
            },
            'cache': {
                'strategy': self.config.cache.strategy.value,
                'hot_path_enabled': self.config.cache.hot_path_enabled,
                'ttl_seconds': self.config.cache.ttl_seconds,
                'max_cache_size_mb': self.config.cache.max_cache_size_mb,
                'eviction_policy': self.config.cache.eviction_policy,
                'prefetch_enabled': self.config.cache.prefetch_enabled
            },
            'hub_nodes': [
                {
                    'entity_label': hub.entity_label,
                    'degree_threshold': hub.degree_threshold,
                    'partition_strategy': hub.partition_strategy,
                    'partition_count': hub.partition_count,
                    'index_required': hub.index_required
                }
                for hub in self.config.hub_nodes
            ],
            'traversal_patterns': [
                {
                    'name': pattern.name,
                    'cypher_pattern': pattern.cypher_pattern,
                    'estimated_frequency': pattern.estimated_frequency,
                    'edge_types': pattern.edge_types,
                    'optimization_priority': pattern.optimization_priority
                }
                for pattern in self.config.traversal_patterns
            ],
            'performance_monitoring': self.config.performance_monitoring,
            'metadata': self.config.metadata
        }
    
    def to_yaml(self, filepath: Optional[Path] = None) -> str:
        """
        Export configuration as YAML
        
        Args:
            filepath: Optional path to write YAML file
            
        Returns:
            YAML string
        """
        config_dict = self.to_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(yaml_str)
            logger.info(f"Graphity config written to {filepath}")
        
        return yaml_str
    
    @classmethod
    def from_yaml(cls, filepath: Path) -> 'GraphityConfigBuilder':
        """
        Load configuration from YAML file
        
        Args:
            filepath: Path to YAML config file
            
        Returns:
            GraphityConfigBuilder with loaded config
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        builder = cls()
        
        # Apply loaded config
        if not data.get('enabled', True):
            builder.disable_graphity()
        
        # Edge grouping
        if 'edge_grouping' in data:
            eg = data['edge_grouping']
            builder.with_edge_grouping(
                EdgeGroupingStrategy(eg['strategy']),
                eg.get('chunk_size')
            )
        
        # Cache
        if 'cache' in data:
            cache = data['cache']
            builder.with_cache(
                CacheStrategy(cache['strategy']),
                cache.get('ttl_seconds'),
                cache.get('max_cache_size_mb')
            )
        
        # Hub nodes (clear defaults first)
        if 'hub_nodes' in data:
            builder.config.hub_nodes = []
            for hub in data['hub_nodes']:
                builder.add_hub_entity(
                    hub['entity_label'],
                    hub['degree_threshold'],
                    hub.get('partition_count', 4)
                )
        
        # Traversal patterns (clear defaults first)
        if 'traversal_patterns' in data:
            builder.config.traversal_patterns = []
            for pattern in data['traversal_patterns']:
                builder.add_traversal_pattern(
                    pattern['name'],
                    pattern['cypher_pattern'],
                    pattern['estimated_frequency'],
                    pattern['edge_types']
                )
        
        return builder


class GraphityOptimizer:
    """
    Analyzes graph structure and provides Graphity optimization recommendations
    """
    
    def __init__(self, graph_stats: Dict[str, Any]):
        """
        Initialize optimizer with graph statistics
        
        Args:
            graph_stats: Graph statistics dictionary with node/edge counts
        """
        self.stats = graph_stats
    
    def recommend_chunk_size(self) -> int:
        """
        Recommend optimal chunk size based on graph size
        
        Returns:
            Recommended chunk size
        """
        total_edges = self.stats.get('total_edges', 0)
        
        if total_edges < 10_000:
            return 500
        elif total_edges < 100_000:
            return 1000
        elif total_edges < 1_000_000:
            return 2000
        else:
            return 5000
    
    def recommend_cache_size(self) -> int:
        """
        Recommend cache size based on graph size and available memory
        
        Returns:
            Recommended cache size in MB
        """
        total_nodes = self.stats.get('total_nodes', 0)
        
        if total_nodes < 10_000:
            return 256
        elif total_nodes < 100_000:
            return 512
        elif total_nodes < 1_000_000:
            return 1024
        else:
            return 2048
    
    def identify_hub_candidates(self, degree_threshold: int = 50) -> List[Dict[str, Any]]:
        """
        Identify entity labels that should be configured as hubs
        
        Args:
            degree_threshold: Min degree to consider as hub
            
        Returns:
            List of hub candidate entities with statistics
        """
        candidates = []
        
        for label, stats in self.stats.get('node_stats', {}).items():
            avg_degree = stats.get('avg_degree', 0)
            max_degree = stats.get('max_degree', 0)
            
            if max_degree >= degree_threshold:
                candidates.append({
                    'entity_label': label,
                    'avg_degree': avg_degree,
                    'max_degree': max_degree,
                    'recommended_threshold': max(degree_threshold, avg_degree * 2),
                    'recommended_partitions': self._recommend_partitions(max_degree)
                })
        
        return sorted(candidates, key=lambda x: x['max_degree'], reverse=True)
    
    def _recommend_partitions(self, max_degree: int) -> int:
        """Recommend partition count based on max degree"""
        if max_degree < 100:
            return 2
        elif max_degree < 1000:
            return 4
        elif max_degree < 10000:
            return 8
        else:
            return 16


# Example usage and CLI
if __name__ == "__main__":
    import sys
    
    # Build default E2I configuration
    builder = GraphityConfigBuilder()
    config = builder.build()
    
    # Export to YAML
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
        builder.to_yaml(output_path)
        print(f"Graphity configuration written to {output_path}")
    else:
        print(builder.to_yaml())
    
    # Example: Custom configuration
    print("\n" + "="*70)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*70 + "\n")
    
    custom_builder = (GraphityConfigBuilder()
                     .with_edge_grouping(EdgeGroupingStrategy.BY_TYPE, chunk_size=2000)
                     .with_cache(CacheStrategy.ADAPTIVE, ttl_seconds=7200, max_size_mb=1024)
                     .add_hub_entity('Organization', degree_threshold=200, partition_count=8))
    
    print(custom_builder.to_yaml())
