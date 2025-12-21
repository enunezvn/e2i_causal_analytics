"""
E2I Hybrid RAG - Configuration Models

This module defines configuration classes for the hybrid RAG system:
- HybridSearchConfig: Weight distribution and search parameters
- EmbeddingConfig: Embedding model configuration
- RAGConfig: Overall RAG system configuration
- FalkorDBConfig: FalkorDB connection configuration

Part of Phase 1, Checkpoint 1.1.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HybridSearchConfig:
    """
    Configuration for hybrid search behavior.

    Controls weight distribution, per-source limits, timeouts,
    and fusion parameters.
    """

    # Weight distribution (must sum to 1.0)
    vector_weight: float = 0.4
    fulltext_weight: float = 0.2
    graph_weight: float = 0.4

    # Per-source result limits
    vector_top_k: int = 20
    fulltext_top_k: int = 20
    graph_top_k: int = 20

    # Final output limit
    final_top_k: int = 10

    # Timeouts (milliseconds)
    vector_timeout_ms: int = 2000
    fulltext_timeout_ms: int = 1000
    graph_timeout_ms: int = 3000

    # Reciprocal Rank Fusion constant
    # Higher k = more weight to lower-ranked results
    rrf_k: int = 60

    # Graph boost for causally-connected results
    graph_boost_factor: float = 1.3

    # Minimum similarity thresholds
    vector_min_similarity: float = 0.3
    fulltext_min_rank: float = 0.1
    graph_min_relevance: float = 0.2

    def validate(self) -> None:
        """Validate configuration values."""
        total = self.vector_weight + self.fulltext_weight + self.graph_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")

        if any(w < 0 for w in [self.vector_weight, self.fulltext_weight, self.graph_weight]):
            raise ValueError("All weights must be non-negative")

        if any(k <= 0 for k in [self.vector_top_k, self.fulltext_top_k, self.graph_top_k]):
            raise ValueError("All top_k values must be positive")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        if self.graph_boost_factor < 1.0:
            raise ValueError("graph_boost_factor must be >= 1.0")

    def __post_init__(self):
        """Validate on initialization."""
        self.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridSearchConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def fusion_weights(self) -> Dict[str, float]:
        """Get fusion weights as a dictionary for the retriever."""
        return {
            "vector": self.vector_weight,
            "fulltext": self.fulltext_weight,
            "graph": self.graph_weight,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vector_weight": self.vector_weight,
            "fulltext_weight": self.fulltext_weight,
            "graph_weight": self.graph_weight,
            "vector_top_k": self.vector_top_k,
            "fulltext_top_k": self.fulltext_top_k,
            "graph_top_k": self.graph_top_k,
            "final_top_k": self.final_top_k,
            "vector_timeout_ms": self.vector_timeout_ms,
            "fulltext_timeout_ms": self.fulltext_timeout_ms,
            "graph_timeout_ms": self.graph_timeout_ms,
            "rrf_k": self.rrf_k,
            "graph_boost_factor": self.graph_boost_factor,
            "vector_min_similarity": self.vector_min_similarity,
            "fulltext_min_rank": self.fulltext_min_rank,
            "graph_min_relevance": self.graph_min_relevance,
        }


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding model.

    Supports both OpenAI API and local sentence-transformers models.
    """

    # Model choice
    model_name: str = "text-embedding-3-small"  # OpenAI default
    model_provider: str = "openai"  # "openai" or "sentence_transformers"

    # OpenAI-specific
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None

    # Embedding dimensions
    embedding_dimension: int = 1536  # OpenAI text-embedding-3-small

    # Batching
    batch_size: int = 100
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Local model path (for sentence_transformers)
    local_model_path: Optional[str] = None

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.model_provider == "openai" and not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_base_url:
            self.api_base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    def validate(self) -> None:
        """Validate configuration."""
        if self.model_provider == "openai" and not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI embeddings")

        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create from environment variables."""
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            model_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base_url=os.getenv("OPENAI_API_BASE"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
        )


@dataclass
class FalkorDBConfig:
    """
    Configuration for FalkorDB graph database connection.
    """

    host: str = "localhost"
    port: int = 6381  # e2i FalkorDB external port (6379=redis internal, 6380=auto-claude, 6381=e2i)
    graph_name: str = "e2i_knowledge"
    password: Optional[str] = None

    # Connection pooling
    max_connections: int = 10
    connection_timeout_seconds: float = 5.0

    # Query settings
    default_query_timeout_ms: int = 3000
    max_path_length: int = 5

    def __post_init__(self):
        """Load from environment if not provided."""
        if not self.password:
            self.password = os.getenv("FALKORDB_PASSWORD")

    @classmethod
    def from_env(cls) -> "FalkorDBConfig":
        """Create from environment variables."""
        return cls(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "6381")),
            graph_name=os.getenv("FALKORDB_GRAPH", "e2i_knowledge"),
            password=os.getenv("FALKORDB_PASSWORD"),
            max_connections=int(os.getenv("FALKORDB_MAX_CONNECTIONS", "10")),
            connection_timeout_seconds=float(os.getenv("FALKORDB_TIMEOUT", "5.0")),
        )

    def get_connection_url(self) -> str:
        """Build Redis connection URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}"


@dataclass
class HealthMonitorConfig:
    """
    Configuration for health monitoring.
    """

    # Check intervals
    check_interval_seconds: float = 30.0

    # Thresholds
    degraded_latency_ms: float = 2000.0  # Mark degraded if latency > this
    unhealthy_consecutive_failures: int = 3  # Mark unhealthy after N failures

    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_reset_seconds: float = 60.0

    # Alerting
    alert_on_degraded: bool = True
    alert_on_unhealthy: bool = True


@dataclass
class RAGConfig:
    """
    Overall RAG system configuration.

    Combines all sub-configurations for easy management.
    """

    # Sub-configurations
    search: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    falkordb: FalkorDBConfig = field(default_factory=FalkorDBConfig)
    health: HealthMonitorConfig = field(default_factory=HealthMonitorConfig)

    # Supabase connection (from environment)
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    # Feature flags
    enable_graph_search: bool = True
    enable_fulltext_search: bool = True
    enable_vector_search: bool = True

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes

    # Logging
    log_all_queries: bool = False
    log_slow_queries_ms: float = 1000.0

    def __post_init__(self):
        """Load from environment if not provided."""
        if not self.supabase_url:
            self.supabase_url = os.getenv("SUPABASE_URL")
        if not self.supabase_key:
            self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    def validate(self) -> None:
        """Validate all configurations."""
        self.search.validate()
        self.embedding.validate()

        if not self.supabase_url:
            raise ValueError("SUPABASE_URL must be set")
        if not self.supabase_key:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY must be set")

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create complete configuration from environment variables."""
        return cls(
            search=HybridSearchConfig(),
            embedding=EmbeddingConfig.from_env(),
            falkordb=FalkorDBConfig.from_env(),
            health=HealthMonitorConfig(),
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            enable_graph_search=os.getenv("RAG_ENABLE_GRAPH", "true").lower() == "true",
            enable_fulltext_search=os.getenv("RAG_ENABLE_FULLTEXT", "true").lower() == "true",
            enable_vector_search=os.getenv("RAG_ENABLE_VECTOR", "true").lower() == "true",
            cache_enabled=os.getenv("RAG_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("RAG_CACHE_TTL", "300")),
            log_all_queries=os.getenv("RAG_LOG_ALL_QUERIES", "false").lower() == "true",
            log_slow_queries_ms=float(os.getenv("RAG_LOG_SLOW_MS", "1000")),
        )


# Default configurations for different environments
DEFAULT_CONFIG = RAGConfig()

DEVELOPMENT_CONFIG = RAGConfig(
    search=HybridSearchConfig(
        vector_timeout_ms=5000,  # Longer timeouts for dev
        fulltext_timeout_ms=3000,
        graph_timeout_ms=5000,
    ),
    log_all_queries=True,
    cache_enabled=False,  # Disable cache in dev for easier debugging
)

PRODUCTION_CONFIG = RAGConfig(
    search=HybridSearchConfig(
        vector_timeout_ms=2000, fulltext_timeout_ms=1000, graph_timeout_ms=3000
    ),
    log_all_queries=False,
    log_slow_queries_ms=500.0,  # Lower threshold in prod
    cache_enabled=True,
    cache_ttl_seconds=600,  # 10 minutes in prod
)
