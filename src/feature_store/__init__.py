"""
E2I Causal Analytics - Lightweight Feature Store

A lightweight feature store implementation using:
- Supabase PostgreSQL for offline storage
- Redis for online serving with caching
- MLflow for feature tracking and versioning
"""

from .client import FeatureStoreClient
from .feature_analyzer_adapter import (
    FeatureAnalyzerAdapter,
    get_feature_analyzer_adapter,
)
from .models import (
    Feature,
    FeatureGroup,
    FeatureValue,
    FeatureValueType,
    FreshnessStatus,
)
from .monitoring import (
    FeatureRetrievalMetrics,
    LatencyStats,
    LatencyTracker,
    get_latency_tracker,
    init_feature_store_metrics,
    record_batch_size,
    record_cache_hit,
    record_cache_miss,
    record_error,
    track_cache_operation,
    track_db_operation,
    track_retrieval,
)
from .retrieval import FeatureRetriever
from .writer import FeatureWriter

__all__ = [
    # Main client
    "FeatureStoreClient",
    # Models
    "FeatureGroup",
    "Feature",
    "FeatureValue",
    "FeatureValueType",
    "FreshnessStatus",
    # Components
    "FeatureRetriever",
    "FeatureWriter",
    # Feature analyzer integration
    "FeatureAnalyzerAdapter",
    "get_feature_analyzer_adapter",
    # Monitoring (Phase 4 - G17)
    "init_feature_store_metrics",
    "track_retrieval",
    "track_cache_operation",
    "track_db_operation",
    "record_cache_hit",
    "record_cache_miss",
    "record_batch_size",
    "record_error",
    "FeatureRetrievalMetrics",
    "LatencyStats",
    "LatencyTracker",
    "get_latency_tracker",
]
