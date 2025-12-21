"""
E2I Causal Analytics - Lightweight Feature Store

A lightweight feature store implementation using:
- Supabase PostgreSQL for offline storage
- Redis for online serving with caching
- MLflow for feature tracking and versioning
"""

from .client import FeatureStoreClient
from .models import (
    Feature,
    FeatureGroup,
    FeatureValue,
    FeatureValueType,
    FreshnessStatus,
)
from .retrieval import FeatureRetriever
from .writer import FeatureWriter

__all__ = [
    "FeatureStoreClient",
    "FeatureGroup",
    "Feature",
    "FeatureValue",
    "FeatureValueType",
    "FreshnessStatus",
    "FeatureRetriever",
    "FeatureWriter",
]
