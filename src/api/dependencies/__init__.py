"""API Dependencies for FastAPI.

This module provides dependency injection for API routes.
"""

from src.api.dependencies.bentoml_client import (
    BentoMLClient,
    BentoMLClientConfig,
    get_bentoml_client,
)

__all__ = [
    "BentoMLClient",
    "BentoMLClientConfig",
    "get_bentoml_client",
]
