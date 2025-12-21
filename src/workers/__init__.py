"""
Celery Workers Module
=====================

Multi-tier worker architecture for E2I Causal Analytics.

Worker Tiers:
- Light: Quick tasks (API calls, cache, notifications)
- Medium: Standard analytics (reports, aggregations)
- Heavy: Compute-intensive (SHAP, causal, ML, twins)
"""

from .celery_app import celery_app

__all__ = ["celery_app"]
