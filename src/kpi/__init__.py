"""
E2I Causal Analytics - KPI Calculation Service

This module provides on-demand KPI calculation with caching and
causal library routing for the 46 KPIs defined in the E2I framework.
"""

from src.kpi.models import KPIResult, KPIMetadata, KPIThreshold, CausalLibrary
from src.kpi.registry import KPIRegistry
from src.kpi.calculator import KPICalculator
from src.kpi.cache import KPICache
from src.kpi.router import CausalLibraryRouter

__all__ = [
    "KPIResult",
    "KPIMetadata",
    "KPIThreshold",
    "CausalLibrary",
    "KPIRegistry",
    "KPICalculator",
    "KPICache",
    "CausalLibraryRouter",
]
