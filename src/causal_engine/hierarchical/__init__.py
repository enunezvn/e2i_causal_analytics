"""
E2I Causal Analytics - Hierarchical Nesting Module
===================================================

B9: Hierarchical analysis combining CausalML segments with EconML CATE.

This module implements Pattern 4 from the Data Architecture:
- EconML within CausalML segments for fine-grained CATE estimation

Components:
- HierarchicalAnalyzer: Orchestrates nested analysis
- SegmentCATECalculator: EconML per CausalML segment
- NestedConfidenceInterval: Combines segment-level CIs

Author: E2I Causal Analytics Team
"""

from .analyzer import (
    HierarchicalAnalyzer,
    HierarchicalConfig,
    HierarchicalResult,
    SegmentResult,
)
from .nested_ci import (
    AggregationMethod,
    NestedCIConfig,
    NestedCIResult,
    NestedConfidenceInterval,
)
from .segment_cate import (
    SegmentCATECalculator,
    SegmentCATEConfig,
    SegmentCATEResult,
)

__all__ = [
    # Analyzer
    "HierarchicalAnalyzer",
    "HierarchicalConfig",
    "HierarchicalResult",
    "SegmentResult",
    # Segment CATE
    "SegmentCATECalculator",
    "SegmentCATEConfig",
    "SegmentCATEResult",
    # Nested CI
    "NestedConfidenceInterval",
    "NestedCIConfig",
    "NestedCIResult",
    "AggregationMethod",
]
