# src/e2i/agents/orchestrator/classifier/__init__.py
"""
Orchestrator Classifier Module

4-stage classification pipeline for query routing:
- Stage 1: Feature Extraction
- Stage 2: Domain Mapping
- Stage 3: Dependency Detection
- Stage 4: Pattern Selection
"""

from .dependency_detector import DependencyDetector
from .domain_mapper import DomainMapper
from .feature_extractor import FeatureExtractor
from .pattern_selector import PatternSelector
from .pipeline import ClassificationPipeline
from .schemas import (
    ClassificationResult,
    Dependency,
    DependencyAnalysis,
    DependencyType,
    Domain,
    DomainMapping,
    ExtractedFeatures,
    RoutingPattern,
    SubQuestion,
)

__all__ = [
    # Enums
    "Domain",
    "RoutingPattern",
    "DependencyType",
    # Schemas
    "ExtractedFeatures",
    "DomainMapping",
    "DependencyAnalysis",
    "ClassificationResult",
    "SubQuestion",
    "Dependency",
    # Components
    "FeatureExtractor",
    "DomainMapper",
    "DependencyDetector",
    "PatternSelector",
    "ClassificationPipeline",
]
