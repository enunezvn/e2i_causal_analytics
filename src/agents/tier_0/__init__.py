"""Tier 0: ML Foundation Pipeline.

This package provides end-to-end orchestration of the 7 ML Foundation agents:
1. ScopeDefiner - Transform business objectives into ML specifications
2. DataPreparer - Validate data quality (QC GATE)
3. ModelSelector - Select optimal algorithm
4. ModelTrainer - Train with HPO and validation
5. FeatureAnalyzer - SHAP-based interpretability
6. ModelDeployer - Deploy to environments
7. ObservabilityConnector - Cross-cutting telemetry

Usage:
    from src.agents.tier_0 import MLFoundationPipeline

    pipeline = MLFoundationPipeline()
    result = await pipeline.run({
        "problem_description": "Predict HCP conversion",
        "business_objective": "Increase market share",
        "target_outcome": "conversion",
        "data_source": "business_metrics",
    })
"""

from .pipeline import MLFoundationPipeline, PipelineConfig, PipelineResult, PipelineStage

__all__ = [
    "MLFoundationPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
]
