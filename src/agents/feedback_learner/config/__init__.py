"""
Configuration module for Feedback Learner self-improvement.

This module provides:
- Pydantic models for self-improvement configuration
- YAML config loader with validation
- Type-safe access to rubric criteria, decision thresholds, and safety controls
"""

from .loader import (
    BehaviorTrackingConfig,
    CooldownConfig,
    DecisionFrameworkConfig,
    HumanOverrideConfig,
    ImprovableComponentConfig,
    ReflectionLoopConfig,
    RubricConfig,
    RubricCriterion,
    SafetyControlsConfig,
    SelfImprovementConfig,
    VersionControlConfig,
    load_self_improvement_config,
)

__all__ = [
    "RubricCriterion",
    "RubricConfig",
    "DecisionFrameworkConfig",
    "CooldownConfig",
    "VersionControlConfig",
    "HumanOverrideConfig",
    "SafetyControlsConfig",
    "ReflectionLoopConfig",
    "BehaviorTrackingConfig",
    "ImprovableComponentConfig",
    "SelfImprovementConfig",
    "load_self_improvement_config",
]
