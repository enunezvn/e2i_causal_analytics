"""
Self-Improvement Configuration Loader.

Loads and validates the self_improvement.yaml configuration file
using Pydantic models for type-safe access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Rubric Configuration Models
# =============================================================================


class RubricCriterion(BaseModel):
    """Configuration for a single rubric criterion."""

    weight: float = Field(..., ge=0, le=1, description="Weight of this criterion (0-1)")
    description: str = Field(..., description="Description of what this criterion evaluates")
    scoring_guide: Dict[int, str] = Field(
        default_factory=dict, description="Scoring guide from 1-5"
    )

    @field_validator("scoring_guide")
    @classmethod
    def validate_scoring_guide(cls, v: Dict[int, str]) -> Dict[int, str]:
        """Ensure scoring guide has keys 1-5."""
        expected_keys = {1, 2, 3, 4, 5}
        actual_keys = set(v.keys())
        if actual_keys != expected_keys:
            logger.warning(f"Scoring guide should have keys 1-5, got {sorted(actual_keys)}")
        return v


class RubricConfig(BaseModel):
    """Configuration for the evaluation rubric."""

    version: str = Field(default="1.0.0", description="Rubric version")
    criteria: Dict[str, RubricCriterion] = Field(
        default_factory=dict, description="Rubric criteria configurations"
    )

    def get_criterion_weights(self) -> Dict[str, float]:
        """Get a dictionary of criterion names to weights."""
        return {name: criterion.weight for name, criterion in self.criteria.items()}

    def get_total_weight(self) -> float:
        """Get total weight (should sum to 1.0)."""
        return sum(c.weight for c in self.criteria.values())


# =============================================================================
# Decision Framework Configuration
# =============================================================================


class OverrideCondition(BaseModel):
    """Configuration for a decision override condition."""

    condition: str = Field(..., description="Condition type")
    threshold: Optional[float] = Field(None, description="Threshold value")
    occurrence_count: Optional[int] = Field(None, description="Occurrence count")
    std_dev_threshold: Optional[float] = Field(None, description="Standard deviation threshold")
    feedback_score_threshold: Optional[int] = Field(None, description="Feedback score threshold")
    action: str = Field(..., description="Action to take when condition is met")


class DecisionFrameworkConfig(BaseModel):
    """Configuration for the decision framework."""

    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "acceptable": 4.0,
            "suggestion": 3.0,
            "auto_update": 2.0,
            "escalate": 2.0,
        },
        description="Decision thresholds",
    )
    override_conditions: List[OverrideCondition] = Field(
        default_factory=list, description="Override conditions"
    )

    def get_decision(
        self, score: float
    ) -> Literal["acceptable", "suggestion", "auto_update", "escalate"]:
        """Determine the decision based on score."""
        if score >= self.thresholds.get("acceptable", 4.0):
            return "acceptable"
        elif score >= self.thresholds.get("suggestion", 3.0):
            return "suggestion"
        elif score >= self.thresholds.get("auto_update", 2.0):
            return "auto_update"
        else:
            return "escalate"


# =============================================================================
# Safety Controls Configuration
# =============================================================================


class CooldownConfig(BaseModel):
    """Cooldown configuration for self-improvement."""

    minimum_hours_between_updates: int = Field(default=6, ge=0)
    maximum_updates_per_day: int = Field(default=3, ge=1)
    burn_in_hours_after_major_change: int = Field(default=24, ge=0)


class VersionControlConfig(BaseModel):
    """Version control configuration for prompt versions."""

    versions_to_keep: int = Field(default=5, ge=1)
    auto_rollback_score_drop_threshold: float = Field(default=0.5, ge=0)
    rollback_window_hours: int = Field(default=48, ge=0)


class HumanOverrideConfig(BaseModel):
    """Human override configuration."""

    require_approval_mode: bool = Field(default=False)
    pause_auto_updates: bool = Field(default=False)
    escalation_email: str = Field(default="")


class SafetyControlsConfig(BaseModel):
    """Safety controls for self-improvement."""

    cooldown: CooldownConfig = Field(default_factory=CooldownConfig)
    version_control: VersionControlConfig = Field(default_factory=VersionControlConfig)
    human_override: HumanOverrideConfig = Field(default_factory=HumanOverrideConfig)
    north_star_guardrails: List[str] = Field(
        default_factory=list,
        description="Immutable guardrails that cannot be modified by self-improvement",
    )

    def is_auto_update_paused(self) -> bool:
        """Check if auto-updates are paused."""
        return self.human_override.pause_auto_updates

    def requires_approval(self) -> bool:
        """Check if all changes require human approval."""
        return self.human_override.require_approval_mode


# =============================================================================
# Reflection Loop Configuration
# =============================================================================


class TriggerConfig(BaseModel):
    """Configuration for reflection loop triggers."""

    mode: Literal["async", "sync", "scheduled"] = Field(default="async")
    batch_size: int = Field(default=5, ge=1)
    schedule_minutes: int = Field(default=15, ge=1)


class EvaluationModelConfig(BaseModel):
    """Configuration for evaluation model."""

    model: str = Field(default="claude-haiku-4-5-20251001")
    max_tokens: int = Field(default=1500, ge=100)
    temperature: float = Field(default=0.3, ge=0, le=1)


class PromptImprovementConfig(BaseModel):
    """Configuration for prompt improvement."""

    model: str = Field(default="claude-sonnet-4-5-20250929")
    max_tokens: int = Field(default=2000, ge=100)
    include_examples: bool = Field(default=True)
    query_knowledge_store: bool = Field(default=True)


class ReflectionLoopConfig(BaseModel):
    """Configuration for the reflection loop."""

    trigger: TriggerConfig = Field(default_factory=TriggerConfig)
    evaluation: EvaluationModelConfig = Field(default_factory=EvaluationModelConfig)
    prompt_improvement: PromptImprovementConfig = Field(default_factory=PromptImprovementConfig)


# =============================================================================
# Behavior Tracking Configuration
# =============================================================================


class PatternConfig(BaseModel):
    """Configuration for a tracked pattern."""

    type: str = Field(..., description="Pattern type identifier")
    description: str = Field(..., description="Human-readable description")
    signal: str = Field(..., description="What this pattern signals")


class BehaviorAnalysisConfig(BaseModel):
    """Configuration for behavior analysis."""

    minimum_occurrences: int = Field(default=5, ge=1)
    analysis_window_days: int = Field(default=7, ge=1)
    correlation_threshold: float = Field(default=0.3, ge=0, le=1)


class BehaviorTrackingConfig(BaseModel):
    """Configuration for behavior tracking."""

    enabled: bool = Field(default=True)
    patterns_to_track: Dict[str, List[PatternConfig]] = Field(default_factory=dict)
    analysis: BehaviorAnalysisConfig = Field(default_factory=BehaviorAnalysisConfig)


# =============================================================================
# Improvable Components Configuration
# =============================================================================


class ComponentConfig(BaseModel):
    """Configuration for an improvable component."""

    component: str = Field(..., description="Component name")
    path: str = Field(..., description="Path to the component")
    risk_level: Literal["low", "medium", "high"] = Field(default="medium")
    auto_update_allowed: bool = Field(default=False)


class ImprovableComponentConfig(BaseModel):
    """Configuration for improvable components."""

    prompts: List[ComponentConfig] = Field(default_factory=list)
    configurations: List[ComponentConfig] = Field(default_factory=list)
    never_modify: List[str] = Field(default_factory=list)

    def can_auto_update(self, component_name: str) -> bool:
        """Check if a component can be auto-updated."""
        for comp in self.prompts + self.configurations:
            if comp.component == component_name:
                return comp.auto_update_allowed
        return False

    def is_protected(self, path: str) -> bool:
        """Check if a path is protected from modification."""
        return any(protected in path for protected in self.never_modify)


# =============================================================================
# Main Configuration Model
# =============================================================================


class SelfImprovementConfig(BaseModel):
    """Complete self-improvement configuration."""

    rubric: RubricConfig = Field(default_factory=RubricConfig)
    decision_framework: DecisionFrameworkConfig = Field(default_factory=DecisionFrameworkConfig)
    safety_controls: SafetyControlsConfig = Field(default_factory=SafetyControlsConfig)
    reflection_loop: ReflectionLoopConfig = Field(default_factory=ReflectionLoopConfig)
    behavior_tracking: BehaviorTrackingConfig = Field(default_factory=BehaviorTrackingConfig)
    improvable_components: ImprovableComponentConfig = Field(
        default_factory=ImprovableComponentConfig
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SelfImprovementConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found at {path}, using defaults")
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def get_criterion_weight(self, criterion_name: str) -> float:
        """Get the weight for a specific criterion."""
        if criterion_name in self.rubric.criteria:
            return self.rubric.criteria[criterion_name].weight
        return 0.0

    def get_all_criterion_weights(self) -> Dict[str, float]:
        """Get all criterion weights."""
        return self.rubric.get_criterion_weights()


# =============================================================================
# Config Loader Function
# =============================================================================


_cached_config: Optional[SelfImprovementConfig] = None


def load_self_improvement_config(
    config_path: Optional[Path | str] = None,
    force_reload: bool = False,
) -> SelfImprovementConfig:
    """
    Load the self-improvement configuration.

    Args:
        config_path: Optional path to config file. If not provided,
                     uses default location at config/self_improvement.yaml
        force_reload: If True, reload config even if cached

    Returns:
        SelfImprovementConfig instance
    """
    global _cached_config

    if _cached_config is not None and not force_reload:
        return _cached_config

    if config_path is None:
        # Default to config/self_improvement.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        config_path = project_root / "config" / "self_improvement.yaml"

    _cached_config = SelfImprovementConfig.from_yaml(config_path)
    logger.info(f"Loaded self-improvement config from {config_path}")

    return _cached_config


def get_rubric_weights() -> Dict[str, float]:
    """
    Convenience function to get rubric criterion weights.

    Returns:
        Dictionary mapping criterion names to their weights
    """
    config = load_self_improvement_config()
    return config.get_all_criterion_weights()


def get_decision_thresholds() -> Dict[str, float]:
    """
    Convenience function to get decision thresholds.

    Returns:
        Dictionary mapping decision types to score thresholds
    """
    config = load_self_improvement_config()
    return config.decision_framework.thresholds
