"""
Simulation Engine
=================

Executes intervention simulations on digital twin populations.
Applies treatment effects based on intervention configuration and
estimates counterfactual outcomes.

The simulation follows these steps:
1. Apply population filters to select relevant twins
2. Estimate treatment effect based on intervention type
3. Apply heterogeneous effects by subgroup
4. Calculate aggregate statistics (ATE, CI)
5. Generate recommendation based on thresholds
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from uuid import UUID
import time

import numpy as np
from scipy import stats

from .models.twin_models import TwinType, Brand, TwinPopulation, DigitalTwin
from .models.simulation_models import (
    SimulationStatus,
    SimulationRecommendation,
    InterventionConfig,
    PopulationFilter,
    EffectHeterogeneity,
    SimulationResult,
)

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Simulates intervention effects on digital twin populations.
    
    The engine applies treatment effects to twins based on their features,
    accounting for heterogeneity across subgroups. Results are used to
    pre-screen experiments before real-world deployment.
    
    Attributes:
        population: TwinPopulation to simulate on
        model_id: ID of the twin generator model
        min_effect_threshold: Minimum ATE to recommend deployment
        confidence_threshold: Minimum confidence for recommendations
    
    Example:
        >>> engine = SimulationEngine(twin_population)
        >>> config = InterventionConfig(
        ...     intervention_type="email_campaign",
        ...     channel="email",
        ...     frequency="weekly",
        ...     duration_weeks=8
        ... )
        >>> result = engine.simulate(config)
        >>> print(result.recommendation)
    """
    
    # Effect size parameters by intervention type (simplified model)
    INTERVENTION_EFFECTS = {
        "email_campaign": {
            "base_effect": 0.05,
            "variance": 0.02,
            "channel_multiplier": {"email": 1.0, "digital": 0.8},
        },
        "call_frequency_increase": {
            "base_effect": 0.08,
            "variance": 0.03,
            "intensity_factor": 0.02,  # Additional effect per call
        },
        "speaker_program_invitation": {
            "base_effect": 0.12,
            "variance": 0.04,
            "tier_multiplier": {1: 1.5, 2: 1.2, 3: 1.0, 4: 0.8, 5: 0.6},
        },
        "sample_distribution": {
            "base_effect": 0.03,
            "variance": 0.01,
        },
        "peer_influence_activation": {
            "base_effect": 0.10,
            "variance": 0.04,
            "influence_threshold": 0.7,  # Min peer influence score
        },
        "digital_engagement": {
            "base_effect": 0.06,
            "variance": 0.02,
        },
    }
    
    # Thresholds for recommendations
    DEFAULT_MIN_EFFECT_THRESHOLD = 0.05  # 5% minimum effect
    DEFAULT_CONFIDENCE_THRESHOLD = 0.70
    
    def __init__(
        self,
        population: TwinPopulation,
        min_effect_threshold: float = DEFAULT_MIN_EFFECT_THRESHOLD,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        model_fidelity_score: Optional[float] = None,
    ):
        """
        Initialize simulation engine.
        
        Args:
            population: Twin population to simulate on
            min_effect_threshold: Minimum ATE to recommend deployment
            confidence_threshold: Minimum confidence required
            model_fidelity_score: Fidelity score of generator model
        """
        self.population = population
        self.model_id = population.model_id
        self.min_effect_threshold = min_effect_threshold
        self.confidence_threshold = confidence_threshold
        self.model_fidelity_score = model_fidelity_score
        
        logger.info(
            f"Initialized SimulationEngine with {len(population)} twins "
            f"(min_effect={min_effect_threshold}, confidence={confidence_threshold})"
        )
    
    def simulate(
        self,
        intervention_config: InterventionConfig,
        population_filter: Optional[PopulationFilter] = None,
        confidence_level: float = 0.95,
        calculate_heterogeneity: bool = True,
    ) -> SimulationResult:
        """
        Run intervention simulation.
        
        Args:
            intervention_config: Configuration of intervention to simulate
            population_filter: Optional filters to subset population
            confidence_level: Confidence level for CI calculation
            calculate_heterogeneity: Whether to compute subgroup effects
        
        Returns:
            SimulationResult with ATE, CI, and recommendation
        """
        start_time = time.time()
        
        logger.info(
            f"Starting simulation: {intervention_config.intervention_type} "
            f"on {len(self.population)} twins"
        )
        
        # Apply population filters
        filtered_population = self._apply_filters(population_filter)
        n_twins = len(filtered_population.twins)
        
        if n_twins < 100:
            return self._create_error_result(
                intervention_config,
                population_filter or PopulationFilter(),
                "Insufficient twins after filtering (need >= 100)",
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        
        # Simulate treatment effects
        treatment_effects = self._simulate_effects(
            filtered_population.twins,
            intervention_config,
        )
        
        # Calculate aggregate statistics
        ate, ci_lower, ci_upper, std_error = self._calculate_statistics(
            treatment_effects,
            confidence_level,
        )
        
        # Calculate heterogeneous effects
        heterogeneity = EffectHeterogeneity()
        if calculate_heterogeneity:
            heterogeneity = self._calculate_heterogeneity(
                filtered_population.twins,
                treatment_effects,
            )
        
        # Generate recommendation
        recommendation, rationale = self._generate_recommendation(
            ate, ci_lower, ci_upper, n_twins
        )
        
        # Calculate recommended sample size for real experiment
        recommended_n = self._calculate_recommended_sample_size(ate, std_error)
        
        # Check fidelity warnings
        fidelity_warning = False
        fidelity_warning_reason = None
        if self.model_fidelity_score and self.model_fidelity_score < 0.7:
            fidelity_warning = True
            fidelity_warning_reason = (
                f"Model fidelity ({self.model_fidelity_score:.2f}) "
                "below threshold (0.70). Results may be unreliable."
            )
        
        # Calculate confidence score
        simulation_confidence = self._calculate_simulation_confidence(
            n_twins, std_error, ate
        )
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        result = SimulationResult(
            model_id=self.model_id,
            intervention_config=intervention_config,
            population_filters=population_filter or PopulationFilter(),
            twin_count=n_twins,
            simulated_ate=ate,
            simulated_ci_lower=ci_lower,
            simulated_ci_upper=ci_upper,
            simulated_std_error=std_error,
            effect_heterogeneity=heterogeneity,
            recommendation=recommendation,
            recommendation_rationale=rationale,
            recommended_sample_size=recommended_n,
            recommended_duration_weeks=intervention_config.duration_weeks,
            simulation_confidence=simulation_confidence,
            fidelity_warning=fidelity_warning,
            fidelity_warning_reason=fidelity_warning_reason,
            model_fidelity_score=self.model_fidelity_score,
            status=SimulationStatus.COMPLETED,
            execution_time_ms=execution_time_ms,
            completed_at=datetime.utcnow(),
        )
        
        logger.info(
            f"Simulation complete: ATE={ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], "
            f"recommendation={recommendation.value}, time={execution_time_ms}ms"
        )
        
        return result
    
    def _apply_filters(
        self,
        filters: Optional[PopulationFilter]
    ) -> TwinPopulation:
        """Apply population filters to select twins."""
        if not filters:
            return self.population
        
        filtered_twins = []
        for twin in self.population.twins:
            if self._twin_matches_filter(twin, filters):
                filtered_twins.append(twin)
        
        return TwinPopulation(
            twin_type=self.population.twin_type,
            brand=self.population.brand,
            twins=filtered_twins,
            size=len(filtered_twins),
            model_id=self.model_id,
            generation_config=self.population.generation_config,
        )
    
    def _twin_matches_filter(
        self,
        twin: DigitalTwin,
        filters: PopulationFilter
    ) -> bool:
        """Check if twin matches all filter criteria."""
        features = twin.features
        
        if filters.specialties and features.get("specialty") not in filters.specialties:
            return False
        if filters.deciles and features.get("decile") not in filters.deciles:
            return False
        if filters.regions and features.get("region") not in filters.regions:
            return False
        if filters.adoption_stages and features.get("adoption_stage") not in filters.adoption_stages:
            return False
        if filters.min_baseline_outcome and twin.baseline_outcome < filters.min_baseline_outcome:
            return False
        if filters.max_baseline_outcome and twin.baseline_outcome > filters.max_baseline_outcome:
            return False
        
        return True
    
    def _simulate_effects(
        self,
        twins: List[DigitalTwin],
        config: InterventionConfig,
    ) -> List[float]:
        """Simulate treatment effect for each twin."""
        intervention_type = config.intervention_type
        
        # Get intervention parameters
        params = self.INTERVENTION_EFFECTS.get(
            intervention_type,
            {"base_effect": 0.05, "variance": 0.02}
        )
        
        effects = []
        for twin in twins:
            effect = self._calculate_individual_effect(twin, config, params)
            effects.append(effect)
        
        return effects
    
    def _calculate_individual_effect(
        self,
        twin: DigitalTwin,
        config: InterventionConfig,
        params: Dict[str, Any],
    ) -> float:
        """Calculate treatment effect for individual twin."""
        features = twin.features
        base_effect = params["base_effect"]
        variance = params["variance"]
        
        # Apply intervention-specific modifiers
        effect_multiplier = 1.0
        
        # Decile-based multiplier (higher deciles = lower effect)
        decile = features.get("decile", 5)
        effect_multiplier *= 1.2 - (decile - 1) * 0.04
        
        # Engagement-based multiplier
        engagement = features.get("digital_engagement_score", 0.5)
        effect_multiplier *= 0.8 + 0.4 * engagement
        
        # Adoption stage multiplier
        adoption_stage = features.get("adoption_stage", "early_majority")
        adoption_multipliers = {
            "innovator": 0.6,  # Already adopted, less room to grow
            "early_adopter": 0.8,
            "early_majority": 1.0,
            "late_majority": 1.2,
            "laggard": 1.4,
        }
        effect_multiplier *= adoption_multipliers.get(adoption_stage, 1.0)
        
        # Intensity multiplier from config
        effect_multiplier *= config.intensity_multiplier
        
        # Duration adjustment (longer = stronger, with diminishing returns)
        duration_factor = np.log1p(config.duration_weeks) / np.log1p(8)  # Normalized to 8 weeks
        effect_multiplier *= duration_factor
        
        # Channel-specific adjustments
        if "channel_multiplier" in params and config.channel:
            channel_mult = params["channel_multiplier"].get(config.channel, 1.0)
            effect_multiplier *= channel_mult
        
        # Apply propensity weighting (higher propensity = more responsive)
        effect_multiplier *= 0.8 + 0.4 * twin.baseline_propensity
        
        # Calculate final effect with noise
        effect = base_effect * effect_multiplier
        effect += np.random.normal(0, variance)
        
        return effect
    
    def _calculate_statistics(
        self,
        effects: List[float],
        confidence_level: float,
    ) -> Tuple[float, float, float, float]:
        """Calculate aggregate statistics from effects."""
        effects_array = np.array(effects)
        
        ate = float(np.mean(effects_array))
        std_error = float(stats.sem(effects_array))
        
        # Calculate confidence interval
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ci_lower = ate - z_score * std_error
        ci_upper = ate + z_score * std_error
        
        return ate, ci_lower, ci_upper, std_error
    
    def _calculate_heterogeneity(
        self,
        twins: List[DigitalTwin],
        effects: List[float],
    ) -> EffectHeterogeneity:
        """Calculate heterogeneous effects by subgroup."""
        heterogeneity = EffectHeterogeneity()
        
        # Group by specialty
        specialty_groups: Dict[str, List[float]] = {}
        decile_groups: Dict[str, List[float]] = {}
        region_groups: Dict[str, List[float]] = {}
        adoption_groups: Dict[str, List[float]] = {}
        
        for twin, effect in zip(twins, effects):
            features = twin.features
            
            specialty = features.get("specialty", "unknown")
            specialty_groups.setdefault(specialty, []).append(effect)
            
            decile = str(features.get("decile", "unknown"))
            decile_groups.setdefault(decile, []).append(effect)
            
            region = features.get("region", "unknown")
            region_groups.setdefault(region, []).append(effect)
            
            adoption = features.get("adoption_stage", "unknown")
            adoption_groups.setdefault(adoption, []).append(effect)
        
        # Calculate stats for each group
        def calc_group_stats(groups: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
            result = {}
            for name, group_effects in groups.items():
                if len(group_effects) >= 10:  # Min sample size
                    result[name] = {
                        "ate": float(np.mean(group_effects)),
                        "std": float(np.std(group_effects)),
                        "n": len(group_effects),
                    }
            return result
        
        heterogeneity.by_specialty = calc_group_stats(specialty_groups)
        heterogeneity.by_decile = calc_group_stats(decile_groups)
        heterogeneity.by_region = calc_group_stats(region_groups)
        heterogeneity.by_adoption_stage = calc_group_stats(adoption_groups)
        
        return heterogeneity
    
    def _generate_recommendation(
        self,
        ate: float,
        ci_lower: float,
        ci_upper: float,
        n_twins: int,
    ) -> Tuple[SimulationRecommendation, str]:
        """Generate recommendation based on simulation results."""
        
        # Check if effect is too small
        if abs(ate) < self.min_effect_threshold:
            return (
                SimulationRecommendation.SKIP,
                f"Simulated ATE ({ate:.4f}) below minimum threshold "
                f"({self.min_effect_threshold}). Predicted impact insufficient "
                f"to justify experiment costs."
            )
        
        # Check if CI includes zero (not statistically significant)
        if ci_lower <= 0 <= ci_upper:
            return (
                SimulationRecommendation.REFINE,
                f"Effect not statistically significant (CI includes zero: "
                f"[{ci_lower:.4f}, {ci_upper:.4f}]). Consider refining "
                f"intervention design or increasing duration."
            )
        
        # Check if wide confidence interval (high uncertainty)
        ci_width = ci_upper - ci_lower
        if ci_width > abs(ate):
            return (
                SimulationRecommendation.REFINE,
                f"High uncertainty in effect estimate (CI width {ci_width:.4f} "
                f"> ATE {abs(ate):.4f}). Consider more targeted population "
                f"or stronger intervention."
            )
        
        # Recommend deployment
        effect_direction = "positive" if ate > 0 else "negative"
        return (
            SimulationRecommendation.DEPLOY,
            f"Simulation predicts {effect_direction} effect (ATE={ate:.4f}, "
            f"CI=[{ci_lower:.4f}, {ci_upper:.4f}]). Effect exceeds minimum "
            f"threshold and is statistically significant. Proceed with "
            f"real-world A/B test."
        )
    
    def _calculate_recommended_sample_size(
        self,
        ate: float,
        std_error: float,
        power: float = 0.8,
        alpha: float = 0.05,
    ) -> int:
        """Calculate recommended sample size for real experiment."""
        if abs(ate) < 0.001:
            return 10000  # Default for very small effects
        
        # Standard sample size calculation for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Estimate variance from simulation
        variance = (std_error * np.sqrt(len(self.population.twins))) ** 2
        
        n = int(2 * ((z_alpha + z_beta) ** 2) * variance / (ate ** 2))
        
        # Apply minimum and maximum bounds
        n = max(100, min(n, 50000))
        
        return n
    
    def _calculate_simulation_confidence(
        self,
        n_twins: int,
        std_error: float,
        ate: float,
    ) -> float:
        """Calculate confidence score for simulation results."""
        # Factors contributing to confidence:
        # 1. Sample size (more = better)
        size_score = min(1.0, n_twins / 10000)
        
        # 2. Precision (lower std error = better)
        precision_score = max(0, 1 - std_error / (abs(ate) + 0.001))
        
        # 3. Model fidelity
        fidelity_score = self.model_fidelity_score or 0.7
        
        # Weighted average
        confidence = 0.3 * size_score + 0.3 * precision_score + 0.4 * fidelity_score
        
        return min(1.0, max(0.0, confidence))
    
    def _create_error_result(
        self,
        config: InterventionConfig,
        filters: PopulationFilter,
        error_message: str,
        execution_time_ms: int,
    ) -> SimulationResult:
        """Create error result when simulation cannot complete."""
        return SimulationResult(
            model_id=self.model_id,
            intervention_config=config,
            population_filters=filters,
            twin_count=0,
            simulated_ate=0.0,
            simulated_ci_lower=0.0,
            simulated_ci_upper=0.0,
            simulated_std_error=0.0,
            recommendation=SimulationRecommendation.REFINE,
            recommendation_rationale=error_message,
            simulation_confidence=0.0,
            status=SimulationStatus.FAILED,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
        )
