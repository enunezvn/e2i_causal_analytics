"""
Simulation Engine
=================

Executes intervention simulations on digital twin populations using the real
uplift effect engine (``src.digital_twin.effect``). Fail-closed: a fabricated
ATE is never emitted; bad/insufficient data yields a FAILED result.

The simulation follows these steps:
1. Apply population filters to select relevant twins
2. Fit an uplift model on a labeled (treatment, outcome, confounders) frame
   from the effect provider and score per-twin uplift over the population
3. Derive heterogeneous effects from the per-twin uplift scores
4. Use the estimate's CI-based ATE bounds
5. Generate a DEPLOY / REFINE / SKIP recommendation from the CI-based policy
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from src.causal_engine.errors import EstimationError
from src.digital_twin.effect import (
    EffectDataProvider,
    EffectDataUnavailable,
    PolicyThresholds,
    RecommendationPolicy,
    SyntheticEffectDataProvider,
    TwinEffectEstimator,
)

from .models.simulation_models import (
    EffectHeterogeneity,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
)
from .models.twin_models import DigitalTwin, TwinPopulation

# Type hint for optional cache import
TYPE_CHECKING = False
if TYPE_CHECKING:
    from .simulation_cache import SimulationCache

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

    # Thresholds for recommendations
    DEFAULT_MIN_EFFECT_THRESHOLD = 0.05  # 5% minimum effect
    DEFAULT_CONFIDENCE_THRESHOLD = 0.70

    def __init__(
        self,
        population: TwinPopulation,
        min_effect_threshold: float = DEFAULT_MIN_EFFECT_THRESHOLD,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        model_fidelity_score: Optional[float] = None,
        cache: Optional["SimulationCache"] = None,
        effect_provider: Optional[EffectDataProvider] = None,
        effect_estimator: Optional[TwinEffectEstimator] = None,
    ):
        """
        Initialize simulation engine.

        Args:
            population: Twin population to simulate on
            min_effect_threshold: Minimum ATE to recommend deployment
            confidence_threshold: Minimum confidence required
            model_fidelity_score: Fidelity score of generator model
            cache: Optional simulation cache for result caching
            effect_provider: Labeled-data provider for uplift fitting
                (defaults to the synthetic known-effect DGP). Injectable for tests.
            effect_estimator: Uplift effect estimator (defaults to the real
                TwinEffectEstimator). Injectable for tests.
        """
        self.population = population
        self.model_id = population.model_id
        self.min_effect_threshold = min_effect_threshold
        self.confidence_threshold = confidence_threshold
        self.model_fidelity_score = model_fidelity_score
        self._cache = cache
        self._effect_provider = effect_provider or SyntheticEffectDataProvider()
        self._effect_estimator = effect_estimator or TwinEffectEstimator()

        logger.info(
            f"Initialized SimulationEngine with {len(population)} twins "
            f"(min_effect={min_effect_threshold}, confidence={confidence_threshold}, "
            f"cache={'enabled' if cache else 'disabled'})"
        )

    def simulate(
        self,
        intervention_config: InterventionConfig,
        population_filter: Optional[PopulationFilter] = None,
        confidence_level: float = 0.95,
        calculate_heterogeneity: bool = True,
        use_cache: bool = True,
    ) -> SimulationResult:
        """
        Run intervention simulation.

        Args:
            intervention_config: Configuration of intervention to simulate
            population_filter: Optional filters to subset population
            confidence_level: Confidence level for CI calculation
            calculate_heterogeneity: Whether to compute subgroup effects
            use_cache: Whether to use cache for results (default True)

        Returns:
            SimulationResult with ATE, CI, and recommendation
        """
        start_time = time.time()

        logger.info(
            f"Starting simulation: {intervention_config.intervention_type} "
            f"on {len(self.population)} twins"
        )

        # Check cache first if enabled
        if use_cache and self._cache and self.model_id:
            try:
                import asyncio

                # Run async cache lookup synchronously
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a task
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._cache.get_cached_result(
                                intervention_config, population_filter, self.model_id
                            ),
                        )
                        cached_result = future.result(timeout=5)
                else:
                    cached_result = loop.run_until_complete(
                        self._cache.get_cached_result(
                            intervention_config, population_filter, self.model_id
                        )
                    )

                if cached_result:
                    logger.info(
                        f"Returning cached simulation result "
                        f"(ATE={cached_result.simulated_ate:.4f})"
                    )
                    return cached_result
            except Exception as e:
                logger.debug(f"Cache lookup failed, proceeding with simulation: {e}")

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

        # Estimate the real uplift effect (fail-closed: no fabricated ATE).
        # The provider supplies a labeled (treatment, outcome, confounders) frame;
        # the estimator fits an uplift model on it and scores the twin population.
        twins = filtered_population.twins
        twin_df = pd.DataFrame([t.features for t in twins])
        try:
            frame = self._effect_provider.get_training_frame(
                intervention_config.intervention_type,
                brand=str(self.population.brand),
                twin_type=str(self.population.twin_type),
                reference_covariates=twin_df,
            )
            estimate = self._effect_estimator.estimate(frame, twin_df)
        except (EffectDataUnavailable, EstimationError) as e:
            return self._create_error_result(
                intervention_config,
                population_filter or PopulationFilter(),
                f"Effect estimation failed: {e}",
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

        treatment_effects = list(estimate.per_twin_uplift.ravel())
        ate = estimate.ate
        ci_lower = estimate.ate_ci_lower
        ci_upper = estimate.ate_ci_upper
        # SE consistent with the training-evidence CI (CI = ate +/- 1.96*SE),
        # so it does not shrink with the twin count.
        std_error = float((ci_upper - ci_lower) / (2 * 1.96))

        # Calculate heterogeneous effects from the per-twin uplift scores
        heterogeneity = EffectHeterogeneity()
        if calculate_heterogeneity:
            heterogeneity = self._calculate_heterogeneity(twins, treatment_effects)

        # Generate recommendation from the CI-based policy
        baseline_rate = float(np.mean([t.baseline_propensity for t in twins]))
        rec, rationale, recommended_n = RecommendationPolicy(
            PolicyThresholds(min_effect=self.min_effect_threshold)
        ).decide(estimate, baseline_rate=baseline_rate)
        recommendation = SimulationRecommendation(rec.value)

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
        simulation_confidence = self._calculate_simulation_confidence(n_twins, std_error, ate)

        execution_time_ms = int((time.time() - start_time) * 1000)

        result = SimulationResult(
            model_id=self.model_id or uuid4(),
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
            data_provenance=estimate.data_provenance,
            status=SimulationStatus.COMPLETED,
            execution_time_ms=execution_time_ms,
            completed_at=datetime.now(timezone.utc),
        )

        logger.info(
            f"Simulation complete: ATE={ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], "
            f"recommendation={recommendation.value}, time={execution_time_ms}ms"
        )

        # Cache the result if caching is enabled
        if use_cache and self._cache and self.model_id:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        executor.submit(
                            asyncio.run,
                            self._cache.cache_result(result),
                        )
                else:
                    loop.run_until_complete(self._cache.cache_result(result))
                logger.debug("Cached simulation result")
            except Exception as e:
                logger.debug(f"Failed to cache simulation result: {e}")

        return result

    def _apply_filters(self, filters: Optional[PopulationFilter]) -> TwinPopulation:
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

    def _twin_matches_filter(self, twin: DigitalTwin, filters: PopulationFilter) -> bool:
        """Check if twin matches all filter criteria."""
        features = twin.features

        if filters.specialties and features.get("specialty") not in filters.specialties:
            return False
        if filters.deciles and features.get("decile") not in filters.deciles:
            return False
        if filters.regions and features.get("region") not in filters.regions:
            return False
        if (
            filters.adoption_stages
            and features.get("adoption_stage") not in filters.adoption_stages
        ):
            return False
        if filters.min_baseline_outcome and twin.baseline_outcome < filters.min_baseline_outcome:
            return False
        if filters.max_baseline_outcome and twin.baseline_outcome > filters.max_baseline_outcome:
            return False

        return True

    def _calculate_heterogeneity(
        self,
        twins: List[DigitalTwin],
        effects: List[float],
    ) -> EffectHeterogeneity:
        """Calculate heterogeneous effects by subgroup."""
        heterogeneity = EffectHeterogeneity()

        # Group by specialty
        specialty_groups: dict[str, List[float]] = {}
        decile_groups: dict[str, List[float]] = {}
        region_groups: dict[str, List[float]] = {}
        adoption_groups: dict[str, List[float]] = {}

        for twin, effect in zip(twins, effects, strict=False):
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
        def calc_group_stats(groups: dict[str, List[float]]) -> dict[str, dict[str, float]]:
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
            model_id=self.model_id or uuid4(),
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
