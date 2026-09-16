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
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from src.causal_engine.errors import EstimationError
from src.digital_twin.effect import (
    SUBGROUP_AXES,
    EffectDataProvider,
    EffectDataUnavailable,
    EffectEstimate,
    PolicyThresholds,
    RecommendationPolicy,
    TwinEffectEstimator,
    experiment_size,
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

# Subgroup axes whose twin feature is a NUMBER rather than a label, so its group key is
# stringified: decile 1 and "1" are the same decile. Every other axis in SUBGROUP_AXES keys
# on the feature value itself, which is what it has always done — see _calculate_heterogeneity.
_NUMERIC_AXES = frozenset({"decile"})

# Training rows at which the confidence heuristic's evidence term saturates (#2104). The
# same knee the causalml executor uses (``n_train / 1000``) and the estimator's default
# minimum training size (``DEFAULT_MIN_TRAINING_SAMPLES``): a region-targeted cohort
# estimate on a region under the knee (~800-1000 rows live) scores below a cohort-wide
# one (~4000 rows); south (~1500 rows) saturates with the cohort.
# What the twin count can still reach, by path: on the COHORT path nothing — n_train is
# the cohort rows, the CI is the DML inference interval on them and the ATE is a CATE
# average over them. On the SYNTHETIC path the twins ARE the data by construction: the
# provider draws its training frame from the twins' covariates (n_train = the provider's
# frame size), the uplift model is refit on that draw and the ATE is the mean of the
# per-twin predictions, so the precision term varies with the twin sample.
CONFIDENCE_N_SATURATION = 1000.0


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
        >>> engine = SimulationEngine(
        ...     twin_population,
        ...     effect_provider=cohort_provider,
        ...     effect_estimator=CohortCausalEstimator(),
        ... )
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
        *,
        effect_provider: EffectDataProvider,
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
            effect_provider: Labeled-data provider for uplift fitting. Required, with no
                default: a synthetic default returned its planted effect as the estimate
                for any caller that forgot it (#2025). Production passes the cohort
                provider; a test that wants the known-effect DGP passes
                ``SyntheticEffectDataProvider`` explicitly.
            effect_estimator: Uplift effect estimator (defaults to the real
                TwinEffectEstimator). Injectable for tests.
        """
        self.population = population
        self.model_id = population.model_id
        self.min_effect_threshold = min_effect_threshold
        self.confidence_threshold = confidence_threshold
        self.model_fidelity_score = model_fidelity_score
        self._cache = cache
        self._effect_provider = effect_provider
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
            # Only EffectDataUnavailable names a cause (#2021 9b). EstimationError's ``details`` is
            # a diagnostic dict that may hold text, so nothing of it is carried.
            unavailable = e if isinstance(e, EffectDataUnavailable) else None
            return self._create_error_result(
                intervention_config,
                population_filter or PopulationFilter(),
                f"Effect estimation failed: {e}",
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_cause=(
                    str(unavailable.cause)
                    if unavailable is not None and unavailable.cause is not None
                    else None
                ),
                error_details=unavailable.details if unavailable is not None else None,
            )

        treatment_effects = list(estimate.per_twin_uplift.ravel())
        ate = estimate.ate
        ci_lower = estimate.ate_ci_lower
        ci_upper = estimate.ate_ci_upper
        # SE consistent with the training-evidence CI (CI = ate +/- 1.96*SE), so it does
        # not shrink as more twins are scored (synthetic path: the frame is drawn from
        # the twins, so a different twin sample can still refit to a different width).
        std_error = float((ci_upper - ci_lower) / (2 * 1.96))

        # Calculate heterogeneous effects from the per-twin uplift scores
        heterogeneity = EffectHeterogeneity()
        if calculate_heterogeneity:
            heterogeneity = self._calculate_heterogeneity(twins, treatment_effects, estimate)

        # Generate recommendation from the CI-based policy. The experiment is sized by the
        # rule the chat simulator shares (#2015): the outcome's comparison-arm spread in the
        # effect provider's frame, never the twins' propensity. No size -> None, and the
        # reason joins the rationale the page shows.
        # The size is scoped to whatever the estimate is scoped to (#2023): sizing a
        # region-targeted effect on the whole cohort's spread would state a number for a
        # different population than the effect it is powering for.
        policy = PolicyThresholds(min_effect=self.min_effect_threshold)
        rec, rationale = RecommendationPolicy(policy).decide(estimate)
        recommended_n, size_note = experiment_size(
            frame, ate, regions=estimate.target_regions, thresholds=policy
        )
        if recommended_n is None:
            rationale = f"{rationale} {size_note}"
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

        # Confidence follows the estimate's own evidence, not the twin count as such
        # (#2104; see CONFIDENCE_N_SATURATION for what each path's evidence is).
        simulation_confidence = self._calculate_simulation_confidence(
            estimate.n_train, std_error, ate
        )

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
            target_regions=list(estimate.target_regions),
            cohort_ate=estimate.cohort_ate,
            cohort_ci_lower=estimate.cohort_ci_lower,
            cohort_ci_upper=estimate.cohort_ci_upper,
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
        estimate: EffectEstimate,
    ) -> EffectHeterogeneity:
        """Subgroup effects on the axes the ESTIMATE resolves — and only those (#2054).

        Averaging ``per_twin_uplift`` over twin subgroups is a real subgroup effect only
        when the score varies WITHIN a subgroup. ``TwinEffectEstimator`` scores each twin
        over all its covariates, so it does, and all four axes are reported as before.
        ``CohortCausalEstimator`` fits region as its only heterogeneity axis, so its score
        is a step function of region: every twin in a region carries the same value, and a
        ``by_specialty`` average is then just the twin region-mixture mean. Since specialty,
        decile and adoption_stage are drawn independently of region, every such group
        converges to the SAME number and the spread between them is sampling noise in the
        twin draw (measured: 0.049 at 100 twins, 0.002 at 100k, while region is invariant).

        So each estimator declares what it resolves (``EffectEstimate.cate_by_axis``) and
        an undeclared axis is reported as ``{}`` — the fail-closed answer this codebase
        already uses for an effect it cannot support. Where the estimator precomputed the
        group effects itself they are reported verbatim, with its own evidence rows as
        ``n``, so the numbers do not move with the twin count.
        """
        heterogeneity = EffectHeterogeneity()

        # One bucket per axis, keyed by the twin's value on it. Driven by SUBGROUP_AXES so an
        # axis added there is grouped and reported without a second edit here.
        twin_groups: dict[str, dict[str, List[float]]] = {axis: {} for axis in SUBGROUP_AXES}

        for twin, effect in zip(twins, effects, strict=False):
            for axis, groups in twin_groups.items():
                key = twin.features.get(axis, "unknown")
                # A LABEL axis keys on the value itself. ``features`` is ``Dict[str, Any]``,
                # so coercing would merge distinct labels — int 1 with str "1" — into one
                # group whose ATE is the average of two different effects.
                groups.setdefault(str(key) if axis in _NUMERIC_AXES else key, []).append(effect)

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

        def declared_stats(axis: str) -> dict[str, dict[str, float]]:
            """The estimator's own group effects for ``axis``, with its own evidence rows.
            ``std`` is 0.0 because this estimate assigns one effect per group: the spread
            WITHIN a group, under this estimate, is exactly zero. Reporting a twin-draw
            std instead would describe the twin mixture, not the effect."""
            counts = estimate.n_by_axis.get(axis, {})
            if not counts:
                # A declared axis with no evidence counts reports nothing (fail-closed).
                # That is a bug in the estimator, not a data condition, so say so loudly
                # rather than let a resolved axis vanish from the response in silence.
                logger.warning(
                    "%s declared cate_by_axis[%r] with no n_by_axis counts; reporting no "
                    "subgroup effects for that axis",
                    estimate.estimator_type,
                    axis,
                )
            return {
                name: {"ate": float(ate), "std": 0.0, "n": int(counts[name])}
                for name, ate in estimate.cate_by_axis[axis].items()
                if int(counts.get(name, 0)) >= 10  # Min sample size, on the real evidence
            }

        def axis_stats(axis: str, groups: dict[str, List[float]]) -> dict[str, dict[str, float]]:
            if axis not in estimate.cate_by_axis:
                return {}  # not resolved by this estimate -> no subgroup number at all
            if estimate.cate_by_axis[axis]:
                return declared_stats(axis)
            # Declared with nothing precomputed: the per-twin scores resolve this axis.
            return calc_group_stats(groups)

        # Every axis in SUBGROUP_AXES gets reported, so a new one cannot be declared by an
        # estimator and then silently dropped here. An axis with no matching ``by_<axis>``
        # field fails loudly on assignment rather than being ignored.
        for axis, groups in twin_groups.items():
            setattr(heterogeneity, f"by_{axis}", axis_stats(axis, groups))

        return heterogeneity

    def _calculate_simulation_confidence(
        self,
        n_train: int,
        std_error: float,
        ate: float,
    ) -> float:
        """Calculate confidence score for simulation results."""
        # Factors contributing to confidence:
        # 1. Evidence: the rows the estimator fit on — the cohort rows, or the synthetic
        #    provider's frame — not the twin count, which on the cohort path is only a
        #    compute knob (#2104). Saturates at CONFIDENCE_N_SATURATION.
        size_score = min(1.0, n_train / CONFIDENCE_N_SATURATION)

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
        error_cause: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """Create error result when simulation cannot complete.

        ``error_cause`` / ``error_details`` are the effect engine's cause and counts, when it
        named one; ``error_message`` is unchanged by them.
        """
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
            error_cause=error_cause,
            error_details=dict(error_details or {}),
            execution_time_ms=execution_time_ms,
        )
