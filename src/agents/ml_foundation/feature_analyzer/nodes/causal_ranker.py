"""Causal Ranker Node - NO LLM.

Integrates DriverRanker to compare causal vs predictive feature importance.
Uses DiscoveryRunner for DAG learning and DriverRanker for comparison.

This is a deterministic computation node with no LLM calls.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from src.causal_engine.discovery.base import DiscoveryAlgorithmType, DiscoveryConfig
from src.causal_engine.discovery.driver_ranker import DriverRanker
from src.causal_engine.discovery.gate import DiscoveryGate, GateConfig
from src.causal_engine.discovery.runner import DiscoveryRunner


def _build_discovery_config(config_dict: Optional[Dict[str, Any]]) -> DiscoveryConfig:
    """Build DiscoveryConfig from dict, handling algorithm type conversion.

    Args:
        config_dict: Configuration dictionary or None

    Returns:
        DiscoveryConfig instance
    """
    if config_dict is None:
        return DiscoveryConfig()

    # Convert algorithm strings to enum if needed
    if "algorithms" in config_dict:
        algos = config_dict["algorithms"]
        converted = []
        for algo in algos:
            if isinstance(algo, str):
                try:
                    converted.append(DiscoveryAlgorithmType(algo.lower()))
                except ValueError:
                    logger.warning(f"Unknown algorithm '{algo}', skipping")
            else:
                converted.append(algo)
        config_dict = {**config_dict, "algorithms": converted}

    return DiscoveryConfig(**config_dict)


async def rank_causal_drivers(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compare causal vs predictive feature importance using DriverRanker.

    This node:
    1. Checks if discovery is enabled
    2. Runs causal discovery using DiscoveryRunner
    3. Evaluates discovery quality with DiscoveryGate
    4. Compares causal vs predictive importance with DriverRanker
    5. Returns rankings, divergent features, and correlations

    Args:
        state: Current agent state with:
            - discovery_enabled: bool - Whether to run discovery
            - discovery_config: dict - DiscoveryConfig parameters
            - X_sample: Training data for discovery
            - causal_target_variable: Target for causal analysis
            - shap_values: SHAP values for predictive importance
            - feature_names: Feature names
            - global_importance: Dict of predictive importance scores

    Returns:
        State updates with causal rankings and comparison metrics
    """
    start_time = time.time()

    # Check if discovery is enabled
    if not state.get("discovery_enabled", False):
        logger.debug("Causal discovery disabled, skipping causal ranker node")
        return {}

    try:
        # Get required inputs
        X_sample = state.get("X_sample")
        target = state.get("causal_target_variable")
        shap_values = state.get("shap_values")
        feature_names = state.get("feature_names", [])
        state.get("global_importance", {})

        # Validate inputs
        if X_sample is None:
            return {
                "error": "Missing X_sample for causal discovery",
                "error_type": "missing_data",
                "discovery_gate_decision": "reject",
            }

        if not target:
            # Fallback: use y_sample column name if available
            y_sample = state.get("y_sample")
            if y_sample is not None and hasattr(y_sample, "name") and y_sample.name:
                target = y_sample.name
            else:
                target = "target"
                logger.warning(f"No causal_target_variable specified, using '{target}'")

        if shap_values is None:
            return {
                "error": "Missing shap_values for predictive comparison",
                "error_type": "missing_shap",
                "discovery_gate_decision": "reject",
            }

        # Prepare data for discovery
        if isinstance(X_sample, np.ndarray):
            if not feature_names:
                feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]
            data_df = pd.DataFrame(X_sample, columns=feature_names)
        else:
            data_df = X_sample
            if not feature_names:
                feature_names = list(data_df.columns)

        # Add target to data if we have y_sample
        y_sample = state.get("y_sample")
        if y_sample is not None:
            if isinstance(y_sample, np.ndarray):
                data_df[target] = y_sample
            else:
                data_df[target] = y_sample.values if hasattr(y_sample, "values") else y_sample

        # Build discovery config
        config_dict = state.get("discovery_config", {})
        discovery_config = _build_discovery_config(config_dict)

        logger.info(
            f"Running causal discovery with {len(discovery_config.algorithms)} algorithms "
            f"on {len(feature_names)} features"
        )

        # Run discovery
        runner = DiscoveryRunner()
        discovery_result = await runner.discover_dag(data_df, discovery_config)

        # Evaluate with gate
        gate_config = GateConfig(
            accept_threshold=config_dict.get("accept_threshold", 0.8),
            review_threshold=config_dict.get("review_threshold", 0.5),
            augment_edge_threshold=config_dict.get("augment_edge_threshold", 0.9),
        )
        gate = DiscoveryGate(config=gate_config)
        gate_evaluation = gate.evaluate(discovery_result)

        # Rank drivers if discovery succeeded
        if discovery_result.success and discovery_result.ensemble_dag is not None:
            ranker = DriverRanker()

            # Get feature names without target
            ranking_features = [f for f in feature_names if f != target]

            # Filter SHAP values to match feature order
            if (
                isinstance(shap_values, np.ndarray)
                and len(ranking_features) == shap_values.shape[1]
            ):
                filtered_shap = shap_values
            else:
                # Reorder SHAP values to match ranking_features
                shap_indices = [
                    feature_names.index(f) for f in ranking_features if f in feature_names
                ]
                filtered_shap = (
                    shap_values[:, shap_indices] if len(shap_indices) > 0 else shap_values
                )

            ranking_result = ranker.rank_from_discovery_result(
                result=discovery_result,
                target=target,
                shap_values=filtered_shap,
            )

            # Build output
            causal_rankings = [
                {
                    "feature_name": r.feature_name,
                    "causal_rank": r.causal_rank,
                    "predictive_rank": r.predictive_rank,
                    "causal_score": r.causal_score,
                    "predictive_score": r.predictive_score,
                    "rank_difference": r.rank_difference,
                    "is_direct_cause": r.is_direct_cause,
                    "path_length": r.path_length,
                }
                for r in ranking_result.rankings
            ]

            # Build causal importance dict
            causal_importance = {r.feature_name: r.causal_score for r in ranking_result.rankings}
            causal_importance_ranked = sorted(
                causal_importance.items(), key=lambda x: x[1], reverse=True
            )

            # Identify divergent features (rank difference > 3)
            divergent_threshold = config_dict.get("divergent_threshold", 3)
            divergent_features = [
                r.feature_name
                for r in ranking_result.rankings
                if abs(r.rank_difference) > divergent_threshold
            ]

            # Direct cause features
            direct_cause_features = [
                r.feature_name for r in ranking_result.rankings if r.is_direct_cause
            ]

            computation_time = time.time() - start_time

            logger.info(
                f"Causal discovery complete: gate={gate_evaluation.decision.value}, "
                f"correlation={ranking_result.rank_correlation:.3f}, "
                f"divergent={len(divergent_features)}, time={computation_time:.2f}s"
            )

            return {
                # Discovery results
                "discovery_result": discovery_result.to_dict(),
                "discovery_gate_decision": gate_evaluation.decision.value,
                "discovery_gate_confidence": gate_evaluation.confidence,
                "discovery_gate_reasons": gate_evaluation.reasons,
                # Causal rankings
                "causal_target_variable": target,
                "causal_rankings": causal_rankings,
                "rank_correlation": ranking_result.rank_correlation,
                # Feature categorization
                "divergent_features": divergent_features,
                "causal_only_features": ranking_result.causal_only_features,
                "predictive_only_features": ranking_result.predictive_only_features,
                "concordant_features": ranking_result.concordant_features,
                # Causal importance
                "causal_importance": causal_importance,
                "causal_importance_ranked": causal_importance_ranked,
                "direct_cause_features": direct_cause_features,
                # Metadata
                "causal_ranker_time_seconds": computation_time,
            }

        else:
            # Discovery failed
            computation_time = time.time() - start_time
            logger.warning(
                f"Causal discovery did not succeed: gate={gate_evaluation.decision.value}"
            )

            return {
                "discovery_result": discovery_result.to_dict() if discovery_result else None,
                "discovery_gate_decision": gate_evaluation.decision.value,
                "discovery_gate_confidence": gate_evaluation.confidence,
                "discovery_gate_reasons": gate_evaluation.reasons,
                "causal_rankings": [],
                "rank_correlation": 0.0,
                "divergent_features": [],
                "causal_only_features": [],
                "predictive_only_features": [],
                "concordant_features": [],
                "causal_importance": {},
                "causal_importance_ranked": [],
                "direct_cause_features": [],
                "causal_ranker_time_seconds": computation_time,
            }

    except Exception as e:
        computation_time = time.time() - start_time
        logger.exception(f"Causal ranker failed: {e}")
        return {
            "error": f"Causal ranking failed: {str(e)}",
            "error_type": "causal_ranker_error",
            "error_details": {"exception": str(e)},
            "discovery_gate_decision": "reject",
            "causal_ranker_time_seconds": computation_time,
        }
