"""Interaction Detection Node - NO LLM.

Detects feature interactions using correlation-based methods.
This is a deterministic computation node with no LLM calls.
"""

import time
from typing import Any, Dict, List, Tuple

import numpy as np


async def detect_interactions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Detect feature interactions from SHAP values.

    This node:
    1. Computes pairwise correlations between SHAP values
    2. Identifies strong interactions (high correlation)
    3. Ranks interactions by strength
    4. Returns top interactions for interpretation

    Args:
        state: Current agent state with SHAP values

    Returns:
        State updates with interaction matrix and top interactions
    """
    start_time = time.time()

    try:
        # Extract inputs
        shap_values = state.get("shap_values")
        feature_names = state.get("feature_names", [])
        compute_interactions = state.get("compute_interactions", True)

        if not compute_interactions:
            # Skip interaction detection
            return {
                "interaction_matrix": {},
                "top_interactions_raw": [],
                "interaction_computation_time_seconds": 0.0,
                "interaction_method": "skipped",
            }

        if shap_values is None or len(feature_names) == 0:
            return {
                "error": "Missing SHAP values or feature names for interaction detection",
                "error_type": "missing_shap_data",
                "status": "failed",
            }

        # Compute interaction matrix using correlation method
        interaction_matrix, interaction_method = _compute_correlation_interactions(
            shap_values, feature_names
        )

        # Extract top interactions (sorted by absolute strength)
        top_interactions_raw = _extract_top_interactions(interaction_matrix, top_k=10)

        computation_time = time.time() - start_time

        return {
            "interaction_matrix": interaction_matrix,
            "top_interactions_raw": top_interactions_raw,
            "interaction_computation_time_seconds": computation_time,
            "interaction_method": interaction_method,
        }

    except Exception as e:
        return {
            "error": f"Interaction detection failed: {str(e)}",
            "error_type": "interaction_detection_error",
            "error_details": {"exception": str(e)},
            "status": "failed",
        }


def _compute_correlation_interactions(
    shap_values: np.ndarray, feature_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """Compute feature interactions using SHAP value correlations.

    Strong correlations between SHAP values indicate interactions:
    - Positive correlation: Features interact to amplify effects
    - Negative correlation: Features interact to cancel effects

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names

    Returns:
        Tuple of (interaction_matrix, method_name)
    """
    n_features = shap_values.shape[1]

    # Compute correlation matrix between SHAP value columns
    correlation_matrix = np.corrcoef(shap_values.T)

    # Build interaction matrix as nested dict
    interaction_matrix = {}

    for i in range(n_features):
        feature_i = feature_names[i]
        interaction_matrix[feature_i] = {}

        for j in range(n_features):
            if i == j:
                continue  # Skip self-interactions

            feature_j = feature_names[j]
            correlation = correlation_matrix[i, j]

            # Store interaction strength (correlation coefficient)
            # Only store if abs(correlation) > threshold
            if abs(correlation) > 0.1:  # Weak correlation threshold
                interaction_matrix[feature_i][feature_j] = float(correlation)

    return interaction_matrix, "correlation"


def _extract_top_interactions(
    interaction_matrix: Dict[str, Dict[str, float]], top_k: int = 10
) -> List[Tuple[str, str, float]]:
    """Extract top-k feature interactions by strength.

    Args:
        interaction_matrix: Nested dict of interactions
        top_k: Number of top interactions to return

    Returns:
        List of (feature1, feature2, strength) tuples sorted by |strength|
    """
    interactions = []

    # Collect all interactions (avoiding duplicates)
    seen_pairs = set()

    for feature_i, interactions_dict in interaction_matrix.items():
        for feature_j, strength in interactions_dict.items():
            # Create sorted pair to avoid duplicates
            pair = tuple(sorted([feature_i, feature_j]))

            if pair not in seen_pairs:
                seen_pairs.add(pair)
                interactions.append((feature_i, feature_j, strength))

    # Sort by absolute strength
    interactions_sorted = sorted(interactions, key=lambda x: abs(x[2]), reverse=True)

    return interactions_sorted[:top_k]
