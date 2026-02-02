"""
Ground Truth Causal Effects Storage

Stores known TRUE_ATE and CATE values for each DGP/brand combination
to enable pipeline validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from ..config import DGP_CONFIGS, Brand, DGPType


@dataclass
class GroundTruthEffect:
    """
    Ground truth causal effect for a specific DGP/brand combination.

    Stores the TRUE_ATE that the pipeline must recover.
    """

    brand: Brand
    dgp_type: DGPType
    true_ate: float
    tolerance: float
    confounders: List[str]
    treatment_variable: str
    outcome_variable: str
    generation_timestamp: datetime = field(default_factory=datetime.now)

    # For heterogeneous effects
    cate_by_segment: Optional[Dict[str, float]] = None

    # Metadata
    n_samples: int = 0
    data_split_counts: Dict[str, int] = field(default_factory=dict)

    def is_estimate_valid(self, estimated_ate: float) -> bool:
        """Check if an estimated ATE is within tolerance of ground truth."""
        return abs(estimated_ate - self.true_ate) <= self.tolerance

    def get_error(self, estimated_ate: float) -> float:
        """Get the absolute error between estimate and ground truth."""
        return abs(estimated_ate - self.true_ate)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "brand": self.brand.value if isinstance(self.brand, Brand) else self.brand,
            "dgp_type": (
                self.dgp_type.value if isinstance(self.dgp_type, DGPType) else self.dgp_type
            ),
            "true_ate": self.true_ate,
            "tolerance": self.tolerance,
            "confounders": self.confounders,
            "treatment_variable": self.treatment_variable,
            "outcome_variable": self.outcome_variable,
            "generation_timestamp": self.generation_timestamp.isoformat(),
            "cate_by_segment": self.cate_by_segment,
            "n_samples": self.n_samples,
            "data_split_counts": self.data_split_counts,
        }


class GroundTruthStore:
    """
    In-memory store for ground truth causal effects.

    Tracks all generated datasets and their known effects.
    """

    def __init__(self):
        self._effects: Dict[str, GroundTruthEffect] = {}

    def _make_key(self, brand: Brand, dgp_type: DGPType) -> str:
        """Create a unique key for brand/dgp combination."""
        brand_val = brand.value if isinstance(brand, Brand) else brand
        dgp_val = dgp_type.value if isinstance(dgp_type, DGPType) else dgp_type
        return f"{brand_val}_{dgp_val}"

    def store(self, effect: GroundTruthEffect) -> None:
        """Store a ground truth effect."""
        key = self._make_key(effect.brand, effect.dgp_type)
        self._effects[key] = effect

    def get(self, brand: Brand, dgp_type: DGPType) -> Optional[GroundTruthEffect]:
        """Retrieve a ground truth effect."""
        key = self._make_key(brand, dgp_type)
        return self._effects.get(key)

    def get_all(self) -> List[GroundTruthEffect]:
        """Get all stored ground truth effects."""
        return list(self._effects.values())

    def validate_estimate(self, brand: Brand, dgp_type: DGPType, estimated_ate: float) -> Dict:
        """
        Validate an estimated ATE against ground truth.

        Returns:
            Dict with validation results including:
            - is_valid: bool
            - true_ate: float
            - estimated_ate: float
            - error: float
            - tolerance: float
        """
        effect = self.get(brand, dgp_type)
        if effect is None:
            return {
                "is_valid": False,
                "error": "No ground truth found for brand/dgp combination",
                "brand": brand.value if isinstance(brand, Brand) else brand,
                "dgp_type": dgp_type.value if isinstance(dgp_type, DGPType) else dgp_type,
            }

        return {
            "is_valid": effect.is_estimate_valid(estimated_ate),
            "true_ate": effect.true_ate,
            "estimated_ate": estimated_ate,
            "error": effect.get_error(estimated_ate),
            "tolerance": effect.tolerance,
            "brand": brand.value if isinstance(brand, Brand) else brand,
            "dgp_type": dgp_type.value if isinstance(dgp_type, DGPType) else dgp_type,
        }

    def clear(self) -> None:
        """Clear all stored effects."""
        self._effects.clear()


# Global store instance
_GLOBAL_STORE = GroundTruthStore()


def get_ground_truth(brand: Brand, dgp_type: DGPType) -> Optional[GroundTruthEffect]:
    """Get ground truth from the global store."""
    return _GLOBAL_STORE.get(brand, dgp_type)


def validate_estimate(brand: Brand, dgp_type: DGPType, estimated_ate: float) -> Dict:
    """Validate an estimate against the global store."""
    return _GLOBAL_STORE.validate_estimate(brand, dgp_type, estimated_ate)


def create_ground_truth_from_dgp_config(
    brand: Brand,
    dgp_type: DGPType,
    n_samples: int = 0,
    data_split_counts: Optional[Dict[str, int]] = None,
) -> GroundTruthEffect:
    """
    Create a GroundTruthEffect from a DGP configuration.

    Args:
        brand: The pharmaceutical brand
        dgp_type: The DGP type
        n_samples: Number of samples generated
        data_split_counts: Count of samples in each split

    Returns:
        GroundTruthEffect with known causal parameters
    """
    dgp_config = DGP_CONFIGS[dgp_type]

    return GroundTruthEffect(
        brand=brand,
        dgp_type=dgp_type,
        true_ate=dgp_config.true_ate,
        tolerance=dgp_config.tolerance,
        confounders=dgp_config.confounders,
        treatment_variable=dgp_config.treatment_variable,
        outcome_variable=dgp_config.outcome_variable,
        cate_by_segment=dgp_config.cate_by_segment,
        n_samples=n_samples,
        data_split_counts=data_split_counts or {},
    )


def store_ground_truth(effect: GroundTruthEffect) -> None:
    """Store a ground truth effect in the global store."""
    _GLOBAL_STORE.store(effect)


def get_global_store() -> GroundTruthStore:
    """Get the global ground truth store."""
    return _GLOBAL_STORE
