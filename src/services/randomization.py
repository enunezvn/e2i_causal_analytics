"""
Randomization Service for A/B Testing.

Phase 15: A/B Testing Infrastructure

Provides randomization algorithms for experiment assignment:
- Simple random assignment
- Stratified randomization
- Block randomization
- Multi-arm allocation
- Deterministic hash-based assignment for reproducibility
"""

import hashlib
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class RandomizationMethod(str, Enum):
    """Randomization method types."""

    SIMPLE = "simple"
    STRATIFIED = "stratified"
    BLOCK = "block"
    CLUSTER = "cluster"
    ADAPTIVE = "adaptive"


class UnitType(str, Enum):
    """Unit types for experiment assignment."""

    HCP = "hcp"
    PATIENT = "patient"
    TERRITORY = "territory"
    ACCOUNT = "account"


@dataclass
class AssignmentResult:
    """Result of a randomization assignment."""

    unit_id: str
    unit_type: UnitType
    variant: str
    experiment_id: UUID
    randomization_method: RandomizationMethod
    assigned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stratification_key: Optional[Dict[str, Any]] = None
    block_id: Optional[str] = None
    assignment_hash: Optional[str] = None


@dataclass
class RandomizationConfig:
    """Configuration for randomization."""

    # Allocation ratios per variant
    allocation_ratio: Dict[str, float] = field(
        default_factory=lambda: {"control": 0.5, "treatment": 0.5}
    )

    # Stratification columns (for stratified randomization)
    strata_columns: List[str] = field(default_factory=list)

    # Block size (for block randomization)
    block_size: int = 10

    # Salt for deterministic hashing
    salt: str = "e2i_experiment_salt"

    # Whether to use deterministic assignment
    deterministic: bool = True

    def __post_init__(self):
        """Validate allocation ratios."""
        total = sum(self.allocation_ratio.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Allocation ratios must sum to 1.0, got {total}")


class RandomizationService:
    """
    Service for randomizing units to experiment variants.

    Supports multiple randomization methods to ensure balance
    across covariates and reproducible assignments.
    """

    def __init__(self, config: Optional[RandomizationConfig] = None):
        """
        Initialize randomization service.

        Args:
            config: Randomization configuration
        """
        self.config = config or RandomizationConfig()
        self._block_counters: Dict[str, Dict[str, int]] = {}

    def _generate_assignment_hash(
        self,
        experiment_id: UUID,
        unit_id: str,
        salt: Optional[str] = None,
    ) -> str:
        """
        Generate deterministic assignment hash.

        Args:
            experiment_id: Experiment UUID
            unit_id: Unit identifier
            salt: Optional salt override

        Returns:
            SHA-256 hash string
        """
        salt = salt or self.config.salt
        data = f"{experiment_id}:{unit_id}:{salt}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _hash_to_unit_interval(self, hash_str: str) -> float:
        """
        Convert hash to value in [0, 1) interval.

        Args:
            hash_str: SHA-256 hash string

        Returns:
            Float in [0, 1)
        """
        # Use first 16 hex chars (64 bits) for precision
        hash_int = int(hash_str[:16], 16)
        return hash_int / (2**64)

    def _select_variant(
        self,
        random_value: float,
        allocation_ratio: Dict[str, float],
    ) -> str:
        """
        Select variant based on random value and allocation ratios.

        Args:
            random_value: Random value in [0, 1)
            allocation_ratio: Variant allocation ratios

        Returns:
            Selected variant name
        """
        cumulative = 0.0
        for variant, ratio in sorted(allocation_ratio.items()):
            cumulative += ratio
            if random_value < cumulative:
                return variant

        # Fallback to last variant (shouldn't happen with valid ratios)
        return list(allocation_ratio.keys())[-1]

    async def simple_randomize(
        self,
        experiment_id: UUID,
        units: List[Dict[str, Any]],
        unit_type: UnitType = UnitType.HCP,
        allocation_ratio: Optional[Dict[str, float]] = None,
    ) -> List[AssignmentResult]:
        """
        Simple random assignment.

        Each unit is independently assigned based on allocation ratios.

        Args:
            experiment_id: Experiment UUID
            units: List of unit dictionaries with 'id' key
            unit_type: Type of unit
            allocation_ratio: Optional override for allocation ratios

        Returns:
            List of assignment results
        """
        ratio = allocation_ratio or self.config.allocation_ratio
        results = []

        for unit in units:
            unit_id = str(unit.get("id") or unit.get("unit_id"))

            if self.config.deterministic:
                hash_str = self._generate_assignment_hash(experiment_id, unit_id)
                random_value = self._hash_to_unit_interval(hash_str)
            else:
                random_value = random.random()
                hash_str = None

            variant = self._select_variant(random_value, ratio)

            results.append(
                AssignmentResult(
                    unit_id=unit_id,
                    unit_type=unit_type,
                    variant=variant,
                    experiment_id=experiment_id,
                    randomization_method=RandomizationMethod.SIMPLE,
                    assignment_hash=hash_str,
                )
            )

        logger.info(
            f"Simple randomization complete: {len(results)} units assigned "
            f"to experiment {experiment_id}"
        )
        return results

    async def stratified_randomize(
        self,
        experiment_id: UUID,
        units: List[Dict[str, Any]],
        strata_columns: List[str],
        unit_type: UnitType = UnitType.HCP,
        allocation_ratio: Optional[Dict[str, float]] = None,
    ) -> List[AssignmentResult]:
        """
        Stratified randomization.

        Ensures balanced assignment within each stratum defined by covariates.

        Args:
            experiment_id: Experiment UUID
            units: List of unit dictionaries with 'id' and stratification columns
            strata_columns: Columns to stratify on
            unit_type: Type of unit
            allocation_ratio: Optional override for allocation ratios

        Returns:
            List of assignment results
        """
        ratio = allocation_ratio or self.config.allocation_ratio
        results = []

        # Group units by strata
        strata_groups: Dict[str, List[Dict]] = {}
        for unit in units:
            strata_key = tuple(str(unit.get(col, "missing")) for col in strata_columns)
            strata_key_str = "|".join(strata_key)
            if strata_key_str not in strata_groups:
                strata_groups[strata_key_str] = []
            strata_groups[strata_key_str].append(unit)

        # Randomize within each stratum
        for strata_key_str, stratum_units in strata_groups.items():
            strata_key_dict = dict(zip(strata_columns, strata_key_str.split("|"), strict=False))

            for unit in stratum_units:
                unit_id = str(unit.get("id") or unit.get("unit_id"))

                if self.config.deterministic:
                    # Include stratum in hash for balance within strata
                    hash_str = self._generate_assignment_hash(
                        experiment_id,
                        f"{unit_id}:{strata_key_str}",
                    )
                    random_value = self._hash_to_unit_interval(hash_str)
                else:
                    random_value = random.random()
                    hash_str = None

                variant = self._select_variant(random_value, ratio)

                results.append(
                    AssignmentResult(
                        unit_id=unit_id,
                        unit_type=unit_type,
                        variant=variant,
                        experiment_id=experiment_id,
                        randomization_method=RandomizationMethod.STRATIFIED,
                        stratification_key=strata_key_dict,
                        assignment_hash=hash_str,
                    )
                )

        logger.info(
            f"Stratified randomization complete: {len(results)} units in "
            f"{len(strata_groups)} strata assigned to experiment {experiment_id}"
        )
        return results

    async def block_randomize(
        self,
        experiment_id: UUID,
        units: List[Dict[str, Any]],
        block_size: Optional[int] = None,
        unit_type: UnitType = UnitType.HCP,
        allocation_ratio: Optional[Dict[str, float]] = None,
    ) -> List[AssignmentResult]:
        """
        Block randomization.

        Ensures balanced assignment within fixed-size blocks.

        Args:
            experiment_id: Experiment UUID
            units: List of unit dictionaries with 'id' key
            block_size: Size of each randomization block
            unit_type: Type of unit
            allocation_ratio: Optional override for allocation ratios

        Returns:
            List of assignment results
        """
        ratio = allocation_ratio or self.config.allocation_ratio
        block_size = block_size or self.config.block_size
        results = []

        # Generate block assignments
        blocks = self._generate_blocks(
            num_units=len(units),
            block_size=block_size,
            allocation_ratio=ratio,
            experiment_id=experiment_id,
        )

        for i, unit in enumerate(units):
            unit_id = str(unit.get("id") or unit.get("unit_id"))
            block_num = i // block_size
            block_id = f"{experiment_id}:block_{block_num}"
            variant = blocks[i]

            hash_str = self._generate_assignment_hash(experiment_id, unit_id)

            results.append(
                AssignmentResult(
                    unit_id=unit_id,
                    unit_type=unit_type,
                    variant=variant,
                    experiment_id=experiment_id,
                    randomization_method=RandomizationMethod.BLOCK,
                    block_id=block_id,
                    assignment_hash=hash_str,
                )
            )

        logger.info(
            f"Block randomization complete: {len(results)} units in "
            f"{(len(units) + block_size - 1) // block_size} blocks "
            f"assigned to experiment {experiment_id}"
        )
        return results

    def _generate_blocks(
        self,
        num_units: int,
        block_size: int,
        allocation_ratio: Dict[str, float],
        experiment_id: UUID,
    ) -> List[str]:
        """
        Generate block assignments for all units.

        Args:
            num_units: Total number of units
            block_size: Size of each block
            allocation_ratio: Variant allocation ratios
            experiment_id: For deterministic shuffling

        Returns:
            List of variant assignments
        """
        assignments = []

        # Calculate assignments per variant per block
        variant_counts = {
            variant: max(1, round(ratio * block_size))
            for variant, ratio in allocation_ratio.items()
        }

        # Adjust for rounding errors
        total = sum(variant_counts.values())
        if total != block_size:
            # Add/remove from first variant
            first_variant = list(variant_counts.keys())[0]
            variant_counts[first_variant] += block_size - total

        num_blocks = (num_units + block_size - 1) // block_size

        for block_num in range(num_blocks):
            # Create block with correct counts
            block = []
            for variant, count in variant_counts.items():
                block.extend([variant] * count)

            # Shuffle block deterministically
            seed = int(hashlib.md5(f"{experiment_id}:block_{block_num}".encode()).hexdigest(), 16)
            rng = random.Random(seed)
            rng.shuffle(block)

            assignments.extend(block)

        return assignments[:num_units]

    async def multi_arm_allocate(
        self,
        experiment_id: UUID,
        unit: Dict[str, Any],
        arm_probabilities: Dict[str, float],
        unit_type: UnitType = UnitType.HCP,
    ) -> AssignmentResult:
        """
        Multi-arm allocation for a single unit.

        Supports experiments with more than two treatment arms.

        Args:
            experiment_id: Experiment UUID
            unit: Unit dictionary with 'id' key
            arm_probabilities: Probability for each arm
            unit_type: Type of unit

        Returns:
            Assignment result
        """
        unit_id = str(unit.get("id") or unit.get("unit_id"))

        # Validate probabilities
        total_prob = sum(arm_probabilities.values())
        if abs(total_prob - 1.0) > 0.001:
            raise ValueError(f"Arm probabilities must sum to 1.0, got {total_prob}")

        if self.config.deterministic:
            hash_str = self._generate_assignment_hash(experiment_id, unit_id)
            random_value = self._hash_to_unit_interval(hash_str)
        else:
            random_value = random.random()
            hash_str = None

        variant = self._select_variant(random_value, arm_probabilities)

        return AssignmentResult(
            unit_id=unit_id,
            unit_type=unit_type,
            variant=variant,
            experiment_id=experiment_id,
            randomization_method=RandomizationMethod.SIMPLE,
            assignment_hash=hash_str,
        )

    async def batch_multi_arm_allocate(
        self,
        experiment_id: UUID,
        units: List[Dict[str, Any]],
        arm_probabilities: Dict[str, float],
        unit_type: UnitType = UnitType.HCP,
    ) -> List[AssignmentResult]:
        """
        Multi-arm allocation for multiple units.

        Args:
            experiment_id: Experiment UUID
            units: List of unit dictionaries
            arm_probabilities: Probability for each arm
            unit_type: Type of unit

        Returns:
            List of assignment results
        """
        results = []
        for unit in units:
            result = await self.multi_arm_allocate(
                experiment_id=experiment_id,
                unit=unit,
                arm_probabilities=arm_probabilities,
                unit_type=unit_type,
            )
            results.append(result)
        return results

    def verify_assignment(
        self,
        experiment_id: UUID,
        unit_id: str,
        expected_variant: str,
        salt: Optional[str] = None,
    ) -> bool:
        """
        Verify a unit's assignment is correct.

        Useful for auditing and debugging randomization issues.

        Args:
            experiment_id: Experiment UUID
            unit_id: Unit identifier
            expected_variant: Expected variant assignment
            salt: Optional salt override

        Returns:
            True if assignment matches
        """
        hash_str = self._generate_assignment_hash(experiment_id, unit_id, salt)
        random_value = self._hash_to_unit_interval(hash_str)
        actual_variant = self._select_variant(random_value, self.config.allocation_ratio)
        return actual_variant == expected_variant

    def get_allocation_summary(
        self,
        assignments: List[AssignmentResult],
    ) -> Dict[str, Any]:
        """
        Get summary statistics for assignments.

        Args:
            assignments: List of assignment results

        Returns:
            Summary statistics
        """
        variant_counts: Dict[str, int] = {}
        strata_counts: Dict[str, Dict[str, int]] = {}

        for assignment in assignments:
            # Count by variant
            variant_counts[assignment.variant] = variant_counts.get(assignment.variant, 0) + 1

            # Count by strata if stratified
            if assignment.stratification_key:
                strata_key = "|".join(
                    f"{k}={v}" for k, v in sorted(assignment.stratification_key.items())
                )
                if strata_key not in strata_counts:
                    strata_counts[strata_key] = {}
                strata_counts[strata_key][assignment.variant] = (
                    strata_counts[strata_key].get(assignment.variant, 0) + 1
                )

        total = sum(variant_counts.values())
        return {
            "total_units": total,
            "variant_counts": variant_counts,
            "variant_proportions": {v: c / total for v, c in variant_counts.items()}
            if total > 0
            else {},
            "strata_counts": strata_counts if strata_counts else None,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def get_randomization_service(
    config: Optional[RandomizationConfig] = None,
) -> RandomizationService:
    """Get randomization service instance."""
    return RandomizationService(config)
