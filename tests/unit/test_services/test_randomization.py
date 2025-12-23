"""
Unit Tests for Randomization Service (Phase 15).

Tests cover:
- Simple randomization
- Stratified randomization
- Block randomization
- Multi-arm allocation
- Deterministic hash-based assignments
- Allocation balance verification
"""

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from src.services.randomization import (
    AssignmentResult,
    RandomizationConfig,
    RandomizationMethod,
    RandomizationService,
    UnitType,
    get_randomization_service,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> RandomizationConfig:
    """Create default randomization configuration."""
    return RandomizationConfig(
        allocation_ratio={"control": 0.5, "treatment": 0.5},
        strata_columns=[],
        block_size=10,
        salt="test_salt",
        deterministic=True,
    )


@pytest.fixture
def multi_arm_config() -> RandomizationConfig:
    """Create multi-arm allocation configuration."""
    return RandomizationConfig(
        allocation_ratio={
            "control": 0.25,
            "treatment_a": 0.25,
            "treatment_b": 0.25,
            "treatment_c": 0.25,
        },
        deterministic=True,
    )


@pytest.fixture
def service(default_config: RandomizationConfig) -> RandomizationService:
    """Create randomization service instance."""
    return RandomizationService(config=default_config)


@pytest.fixture
def multi_arm_service(multi_arm_config: RandomizationConfig) -> RandomizationService:
    """Create multi-arm randomization service instance."""
    return RandomizationService(config=multi_arm_config)


@pytest.fixture
def experiment_id() -> UUID:
    """Create test experiment ID."""
    return uuid4()


@pytest.fixture
def sample_units() -> List[Dict[str, Any]]:
    """Create sample units for randomization."""
    return [{"id": f"unit_{i}", "region": "north" if i % 2 == 0 else "south"} for i in range(100)]


@pytest.fixture
def stratified_units() -> List[Dict[str, Any]]:
    """Create units with stratification attributes."""
    units = []
    for i in range(100):
        units.append({
            "id": f"unit_{i}",
            "region": ["north", "south", "east", "west"][i % 4],
            "tier": ["high", "medium", "low"][i % 3],
        })
    return units


# =============================================================================
# UNIT TYPE TESTS
# =============================================================================


class TestUnitType:
    """Tests for UnitType enum."""

    def test_hcp_type_exists(self):
        """Test HCP type is defined."""
        assert UnitType.HCP == "hcp"

    def test_patient_type_exists(self):
        """Test PATIENT type is defined."""
        assert UnitType.PATIENT == "patient"

    def test_territory_type_exists(self):
        """Test TERRITORY type is defined."""
        assert UnitType.TERRITORY == "territory"

    def test_account_type_exists(self):
        """Test ACCOUNT type is defined."""
        assert UnitType.ACCOUNT == "account"


# =============================================================================
# RANDOMIZATION METHOD TESTS
# =============================================================================


class TestRandomizationMethod:
    """Tests for RandomizationMethod enum."""

    def test_simple_method_exists(self):
        """Test SIMPLE method is defined."""
        assert RandomizationMethod.SIMPLE == "simple"

    def test_stratified_method_exists(self):
        """Test STRATIFIED method is defined."""
        assert RandomizationMethod.STRATIFIED == "stratified"

    def test_block_method_exists(self):
        """Test BLOCK method is defined."""
        assert RandomizationMethod.BLOCK == "block"

    def test_cluster_method_exists(self):
        """Test CLUSTER method is defined."""
        assert RandomizationMethod.CLUSTER == "cluster"


# =============================================================================
# RANDOMIZATION CONFIG TESTS
# =============================================================================


class TestRandomizationConfig:
    """Tests for RandomizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RandomizationConfig()
        assert config.allocation_ratio == {"control": 0.5, "treatment": 0.5}
        assert config.block_size == 10
        assert config.deterministic is True

    def test_custom_allocation_ratio(self):
        """Test custom allocation ratio."""
        config = RandomizationConfig(
            allocation_ratio={"control": 0.3, "treatment": 0.7}
        )
        assert config.allocation_ratio["control"] == 0.3
        assert config.allocation_ratio["treatment"] == 0.7

    def test_invalid_allocation_ratio_raises_error(self):
        """Test that allocation ratios not summing to 1.0 raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RandomizationConfig(allocation_ratio={"control": 0.3, "treatment": 0.3})

    def test_strata_columns_default_empty(self):
        """Test strata columns default to empty list."""
        config = RandomizationConfig()
        assert config.strata_columns == []


# =============================================================================
# SIMPLE RANDOMIZATION TESTS
# =============================================================================


class TestSimpleRandomization:
    """Tests for simple randomization."""

    @pytest.mark.asyncio
    async def test_simple_randomize_returns_results(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test simple randomization returns assignment results."""
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )

        assert len(results) == len(sample_units)
        assert all(isinstance(r, AssignmentResult) for r in results)

    @pytest.mark.asyncio
    async def test_simple_randomize_assigns_correct_variants(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test simple randomization assigns control/treatment."""
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )

        variants = {r.variant for r in results}
        assert variants == {"control", "treatment"}

    @pytest.mark.asyncio
    async def test_simple_randomize_deterministic(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test deterministic randomization produces same results."""
        results1 = await service.simple_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )
        results2 = await service.simple_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )

        for r1, r2 in zip(results1, results2):
            assert r1.variant == r2.variant
            assert r1.assignment_hash == r2.assignment_hash

    @pytest.mark.asyncio
    async def test_simple_randomize_balance(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test simple randomization achieves approximate balance."""
        units = [{"id": f"unit_{i}"} for i in range(1000)]
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=units,
        )

        counts = Counter(r.variant for r in results)
        # With 1000 units and 50/50 split, expect approximately 500 each
        # Allow for some variance (40-60%)
        assert 400 <= counts["control"] <= 600
        assert 400 <= counts["treatment"] <= 600

    @pytest.mark.asyncio
    async def test_simple_randomize_with_custom_ratio(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test simple randomization with custom allocation ratio."""
        units = [{"id": f"unit_{i}"} for i in range(1000)]
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=units,
            allocation_ratio={"control": 0.3, "treatment": 0.7},
        )

        counts = Counter(r.variant for r in results)
        # Expect approximately 30% control, 70% treatment
        assert 200 <= counts["control"] <= 400
        assert 600 <= counts["treatment"] <= 800

    @pytest.mark.asyncio
    async def test_simple_randomize_sets_correct_method(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test simple randomization sets SIMPLE method."""
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )

        assert all(r.randomization_method == RandomizationMethod.SIMPLE for r in results)

    @pytest.mark.asyncio
    async def test_simple_randomize_sets_experiment_id(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test simple randomization sets experiment ID."""
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )

        assert all(r.experiment_id == experiment_id for r in results)


# =============================================================================
# STRATIFIED RANDOMIZATION TESTS
# =============================================================================


class TestStratifiedRandomization:
    """Tests for stratified randomization."""

    @pytest.mark.asyncio
    async def test_stratified_randomize_returns_results(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        stratified_units: List[Dict],
    ):
        """Test stratified randomization returns results."""
        results = await service.stratified_randomize(
            experiment_id=experiment_id,
            units=stratified_units,
            strata_columns=["region"],
        )

        assert len(results) == len(stratified_units)

    @pytest.mark.asyncio
    async def test_stratified_randomize_sets_stratification_key(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        stratified_units: List[Dict],
    ):
        """Test stratified randomization sets stratification key."""
        results = await service.stratified_randomize(
            experiment_id=experiment_id,
            units=stratified_units,
            strata_columns=["region"],
        )

        for result in results:
            assert result.stratification_key is not None
            assert "region" in result.stratification_key

    @pytest.mark.asyncio
    async def test_stratified_randomize_balance_within_strata(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test stratified randomization achieves balance within each stratum."""
        # Create units with known strata distribution
        units = []
        for i in range(400):
            units.append({
                "id": f"unit_{i}",
                "region": ["north", "south"][i % 2],
            })

        results = await service.stratified_randomize(
            experiment_id=experiment_id,
            units=units,
            strata_columns=["region"],
        )

        # Check balance within each stratum
        north_results = [r for r in results if r.stratification_key.get("region") == "north"]
        south_results = [r for r in results if r.stratification_key.get("region") == "south"]

        north_counts = Counter(r.variant for r in north_results)
        south_counts = Counter(r.variant for r in south_results)

        # Expect approximate balance within each stratum
        assert 80 <= north_counts["control"] <= 120
        assert 80 <= south_counts["control"] <= 120

    @pytest.mark.asyncio
    async def test_stratified_randomize_with_multiple_strata(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        stratified_units: List[Dict],
    ):
        """Test stratified randomization with multiple strata columns."""
        results = await service.stratified_randomize(
            experiment_id=experiment_id,
            units=stratified_units,
            strata_columns=["region", "tier"],
        )

        for result in results:
            assert result.stratification_key is not None
            assert "region" in result.stratification_key
            assert "tier" in result.stratification_key

    @pytest.mark.asyncio
    async def test_stratified_randomize_sets_correct_method(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        stratified_units: List[Dict],
    ):
        """Test stratified randomization sets STRATIFIED method."""
        results = await service.stratified_randomize(
            experiment_id=experiment_id,
            units=stratified_units,
            strata_columns=["region"],
        )

        assert all(r.randomization_method == RandomizationMethod.STRATIFIED for r in results)


# =============================================================================
# BLOCK RANDOMIZATION TESTS
# =============================================================================


class TestBlockRandomization:
    """Tests for block randomization."""

    @pytest.mark.asyncio
    async def test_block_randomize_returns_results(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test block randomization returns results."""
        results = await service.block_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )

        assert len(results) == len(sample_units)

    @pytest.mark.asyncio
    async def test_block_randomize_sets_block_id(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test block randomization sets block ID."""
        results = await service.block_randomize(
            experiment_id=experiment_id,
            units=sample_units,
            block_size=10,
        )

        for result in results:
            assert result.block_id is not None
            assert "block_" in result.block_id

    @pytest.mark.asyncio
    async def test_block_randomize_balance_within_blocks(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test block randomization achieves exact balance within complete blocks."""
        # Create exactly 40 units (4 complete blocks of 10)
        units = [{"id": f"unit_{i}"} for i in range(40)]

        results = await service.block_randomize(
            experiment_id=experiment_id,
            units=units,
            block_size=10,
        )

        # Group by block
        blocks: Dict[str, List[AssignmentResult]] = {}
        for result in results:
            if result.block_id not in blocks:
                blocks[result.block_id] = []
            blocks[result.block_id].append(result)

        # Each complete block should have exactly 5 control and 5 treatment
        for block_id, block_results in blocks.items():
            if len(block_results) == 10:  # Complete block
                counts = Counter(r.variant for r in block_results)
                assert counts["control"] == 5
                assert counts["treatment"] == 5

    @pytest.mark.asyncio
    async def test_block_randomize_custom_block_size(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test block randomization with custom block size."""
        results = await service.block_randomize(
            experiment_id=experiment_id,
            units=sample_units,
            block_size=20,
        )

        # Count unique blocks
        block_ids = set(r.block_id for r in results)
        expected_blocks = (len(sample_units) + 19) // 20
        assert len(block_ids) == expected_blocks

    @pytest.mark.asyncio
    async def test_block_randomize_sets_correct_method(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test block randomization sets BLOCK method."""
        results = await service.block_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )

        assert all(r.randomization_method == RandomizationMethod.BLOCK for r in results)


# =============================================================================
# MULTI-ARM ALLOCATION TESTS
# =============================================================================


class TestMultiArmAllocation:
    """Tests for multi-arm allocation."""

    @pytest.mark.asyncio
    async def test_multi_arm_allocate_single_unit(
        self,
        multi_arm_service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test multi-arm allocation for single unit."""
        unit = {"id": "test_unit"}
        arm_probabilities = {
            "control": 0.25,
            "treatment_a": 0.25,
            "treatment_b": 0.25,
            "treatment_c": 0.25,
        }

        result = await multi_arm_service.multi_arm_allocate(
            experiment_id=experiment_id,
            unit=unit,
            arm_probabilities=arm_probabilities,
        )

        assert isinstance(result, AssignmentResult)
        assert result.variant in arm_probabilities.keys()

    @pytest.mark.asyncio
    async def test_multi_arm_allocate_deterministic(
        self,
        multi_arm_service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test multi-arm allocation is deterministic."""
        unit = {"id": "test_unit"}
        arm_probabilities = {
            "control": 0.25,
            "treatment_a": 0.25,
            "treatment_b": 0.25,
            "treatment_c": 0.25,
        }

        result1 = await multi_arm_service.multi_arm_allocate(
            experiment_id=experiment_id,
            unit=unit,
            arm_probabilities=arm_probabilities,
        )
        result2 = await multi_arm_service.multi_arm_allocate(
            experiment_id=experiment_id,
            unit=unit,
            arm_probabilities=arm_probabilities,
        )

        assert result1.variant == result2.variant

    @pytest.mark.asyncio
    async def test_multi_arm_allocate_invalid_probabilities(
        self,
        multi_arm_service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test multi-arm allocation raises error for invalid probabilities."""
        unit = {"id": "test_unit"}
        arm_probabilities = {"control": 0.3, "treatment": 0.3}  # Sums to 0.6

        with pytest.raises(ValueError, match="must sum to 1.0"):
            await multi_arm_service.multi_arm_allocate(
                experiment_id=experiment_id,
                unit=unit,
                arm_probabilities=arm_probabilities,
            )

    @pytest.mark.asyncio
    async def test_batch_multi_arm_allocate(
        self,
        multi_arm_service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test batch multi-arm allocation."""
        units = [{"id": f"unit_{i}"} for i in range(100)]
        arm_probabilities = {
            "control": 0.25,
            "treatment_a": 0.25,
            "treatment_b": 0.25,
            "treatment_c": 0.25,
        }

        results = await multi_arm_service.batch_multi_arm_allocate(
            experiment_id=experiment_id,
            units=units,
            arm_probabilities=arm_probabilities,
        )

        assert len(results) == 100
        # Check all arms are represented
        variants = set(r.variant for r in results)
        assert len(variants) >= 2  # At least 2 arms should be present


# =============================================================================
# HASH FUNCTION TESTS
# =============================================================================


class TestHashFunctions:
    """Tests for hash-based assignment functions."""

    def test_generate_assignment_hash_deterministic(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test hash generation is deterministic."""
        hash1 = service._generate_assignment_hash(experiment_id, "unit_1")
        hash2 = service._generate_assignment_hash(experiment_id, "unit_1")

        assert hash1 == hash2

    def test_generate_assignment_hash_different_units(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test different units get different hashes."""
        hash1 = service._generate_assignment_hash(experiment_id, "unit_1")
        hash2 = service._generate_assignment_hash(experiment_id, "unit_2")

        assert hash1 != hash2

    def test_generate_assignment_hash_different_experiments(
        self,
        service: RandomizationService,
    ):
        """Test same unit in different experiments gets different hashes."""
        exp1 = uuid4()
        exp2 = uuid4()

        hash1 = service._generate_assignment_hash(exp1, "unit_1")
        hash2 = service._generate_assignment_hash(exp2, "unit_1")

        assert hash1 != hash2

    def test_hash_to_unit_interval_range(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test hash to unit interval produces values in [0, 1)."""
        for i in range(100):
            hash_str = service._generate_assignment_hash(experiment_id, f"unit_{i}")
            value = service._hash_to_unit_interval(hash_str)
            assert 0.0 <= value < 1.0

    def test_hash_to_unit_interval_distribution(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test hash to unit interval produces roughly uniform distribution."""
        values = []
        for i in range(1000):
            hash_str = service._generate_assignment_hash(experiment_id, f"unit_{i}")
            values.append(service._hash_to_unit_interval(hash_str))

        # Check distribution is roughly uniform
        quartiles = [
            sum(1 for v in values if 0.0 <= v < 0.25),
            sum(1 for v in values if 0.25 <= v < 0.5),
            sum(1 for v in values if 0.5 <= v < 0.75),
            sum(1 for v in values if 0.75 <= v < 1.0),
        ]

        # Each quartile should have roughly 250 values
        for q in quartiles:
            assert 150 <= q <= 350  # Allow variance


# =============================================================================
# VARIANT SELECTION TESTS
# =============================================================================


class TestVariantSelection:
    """Tests for variant selection logic."""

    def test_select_variant_control(
        self,
        service: RandomizationService,
    ):
        """Test variant selection returns control for low values."""
        ratio = {"control": 0.5, "treatment": 0.5}

        variant = service._select_variant(0.1, ratio)
        assert variant == "control"

    def test_select_variant_treatment(
        self,
        service: RandomizationService,
    ):
        """Test variant selection returns treatment for high values."""
        ratio = {"control": 0.5, "treatment": 0.5}

        variant = service._select_variant(0.9, ratio)
        assert variant == "treatment"

    def test_select_variant_boundary(
        self,
        service: RandomizationService,
    ):
        """Test variant selection at boundary."""
        ratio = {"control": 0.5, "treatment": 0.5}

        # At exactly 0.5, should be treatment (>= boundary)
        variant = service._select_variant(0.5, ratio)
        assert variant == "treatment"

    def test_select_variant_multi_arm(
        self,
        service: RandomizationService,
    ):
        """Test variant selection with multiple arms."""
        ratio = {
            "control": 0.25,
            "treatment_a": 0.25,
            "treatment_b": 0.25,
            "treatment_c": 0.25,
        }

        # Test different ranges
        assert service._select_variant(0.1, ratio) == "control"
        assert service._select_variant(0.3, ratio) == "treatment_a"
        assert service._select_variant(0.55, ratio) == "treatment_b"
        assert service._select_variant(0.9, ratio) == "treatment_c"


# =============================================================================
# VERIFICATION AND SUMMARY TESTS
# =============================================================================


class TestVerificationAndSummary:
    """Tests for assignment verification and summary functions."""

    def test_verify_assignment_correct(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test verifying correct assignment returns True."""
        # First, get the actual assignment
        hash_str = service._generate_assignment_hash(experiment_id, "unit_1")
        random_value = service._hash_to_unit_interval(hash_str)
        expected_variant = service._select_variant(random_value, service.config.allocation_ratio)

        result = service.verify_assignment(
            experiment_id=experiment_id,
            unit_id="unit_1",
            expected_variant=expected_variant,
        )

        assert result is True

    def test_verify_assignment_incorrect(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test verifying incorrect assignment returns False."""
        # First, get the actual assignment
        hash_str = service._generate_assignment_hash(experiment_id, "unit_1")
        random_value = service._hash_to_unit_interval(hash_str)
        actual_variant = service._select_variant(random_value, service.config.allocation_ratio)

        # Use the opposite variant
        wrong_variant = "treatment" if actual_variant == "control" else "control"

        result = service.verify_assignment(
            experiment_id=experiment_id,
            unit_id="unit_1",
            expected_variant=wrong_variant,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_allocation_summary(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test allocation summary generation."""
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )

        summary = service.get_allocation_summary(results)

        assert summary["total_units"] == len(sample_units)
        assert "variant_counts" in summary
        assert "variant_proportions" in summary
        assert "control" in summary["variant_counts"]
        assert "treatment" in summary["variant_counts"]

    @pytest.mark.asyncio
    async def test_get_allocation_summary_with_strata(
        self,
        service: RandomizationService,
        experiment_id: UUID,
        stratified_units: List[Dict],
    ):
        """Test allocation summary includes strata counts for stratified randomization."""
        results = await service.stratified_randomize(
            experiment_id=experiment_id,
            units=stratified_units,
            strata_columns=["region"],
        )

        summary = service.get_allocation_summary(results)

        assert summary["strata_counts"] is not None
        assert len(summary["strata_counts"]) > 0


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunction:
    """Tests for factory function."""

    def test_get_randomization_service_default(self):
        """Test factory creates service with default config."""
        service = get_randomization_service()

        assert isinstance(service, RandomizationService)
        assert service.config is not None

    def test_get_randomization_service_custom_config(self):
        """Test factory creates service with custom config."""
        config = RandomizationConfig(
            allocation_ratio={"control": 0.3, "treatment": 0.7}
        )

        service = get_randomization_service(config=config)

        assert service.config.allocation_ratio["control"] == 0.3


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_units_list(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test randomization with empty units list."""
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=[],
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_single_unit(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test randomization with single unit."""
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=[{"id": "only_unit"}],
        )

        assert len(results) == 1
        assert results[0].variant in ["control", "treatment"]

    @pytest.mark.asyncio
    async def test_unit_with_unit_id_key(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test units can use 'unit_id' key instead of 'id'."""
        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=[{"unit_id": "test_unit"}],
        )

        assert len(results) == 1
        assert results[0].unit_id == "test_unit"

    @pytest.mark.asyncio
    async def test_non_deterministic_mode(
        self,
        experiment_id: UUID,
        sample_units: List[Dict],
    ):
        """Test non-deterministic randomization mode."""
        config = RandomizationConfig(deterministic=False)
        service = RandomizationService(config=config)

        results = await service.simple_randomize(
            experiment_id=experiment_id,
            units=sample_units,
        )

        # Results should not have assignment hash in non-deterministic mode
        for result in results:
            assert result.assignment_hash is None

    @pytest.mark.asyncio
    async def test_different_unit_types(
        self,
        service: RandomizationService,
        experiment_id: UUID,
    ):
        """Test randomization with different unit types."""
        for unit_type in UnitType:
            results = await service.simple_randomize(
                experiment_id=experiment_id,
                units=[{"id": "test"}],
                unit_type=unit_type,
            )

            assert results[0].unit_type == unit_type
