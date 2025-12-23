"""Tests for Feast Migration Validation Script.

Tests cover:
- MigrationValidator initialization
- Feature parity validation
- Shadow mode comparison
- Performance benchmarking
- Export functionality
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.feast_validate_migration import MigrationValidator


class TestMigrationValidatorInit:
    """Test MigrationValidator initialization."""

    def test_validator_creation_defaults(self):
        """Test validator can be created with defaults."""
        validator = MigrationValidator()

        assert validator.custom_client is None
        assert validator.feast_client is None
        assert validator._initialized is False

    def test_validator_creation_with_clients(self):
        """Test validator with provided clients."""
        mock_custom = MagicMock()
        mock_feast = MagicMock()

        validator = MigrationValidator(
            custom_client=mock_custom,
            feast_client=mock_feast,
        )

        assert validator.custom_client is mock_custom
        assert validator.feast_client is mock_feast


class TestMigrationValidatorInitialize:
    """Test MigrationValidator initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        mock_feast = MagicMock()
        mock_feast.initialize = AsyncMock()

        validator = MigrationValidator(
            custom_client=MagicMock(),
            feast_client=mock_feast,
        )

        result = await validator.initialize()

        assert result is True
        assert validator._initialized is True
        mock_feast.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test that initialization is skipped if already done."""
        mock_feast = MagicMock()
        mock_feast.initialize = AsyncMock()

        validator = MigrationValidator(
            custom_client=MagicMock(),
            feast_client=mock_feast,
        )
        validator._initialized = True

        result = await validator.initialize()

        assert result is True
        mock_feast.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure handling."""
        mock_feast = MagicMock()
        mock_feast.initialize = AsyncMock(side_effect=Exception("Init failed"))

        validator = MigrationValidator(
            custom_client=MagicMock(),
            feast_client=mock_feast,
        )

        result = await validator.initialize()

        assert result is False
        assert validator._initialized is False


class TestFeatureParityValidation:
    """Test feature parity validation."""

    @pytest.mark.asyncio
    async def test_validate_parity_init_failure(self):
        """Test parity validation when init fails."""
        mock_feast = MagicMock()
        mock_feast.initialize = AsyncMock(side_effect=Exception("Init failed"))

        validator = MigrationValidator(
            custom_client=MagicMock(),
            feast_client=mock_feast,
        )

        result = await validator.validate_feature_parity(
            entity_ids=[{"hcp_id": "123"}],
            feature_refs=["test:feature"],
        )

        assert result["status"] == "failed"
        assert "Initialization failed" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_parity_all_match(self):
        """Test parity validation when all features match."""
        mock_custom = MagicMock()
        mock_feast = MagicMock()
        mock_feast.initialize = AsyncMock()
        mock_feast.get_online_features = AsyncMock(
            return_value={"test__feature": [0.5]}
        )

        validator = MigrationValidator(
            custom_client=mock_custom,
            feast_client=mock_feast,
        )

        # Mock custom feature retrieval
        with patch.object(
            validator, "_get_custom_features", new_callable=AsyncMock
        ) as mock_get_custom:
            mock_get_custom.return_value = {"test:feature": 0.5}

            result = await validator.validate_feature_parity(
                entity_ids=[{"hcp_id": "123"}],
                feature_refs=["test:feature"],
            )

            assert result["status"] == "completed"
            assert result["matches"] == 1
            assert result["match_rate"] == 1.0
            assert result["parity_achieved"] is True

    @pytest.mark.asyncio
    async def test_validate_parity_mismatch(self):
        """Test parity validation when features don't match."""
        mock_custom = MagicMock()
        mock_feast = MagicMock()
        mock_feast.initialize = AsyncMock()
        mock_feast.get_online_features = AsyncMock(
            return_value={"test__feature": [0.8]}
        )

        validator = MigrationValidator(
            custom_client=mock_custom,
            feast_client=mock_feast,
        )

        # Mock custom feature retrieval with different value
        with patch.object(
            validator, "_get_custom_features", new_callable=AsyncMock
        ) as mock_get_custom:
            mock_get_custom.return_value = {"test:feature": 0.3}

            result = await validator.validate_feature_parity(
                entity_ids=[{"hcp_id": "123"}],
                feature_refs=["test:feature"],
            )

            assert result["status"] == "completed"
            assert result["matches"] == 0
            assert len(result["mismatches"]) == 1
            assert result["parity_achieved"] is False


class TestFeatureComparison:
    """Test feature comparison logic."""

    def test_compare_features_match(self):
        """Test comparison when features match."""
        validator = MigrationValidator()

        result = validator._compare_features(
            custom={"feat1": 0.5, "feat2": "value"},
            feast={"feat1": 0.5, "feat2": "value"},
            tolerance=0.001,
        )

        assert result["match"] is True
        assert len(result["differences"]) == 0

    def test_compare_features_numeric_within_tolerance(self):
        """Test comparison when numeric values are within tolerance."""
        validator = MigrationValidator()

        result = validator._compare_features(
            custom={"feat1": 0.50001},
            feast={"feat1": 0.50002},
            tolerance=0.001,
        )

        assert result["match"] is True

    def test_compare_features_numeric_outside_tolerance(self):
        """Test comparison when numeric values are outside tolerance."""
        validator = MigrationValidator()

        result = validator._compare_features(
            custom={"feat1": 0.5},
            feast={"feat1": 0.6},
            tolerance=0.001,
        )

        assert result["match"] is False
        assert len(result["differences"]) == 1
        assert result["differences"][0]["reason"] == "value_mismatch"

    def test_compare_features_missing_in_custom(self):
        """Test comparison when feature missing in custom."""
        validator = MigrationValidator()

        result = validator._compare_features(
            custom={},
            feast={"feat1": 0.5},
            tolerance=0.001,
        )

        assert result["match"] is False
        assert result["differences"][0]["reason"] == "missing_in_custom"

    def test_compare_features_missing_in_feast(self):
        """Test comparison when feature missing in Feast."""
        validator = MigrationValidator()

        result = validator._compare_features(
            custom={"feat1": 0.5},
            feast={},
            tolerance=0.001,
        )

        assert result["match"] is False
        assert result["differences"][0]["reason"] == "missing_in_feast"


class TestShadowMode:
    """Test shadow mode comparison."""

    @pytest.mark.asyncio
    async def test_shadow_mode_success(self):
        """Test successful shadow mode run."""
        mock_custom = MagicMock()
        mock_feast = MagicMock()
        mock_feast.initialize = AsyncMock()
        mock_feast.get_online_features = AsyncMock(
            return_value={"test__feature": [0.5]}
        )

        validator = MigrationValidator(
            custom_client=mock_custom,
            feast_client=mock_feast,
        )

        entity_df = pd.DataFrame({
            "hcp_id": ["hcp_001", "hcp_002"],
            "brand_id": ["brand1", "brand1"],
            "event_timestamp": [datetime.now(timezone.utc)] * 2,
        })

        with patch.object(
            validator, "_get_custom_features", new_callable=AsyncMock
        ) as mock_get_custom:
            mock_get_custom.return_value = {"test:feature": 0.5}

            result = await validator.run_shadow_mode(
                entity_df=entity_df,
                feature_refs=["test:feature"],
                sample_size=2,
            )

            assert result["status"] == "completed"
            assert result["mode"] == "shadow"
            assert result["sample_size"] == 2
            assert "statistics" in result
            assert "custom" in result["statistics"]
            assert "feast" in result["statistics"]
            assert "parity" in result["statistics"]

    @pytest.mark.asyncio
    async def test_shadow_mode_sampling(self):
        """Test shadow mode sampling when df is large."""
        mock_custom = MagicMock()
        mock_feast = MagicMock()
        mock_feast.initialize = AsyncMock()
        mock_feast.get_online_features = AsyncMock(
            return_value={"test__feature": [0.5]}
        )

        validator = MigrationValidator(
            custom_client=mock_custom,
            feast_client=mock_feast,
        )

        # Create large DataFrame
        entity_df = pd.DataFrame({
            "hcp_id": [f"hcp_{i:03d}" for i in range(1000)],
            "brand_id": ["brand1"] * 1000,
            "event_timestamp": [datetime.now(timezone.utc)] * 1000,
        })

        with patch.object(
            validator, "_get_custom_features", new_callable=AsyncMock
        ) as mock_get_custom:
            mock_get_custom.return_value = {"test:feature": 0.5}

            result = await validator.run_shadow_mode(
                entity_df=entity_df,
                feature_refs=["test:feature"],
                sample_size=10,  # Sample only 10
            )

            assert result["sample_size"] == 10
            assert len(result["custom_latencies_ms"]) == 10


class TestBenchmark:
    """Test performance benchmarking."""

    @pytest.mark.asyncio
    async def test_benchmark_success(self):
        """Test successful benchmark run."""
        mock_custom = MagicMock()
        mock_feast = MagicMock()
        mock_feast.initialize = AsyncMock()
        mock_feast.get_online_features = AsyncMock(
            return_value={"test__feature": [0.5] * 10}
        )

        validator = MigrationValidator(
            custom_client=mock_custom,
            feast_client=mock_feast,
        )

        with patch.object(
            validator, "_get_custom_features", new_callable=AsyncMock
        ) as mock_get_custom:
            mock_get_custom.return_value = {"test:feature": 0.5}

            result = await validator.benchmark_performance(
                entity_count=10,
                iterations=2,
            )

            assert result["status"] == "completed"
            assert result["entity_count"] == 10
            assert result["iterations"] == 2
            assert "custom_store" in result
            assert "feast" in result
            assert "mean_seconds" in result["custom_store"]
            assert "qps" in result["custom_store"]


class TestPercentile:
    """Test percentile calculation."""

    def test_percentile_p50(self):
        """Test 50th percentile calculation."""
        validator = MigrationValidator()
        data = [1, 2, 3, 4, 5]

        result = validator._percentile(data, 50)

        assert result == 3.0

    def test_percentile_p95(self):
        """Test 95th percentile calculation."""
        validator = MigrationValidator()
        data = list(range(1, 101))  # 1 to 100

        result = validator._percentile(data, 95)

        assert result >= 95.0

    def test_percentile_empty(self):
        """Test percentile with empty data."""
        validator = MigrationValidator()

        result = validator._percentile([], 50)

        assert result == 0.0


class TestValidatorLifecycle:
    """Test validator lifecycle operations."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the validator."""
        mock_feast = MagicMock()
        mock_feast.close = AsyncMock()

        validator = MigrationValidator(
            custom_client=MagicMock(),
            feast_client=mock_feast,
        )
        validator._initialized = True

        await validator.close()

        mock_feast.close.assert_called_once()
        assert validator._initialized is False

    @pytest.mark.asyncio
    async def test_close_no_feast_client(self):
        """Test closing when no Feast client."""
        validator = MigrationValidator()

        # Should not raise
        await validator.close()
