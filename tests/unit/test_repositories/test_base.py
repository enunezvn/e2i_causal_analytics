"""
Unit tests for BaseRepository and SplitAwareRepository.

Tests CRUD operations, split-aware querying, and data leakage prevention.
"""

import hashlib
from typing import Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel


# Mock model for testing
class MockModel(BaseModel):
    """Mock Pydantic model for testing."""

    id: Optional[str] = None
    name: str = "test"
    value: int = 0
    split_assignment: Optional[str] = None


# Concrete implementation for testing
class TestRepository:
    """Concrete BaseRepository for testing."""

    def __init__(self, supabase_client=None):
        from src.repositories.base import BaseRepository

        class ConcreteRepo(BaseRepository[MockModel]):
            table_name = "test_table"
            model_class = MockModel

        self.repo = ConcreteRepo(supabase_client)


class TestSplitRepository:
    """Concrete SplitAwareRepository for testing."""

    def __init__(self, supabase_client=None):
        from src.repositories.base import SplitAwareRepository

        class ConcreteRepo(SplitAwareRepository[MockModel]):
            table_name = "test_split_table"
            model_class = MockModel

        self.repo = ConcreteRepo(supabase_client)


@pytest.mark.unit
class TestBaseRepository:
    """Tests for BaseRepository base class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        test_repo = TestRepository(supabase_client=mock_client)
        return test_repo.repo

    @pytest.fixture
    def sample_data(self):
        """Sample database record."""
        return {
            "id": str(uuid4()),
            "name": "test_record",
            "value": 42,
            "split_assignment": "train",
        }


@pytest.mark.unit
class TestGetById(TestBaseRepository):
    """Tests for get_by_id method."""

    @pytest.mark.asyncio
    async def test_returns_model_when_found(self, repo, mock_client, sample_data):
        """Test that model is returned when record exists."""
        mock_result = MagicMock()
        mock_result.data = [sample_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.execute = mock_execute

        result = await repo.get_by_id(sample_data["id"])

        assert result is not None
        assert result.id == sample_data["id"]
        assert result.name == sample_data["name"]
        assert result.value == sample_data["value"]

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, repo, mock_client):
        """Test that None is returned when record doesn't exist."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.execute = mock_execute

        result = await repo.get_by_id("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_applies_split_filter_when_provided(self, repo, mock_client, sample_data):
        """Test that split filter is applied when provided."""
        mock_result = MagicMock()
        mock_result.data = [sample_data]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_eq_split = MagicMock()
        mock_eq_split.execute = mock_execute
        mock_eq_id = MagicMock()
        mock_eq_id.eq.return_value = mock_eq_split
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_id

        result = await repo.get_by_id(sample_data["id"], split="train")

        assert result is not None
        # Verify split filter was applied
        mock_eq_id.eq.assert_called_once_with("split_assignment", "train")

    @pytest.mark.asyncio
    async def test_returns_none_without_client(self):
        """Test that None is returned when client is None."""
        test_repo = TestRepository(supabase_client=None)
        result = await test_repo.repo.get_by_id("some-id")
        assert result is None


@pytest.mark.unit
class TestGetMany(TestBaseRepository):
    """Tests for get_many method."""

    @pytest.mark.asyncio
    async def test_returns_list_of_models(self, repo, mock_client, sample_data):
        """Test that list of models is returned."""
        mock_result = MagicMock()
        mock_result.data = [sample_data, {**sample_data, "id": str(uuid4())}]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_many(filters={"name": "test_record"})

        assert len(result) == 2
        assert all(isinstance(item, MockModel) for item in result)

    @pytest.mark.asyncio
    async def test_applies_filters(self, repo, mock_client, sample_data):
        """Test that filters are applied correctly."""
        mock_result = MagicMock()
        mock_result.data = [sample_data]
        mock_execute = AsyncMock(return_value=mock_result)

        # Create a chain of mocks
        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_value = MagicMock()
        mock_eq_value.limit.return_value = mock_limit
        mock_eq_name = MagicMock()
        mock_eq_name.eq.return_value = mock_eq_value
        mock_select = MagicMock()
        mock_select.eq.return_value = mock_eq_name
        mock_client.table.return_value.select.return_value = mock_select

        result = await repo.get_many(filters={"name": "test", "value": 42})

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_applies_split_filter(self, repo, mock_client, sample_data):
        """Test that split filter is applied when provided."""
        mock_result = MagicMock()
        mock_result.data = [sample_data]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_split = MagicMock()
        mock_eq_split.limit.return_value = mock_limit
        mock_eq_name = MagicMock()
        mock_eq_name.eq.return_value = mock_eq_split
        mock_select = MagicMock()
        mock_select.eq.return_value = mock_eq_name
        mock_client.table.return_value.select.return_value = mock_select

        result = await repo.get_many(filters={"name": "test"}, split="train")

        assert len(result) == 1
        # Verify split filter was called
        mock_eq_name.eq.assert_called_with("split_assignment", "train")

    @pytest.mark.asyncio
    async def test_respects_limit_and_offset(self, repo, mock_client, sample_data):
        """Test that limit and offset are applied."""
        mock_result = MagicMock()
        mock_result.data = [sample_data]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_many(filters={"name": "test"}, limit=50, offset=10)

        assert len(result) == 1
        mock_eq.limit.assert_called_once_with(50)
        mock_limit.offset.assert_called_once_with(10)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        test_repo = TestRepository(supabase_client=None)
        result = await test_repo.repo.get_many(filters={})
        assert result == []


@pytest.mark.unit
class TestCreate(TestBaseRepository):
    """Tests for create method."""

    @pytest.mark.asyncio
    async def test_creates_and_returns_model(self, repo, mock_client, sample_data):
        """Test that model is created and returned."""
        entity = MockModel(**sample_data)
        mock_result = MagicMock()
        mock_result.data = [sample_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        result = await repo.create(entity)

        assert result.id == sample_data["id"]
        assert result.name == sample_data["name"]
        mock_client.table.assert_called_with("test_table")

    @pytest.mark.asyncio
    async def test_converts_pydantic_model_to_dict(self, repo, mock_client, sample_data):
        """Test that Pydantic model is converted to dict before insert."""
        entity = MockModel(**sample_data)
        mock_result = MagicMock()
        mock_result.data = [sample_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_insert = MagicMock()
        mock_insert.execute = mock_execute
        mock_client.table.return_value.insert.return_value = mock_insert

        await repo.create(entity)

        # Verify insert was called with dict
        mock_client.table.return_value.insert.assert_called_once()
        call_args = mock_client.table.return_value.insert.call_args
        assert isinstance(call_args[0][0], dict)

    @pytest.mark.asyncio
    async def test_returns_entity_without_client(self):
        """Test that entity is returned unchanged when client is None."""
        test_repo = TestRepository(supabase_client=None)
        entity = MockModel(id=str(uuid4()), name="test", value=42)
        result = await test_repo.repo.create(entity)
        assert result.id == entity.id


@pytest.mark.unit
class TestUpdate(TestBaseRepository):
    """Tests for update method."""

    @pytest.mark.asyncio
    async def test_updates_and_returns_model(self, repo, mock_client, sample_data):
        """Test that record is updated and returned."""
        updated_data = {**sample_data, "value": 100}
        mock_result = MagicMock()
        mock_result.data = [updated_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.update(sample_data["id"], {"value": 100})

        assert result is not None
        assert result.value == 100
        mock_client.table.return_value.update.assert_called_once_with({"value": 100})

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, repo, mock_client):
        """Test that None is returned when record doesn't exist."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.update("nonexistent-id", {"value": 100})

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_without_client(self):
        """Test that None is returned when client is None."""
        test_repo = TestRepository(supabase_client=None)
        result = await test_repo.repo.update("some-id", {"value": 100})
        assert result is None


@pytest.mark.unit
class TestDelete(TestBaseRepository):
    """Tests for delete method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self, repo, mock_client, sample_data):
        """Test that True is returned when record is deleted."""
        mock_result = MagicMock()
        mock_result.data = [sample_data]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.delete.return_value.eq.return_value.execute = mock_execute

        result = await repo.delete(sample_data["id"])

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, repo, mock_client):
        """Test that False is returned when record doesn't exist."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.delete.return_value.eq.return_value.execute = mock_execute

        result = await repo.delete("nonexistent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_without_client(self):
        """Test that False is returned when client is None."""
        test_repo = TestRepository(supabase_client=None)
        result = await test_repo.repo.delete("some-id")
        assert result is False


@pytest.mark.unit
class TestToModel(TestBaseRepository):
    """Tests for _to_model method."""

    def test_converts_dict_to_pydantic_model(self, repo, sample_data):
        """Test that dict is converted to Pydantic model."""
        result = repo._to_model(sample_data)

        assert isinstance(result, MockModel)
        assert result.id == sample_data["id"]
        assert result.name == sample_data["name"]

    def test_returns_dict_when_no_model_class(self, mock_client, sample_data):
        """Test that dict is returned when model_class is None."""
        from src.repositories.base import BaseRepository

        class NoModelRepo(BaseRepository):
            table_name = "test"
            model_class = None

        repo = NoModelRepo(supabase_client=mock_client)
        result = repo._to_model(sample_data)

        assert result == sample_data


@pytest.mark.unit
class TestSplitAwareRepository:
    """Tests for SplitAwareRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create split-aware repository with mock client."""
        test_repo = TestSplitRepository(supabase_client=mock_client)
        return test_repo.repo

    @pytest.fixture
    def sample_train_data(self):
        """Sample training data."""
        return [
            {"id": str(uuid4()), "name": "train1", "value": 1, "split_assignment": "train"},
            {"id": str(uuid4()), "name": "train2", "value": 2, "split_assignment": "train"},
        ]

    @pytest.fixture
    def sample_validation_data(self):
        """Sample validation data."""
        return [
            {"id": str(uuid4()), "name": "val1", "value": 10, "split_assignment": "validation"},
        ]


@pytest.mark.unit
class TestGetTrainingData(TestSplitAwareRepository):
    """Tests for get_training_data method."""

    @pytest.mark.asyncio
    async def test_returns_only_training_split(self, repo, mock_client, sample_train_data):
        """Test that only training split data is returned."""
        mock_result = MagicMock()
        mock_result.data = sample_train_data
        mock_execute = AsyncMock(return_value=mock_result)

        # Create proper mock chain for split-aware query
        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_split = MagicMock()
        mock_eq_split.limit.return_value = mock_limit
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_split

        result = await repo.get_training_data()

        assert len(result) == 2
        assert all(item.split_assignment == "train" for item in result)

    @pytest.mark.asyncio
    async def test_applies_additional_filters(self, repo, mock_client, sample_train_data):
        """Test that additional filters are applied."""
        mock_result = MagicMock()
        mock_result.data = [sample_train_data[0]]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_split = MagicMock()
        mock_eq_split.limit.return_value = mock_limit
        mock_eq_value = MagicMock()
        mock_eq_value.eq.return_value = mock_eq_split
        mock_select = MagicMock()
        mock_select.eq.return_value = mock_eq_value
        mock_client.table.return_value.select.return_value = mock_select

        result = await repo.get_training_data(filters={"value": 1})

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_train_data):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_train_data[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_training_data(limit=1)

        assert len(result) == 1
        mock_eq.limit.assert_called_with(1)


@pytest.mark.unit
class TestGetValidationData(TestSplitAwareRepository):
    """Tests for get_validation_data method."""

    @pytest.mark.asyncio
    async def test_returns_only_validation_split(self, repo, mock_client, sample_validation_data):
        """Test that only validation split data is returned."""
        mock_result = MagicMock()
        mock_result.data = sample_validation_data
        mock_execute = AsyncMock(return_value=mock_result)

        # Create proper mock chain for split-aware query
        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_split = MagicMock()
        mock_eq_split.limit.return_value = mock_limit
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_split

        result = await repo.get_validation_data()

        assert len(result) == 1
        assert result[0].split_assignment == "validation"

    @pytest.mark.asyncio
    async def test_applies_additional_filters(self, repo, mock_client, sample_validation_data):
        """Test that additional filters are applied."""
        mock_result = MagicMock()
        mock_result.data = sample_validation_data
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_split = MagicMock()
        mock_eq_split.limit.return_value = mock_limit
        mock_eq_value = MagicMock()
        mock_eq_value.eq.return_value = mock_eq_split
        mock_select = MagicMock()
        mock_select.eq.return_value = mock_eq_value
        mock_client.table.return_value.select.return_value = mock_select

        result = await repo.get_validation_data(filters={"value": 10})

        assert len(result) == 1


@pytest.mark.unit
class TestAssignSplit:
    """Tests for assign_split static method."""

    def test_assigns_deterministically(self):
        """Test that same patient_id always gets same split."""
        from src.repositories.base import SplitAwareRepository

        patient_id = "patient-123"
        split1 = SplitAwareRepository.assign_split(patient_id)
        split2 = SplitAwareRepository.assign_split(patient_id)
        split3 = SplitAwareRepository.assign_split(patient_id)

        assert split1 == split2 == split3

    def test_split_distribution(self):
        """Test that split distribution matches expected ratios."""
        from src.repositories.base import SplitAwareRepository

        # Generate 10000 patient IDs and check distribution
        splits = {"train": 0, "validation": 0, "test": 0, "holdout": 0}

        for i in range(10000):
            patient_id = f"patient-{i}"
            split = SplitAwareRepository.assign_split(patient_id)
            splits[split] += 1

        # Expected ratios: train=60%, validation=20%, test=15%, holdout=5%
        # Allow 2% tolerance
        assert 0.58 <= splits["train"] / 10000 <= 0.62
        assert 0.18 <= splits["validation"] / 10000 <= 0.22
        assert 0.13 <= splits["test"] / 10000 <= 0.17
        assert 0.03 <= splits["holdout"] / 10000 <= 0.07

    def test_returns_valid_split(self):
        """Test that only valid split names are returned."""
        from src.repositories.base import SplitAwareRepository

        valid_splits = {"train", "validation", "test", "holdout"}

        for i in range(100):
            patient_id = f"patient-{i}"
            split = SplitAwareRepository.assign_split(patient_id)
            assert split in valid_splits

    def test_different_patients_can_have_different_splits(self):
        """Test that different patients can have different splits."""
        from src.repositories.base import SplitAwareRepository

        splits = set()
        for i in range(100):
            patient_id = f"patient-{i}"
            split = SplitAwareRepository.assign_split(patient_id)
            splits.add(split)

        # Should have at least 2 different splits in 100 patients
        assert len(splits) >= 2

    def test_hash_based_assignment(self):
        """Test that assignment is based on MD5 hash."""
        from src.repositories.base import SplitAwareRepository

        patient_id = "test-patient-456"

        # Calculate expected split manually
        hash_val = int(hashlib.md5(patient_id.encode()).hexdigest(), 16)
        normalized = hash_val / (2**128)

        if normalized < 0.60:
            expected_split = "train"
        elif normalized < 0.80:
            expected_split = "validation"
        elif normalized < 0.95:
            expected_split = "test"
        else:
            expected_split = "holdout"

        actual_split = SplitAwareRepository.assign_split(patient_id)
        assert actual_split == expected_split
