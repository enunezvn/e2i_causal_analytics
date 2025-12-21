"""
Base repository implementations with split-aware querying.

CRITICAL: All queries must respect ML splits to prevent data leakage.
"""

import hashlib
from abc import ABC
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """
    Base repository with common CRUD operations.

    All subclasses must define:
    - table_name: The Supabase table name
    - model_class: The Pydantic model class
    """

    table_name: str
    model_class: Type[T]

    def __init__(self, supabase_client=None):
        """
        Initialize repository with Supabase client.

        Args:
            supabase_client: Supabase client instance
        """
        self.client = supabase_client

    async def get_by_id(
        self,
        id: str,
        split: Optional[str] = None,
    ) -> Optional[T]:
        """
        Get a single record by ID.

        Args:
            id: Record UUID
            split: Optional ML split filter

        Returns:
            Model instance or None
        """
        if not self.client:
            return None

        query = self.client.table(self.table_name).select("*").eq("id", id)
        if split:
            query = query.eq("split_assignment", split)

        result = await query.execute()
        return self._to_model(result.data[0]) if result.data else None

    async def get_many(
        self,
        filters: Dict[str, Any],
        split: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """
        Get multiple records with filters.

        Args:
            filters: Column-value filters
            split: Optional ML split filter
            limit: Maximum records to return
            offset: Pagination offset

        Returns:
            List of model instances
        """
        if not self.client:
            return []

        query = self.client.table(self.table_name).select("*")

        for column, value in filters.items():
            query = query.eq(column, value)

        if split:
            query = query.eq("split_assignment", split)

        query = query.limit(limit).offset(offset)
        result = await query.execute()

        return [self._to_model(row) for row in result.data]

    async def create(self, entity: T) -> T:
        """
        Create a new record.

        Args:
            entity: Model instance to create

        Returns:
            Created model instance with ID
        """
        if not self.client:
            return entity

        data = entity.model_dump() if hasattr(entity, "model_dump") else entity
        result = await self.client.table(self.table_name).insert(data).execute()

        return self._to_model(result.data[0]) if result.data else entity

    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[T]:
        """
        Update an existing record.

        Args:
            id: Record UUID
            updates: Column-value updates

        Returns:
            Updated model instance or None
        """
        if not self.client:
            return None

        result = await self.client.table(self.table_name).update(updates).eq("id", id).execute()

        return self._to_model(result.data[0]) if result.data else None

    async def delete(self, id: str) -> bool:
        """
        Delete a record.

        Args:
            id: Record UUID

        Returns:
            True if deleted, False otherwise
        """
        if not self.client:
            return False

        result = await self.client.table(self.table_name).delete().eq("id", id).execute()

        return len(result.data) > 0

    def _to_model(self, data: Dict[str, Any]) -> T:
        """Convert database row to model instance."""
        if self.model_class and hasattr(self.model_class, "model_validate"):
            return self.model_class.model_validate(data)
        return data


class SplitAwareRepository(BaseRepository[T]):
    """
    Repository that enforces ML split boundaries.

    CRITICAL: This repository ensures no data leakage between splits.
    - Training queries only see train split
    - Validation queries only see validation split
    - Test/holdout are NEVER exposed in normal operations
    """

    async def get_training_data(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10000,
    ) -> List[T]:
        """
        Get training split data only.

        Args:
            filters: Additional filters
            limit: Maximum records

        Returns:
            Training data only
        """
        return await self.get_many(filters or {}, split="train", limit=limit)

    async def get_validation_data(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10000,
    ) -> List[T]:
        """
        Get validation split data only.

        Args:
            filters: Additional filters
            limit: Maximum records

        Returns:
            Validation data only
        """
        return await self.get_many(filters or {}, split="validation", limit=limit)

    @staticmethod
    def assign_split(patient_id: str) -> str:
        """
        Assign a patient to a split based on hash.

        RULES:
        1. Same patient always in same split (prevent leakage)
        2. Deterministic based on patient_id

        Split ratios:
        - train: 60%
        - validation: 20%
        - test: 15%
        - holdout: 5%

        Args:
            patient_id: Patient identifier

        Returns:
            Split name: train, validation, test, or holdout
        """
        hash_val = int(hashlib.md5(patient_id.encode()).hexdigest(), 16)
        normalized = hash_val / (2**128)

        if normalized < 0.60:
            return "train"
        elif normalized < 0.80:
            return "validation"
        elif normalized < 0.95:
            return "test"
        else:
            return "holdout"
