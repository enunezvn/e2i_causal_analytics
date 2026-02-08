"""
Feast Feature Store tracking repository.

Provides database access for Feast feature tracking, enabling:
- Feature view configuration tracking
- Materialization job monitoring
- Feature freshness tracking

Tables: ml_feast_feature_views, ml_feast_materialization_jobs, ml_feast_feature_freshness
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, cast
from uuid import UUID, uuid4

from .base import BaseRepository

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class MaterializationJobType(str, Enum):
    """Materialization job type values."""

    FULL = "full"
    INCREMENTAL = "incremental"


class MaterializationStatus(str, Enum):
    """Materialization job status values."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class FreshnessStatus(str, Enum):
    """Feature freshness status values."""

    FRESH = "fresh"
    STALE = "stale"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    """Feature view source type values."""

    BATCH = "batch"
    STREAM = "stream"
    REQUEST = "request"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class FeastFeatureView:
    """Feature view entity matching ml_feast_feature_views table."""

    id: Optional[UUID] = None
    name: str = ""
    project: str = "e2i_causal_analytics"
    description: Optional[str] = None

    # Entity configuration
    entities: List[str] = field(default_factory=list)
    entity_join_keys: List[str] = field(default_factory=list)

    # Feature definitions
    features: List[Dict[str, Any]] = field(default_factory=list)
    feature_count: int = 0

    # Source configuration
    source_type: Optional[str] = None
    source_name: Optional[str] = None
    source_config: Dict[str, Any] = field(default_factory=dict)

    # TTL and online store settings
    ttl_seconds: Optional[int] = None
    online_enabled: bool = True
    batch_source_enabled: bool = True

    # Tags and metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    owner: Optional[str] = None

    # Lifecycle
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": str(self.id) if self.id else None,
            "name": self.name,
            "project": self.project,
            "description": self.description,
            "entities": self.entities,
            "entity_join_keys": self.entity_join_keys,
            "features": self.features,
            "feature_count": self.feature_count,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "source_config": self.source_config,
            "ttl_seconds": self.ttl_seconds,
            "online_enabled": self.online_enabled,
            "batch_source_enabled": self.batch_source_enabled,
            "tags": self.tags,
            "owner": self.owner,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeastFeatureView":
        """Create instance from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            name=data.get("name", ""),
            project=data.get("project", "e2i_causal_analytics"),
            description=data.get("description"),
            entities=data.get("entities") or [],
            entity_join_keys=data.get("entity_join_keys") or [],
            features=data.get("features") or [],
            feature_count=data.get("feature_count", 0),
            source_type=data.get("source_type"),
            source_name=data.get("source_name"),
            source_config=data.get("source_config") or {},
            ttl_seconds=data.get("ttl_seconds"),
            online_enabled=data.get("online_enabled", True),
            batch_source_enabled=data.get("batch_source_enabled", True),
            tags=data.get("tags") or {},
            owner=data.get("owner"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
            deleted_at=datetime.fromisoformat(data["deleted_at"])
            if data.get("deleted_at")
            else None,
        )


@dataclass
class FeastMaterializationJob:
    """Materialization job entity matching ml_feast_materialization_jobs table."""

    id: Optional[UUID] = None
    feature_view_id: Optional[UUID] = None
    feature_view_name: str = ""

    # Job identification
    job_id: Optional[str] = None
    job_type: str = "incremental"

    # Time range materialized
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Execution status
    status: str = "pending"
    error_message: Optional[str] = None

    # Metrics
    rows_materialized: int = 0
    bytes_written: Optional[int] = None
    duration_seconds: Optional[float] = None

    # Online store metrics
    online_store_rows_written: Optional[int] = None
    online_store_latency_ms: Optional[float] = None

    # Resource usage
    cpu_seconds: Optional[float] = None
    memory_peak_mb: Optional[float] = None

    # Timestamps
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": str(self.id) if self.id else None,
            "feature_view_id": str(self.feature_view_id) if self.feature_view_id else None,
            "feature_view_name": self.feature_view_name,
            "job_id": self.job_id,
            "job_type": self.job_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "error_message": self.error_message,
            "rows_materialized": self.rows_materialized,
            "bytes_written": self.bytes_written,
            "duration_seconds": self.duration_seconds,
            "online_store_rows_written": self.online_store_rows_written,
            "online_store_latency_ms": self.online_store_latency_ms,
            "cpu_seconds": self.cpu_seconds,
            "memory_peak_mb": self.memory_peak_mb,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeastMaterializationJob":
        """Create instance from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            feature_view_id=UUID(data["feature_view_id"]) if data.get("feature_view_id") else None,
            feature_view_name=data.get("feature_view_name", ""),
            job_id=data.get("job_id"),
            job_type=data.get("job_type", "incremental"),
            start_time=datetime.fromisoformat(data["start_time"])
            if data.get("start_time")
            else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=data.get("status", "pending"),
            error_message=data.get("error_message"),
            rows_materialized=data.get("rows_materialized", 0),
            bytes_written=data.get("bytes_written"),
            duration_seconds=data.get("duration_seconds"),
            online_store_rows_written=data.get("online_store_rows_written"),
            online_store_latency_ms=data.get("online_store_latency_ms"),
            cpu_seconds=data.get("cpu_seconds"),
            memory_peak_mb=data.get("memory_peak_mb"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
        )


@dataclass
class FeastFeatureFreshness:
    """Feature freshness entity matching ml_feast_feature_freshness table."""

    id: Optional[UUID] = None
    feature_view_id: Optional[UUID] = None
    feature_view_name: str = ""

    # Freshness timestamp
    recorded_at: Optional[datetime] = None

    # Freshness metrics
    last_materialization_time: Optional[datetime] = None
    staleness_seconds: Optional[int] = None
    data_lag_seconds: Optional[int] = None

    # Feature statistics
    null_rate: Optional[float] = None
    unique_count: Optional[int] = None

    # Health status
    freshness_status: str = "unknown"

    # Thresholds used
    staleness_threshold_seconds: Optional[int] = None
    critical_threshold_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": str(self.id) if self.id else None,
            "feature_view_id": str(self.feature_view_id) if self.feature_view_id else None,
            "feature_view_name": self.feature_view_name,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "last_materialization_time": self.last_materialization_time.isoformat()
            if self.last_materialization_time
            else None,
            "staleness_seconds": self.staleness_seconds,
            "data_lag_seconds": self.data_lag_seconds,
            "null_rate": self.null_rate,
            "unique_count": self.unique_count,
            "freshness_status": self.freshness_status,
            "staleness_threshold_seconds": self.staleness_threshold_seconds,
            "critical_threshold_seconds": self.critical_threshold_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeastFeatureFreshness":
        """Create instance from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            feature_view_id=UUID(data["feature_view_id"]) if data.get("feature_view_id") else None,
            feature_view_name=data.get("feature_view_name", ""),
            recorded_at=datetime.fromisoformat(data["recorded_at"])
            if data.get("recorded_at")
            else None,
            last_materialization_time=datetime.fromisoformat(data["last_materialization_time"])
            if data.get("last_materialization_time")
            else None,
            staleness_seconds=data.get("staleness_seconds"),
            data_lag_seconds=data.get("data_lag_seconds"),
            null_rate=data.get("null_rate"),
            unique_count=data.get("unique_count"),
            freshness_status=data.get("freshness_status", "unknown"),
            staleness_threshold_seconds=data.get("staleness_threshold_seconds"),
            critical_threshold_seconds=data.get("critical_threshold_seconds"),
        )


# ============================================================================
# REPOSITORY CLASSES
# ============================================================================


class FeastFeatureViewRepository(BaseRepository[FeastFeatureView]):
    """Repository for Feast feature view records."""

    table_name = "ml_feast_feature_views"
    model_class = FeastFeatureView

    def _to_model(self, data: Dict[str, Any]) -> FeastFeatureView:
        """Convert database row to model."""
        return FeastFeatureView.from_dict(data)

    async def create_feature_view(
        self,
        name: str,
        project: str = "e2i_causal_analytics",
        description: Optional[str] = None,
        entities: Optional[List[str]] = None,
        features: Optional[List[Dict[str, Any]]] = None,
        source_type: Optional[str] = None,
        source_name: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        online_enabled: bool = True,
        owner: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Optional[FeastFeatureView]:
        """Create a new feature view record.

        Args:
            name: Feature view name
            project: Feast project name
            description: Description of the feature view
            entities: List of entity names
            features: List of feature definitions
            source_type: Source type (batch/stream/request)
            source_name: Source name
            ttl_seconds: TTL in seconds
            online_enabled: Whether online store is enabled
            owner: Owner username
            tags: Feature view tags

        Returns:
            Created FeastFeatureView or None
        """
        if not self.client:
            logger.warning("No Supabase client available for feature view creation")
            return None

        try:
            feature_view = FeastFeatureView(
                id=uuid4(),
                name=name,
                project=project,
                description=description,
                entities=entities or [],
                features=features or [],
                feature_count=len(features) if features else 0,
                source_type=source_type,
                source_name=source_name,
                ttl_seconds=ttl_seconds,
                online_enabled=online_enabled,
                owner=owner,
                tags=tags or {},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            data = feature_view.to_dict()
            data = {k: v for k, v in data.items() if v is not None}

            result = self.client.table(self.table_name).insert(data).execute()

            if result.data:
                logger.info(f"Created Feast feature view: {name} (project: {project})")
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to create feature view: {e}")
            return None

    async def get_by_name(
        self,
        name: str,
        project: str = "e2i_causal_analytics",
    ) -> Optional[FeastFeatureView]:
        """Get feature view by name.

        Args:
            name: Feature view name
            project: Feast project name

        Returns:
            Feature view or None
        """
        if not self.client:
            return None

        try:
            result = (
                self.client.table(self.table_name)
                .select("*")
                .eq("name", name)
                .eq("project", project)
                .is_("deleted_at", "null")
                .limit(1)
                .execute()
            )

            if result.data:
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to get feature view by name: {e}")
            return None

    async def get_active_views(
        self,
        project: Optional[str] = None,
        online_only: bool = False,
    ) -> List[FeastFeatureView]:
        """Get all active feature views.

        Args:
            project: Optional project filter
            online_only: Only return views with online store enabled

        Returns:
            List of active feature views
        """
        if not self.client:
            return []

        try:
            query = self.client.table(self.table_name).select("*").is_("deleted_at", "null")

            if project:
                query = query.eq("project", project)

            if online_only:
                query = query.eq("online_enabled", True)

            result = query.execute()
            return [self._to_model(row) for row in result.data]

        except Exception as e:
            logger.error(f"Failed to get active feature views: {e}")
            return []

    async def update_feature_view(
        self,
        view_id: UUID,
        features: Optional[List[Dict[str, Any]]] = None,
        ttl_seconds: Optional[int] = None,
        online_enabled: Optional[bool] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Optional[FeastFeatureView]:
        """Update feature view configuration.

        Args:
            view_id: Feature view UUID
            features: Updated feature definitions
            ttl_seconds: Updated TTL
            online_enabled: Updated online store setting
            tags: Updated tags

        Returns:
            Updated feature view or None
        """
        if not self.client:
            return None

        try:
            update_data: Dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            if features is not None:
                update_data["features"] = features
                update_data["feature_count"] = len(features)
            if ttl_seconds is not None:
                update_data["ttl_seconds"] = ttl_seconds
            if online_enabled is not None:
                update_data["online_enabled"] = online_enabled
            if tags is not None:
                update_data["tags"] = tags

            result = (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("id", str(view_id))
                .execute()
            )

            if result.data:
                logger.info(f"Updated feature view {view_id}")
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to update feature view: {e}")
            return None

    async def soft_delete(self, view_id: UUID) -> bool:
        """Soft delete a feature view.

        Args:
            view_id: Feature view UUID

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            result = (
                self.client.table(self.table_name)
                .update(
                    {
                        "deleted_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                .eq("id", str(view_id))
                .execute()
            )

            if result.data:
                logger.info(f"Soft deleted feature view {view_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to soft delete feature view: {e}")
            return False


class FeastMaterializationRepository(BaseRepository[FeastMaterializationJob]):
    """Repository for Feast materialization job records."""

    table_name = "ml_feast_materialization_jobs"
    model_class = FeastMaterializationJob

    def _to_model(self, data: Dict[str, Any]) -> FeastMaterializationJob:
        """Convert database row to model."""
        return FeastMaterializationJob.from_dict(data)

    async def create_job(
        self,
        feature_view_id: UUID,
        feature_view_name: str,
        start_time: datetime,
        end_time: datetime,
        job_type: str = "incremental",
        job_id: Optional[str] = None,
    ) -> Optional[FeastMaterializationJob]:
        """Create a new materialization job record.

        Args:
            feature_view_id: Feature view UUID
            feature_view_name: Feature view name
            start_time: Materialization start time
            end_time: Materialization end time
            job_type: Job type (full/incremental)
            job_id: External job ID

        Returns:
            Created job or None
        """
        if not self.client:
            logger.warning("No Supabase client available for job creation")
            return None

        try:
            job = FeastMaterializationJob(
                id=uuid4(),
                feature_view_id=feature_view_id,
                feature_view_name=feature_view_name,
                job_id=job_id,
                job_type=job_type,
                start_time=start_time,
                end_time=end_time,
                status="pending",
                created_at=datetime.now(timezone.utc),
            )

            data = job.to_dict()
            data = {k: v for k, v in data.items() if v is not None}

            result = self.client.table(self.table_name).insert(data).execute()

            if result.data:
                logger.info(f"Created materialization job for {feature_view_name}")
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to create materialization job: {e}")
            return None

    async def update_status(
        self,
        job_id: UUID,
        status: MaterializationStatus,
        error_message: Optional[str] = None,
        rows_materialized: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ) -> Optional[FeastMaterializationJob]:
        """Update job status.

        Args:
            job_id: Job UUID
            status: New status
            error_message: Error message if failed
            rows_materialized: Rows materialized
            duration_seconds: Job duration

        Returns:
            Updated job or None
        """
        if not self.client:
            return None

        try:
            update_data: Dict[str, Any] = {"status": status.value}

            if status == MaterializationStatus.RUNNING:
                update_data["started_at"] = datetime.now(timezone.utc).isoformat()
            elif status in (MaterializationStatus.SUCCESS, MaterializationStatus.FAILED):
                update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

            if error_message:
                update_data["error_message"] = error_message
            if rows_materialized is not None:
                update_data["rows_materialized"] = rows_materialized
            if duration_seconds is not None:
                update_data["duration_seconds"] = duration_seconds

            result = (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("id", str(job_id))
                .execute()
            )

            if result.data:
                logger.info(f"Updated materialization job {job_id} to {status.value}")
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
            return None

    async def get_jobs_for_view(
        self,
        feature_view_id: UUID,
        limit: int = 10,
        status: Optional[MaterializationStatus] = None,
    ) -> List[FeastMaterializationJob]:
        """Get jobs for a feature view.

        Args:
            feature_view_id: Feature view UUID
            limit: Maximum jobs to return
            status: Optional status filter

        Returns:
            List of jobs
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("feature_view_id", str(feature_view_id))
                .order("created_at", desc=True)
                .limit(limit)
            )

            if status:
                query = query.eq("status", status.value)

            result = query.execute()
            return [self._to_model(row) for row in result.data]

        except Exception as e:
            logger.error(f"Failed to get jobs for view: {e}")
            return []

    async def get_recent_jobs(
        self,
        days: int = 7,
        status: Optional[MaterializationStatus] = None,
    ) -> List[FeastMaterializationJob]:
        """Get recent materialization jobs.

        Args:
            days: Number of days to look back
            status: Optional status filter

        Returns:
            List of recent jobs
        """
        if not self.client:
            return []

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

            query = (
                self.client.table(self.table_name)
                .select("*")
                .gte("created_at", cutoff.isoformat())
                .order("created_at", desc=True)
            )

            if status:
                query = query.eq("status", status.value)

            result = query.execute()
            return [self._to_model(row) for row in result.data]

        except Exception as e:
            logger.error(f"Failed to get recent jobs: {e}")
            return []

    async def get_summary(
        self,
        feature_view_id: UUID,
        days: int = 7,
    ) -> Dict[str, Any]:
        """Get materialization summary for a feature view.

        Args:
            feature_view_id: Feature view UUID
            days: Days to aggregate

        Returns:
            Summary dictionary
        """
        if not self.client:
            return {}

        try:
            # Try database function first
            result = self.client.rpc(
                "get_feast_materialization_summary",
                {
                    "p_feature_view_id": str(feature_view_id),
                    "p_days": days,
                },
            ).execute()

            if result.data:
                return cast(
                    Dict[str, Any],
                    result.data[0] if isinstance(result.data, list) else result.data,
                )

            # Fallback to manual aggregation
            jobs = await self.get_jobs_for_view(feature_view_id, limit=100)
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            jobs = [j for j in jobs if j.created_at and j.created_at > cutoff]

            if not jobs:
                return {}

            return {
                "total_jobs": len(jobs),
                "successful_jobs": len([j for j in jobs if j.status == "success"]),
                "failed_jobs": len([j for j in jobs if j.status == "failed"]),
                "total_rows_materialized": sum(j.rows_materialized for j in jobs),
                "avg_duration_seconds": sum(j.duration_seconds or 0 for j in jobs) / len(jobs)
                if jobs
                else 0,
                "success_rate": len([j for j in jobs if j.status == "success"]) / len(jobs) * 100
                if jobs
                else 0,
            }

        except Exception as e:
            logger.error(f"Failed to get materialization summary: {e}")
            return {}


class FeastFreshnessRepository(BaseRepository[FeastFeatureFreshness]):
    """Repository for Feast feature freshness records."""

    table_name = "ml_feast_feature_freshness"
    model_class = FeastFeatureFreshness

    def _to_model(self, data: Dict[str, Any]) -> FeastFeatureFreshness:
        """Convert database row to model."""
        return FeastFeatureFreshness.from_dict(data)

    async def record_freshness(
        self,
        feature_view_id: UUID,
        feature_view_name: str,
        last_materialization_time: Optional[datetime] = None,
        staleness_seconds: Optional[int] = None,
        null_rate: Optional[float] = None,
        freshness_status: str = "unknown",
        staleness_threshold_seconds: Optional[int] = None,
    ) -> Optional[FeastFeatureFreshness]:
        """Record a freshness check.

        Args:
            feature_view_id: Feature view UUID
            feature_view_name: Feature view name
            last_materialization_time: Last materialization timestamp
            staleness_seconds: Seconds since last materialization
            null_rate: Null rate in features
            freshness_status: Status (fresh/stale/critical/unknown)
            staleness_threshold_seconds: Threshold for staleness

        Returns:
            Created freshness record or None
        """
        if not self.client:
            logger.warning("No Supabase client available for freshness recording")
            return None

        try:
            freshness = FeastFeatureFreshness(
                id=uuid4(),
                feature_view_id=feature_view_id,
                feature_view_name=feature_view_name,
                recorded_at=datetime.now(timezone.utc),
                last_materialization_time=last_materialization_time,
                staleness_seconds=staleness_seconds,
                null_rate=null_rate,
                freshness_status=freshness_status,
                staleness_threshold_seconds=staleness_threshold_seconds,
            )

            data = freshness.to_dict()
            data = {k: v for k, v in data.items() if v is not None}

            result = self.client.table(self.table_name).insert(data).execute()

            if result.data:
                logger.debug(f"Recorded freshness for {feature_view_name}: {freshness_status}")
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to record freshness: {e}")
            return None

    async def get_latest(
        self,
        feature_view_id: UUID,
    ) -> Optional[FeastFeatureFreshness]:
        """Get latest freshness record for a view.

        Args:
            feature_view_id: Feature view UUID

        Returns:
            Latest freshness record or None
        """
        if not self.client:
            return None

        try:
            result = (
                self.client.table(self.table_name)
                .select("*")
                .eq("feature_view_id", str(feature_view_id))
                .order("recorded_at", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to get latest freshness: {e}")
            return None

    async def get_stale_views(
        self,
        threshold_status: FreshnessStatus = FreshnessStatus.STALE,
    ) -> List[FeastFeatureFreshness]:
        """Get feature views that are stale or critical.

        Args:
            threshold_status: Minimum status to include (STALE includes CRITICAL)

        Returns:
            List of freshness records for stale views
        """
        if not self.client:
            return []

        try:
            statuses = ["critical"]
            if threshold_status == FreshnessStatus.STALE:
                statuses.append("stale")

            result = (
                self.client.table(self.table_name)
                .select("*")
                .in_("freshness_status", statuses)
                .order("recorded_at", desc=True)
                .execute()
            )

            # Deduplicate by feature_view_id, keeping only latest
            seen: Dict[str, FeastFeatureFreshness] = {}
            for row in result.data:
                freshness = self._to_model(row)
                view_id = str(freshness.feature_view_id)
                if view_id not in seen:
                    seen[view_id] = freshness

            return list(seen.values())

        except Exception as e:
            logger.error(f"Failed to get stale views: {e}")
            return []

    async def get_freshness_history(
        self,
        feature_view_id: UUID,
        hours: int = 24,
    ) -> List[FeastFeatureFreshness]:
        """Get freshness history for a view.

        Args:
            feature_view_id: Feature view UUID
            hours: Hours of history

        Returns:
            List of freshness records
        """
        if not self.client:
            return []

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

            result = (
                self.client.table(self.table_name)
                .select("*")
                .eq("feature_view_id", str(feature_view_id))
                .gte("recorded_at", cutoff.isoformat())
                .order("recorded_at", desc=True)
                .execute()
            )

            return [self._to_model(row) for row in result.data]

        except Exception as e:
            logger.error(f"Failed to get freshness history: {e}")
            return []
