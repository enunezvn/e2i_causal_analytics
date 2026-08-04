"""Health Score Agent MLflow Tracker.

This module provides MLflow experiment tracking for the Health Score Agent,
enabling monitoring of health check metrics across different scopes.

Tracked metrics:
- Overall health score and grade
- Component health scores (component, model, pipeline, agent)
- Check latency and scope
- Issue and warning counts
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Mapping, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .agent import HealthScoreOutput
    from .state import HealthScoreState

logger = logging.getLogger(__name__)

# Experiment prefix for Health Score Agent
EXPERIMENT_PREFIX = "e2i_causal/health_score"

# Fallback when MLFLOW_TRACKING_URI is absent. Deliberately a tracking SERVER and
# not MLflow's own "./mlruns" default: every deployment of this platform runs an
# MLflow server (docker/docker-compose.yml `mlflow`), and a silent fall-back to a
# local file store writes a metrics trail nobody ever reads. Matches the twelve
# sibling agent trackers (gap_analyzer, explainer, causal_impact, ...).
DEFAULT_TRACKING_URI = "http://localhost:5000"

# Artifact URI schemes the MLflow client hands off to the tracking server or an
# object store — nothing is written through THIS process's filesystem.
_NON_LOCAL_ARTIFACT_SCHEMES = frozenset(
    {
        "mlflow-artifacts",
        "http",
        "https",
        "s3",
        "gs",
        "abfs",
        "abfss",
        "wasbs",
        "dbfs",
        "ftp",
        "sftp",
    }
)


def resolve_tracking_uri(
    explicit: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve the MLflow tracking URI this tracker should talk to.

    Precedence: explicit argument > ``MLFLOW_TRACKING_URI`` > DEFAULT_TRACKING_URI.

    #1452: this tracker previously left ``mlflow``'s global URI untouched when no
    URI was passed, relying on MLflow's implicit env read. That works in the
    container (compose x-common-env forwards ``MLFLOW_TRACKING_URI=http://mlflow:5000``)
    but silently degrades to a local ``./mlruns`` file store anywhere the env var
    is missing. Resolving explicitly makes the destination testable and uniform.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    source = os.environ if env is None else env
    from_env = (source.get("MLFLOW_TRACKING_URI") or "").strip()
    return from_env or DEFAULT_TRACKING_URI


@dataclass(frozen=True)
class ArtifactDestination:
    """Where ``mlflow.log_artifact`` would actually put bytes for a run.

    Attributes:
        uri: the run's ``artifact_uri`` as reported by the tracking server.
        local_path: filesystem path this process would write to, or ``None``
            when the upload is proxied through the tracking server / an object
            store (the healthy case).
        blocked_root: the nearest existing ancestor of ``local_path`` that this
            process cannot write to, or ``None`` when the write can succeed.
        missing_root: the first component of ``local_path`` that does not exist
            (#1459: the REAL gap to name — e.g. ``/mlflow`` for the legacy
            production URI, whose ``blocked_root`` is merely ``/``). ``None``
            when the write can succeed or when ``local_path`` fully exists.
    """

    uri: str
    local_path: Optional[str]
    blocked_root: Optional[str]
    missing_root: Optional[str] = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked_root is not None


def classify_artifact_destination(artifact_uri: str) -> ArtifactDestination:
    """Classify a run's ``artifact_uri`` as proxied, locally-writable, or blocked.

    #1452 root cause: experiment ``e2i_causal/health_score/default`` (id 9) was
    created before b0a30f11 added ``artifact_location="mlflow-artifacts:/"``, so
    it still carries the tracking server's own filesystem path
    (``/mlflow/artifacts/9``). MLflow hands that path straight to the CLIENT,
    which in the read-only api container tries to ``mkdir('/mlflow')`` and dies
    with ``[Errno 30] Read-only file system``. ``artifact_location`` is
    create-time only — MLflow's UpdateExperiment can rename an experiment but
    cannot rewrite its artifact root — so the client has to detect this rather
    than retry it forever.
    """
    parsed = urlparse(artifact_uri)
    if parsed.scheme in _NON_LOCAL_ARTIFACT_SCHEMES:
        return ArtifactDestination(uri=artifact_uri, local_path=None, blocked_root=None)

    # Bare path or file:// -> the client writes it itself.
    local_path = parsed.path if parsed.scheme == "file" else artifact_uri
    if not local_path:
        return ArtifactDestination(uri=artifact_uri, local_path=None, blocked_root=None)

    probe = os.path.abspath(local_path)
    missing_root: Optional[str] = None
    while not os.path.exists(probe):
        # After the walk this holds the SHALLOWEST nonexistent component —
        # e.g. '/mlflow' for '/mlflow/artifacts/9/<run>/artifacts' (#1459).
        missing_root = probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    writable = os.path.isdir(probe) and os.access(probe, os.W_OK)
    return ArtifactDestination(
        uri=artifact_uri,
        local_path=local_path,
        blocked_root=None if writable else probe,
        missing_root=None if writable else missing_root,
    )


# Persistent-condition reports are deduplicated for the LIFETIME OF THE PROCESS,
# not per tracker instance: src/api/routes/health_score.py builds a fresh
# HealthScoreAgent (and therefore a fresh tracker) on every request, so
# per-instance state would never suppress anything. #1452 asks for "once at
# startup, not a per-run warning forever" — process-wide is the faithful scope.
_reported_conditions: set[str] = set()


def reset_tracking_reports() -> None:
    """Clear the once-per-process report ledger (test seam)."""
    _reported_conditions.clear()


def _report_once(key: str, message: str) -> bool:
    """WARNING on the first occurrence of ``key``, DEBUG on every repeat.

    Returns True when this call was the first (i.e. the WARNING was emitted).
    """
    if key in _reported_conditions:
        logger.debug("%s [recurring; already reported once this process]", message)
        return False
    _reported_conditions.add(key)
    logger.warning(message)
    return True


def report_artifact_write_blocked(artifact_uri: str, run_id: str) -> bool:
    """Report — at most once per process — that artifacts cannot be written.

    Returns True when the destination is blocked AND this was the first report,
    False when the destination is fine or the condition was already surfaced.

    #1459: the message names the MISSING artifact root (e.g. ``/mlflow``)
    separately from the unwritable ancestor (``/`` in production), and it must
    NEVER advise mounting a writable volume — for the real production URI the
    blocked root is ``/``, so that advice told operators to defeat the very
    ``read_only: true`` hardening (docker-compose ``e2i_api``) it cites.
    Recreating the experiment onto the artifact-proxy convention (b0a30f11) is
    the single recommended remediation.
    """
    destination = classify_artifact_destination(artifact_uri)
    if not destination.is_blocked:
        return False

    if destination.missing_root:
        gap = (
            f"but {destination.missing_root!r} does not exist in this container and "
            f"its nearest existing ancestor {destination.blocked_root!r} is not "
            "writable from this process"
        )
    else:
        gap = f"but {destination.blocked_root!r} is not writable from this process"

    return _report_once(
        f"artifact-root-unwritable:{destination.blocked_root}",
        (
            "MLflow artifact logging is disabled for health_score: the experiment's "
            f"artifact_location resolves to the local path {destination.local_path!r}, "
            f"{gap} "
            "(the api container rootfs is read-only by design). Metrics, params and "
            "tags are unaffected and still reach the tracking server. Remediation: "
            "this experiment predates the artifact-proxy convention and MLflow cannot "
            "rewrite artifact_location in place — rename/archive it on the tracking "
            "server so it is recreated with artifact_location='mlflow-artifacts:/'. "
            f"(first seen on run {run_id})"
        ),
    )


@dataclass
class HealthScoreContext:
    """Context for a health score tracking run."""

    run_id: str
    experiment_name: str
    check_scope: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Where the tracking server says this run's artifacts belong (#1452). None
    # for the degraded contexts yielded when MLflow is unavailable.
    artifact_uri: Optional[str] = None


@dataclass
class HealthScoreMetrics:
    """Structured metrics for health score tracking."""

    # Overall metrics
    overall_health_score: float = 0.0
    health_grade: str = "F"

    # Component scores (0-1 scale). F1: None == unmeasured dimension (no real
    # backend). An unmeasured dim is OMITTED from the MLflow metrics below
    # (honest absence) rather than logged as a fabricated 0.0.
    component_health_score: Optional[float] = None
    model_health_score: Optional[float] = None
    pipeline_health_score: Optional[float] = None
    agent_health_score: Optional[float] = None

    # Issue counts
    critical_issues_count: int = 0
    warnings_count: int = 0

    # Execution metadata
    check_scope: str = "full"
    total_latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for MLflow logging.

        F1: per-dimension scores that are None (unmeasured) are OMITTED so
        MLflow records honest absence — never a fabricated 0.0 for a dimension
        that has no real backend. (mlflow.log_metrics requires numeric values.)
        """
        metrics: Dict[str, Any] = {
            "overall_health_score": self.overall_health_score,
            "health_grade_numeric": self._grade_to_numeric(self.health_grade),
            "critical_issues_count": self.critical_issues_count,
            "warnings_count": self.warnings_count,
            "total_latency_ms": self.total_latency_ms,
        }
        for key, value in (
            ("component_health_score", self.component_health_score),
            ("model_health_score", self.model_health_score),
            ("pipeline_health_score", self.pipeline_health_score),
            ("agent_health_score", self.agent_health_score),
        ):
            if value is not None:
                metrics[key] = value
        return metrics

    @staticmethod
    def _grade_to_numeric(grade: str) -> int:
        """Convert letter grade to numeric for trending."""
        grade_map = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
        return grade_map.get(grade, 0)


class HealthScoreMLflowTracker:
    """MLflow tracker for Health Score Agent.

    Provides experiment tracking for health check runs including:
    - Overall and component health scores
    - Issue and warning tracking
    - Latency monitoring
    - Historical trend analysis

    Usage:
        tracker = HealthScoreMLflowTracker()
        async with tracker.start_health_run(
            experiment_name="production",
            check_scope="full"
        ) as ctx:
            # Run health check
            output = await agent.check_health(scope="full")
            # Log results
            await tracker.log_health_result(output, state)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize MLflow tracker.

        Args:
            tracking_uri: Optional MLflow tracking URI. When omitted the URI is
                resolved from ``MLFLOW_TRACKING_URI`` (compose x-common-env
                forwards ``http://mlflow:5000`` to every service), falling back
                to :data:`DEFAULT_TRACKING_URI`.
        """
        self._mlflow = None
        self._tracking_uri = resolve_tracking_uri(tracking_uri)
        self._current_run_id: Optional[str] = None
        self._current_artifact_uri: Optional[str] = None

    @property
    def tracking_uri(self) -> str:
        """The MLflow tracking URI this tracker talks to (always resolved)."""
        return self._tracking_uri

    def _get_mlflow(self):
        """Lazy load MLflow to avoid import errors when not installed."""
        if self._mlflow is None:
            try:
                import mlflow

                # Always pin the resolved URI — never inherit whatever another
                # component left on mlflow's process-global setting (#1452).
                mlflow.set_tracking_uri(self._tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLflow not installed, tracking disabled")
                return None
        return self._mlflow

    @asynccontextmanager
    async def start_health_run(
        self,
        experiment_name: str = "default",
        check_scope: str = "full",
    ) -> AsyncIterator[HealthScoreContext]:
        """Start an MLflow run for health check tracking.

        Args:
            experiment_name: Name of the experiment (e.g., "production", "staging")
            check_scope: Scope of health check ("full", "quick", "models", etc.)

        Yields:
            HealthScoreContext with run information
        """
        mlflow = self._get_mlflow()

        if mlflow is None:
            # Yield a dummy context if MLflow is not available
            yield HealthScoreContext(
                run_id="no-mlflow",
                experiment_name=experiment_name,
                check_scope=check_scope,
            )
            return

        # Create experiment if it doesn't exist
        full_experiment_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
        try:
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    full_experiment_name,
                    artifact_location="mlflow-artifacts:/",
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            # Persistent condition (server down / bad URI) — surface it once per
            # process, not on every health check forever (#1452).
            _report_once(
                f"experiment-unavailable:{full_experiment_name}:{type(e).__name__}",
                (
                    f"Health-score MLflow tracking is degraded: could not create/get "
                    f"experiment {full_experiment_name!r} on {self._tracking_uri!r}: {e}. "
                    "Health checks continue; their metrics trail is not being recorded."
                ),
            )
            yield HealthScoreContext(
                run_id="experiment-error",
                experiment_name=experiment_name,
                check_scope=check_scope,
            )
            return

        # Start MLflow run
        try:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                self._current_run_id = run.info.run_id
                self._current_artifact_uri = getattr(run.info, "artifact_uri", None)

                # Log run parameters
                mlflow.log_params(
                    {
                        "agent": "health_score",
                        "tier": 3,
                        "check_scope": check_scope,
                        "agent_type": "standard",
                    }
                )

                ctx = HealthScoreContext(
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    check_scope=check_scope,
                    artifact_uri=self._current_artifact_uri,
                )

                yield ctx

                self._current_run_id = None
                self._current_artifact_uri = None

        except Exception as e:
            logger.error(f"MLflow run failed: {e}")
            self._current_run_id = None
            self._current_artifact_uri = None
            raise

    async def log_health_result(
        self,
        output: "HealthScoreOutput",
        state: Optional["HealthScoreState"] = None,
    ) -> None:
        """Log health check results to MLflow.

        Args:
            output: HealthScoreOutput from agent execution
            state: Optional final state for detailed logging
        """
        mlflow = self._get_mlflow()
        if mlflow is None or self._current_run_id is None:
            return

        # --- Metrics + tags -------------------------------------------------
        # Deliberately separated from the artifact upload below (#1452): these
        # two failed as one broad try/except, so an artifact-only failure was
        # reported as "Failed to log health metrics" even though every metric
        # had already reached the tracking server. Misattributed errors send
        # people looking in the wrong place.
        try:
            metrics = HealthScoreMetrics(
                overall_health_score=output.overall_health_score,
                health_grade=output.health_grade,
                component_health_score=output.component_health_score,
                model_health_score=output.model_health_score,
                pipeline_health_score=output.pipeline_health_score,
                agent_health_score=output.agent_health_score,
                critical_issues_count=len(output.critical_issues),
                warnings_count=len(output.warnings),
                total_latency_ms=output.total_latency_ms,
            )

            mlflow.log_metrics(metrics.to_dict())

            # Log tags for filtering
            mlflow.set_tags(
                {
                    "health_grade": output.health_grade,
                    "has_critical_issues": str(len(output.critical_issues) > 0).lower(),
                    "has_warnings": str(len(output.warnings) > 0).lower(),
                }
            )

            logger.debug(
                f"Logged health metrics to MLflow run {self._current_run_id}: "
                f"score={output.overall_health_score}, grade={output.health_grade}"
            )
        except Exception as e:
            _report_once(
                f"metrics-log-failed:{type(e).__name__}",
                f"Failed to log health metrics to MLflow ({self._tracking_uri}): {e}",
            )
            return

        # --- Detailed results artifact --------------------------------------
        if not state:
            return

        # Preflight the destination instead of discovering it via OSError on
        # every single run. Experiments created before the artifact-proxy
        # convention hand the CLIENT a server-local path (e.g. /mlflow/...)
        # that the read-only api rootfs can never satisfy.
        artifact_uri = self._current_artifact_uri
        if artifact_uri and classify_artifact_destination(artifact_uri).is_blocked:
            # Warns on the first occurrence in this process, DEBUG thereafter.
            report_artifact_write_blocked(artifact_uri, self._current_run_id or "unknown")
            return

        try:
            artifact_data = {
                "timestamp": output.timestamp,
                "overall_health_score": output.overall_health_score,
                "health_grade": output.health_grade,
                "health_summary": output.health_summary,
                "component_scores": {
                    "component": output.component_health_score,
                    "model": output.model_health_score,
                    "pipeline": output.pipeline_health_score,
                    "agent": output.agent_health_score,
                },
                "critical_issues": output.critical_issues,
                "warnings": output.warnings,
                "component_statuses": state.get("component_statuses", []),
                "model_metrics": state.get("model_metrics", []),
                "pipeline_statuses": state.get("pipeline_statuses", []),
                "agent_statuses": state.get("agent_statuses", []),
            }

            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = os.path.join(tmpdir, "health_check_results.json")
                with open(artifact_path, "w") as f:
                    json.dump(artifact_data, f, indent=2, default=str)
                mlflow.log_artifact(artifact_path)

        except Exception as e:
            _report_once(
                f"artifact-log-failed:{type(e).__name__}",
                (
                    "Failed to log the health-check results artifact to MLflow "
                    f"(run {self._current_run_id}, artifact_uri={artifact_uri!r}): {e}. "
                    "Metrics, params and tags for this run were persisted."
                ),
            )

    async def get_health_history(
        self,
        experiment_name: str = "default",
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query historical health check runs.

        Args:
            experiment_name: Name of the experiment to query
            max_results: Maximum number of results to return

        Returns:
            List of historical health check results
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return []

        try:
            full_experiment_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                return []

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            history = []
            for _, row in runs.iterrows():
                history.append(
                    {
                        "run_id": row["run_id"],
                        "timestamp": row["start_time"],
                        "overall_health_score": row.get("metrics.overall_health_score"),
                        "health_grade_numeric": row.get("metrics.health_grade_numeric"),
                        "component_health_score": row.get("metrics.component_health_score"),
                        "model_health_score": row.get("metrics.model_health_score"),
                        "pipeline_health_score": row.get("metrics.pipeline_health_score"),
                        "agent_health_score": row.get("metrics.agent_health_score"),
                        "critical_issues_count": row.get("metrics.critical_issues_count"),
                        "warnings_count": row.get("metrics.warnings_count"),
                        "total_latency_ms": row.get("metrics.total_latency_ms"),
                        "check_scope": row.get("params.check_scope"),
                    }
                )

            return history

        except Exception as e:
            logger.warning(f"Failed to query health history: {e}")
            return []

    async def get_health_trend(
        self,
        experiment_name: str = "default",
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Get health score trend over time.

        Args:
            experiment_name: Name of the experiment to query
            hours: Number of hours to look back

        Returns:
            Dictionary with trend analysis
        """
        history = await self.get_health_history(experiment_name, max_results=1000)

        if not history:
            return {"trend": "unknown", "data_points": 0}

        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Filter to time window
        recent = [
            h
            for h in history
            if h.get("timestamp")
            and (
                isinstance(h["timestamp"], datetime)
                and h["timestamp"].replace(tzinfo=timezone.utc) > cutoff
            )
        ]

        if len(recent) < 2:
            return {
                "trend": "insufficient_data",
                "data_points": len(recent),
            }

        # Calculate trend
        scores = [
            h["overall_health_score"] for h in recent if h.get("overall_health_score") is not None
        ]

        if not scores:
            return {"trend": "no_scores", "data_points": 0}

        avg_score = sum(scores) / len(scores)
        first_half = scores[: len(scores) // 2]
        second_half = scores[len(scores) // 2 :]

        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0

        if second_avg > first_avg + 5:
            trend = "improving"
        elif second_avg < first_avg - 5:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "data_points": len(recent),
            "avg_score": avg_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "latest_score": scores[0] if scores else None,
        }


def create_tracker(tracking_uri: Optional[str] = None) -> HealthScoreMLflowTracker:
    """Factory function to create a Health Score MLflow tracker.

    Args:
        tracking_uri: Optional MLflow tracking URI. Omit to resolve it from
            ``MLFLOW_TRACKING_URI`` (see :func:`resolve_tracking_uri`).

    Returns:
        Configured HealthScoreMLflowTracker instance
    """
    return HealthScoreMLflowTracker(tracking_uri=tracking_uri)
