"""MetricRecorder — idempotent delete-then-insert wrapper around PerformanceMetricRepository.

Design notes
------------
* ``_resolve_model_id`` is imported at module level so tests can monkeypatch
  it via ``src.mlops.gold_standard_eval.recorder._resolve_model_id``.
* ``split_version`` in P1 is carried ONLY via the delete-scope filter
  (``delete_metrics(model_id, source, split_version)``).  ``record_metrics``
  does not expose a metadata kwarg, so embedding split_version into each row's
  metadata would require extending PerformanceMetricRecord — deferred to a
  later task.  The idempotency guarantee (re-run clears exactly the prior rows
  for this source+split) is fully preserved: the delete scope matches the insert
  scope because both are keyed by (model_id, source).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

# Import the async resolver at module level so it can be monkeypatched in tests.
from src.repositories.drift_monitoring import _resolve_model_id

if TYPE_CHECKING:
    from src.repositories.drift_monitoring import PerformanceMetricRepository

logger = logging.getLogger(__name__)

# Disjoint source tag for structured eval artifacts (confusion matrix / ROC
# curve). Kept separate from the scalar 'holdout' rows so each idempotent delete
# scope stays disjoint, and added to the eval runners' re-run cleanup loops so
# the registry-row replace (FK RESTRICT) is not blocked by stale curve rows.
HOLDOUT_CURVE_SOURCE = "holdout_curve"


class MetricRecorder:
    """Idempotent recorder: delete prior rows for a run, then bulk-insert.

    Usage::

        recorder = MetricRecorder(repo)
        await recorder.record_run(
            model_version="propensity_v2.1.0",
            points=[
                (datetime(2026, 1, 1, tzinfo=timezone.utc), {"auc_roc": 0.82}, 500),
                (datetime(2026, 2, 1, tzinfo=timezone.utc), {"auc_roc": 0.83}, 490),
            ],
            source="backtest_wf",
            split_version="e2i_pilot_v3",
        )

    The delete step fires once with the fully-resolved ``model_id`` to clear any
    rows from a prior run (the ``ml_performance_metrics`` table has no unique
    key, so without the delete step re-runs accumulate duplicate rows).
    """

    def __init__(self, repo: "PerformanceMetricRepository") -> None:
        self.repo = repo

    async def record_run(
        self,
        model_version: str,
        points: list[tuple[datetime, dict[str, float], int]],
        *,
        source: str,
        split_version: str | None = None,
        cis: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Delete prior rows for this (model_version, source, split_version), then insert.

        Args:
            model_version: Model handle or uuid string passed to the repository.
            points: Sequence of ``(measured_at_month, metrics_dict, sample_size)``
                tuples.  Each maps to one ``record_metrics`` call with
                ``measured_at=measured_at_month``.
            source: Tag for the metric origin (``'backtest_wf'`` or ``'holdout'``).
            split_version: Optional cohort/split label.  Used to scope the
                delete filter so only rows from this split are cleared.  NOT
                embedded in individual metric rows in P1 (see module docstring).
                Must be ``None`` until row-level metadata storage is implemented
                (P2); passing a non-None value would cause the delete filter to
                match no rows while the insert still fires, breaking idempotency.
            cis: Optional ``{metric_name: (ci_lower, ci_upper)}`` mapping passed
                through to every ``record_metrics`` call — the repository writes
                the interval into the matching metric row's existing
                ``ci_lower``/``ci_upper`` columns. In practice only the holdout
                headline (a single point) passes this, for the bootstrap
                ``calibration_slope`` CI (B2); trend recordings omit it.
        """
        if split_version is not None:
            raise NotImplementedError(
                "split_version isolation requires writing it to row metadata (P2); pass None"
            )

        # Step 1 — resolve model handle → uuid ONCE
        model_id = await _resolve_model_id(self.repo.client, model_version)
        if model_id is None:
            # Fail-closed: an unresolved handle means the cohort model is not
            # registered. Writing metrics against a NULL model_id would create
            # rows the time-series read path (which resolves by model_id) cannot
            # find, AND break idempotency (the delete scope below is keyed by
            # model_id). Surface the misconfiguration instead.
            raise ValueError(
                f"Cannot record metrics: model handle {model_version!r} did not "
                "resolve to a registered model_id (register the cohort model first)."
            )

        # Step 2 — delete prior rows (idempotency); fires BEFORE any insert
        deleted = await self.repo.delete_metrics(model_id, source, split_version)
        logger.info(
            "MetricRecorder: deleted %d prior row(s) for model_id=%r source=%r split_version=%r",
            deleted,
            model_id,
            source,
            split_version,
        )

        # Step 3 — insert one record_metrics call per point
        for measured_at_month, metrics, sample_size in points:
            await self.repo.record_metrics(
                model_version,
                metrics,
                sample_size,
                measured_at_month,  # window_start = month boundary
                measured_at_month,  # window_end   = same (month-grain; caller refines if needed)
                measured_at=measured_at_month,
                source=source,
                cis=cis,
            )

    async def record_curves(
        self,
        model_version: str,
        curves: list[tuple[str, float, dict]],
        *,
        measured_at: datetime,
        sample_size: int,
        source: str = HOLDOUT_CURVE_SOURCE,
    ) -> None:
        """Idempotently persist structured eval artifacts (confusion / ROC).

        Like ``record_run`` this delete-then-inserts, but under its OWN ``source``
        tag (default ``'holdout_curve'``) so it NEVER clobbers the scalar holdout
        rows (``source='holdout'``) that ``record_run`` writes — the two share a
        model but not a source, so each delete scope is disjoint.

        Args:
            model_version: Model handle/uuid passed to the repository.
            curves: Sequence of ``(kind, scalar_value, payload)`` where ``kind``
                is ``'confusion_matrix'`` / ``'roc_curve'``, ``scalar_value`` is a
                representative metric (accuracy / auc) and ``payload`` is the
                structured dict stored in the row's ``metadata``.
            measured_at: Timestamp for the artifact (the holdout data boundary).
            sample_size: Holdout sample size.
            source: Disjoint source tag for the delete scope (default
                ``'holdout_curve'``).
        """
        model_id = await _resolve_model_id(self.repo.client, model_version)
        if model_id is None:
            # Mirror record_run's fail-closed: an unresolved handle means the
            # cohort model is not registered; a NULL model_id would be unreadable.
            raise ValueError(
                f"Cannot record curves: model handle {model_version!r} did not "
                "resolve to a registered model_id (register the cohort model first)."
            )

        # Idempotency: clear prior curve rows for this (model_id, source) first.
        await self.repo.delete_metrics(model_id, source, None)

        for kind, value, payload in curves:
            await self.repo.record_curve(
                model_version,
                kind,
                value,
                payload,
                sample_size,
                measured_at,  # window_start
                measured_at,  # window_end
                measured_at=measured_at,
                source=source,
            )
